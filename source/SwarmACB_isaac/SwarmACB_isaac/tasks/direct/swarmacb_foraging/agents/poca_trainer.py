# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""POCA Trainer — custom PPO-style training loop for cooperative multi-agent RL.

Orchestrates:
    1. Rollout collection from the Isaac Lab environment.
    2. GAE advantage computation using counterfactual baselines.
    3. Clipped-PPO policy updates (shared actor).
    4. Centralised critic updates with value-clipping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from .poca_networks import Actor, POCACritic
from .poca_buffer import POCARolloutBuffer


# ──────────────────────────────────────────────────────────────────────
#  Training hyperparameters
# ──────────────────────────────────────────────────────────────────────

@dataclass
class POCAConfig:
    """All training hyper-parameters in one place."""

    # Rollout
    horizon: int = 256            # steps per rollout before update
    num_epochs: int = 4           # PPO epochs per rollout
    mini_batch_size: int = 512    # mini-batch size (flattened T*E samples)

    # PPO
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # Optimiser
    lr: float = 3e-4
    lr_schedule: str = "linear"   # "linear" or "constant"

    # Run control
    total_timesteps: int = 10_000_000
    log_interval: int = 10        # print every N updates
    save_interval: int = 50       # save checkpoint every N updates
    checkpoint_dir: str = "checkpoints/poca"

    # Network
    hidden_dim: int = 128
    critic_embed_dim: int = 128
    critic_num_heads: int = 4


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class POCATrainer:
    """End-to-end POCA training loop."""

    def __init__(
        self,
        env,                       # Isaac Lab MARL environment (gymnasium API)
        cfg: POCAConfig | None = None,
    ):
        self.env = env
        self.cfg = cfg or POCAConfig()
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device

        # Environment dimensions
        self.num_envs = self.unwrapped.scene.num_envs
        self.num_agents = self.unwrapped.cfg.num_robots
        self.obs_dim = self.unwrapped.obs_size
        self.act_dim = 2  # [v_left, v_right]
        self.state_dim = self.unwrapped.cfg.state_space

        print(f"[POCA] num_envs={self.num_envs}, agents={self.num_agents}, "
              f"obs={self.obs_dim}, act={self.act_dim}, state={self.state_dim}")

        # ---- Networks ----
        self.actor = Actor(self.obs_dim, self.act_dim, self.cfg.hidden_dim).to(self.device)
        self.critic = POCACritic(
            self.obs_dim, self.state_dim, self.num_agents,
            self.cfg.critic_embed_dim, self.cfg.critic_num_heads,
        ).to(self.device)

        # ---- Optimiser (single optimiser for both networks) ----
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr,
            eps=1e-5,
        )

        # ---- Rollout buffer ----
        self.buffer = POCARolloutBuffer(
            horizon=self.cfg.horizon,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
            device=self.device,
        )

        # ---- Logging ----
        self.global_step = 0
        self.update_count = 0
        self.episode_rewards: list[float] = []

    # ──────────────────────────────────────────────────────────────
    #  Rollout collection
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_rollout(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the environment for *horizon* steps, storing into the buffer.

        Args:
            obs_dict: initial observation dict from env.reset() or last step.

        Returns:
            obs_dict at the end of the rollout (for bootstrapping).
        """
        self.buffer.reset()
        agents = self.unwrapped.cfg.possible_agents

        for t in range(self.cfg.horizon):
            # Stack agent observations → (E, N, obs_dim)
            obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)

            # Global state from critic
            state = self.unwrapped._get_states()  # (E, state_dim)

            # Actor: sample actions for each agent (shared weights)
            all_actions = torch.zeros(self.num_envs, self.num_agents, self.act_dim, device=self.device)
            all_log_probs = torch.zeros(self.num_envs, self.num_agents, device=self.device)

            for i, a in enumerate(agents):
                dist = self.actor.get_dist(obs_stacked[:, i])  # per-agent
                act = dist.sample()
                act = act.clamp(-1.0, 1.0)
                log_p = dist.log_prob(act).sum(dim=-1)
                all_actions[:, i] = act
                all_log_probs[:, i] = log_p

            # Critic: team value + counterfactual baselines
            team_val, baselines = self.critic(obs_stacked, state)
            team_val = team_val.squeeze(-1)  # (E,)

            # Step environment
            action_dict = {a: all_actions[:, i] for i, a in enumerate(agents)}
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info = self.env.step(action_dict)

            # Team reward (all agents share same reward)
            reward = rewards_dict[agents[0]]  # (E,)
            done = (terminated_dict[agents[0]] | truncated_dict[agents[0]]).float()

            # Store
            self.buffer.add(
                obs=obs_stacked,
                actions=all_actions,
                log_probs=all_log_probs,
                reward=reward,
                done=done,
                state=state,
                team_value=team_val,
                agent_baselines=baselines,
            )

            self.global_step += self.num_envs

        # ---- Bootstrap values for GAE ----
        obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
        state = self.unwrapped._get_states()
        last_tv, last_bl = self.critic(obs_stacked, state)
        self.buffer.compute_advantages(last_tv.squeeze(-1), last_bl)

        return obs_dict

    # ──────────────────────────────────────────────────────────────
    #  PPO update
    # ──────────────────────────────────────────────────────────────

    def update(self):
        """Run multiple PPO epochs over the collected rollout."""
        cfg = self.cfg
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(cfg.num_epochs):
            for batch in self.buffer.get_batches(cfg.mini_batch_size):
                obs = batch["obs"]                    # (MB, N, obs)
                actions = batch["actions"]            # (MB, N, act)
                old_logp = batch["old_log_probs"]     # (MB, N)
                advantages = batch["advantages"]      # (MB, N)
                returns = batch["returns"]            # (MB, N)
                states = batch["states"]              # (MB, state)
                old_tv = batch["old_team_values"]     # (MB,)

                MB, N = obs.shape[:2]

                # Normalise advantages (per-agent, across batch)
                adv_mean = advantages.mean()
                adv_std = advantages.std().clamp(min=1e-8)
                advantages = (advantages - adv_mean) / adv_std

                # ---- Actor loss (per-agent, shared weights) ----
                pg_loss = torch.zeros(1, device=self.device)
                ent_loss = torch.zeros(1, device=self.device)

                for i in range(N):
                    new_logp, entropy = self.actor.evaluate(obs[:, i], actions[:, i])
                    ratio = (new_logp - old_logp[:, i]).exp()
                    adv_i = advantages[:, i]

                    surr1 = ratio * adv_i
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_i
                    pg_loss += -torch.min(surr1, surr2).mean()
                    ent_loss += -entropy.mean()

                pg_loss /= N
                ent_loss /= N

                # ---- Critic loss ----
                team_val, baselines = self.critic(obs, states)
                team_val = team_val.squeeze(-1)  # (MB,)

                # Value clipping
                val_clipped = old_tv + (team_val - old_tv).clamp(-cfg.clip_eps, cfg.clip_eps)
                vf_loss1 = (team_val - returns.mean(dim=-1)) ** 2
                vf_loss2 = (val_clipped - returns.mean(dim=-1)) ** 2
                vf_loss = torch.max(vf_loss1, vf_loss2).mean()

                # Baseline loss (per-agent)
                bl_loss = ((baselines - returns) ** 2).mean()

                # ---- Total loss ----
                loss = pg_loss + cfg.entropy_coef * ent_loss + cfg.value_coef * (vf_loss + bl_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += (vf_loss.item() + bl_loss.item())
                total_entropy += -ent_loss.item()
                n_updates += 1

        self.update_count += 1

        return {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ──────────────────────────────────────────────────────────────
    #  LR schedule
    # ──────────────────────────────────────────────────────────────

    def _update_lr(self):
        if self.cfg.lr_schedule == "linear":
            frac = 1.0 - self.global_step / self.cfg.total_timesteps
            frac = max(frac, 0.0)
            new_lr = self.cfg.lr * frac
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

    # ──────────────────────────────────────────────────────────────
    #  Main loop
    # ──────────────────────────────────────────────────────────────

    def train(self):
        """Full training loop."""
        print(f"[POCA] Starting training — {self.cfg.total_timesteps:,} total steps")
        start_time = time.time()

        # Initial reset
        obs_dict, _ = self.env.reset()

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        while self.global_step < self.cfg.total_timesteps:
            # 1. Collect rollout
            obs_dict = self.collect_rollout(obs_dict)

            # 2. PPO update
            self._update_lr()
            metrics = self.update()

            # 3. Logging
            if self.update_count % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed if elapsed > 0 else 0
                print(
                    f"[POCA] update={self.update_count:>5d}  "
                    f"steps={self.global_step:>10,}  "
                    f"pg_loss={metrics['pg_loss']:.4f}  "
                    f"vf_loss={metrics['vf_loss']:.4f}  "
                    f"entropy={metrics['entropy']:.4f}  "
                    f"SPS={sps:.0f}"
                )

            # 4. Checkpointing
            if self.update_count % self.cfg.save_interval == 0:
                self.save_checkpoint(ckpt_dir / f"poca_{self.global_step}.pt")

        # Final save
        self.save_checkpoint(ckpt_dir / "poca_final.pt")
        print(f"[POCA] Training complete — {self.global_step:,} steps in {time.time() - start_time:.0f}s")

    # ──────────────────────────────────────────────────────────────
    #  Checkpoint helpers
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
        }, path)
        print(f"[POCA] Saved checkpoint → {path}")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        print(f"[POCA] Loaded checkpoint ← {path}  (step {self.global_step})")
