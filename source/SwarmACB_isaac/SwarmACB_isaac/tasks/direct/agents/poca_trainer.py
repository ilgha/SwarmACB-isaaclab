# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""POCA Trainer — faithful reproduction of ML-Agents TorchPOCAOptimizer + POCATrainer.

Key design matching ML-Agents
──────────────────────────────
1. Counterfactual baselines via critic.baseline()
2. Lambda-return advantage: advantage = lambda_return − baseline (NOT value)
3. Per-dimension PPO ratio and clipping (ML-Agents .flatten() → per-dim ratio)
4. Loss = policy_loss + 0.5 * (value_loss + 0.5 * baseline_loss) − β * entropy
5. Trust-region clipping on BOTH value and baseline
6. No gradient clipping (ML-Agents doesn't clip gradients for POCA)
7. Constant schedules for lr, ε, β (matching PushBlockCollab.yaml)

Reference:
    ml-agents/mlagents/trainers/poca/optimizer_torch.py
    ml-agents/mlagents/trainers/poca/trainer.py
    ml-agents/mlagents/trainers/torch_entities/utils.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .poca_networks import Actor, DiscreteActor, POCACritic
from .poca_buffer import POCARolloutBuffer


# ──────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class POCAConfig:
    """Training hyper-parameters — loadable from ML-Agents-style YAML."""

    # Rollout
    horizon: int = 1000              # time_horizon in ML-Agents
    num_epochs: int = 3              # num_epoch
    mini_batch_size: int = 2048      # batch_size in ML-Agents

    # PPO / POCA
    clip_eps: float = 0.2            # epsilon
    beta: float = 0.005              # entropy coefficient

    # GAE / lambda-return
    gamma: float = 0.99
    lam: float = 0.95                # lambd in ML-Agents

    # Optimiser
    lr: float = 3e-4
    adam_eps: float = 1e-7

    # Schedules: "linear" or "constant"
    lr_schedule: str = "constant"
    eps_schedule: str = "constant"
    beta_schedule: str = "constant"

    # Run control
    total_timesteps: int = 120_000_000  # max_steps (agent-decisions)
    checkpoint_interval: int = 120_000  # save every N agent-decisions
    summary_freq: int = 120_000         # TensorBoard log every N agent-decisions
    keep_checkpoints: int = 5
    checkpoint_dir: str = "checkpoints/poca"

    # Decision period
    decision_period: int = 1

    # Reward
    reward_strength: float = 1.0     # multiplier on extrinsic reward

    # Network
    hidden_dim: int = 512
    num_layers: int = 2
    critic_num_heads: int = 4

    # TensorBoard
    log_dir: str = "runs/poca"

    # buffer_size hint from YAML (informational only)
    buffer_size_hint: int = 0

    # Legacy aliases used by old CLI (kept for backward compat)
    @property
    def log_interval(self) -> int:
        """Approximate number of updates between logs."""
        return 10

    @property
    def save_interval(self) -> int:
        """Approximate number of updates between saves."""
        return 50


# ──────────────────────────────────────────────────────────────────────
#  Schedule helpers  (kept for optional use with other configs)
# ──────────────────────────────────────────────────────────────────────

class LinearDecay:
    """Linearly decay a value from *initial* to 0 over *total_steps*.

    NOTE: PushBlockCollab.yaml uses constant schedules, so this is unused
    unless explicitly configured.
    """

    def __init__(self, initial: float, total_steps: int):
        self.initial = initial
        self.total_steps = max(total_steps, 1)

    def get(self, step: int) -> float:
        return self.initial * max(1.0 - step / self.total_steps, 0.0)


# ──────────────────────────────────────────────────────────────────────
#  Trust-region loss functions  (matching ML-Agents ModelUtils)
# ──────────────────────────────────────────────────────────────────────

def trust_region_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Clipped value loss matching ML-Agents trust_region_value_loss.

    All inputs: (MB,) or broadcastable.
    """
    clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
    loss_a = (returns - values) ** 2
    loss_b = (returns - clipped) ** 2
    return torch.max(loss_a, loss_b).mean()


def trust_region_policy_loss(
    advantages: torch.Tensor,    # (MB, act_dim) — broadcast from (MB, 1)
    log_probs: torch.Tensor,     # (MB, act_dim) — per-dim
    old_log_probs: torch.Tensor, # (MB, act_dim) — per-dim
    epsilon: float,
) -> torch.Tensor:
    """Clipped policy loss matching ML-Agents trust_region_policy_loss.

    ML-Agents computes the ratio and clips PER ACTION DIMENSION:
        r_theta = exp(log_probs - old_log_probs)    shape: (MB, act_dim)
        advantage is broadcast: (MB, 1) → (MB, act_dim)
        loss = -min(r * adv, clip(r) * adv)

    This is DIFFERENT from standard PPO which sums log_probs first.
    """
    r_theta = (log_probs - old_log_probs).exp()
    p_opt_a = r_theta * advantages
    p_opt_b = r_theta.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages
    return -torch.min(p_opt_a, p_opt_b).mean()


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class POCATrainer:
    """End-to-end POCA training loop."""

    def __init__(self, env, cfg: POCAConfig | None = None):
        self.env = env
        self.cfg = cfg or POCAConfig()
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device

        # ── environment dimensions ────────────────────────────────
        self.num_envs = self.unwrapped.scene.num_envs
        cfg_env = self.unwrapped.cfg
        self.num_agents = getattr(cfg_env, "num_agents",
                                  getattr(cfg_env, "num_robots", None))

        # Detect discrete vs continuous action mode
        self.discrete = getattr(cfg_env, "discrete_actions", False)
        self.num_actions = getattr(cfg_env, "num_actions", 7)  # for discrete

        # Support grid observation: flatten if needed
        sample_obs = self.env.reset()[0][self.unwrapped.cfg.possible_agents[0]]
        if sample_obs.ndim == 4:
            self.obs_dim = int(sample_obs.shape[1] * sample_obs.shape[2] * sample_obs.shape[3])
        else:
            self.obs_dim = sample_obs.shape[1]

        if self.discrete:
            # Discrete: actions stored as (E, N, 1) integers
            # Critic uses one-hot encoded actions → act_dim_critic = num_actions
            self.act_dim = 1                       # storage dimension
            self.act_dim_critic = self.num_actions  # critic entity embedding
        else:
            # Continuous: act_dim used everywhere
            first_agent = self.unwrapped.cfg.possible_agents[0]
            self.act_dim = cfg_env.action_spaces[first_agent]
            self.act_dim_critic = self.act_dim

        # ── networks ──────────────────────────────────────────────
        c = self.cfg
        self.decision_period = c.decision_period

        print(f"[POCA] envs={self.num_envs}  agents={self.num_agents}  "
              f"obs={self.obs_dim}  act={'discrete(' + str(self.num_actions) + ')' if self.discrete else str(self.act_dim)}  "
              f"decision_period={self.decision_period}")

        if self.discrete:
            self.actor = DiscreteActor(
                self.obs_dim, self.num_actions, c.hidden_dim, c.num_layers,
            ).to(self.device)
        else:
            self.actor = Actor(
                self.obs_dim, self.act_dim, c.hidden_dim, c.num_layers,
            ).to(self.device)

        self.critic = POCACritic(
            self.obs_dim, self.act_dim_critic, self.num_agents,
            c.hidden_dim, c.critic_num_heads, c.num_layers,
        ).to(self.device)

        # ── single optimiser (actor + critic) — matches ML-Agents ─
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=c.lr, eps=c.adam_eps,
        )

        # ── schedule support ─────────────────────────────────────
        self._init_lr = c.lr
        self._init_eps = c.clip_eps
        self._init_beta = c.beta

        self.lr_schedule = LinearDecay(c.lr, c.total_timesteps) if c.lr_schedule == "linear" else None
        self.eps_schedule = LinearDecay(c.clip_eps, c.total_timesteps) if c.eps_schedule == "linear" else None
        self.beta_schedule = LinearDecay(c.beta, c.total_timesteps) if c.beta_schedule == "linear" else None

        self.current_lr = c.lr
        self.current_eps = c.clip_eps
        self.current_beta = c.beta

        # ── reward strength ────────────────────────────────────────
        self.reward_strength = c.reward_strength

        # ── step-based checkpoint / log tracking ──────────────────
        self._next_checkpoint_step = c.checkpoint_interval
        self._next_summary_step = c.summary_freq

        # ── rollout buffer ────────────────────────────────────────
        self.buffer = POCARolloutBuffer(
            horizon=c.horizon,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            gamma=c.gamma,
            lam=c.lam,
            device=self.device,
        )

        # ── logging / bookkeeping ─────────────────────────────────
        self.global_step = 0
        self.update_count = 0
        self.writer = SummaryWriter(log_dir=c.log_dir)
        hp_text = "\n".join(f"{k}: {v}" for k, v in vars(c).items())
        self.writer.add_text("hyperparameters", hp_text, 0)

        # episode tracking
        self._episode_reward_acc = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._completed_episode_returns: list[float] = []
        self._completed_episode_lengths: list[float] = []
        self._completed_group_rewards: list[float] = []
        # Rolling reward window (for logging even when no episodes complete)
        self._rollout_reward_history: list[float] = []
        self._max_history = 100  # keep last N rollout rewards

        # Print param count
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"[POCA] Actor params: {actor_params:,}  Critic params: {critic_params:,}")
        print(f"[POCA] TensorBoard → {c.log_dir}")

    # ──────────────────────────────────────────────────────────────
    #  Action encoding helper
    # ──────────────────────────────────────────────────────────────

    def _encode_actions_for_critic(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions for critic input.

        Discrete:   one-hot encode (*, N, 1) int → (*, N, num_actions) float.
        Continuous: pass through (*, N, act_dim) float unchanged.
        """
        if self.discrete:
            # actions: (*, N, 1) long — squeeze last dim then one-hot
            act_idx = actions.squeeze(-1).long()         # (*, N)
            return torch.nn.functional.one_hot(
                act_idx, self.num_actions,
            ).float()                                     # (*, N, num_actions)
        else:
            return actions

    # ──────────────────────────────────────────────────────────────
    #  Schedule helpers
    # ──────────────────────────────────────────────────────────────

    def _apply_schedules(self):
        """Update lr / epsilon / beta according to their schedules."""
        step = self.global_step
        if self.lr_schedule is not None:
            self.current_lr = self.lr_schedule.get(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr
        if self.eps_schedule is not None:
            self.current_eps = self.eps_schedule.get(step)
        if self.beta_schedule is not None:
            self.current_beta = self.beta_schedule.get(step)

    # ──────────────────────────────────────────────────────────────
    #  Rollout collection
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_rollout(self, obs_dict: dict) -> dict:
        """Run *horizon* decisions, each stepping the env *decision_period* times.

        This matches ML-Agents' DecisionRequester behaviour:
        - Agent makes ONE decision (samples action from policy)
        - Environment is stepped decision_period times with that action
        - Velocity is set ONCE on the first step; physics runs naturally after
        - Rewards are accumulated over all sub-steps
        - ONE transition is recorded in the buffer per decision

        Without this, the buffer would store freshly-sampled actions that
        the env never executed (it was using cached actions from 5 steps ago),
        breaking the action↔reward correspondence that PPO requires.
        """
        self.buffer.reset()
        agents = self.unwrapped.cfg.possible_agents
        dp = self.decision_period

        for _ in range(self.cfg.horizon):
            # ── stack observations (taken BEFORE the decision) ────
            obs_stacked = torch.stack(
                [obs_dict[a] for a in agents], dim=1
            )  # (E, N, ...)
            # Flatten grid if needed
            if obs_stacked.ndim == 5:
                # (E, N, C, H, W) -> (E, N, C*H*W)
                obs_stacked = obs_stacked.view(obs_stacked.shape[0], obs_stacked.shape[1], -1)

            # ── sample actions from shared actor (batched over all agents) ──
            # Reshape (E, N, obs) → (E*N, obs) for a SINGLE forward pass
            flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])  # (E*N, obs)
            dist = self.actor.get_dist(flat_obs)

            if self.discrete:
                flat_act = dist.sample()                       # (E*N,)
                flat_logp = dist.log_prob(flat_act)            # (E*N,)
                all_actions = flat_act.view(self.num_envs, self.num_agents, 1)
                all_log_probs = flat_logp.view(self.num_envs, self.num_agents, 1)
            else:
                flat_act = dist.sample()                       # (E*N, act_dim)
                flat_logp = dist.log_prob(flat_act)            # (E*N, act_dim)
                all_actions = flat_act.view(self.num_envs, self.num_agents, self.act_dim)
                all_log_probs = flat_logp.view(self.num_envs, self.num_agents, self.act_dim)

            # ── critic: team value V(s) — obs only ────────────────
            team_val = self.critic.critic_pass(obs_stacked).squeeze(-1)  # (E,)

            # ── baselines: counterfactual b_i ─────────────────────
            critic_actions = self._encode_actions_for_critic(all_actions)
            baselines = self.critic.all_baselines(obs_stacked, critic_actions)  # (E, N)

            # ── step environment decision_period times ────────────
            # Same action for all sub-steps; env applies velocity
            # only on the first sub-step (decision step), then coasts.
            action_dict = {a: all_actions[:, i] for i, a in enumerate(agents)}
            accumulated_reward = torch.zeros(self.num_envs, device=self.device)
            last_done = torch.zeros(self.num_envs, device=self.device)

            for _dp in range(dp):
                obs_dict, rewards_dict, terminated_dict, truncated_dict, info = (
                    self.env.step(action_dict)
                )
                accumulated_reward += rewards_dict[agents[0]]
                step_done = (terminated_dict[agents[0]] | truncated_dict[agents[0]]).float()
                last_done = torch.max(last_done, step_done)

            # ── store ONE transition per decision ─────────────────
            self.buffer.add(
                obs=obs_stacked,
                actions=all_actions,
                log_probs=all_log_probs,  # per-dim!
                reward=accumulated_reward * self.reward_strength,
                done=last_done,
                team_value=team_val,
                baselines=baselines,
            )

            # ── episode reward tracking ───────────────────────────
            self._episode_reward_acc += accumulated_reward
            self._episode_step_count += dp
            done_mask = last_done.bool()
            if done_mask.any():
                self._completed_episode_returns.extend(
                    self._episode_reward_acc[done_mask].tolist()
                )
                self._completed_episode_lengths.extend(
                    self._episode_step_count[done_mask].tolist()
                )
                # Group reward (block scoring only) — read the snapshot saved
                # before auto-reset zeroed episode_group_reward
                self._completed_group_rewards.extend(
                    self.unwrapped.completed_group_reward[done_mask].tolist()
                )
                self._episode_reward_acc[done_mask] = 0.0
                self._episode_step_count[done_mask] = 0.0

            # Count agent-decisions (matching ML-Agents max_steps)
            self.global_step += self.num_envs * self.num_agents

        # ── bootstrap V for lambda-return ─────────────────────────
        obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
        if obs_stacked.ndim == 5:
            obs_stacked = obs_stacked.view(obs_stacked.shape[0], obs_stacked.shape[1], -1)
        last_tv = self.critic.critic_pass(obs_stacked).squeeze(-1)
        self.buffer.compute_returns_and_advantages(last_tv)

        return obs_dict

    # ──────────────────────────────────────────────────────────────
    #  PPO / POCA update
    # ──────────────────────────────────────────────────────────────

    def update(self) -> dict:
        """Run *num_epochs* PPO update epochs over the buffer."""
        cfg = self.cfg

        # Apply schedules
        self._apply_schedules()
        current_eps = self.current_eps
        current_beta = self.current_beta

        total_pol = 0.0
        total_val = 0.0
        total_bl = 0.0
        total_ent = 0.0
        n_updates = 0

        for _epoch in range(cfg.num_epochs):
            for batch in self.buffer.get_batches(cfg.mini_batch_size):
                obs = batch["obs"]                  # (MB, N, obs)
                actions = batch["actions"]          # (MB, N, act)
                old_logp = batch["old_log_probs"]   # (MB, N, act_dim) per-dim!
                advantages = batch["advantages"]    # (MB, N)
                returns = batch["returns"]          # (MB,)
                old_tv = batch["old_team_values"]   # (MB,)
                old_bl = batch["old_baselines"]     # (MB, N)

                MB, N = obs.shape[:2]

                # ── policy loss (batched over all agents, shared actor) ─
                # Reshape (MB, N, obs) → (MB*N, obs) for single forward pass
                flat_obs = obs.reshape(-1, obs.shape[-1])              # (MB*N, obs)
                flat_act = actions.reshape(-1, actions.shape[-1])      # (MB*N, act_dim)
                flat_logp, flat_ent = self.actor.evaluate(flat_obs, flat_act)
                # flat_logp: (MB*N, act_dim), flat_ent: (MB*N,)

                # Reshape back to (MB, N, act_dim) and (MB, N)
                new_logp_all = flat_logp.view(MB, N, -1)              # (MB, N, act_dim)
                ent_all = flat_ent.view(MB, N)                        # (MB, N)

                # Per-dim advantage broadcast: (MB, N) → (MB, N, 1)
                adv_expanded = advantages.unsqueeze(-1)               # (MB, N, 1)

                # Per-dim trust-region policy loss (vectorized over agents)
                # Flatten agent dim into batch: (MB*N, act_dim)
                flat_adv = adv_expanded.reshape(-1, 1)
                flat_new_logp = new_logp_all.reshape(-1, new_logp_all.shape[-1])
                flat_old_logp = old_logp.reshape(-1, old_logp.shape[-1])
                policy_loss = trust_region_policy_loss(
                    flat_adv, flat_new_logp, flat_old_logp, current_eps,
                )
                mean_entropy = ent_all.mean()

                # ── critic: recompute V and baselines ─────────────
                new_tv = self.critic.critic_pass(obs).squeeze(-1)       # (MB,)
                critic_act = self._encode_actions_for_critic(actions)
                new_bl = self.critic.all_baselines(obs, critic_act)     # (MB, N)

                # ── value loss (trust-region clipped) ─────────────
                value_loss = trust_region_value_loss(
                    new_tv, old_tv, returns, current_eps,
                )

                # ── baseline loss (trust-region clipped) ──────────
                # Returns are broadcast to per-agent
                ret_expanded = returns.unsqueeze(-1).expand_as(new_bl)
                old_bl_flat = old_bl.reshape(-1)
                new_bl_flat = new_bl.reshape(-1)
                ret_flat = ret_expanded.reshape(-1)
                baseline_loss = trust_region_value_loss(
                    new_bl_flat, old_bl_flat, ret_flat, current_eps,
                )

                # ── total loss  (matches ML-Agents exactly) ──────
                # loss = policy + 0.5*(value + 0.5*baseline) − β*entropy
                loss = (
                    policy_loss
                    + 0.5 * (value_loss + 0.5 * baseline_loss)
                    - current_beta * mean_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                # NOTE: ML-Agents does NOT clip gradients for POCA
                self.optimizer.step()

                total_pol += policy_loss.item()
                total_val += value_loss.item()
                total_bl += baseline_loss.item()
                total_ent += mean_entropy.item()
                n_updates += 1

        self.update_count += 1
        n = max(n_updates, 1)
        return {
            "policy_loss": total_pol / n,
            "value_loss": total_val / n,
            "baseline_loss": total_bl / n,
            "entropy": total_ent / n,
            "lr": self.current_lr,
            "eps": self.current_eps,
            "beta": self.current_beta,
        }

    # ──────────────────────────────────────────────────────────────
    #  Main training loop
    # ──────────────────────────────────────────────────────────────

    def train(self):
        start_time = time.time()
        obs_dict, _ = self.env.reset()

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(
            total=self.cfg.total_timesteps,
            initial=self.global_step,
            desc="POCA Training",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        while self.global_step < self.cfg.total_timesteps:
            prev_step = self.global_step

            # 1. collect rollout
            obs_dict = self.collect_rollout(obs_dict)

            # 2. update
            metrics = self.update()

            # 3. progress bar
            step_delta = self.global_step - prev_step
            elapsed = time.time() - start_time
            sps = self.global_step / elapsed if elapsed > 0 else 0

            pbar.update(step_delta)
            pbar.set_postfix(
                upd=self.update_count,
                pg=f"{metrics['policy_loss']:.3f}",
                vf=f"{metrics['value_loss']:.3f}",
                bl=f"{metrics['baseline_loss']:.3f}",
                ent=f"{metrics['entropy']:.3f}",
                SPS=f"{sps:.0f}",
            )

            # 4. Accumulate rollout reward history (always, for rolling avg)
            mean_rollout_reward = (
                self.buffer.rewards.sum(dim=0).mean().item()
            )
            self._rollout_reward_history.append(mean_rollout_reward)
            if len(self._rollout_reward_history) > self._max_history:
                self._rollout_reward_history.pop(0)

            # 4b. TensorBoard — gated by summary_freq
            if self.global_step >= self._next_summary_step:
                self._next_summary_step += self.cfg.summary_freq
                s = self.global_step

                # ── Losses & schedule params ──────────────────────
                self.writer.add_scalar(
                    "losses/policy", metrics["policy_loss"], s)
                self.writer.add_scalar(
                    "losses/value", metrics["value_loss"], s)
                self.writer.add_scalar(
                    "losses/baseline",
                    metrics["baseline_loss"], s)
                self.writer.add_scalar(
                    "losses/entropy", metrics["entropy"], s)
                self.writer.add_scalar(
                    "charts/learning_rate", metrics["lr"], s)
                self.writer.add_scalar(
                    "charts/epsilon", metrics["eps"], s)
                self.writer.add_scalar(
                    "charts/beta", metrics["beta"], s)
                self.writer.add_scalar("charts/SPS", sps, s)

                # ── Reward: rollout-level ─────────────────────────
                self.writer.add_scalar(
                    "reward/mean_rollout",
                    mean_rollout_reward, s)
                rolling_avg = (
                    sum(self._rollout_reward_history)
                    / len(self._rollout_reward_history)
                )
                self.writer.add_scalar(
                    "reward/rolling_avg_rollout", rolling_avg, s)

                # ── Reward: in-progress episodes ──────────────────
                self.writer.add_scalar(
                    "reward/mean_in_progress",
                    self._episode_reward_acc.mean().item(), s)
                self.writer.add_scalar(
                    "reward/max_in_progress",
                    self._episode_reward_acc.max().item(), s)

                # ── Reward: completed episodes ────────────────────
                if self._completed_episode_returns:
                    ep = self._completed_episode_returns
                    self.writer.add_scalar(
                        "reward/mean_episode",
                        sum(ep) / len(ep), s)
                    self.writer.add_scalar(
                        "reward/max_episode", max(ep), s)
                    self.writer.add_scalar(
                        "reward/min_episode", min(ep), s)
                    self.writer.add_scalar(
                        "reward/num_episodes", len(ep), s)
                    self._completed_episode_returns.clear()

                # ── Group reward (gate crossings) ─────────────────
                if self._completed_group_rewards:
                    gr = self._completed_group_rewards
                    self.writer.add_scalar(
                        "group_reward/mean_episode",
                        sum(gr) / len(gr), s)
                    self.writer.add_scalar(
                        "group_reward/max_episode", max(gr), s)
                    self.writer.add_scalar(
                        "group_reward/min_episode", min(gr), s)
                    self._completed_group_rewards.clear()

                # ── Episode length ────────────────────────────────
                if self._completed_episode_lengths:
                    el = self._completed_episode_lengths
                    self.writer.add_scalar(
                        "episode/mean_length",
                        sum(el) / len(el), s)
                    self._completed_episode_lengths.clear()

                # ── Critic / advantage diagnostics ────────────────
                self.writer.add_scalar(
                    "charts/mean_value",
                    self.buffer.team_values.mean().item(), s)
                self.writer.add_scalar(
                    "charts/mean_abs_advantage",
                    self.buffer.advantages.abs().mean().item(), s)

            # 5. checkpoint (step-based, matching ML-Agents)
            if self.global_step >= self._next_checkpoint_step:
                self.save_checkpoint(
                    ckpt_dir / f"poca_{self.global_step}.pt",
                )
                self._next_checkpoint_step += (
                    self.cfg.checkpoint_interval
                )
                self._manage_checkpoints(ckpt_dir)

        pbar.close()
        self.writer.close()
        self.save_checkpoint(ckpt_dir / "poca_final.pt")
        elapsed = time.time() - start_time
        print(f"[POCA] Done — {self.global_step:,} steps in {elapsed:.0f}s "
              f"({self.global_step / elapsed:.0f} SPS)")

    # ──────────────────────────────────────────────────────────────
    #  Checkpointing
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            # Save architecture for correct restoration in play
            "hidden_dim": getattr(self.cfg, "hidden_dim", 256),
            "num_layers": getattr(self.cfg, "num_layers", 2),
            "discrete": self.discrete,
            "num_actions": self.num_actions if self.discrete else 0,
            "act_dim": self.act_dim,
        }, path)
        print(f"[POCA] Saved → {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        print(f"[POCA] Loaded ← {path}  (step {self.global_step})")

    def _manage_checkpoints(self, ckpt_dir: Path):
        """Keep only the *keep_checkpoints* most recent numbered checkpoints."""
        keep = self.cfg.keep_checkpoints
        if keep <= 0:
            return  # keep all
        numbered = sorted(
            ckpt_dir.glob("poca_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        # Exclude poca_final.pt from rotation
        numbered = [p for p in numbered if p.stem != "poca_final"]
        while len(numbered) > keep:
            old = numbered.pop(0)
            old.unlink()
            print(f"[POCA] Removed old checkpoint → {old.name}")
