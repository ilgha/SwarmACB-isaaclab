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

from .poca_networks import Actor, DiscreteActor, RecurrentDiscreteActor, POCACritic
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
    adam_eps: float = 1e-8           # ML-Agents uses PyTorch default (1e-8)

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
    seed: int = 0

    # Decision period
    decision_period: int = 1

    # Reward
    reward_strength: float = 1.0     # multiplier on extrinsic reward

    # Network
    hidden_dim: int = 512
    num_layers: int = 2
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2
    critic_num_heads: int = 4
    recurrent: bool = False
    memory_size: int = 128
    sequence_length: int = 128

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

class PolynomialDecay:
    """Polynomial (default=linear) decay matching ML-Agents ModelUtils.polynomial_decay.

    Decays from *initial_value* to *min_value* over *max_step*.
    ML-Agents uses non-zero min values:
        lr      → 1e-10
        epsilon → 0.1
        beta    → 1e-5
    """

    def __init__(self, initial: float, min_value: float, max_step: int, power: float = 1.0):
        self.initial = initial
        self.min_value = min_value
        self.max_step = max(max_step, 1)
        self.power = power

    def get(self, step: int) -> float:
        step = min(step, self.max_step)
        return (self.initial - self.min_value) * (
            1.0 - step / self.max_step
        ) ** self.power + self.min_value


# ──────────────────────────────────────────────────────────────────────
#  Trust-region loss functions  (matching ML-Agents ModelUtils)
# ──────────────────────────────────────────────────────────────────────

def trust_region_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    epsilon: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Clipped value loss matching ML-Agents trust_region_value_loss.

    All inputs: (MB,) or broadcastable.
    """
    clipped = old_values + (values - old_values).clamp(-epsilon, epsilon)
    loss_a = (returns - values) ** 2
    loss_b = (returns - clipped) ** 2
    loss = torch.max(loss_a, loss_b)
    if mask is not None:
        active = mask.to(dtype=loss.dtype)
        return (loss * active).sum() / active.sum().clamp_min(1.0)
    return loss.mean()


def trust_region_policy_loss(
    advantages: torch.Tensor,    # (MB, act_dim) — broadcast from (MB, 1)
    log_probs: torch.Tensor,     # (MB, act_dim) — per-dim
    old_log_probs: torch.Tensor, # (MB, act_dim) — per-dim
    epsilon: float,
    mask: torch.Tensor | None = None,
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
    loss = -torch.min(p_opt_a, p_opt_b)
    if mask is not None:
        active = mask.to(dtype=loss.dtype)
        while active.ndim < loss.ndim:
            active = active.unsqueeze(-1)
        active = active.expand_as(loss)
        return (loss * active).sum() / active.sum().clamp_min(1.0)
    return loss.mean()


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
        self.recurrent = bool(getattr(c, "recurrent", False))
        if self.recurrent and not self.discrete:
            raise ValueError("Recurrent POCA actor is only implemented for discrete actions")

        # Critic state dimension: 5D polar (ρ, cos α, sin α, cos β, sin β)
        # Matches SwarmACB Unity modification: "the value function takes the
        # state and not the observation of the agents."
        self.state_dim = 5

        print(f"[POCA] envs={self.num_envs}  agents={self.num_agents}  "
              f"obs={self.obs_dim}  state={self.state_dim}  "
              f"act={'discrete(' + str(self.num_actions) + ')' if self.discrete else str(self.act_dim)}  "
              f"decision_period={self.decision_period}")
        print("[POCA] Actor uses local obs only; Critic uses 5D polar state")
        if self.recurrent:
            print(f"[POCA] Recurrent actor + critic: LSTM units={c.memory_size // 2}  "
                  f"memory_vector={c.memory_size}  "
                  f"sequence_length={c.sequence_length}")
        print(f"[POCA] Centralized RSA critic: hidden={c.critic_hidden_dim}  "
              f"layers={c.critic_num_layers}  heads={c.critic_num_heads}")

        if self.discrete:
            if self.recurrent:
                self.actor = RecurrentDiscreteActor(
                    self.obs_dim,
                    self.num_actions,
                    c.hidden_dim,
                    c.num_layers,
                    c.memory_size,
                ).to(self.device)
            else:
                self.actor = DiscreteActor(
                    self.obs_dim, self.num_actions, c.hidden_dim, c.num_layers,
                ).to(self.device)
        else:
            self.actor = Actor(
                self.obs_dim, self.act_dim, c.hidden_dim, c.num_layers,
            ).to(self.device)

        if self.recurrent:
            memory_batch = self.num_envs * self.num_agents
            self.actor_memory_h, self.actor_memory_c = self.actor.initial_state(
                memory_batch, self.device,
            )
        else:
            self.actor_memory_h = None
            self.actor_memory_c = None

        self.critic = POCACritic(
            self.state_dim, self.act_dim_critic, self.num_agents,
            c.critic_hidden_dim, c.critic_num_heads, c.critic_num_layers,
            memory_size=c.memory_size if self.recurrent else 0,
        ).to(self.device)

        if self.recurrent:
            self.critic_memory_h, self.critic_memory_c = self.critic.initial_state(
                self.num_envs, self.device,
            )
            self.baseline_memory_h, self.baseline_memory_c = self.critic.initial_state(
                self.num_envs * self.num_agents, self.device,
            )
        else:
            self.critic_memory_h = self.critic_memory_c = None
            self.baseline_memory_h = self.baseline_memory_c = None

        # ── single optimiser (actor + critic) — matches ML-Agents ─
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=c.lr, eps=c.adam_eps,
        )

        # ── schedule support ─────────────────────────────────────
        self._init_lr = c.lr
        self._init_eps = c.clip_eps
        self._init_beta = c.beta

        # ML-Agents polynomial_decay min values:
        #   lr   → 1e-10  (essentially 0)
        #   eps  → 0.1    (trust region never collapses fully)
        #   beta → 1e-5   (tiny entropy bonus remains)
        self.lr_schedule = PolynomialDecay(c.lr, 1e-10, c.total_timesteps) if c.lr_schedule == "linear" else None
        self.eps_schedule = PolynomialDecay(c.clip_eps, 0.1, c.total_timesteps) if c.eps_schedule == "linear" else None
        self.beta_schedule = PolynomialDecay(c.beta, 1e-5, c.total_timesteps) if c.beta_schedule == "linear" else None

        self.current_lr = c.lr
        self.current_eps = c.clip_eps
        self.current_beta = c.beta

        # ── reward strength ────────────────────────────────────────
        self.reward_strength = c.reward_strength

        # ── step-based checkpoint / log tracking ──────────────────
        self._next_checkpoint_step = c.checkpoint_interval
        self._next_summary_step = c.summary_freq

        # ── rollout buffer ────────────────────────────────────────
        # ML-Agents only updates after complete per-agent trajectories have
        # pushed the update buffer strictly past buffer_size. A short terminal
        # trajectory can therefore be followed by another horizon trajectory.
        steps_to_buffer_target = (
            c.buffer_size_hint + self.num_envs * self.num_agents - 1
        ) // (self.num_envs * self.num_agents)
        buffer_capacity = c.horizon + steps_to_buffer_target + 1
        self.buffer = POCARolloutBuffer(
            horizon=buffer_capacity,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            memory_size=self.actor.hidden_size if self.recurrent else 0,
            critic_memory_size=self.critic.hidden_size if self.recurrent else 0,
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

        # Print param count & batch info
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        rollout_experiences = c.horizon * self.num_envs * self.num_agents
        n_batches = (rollout_experiences + c.mini_batch_size - 1) // c.mini_batch_size
        print(f"[POCA] Actor params: {actor_params:,}  "
              f"Critic params: {critic_params:,}")
        print(f"[POCA] Mini-batch: {c.mini_batch_size} focal-agent transitions  "
              f"[{n_batches} batches/epoch x {c.num_epochs} epochs "
              f"= {n_batches * c.num_epochs} updates/rollout]")
        print(f"[POCA] TensorBoard -> {c.log_dir}")

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
    def collect_rollout(
        self,
        obs_dict: dict,
        rollout_steps: int | None = None,
        reset_buffer: bool = True,
    ) -> dict:
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
        if reset_buffer:
            self.buffer.reset()
        agents = self.unwrapped.cfg.possible_agents
        dp = self.decision_period

        steps = self.cfg.horizon if rollout_steps is None else int(rollout_steps)
        for _ in range(steps):
            # ── stack observations (taken BEFORE the decision) ────
            obs_stacked = torch.stack(
                [obs_dict[a] for a in agents], dim=1
            )  # (E, N, ...)
            # Flatten grid if needed
            if obs_stacked.ndim == 5:
                # (E, N, C, H, W) -> (E, N, C*H*W)
                obs_stacked = obs_stacked.view(obs_stacked.shape[0], obs_stacked.shape[1], -1)

            # ── sample actions from shared actor (batched over all agents) ──
            # Actor uses LOCAL observations only (swarm robotics: no global knowledge)
            flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])  # (E*N, obs)
            memory_h = None
            memory_c = None
            if self.recurrent:
                memory_h = self.actor_memory_h.squeeze(0).view(
                    self.num_envs, self.num_agents, -1,
                ).clone()
                memory_c = self.actor_memory_c.squeeze(0).view(
                    self.num_envs, self.num_agents, -1,
                ).clone()
                logits, next_memory = self.actor.step(
                    flat_obs, (self.actor_memory_h, self.actor_memory_c),
                )
                self.actor_memory_h = next_memory[0].detach()
                self.actor_memory_c = next_memory[1].detach()
                dist = torch.distributions.Categorical(logits=logits)
            else:
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

            # ── critic: team value V(s) — uses 5D polar STATE ─────
            critic_state = self.unwrapped.get_critic_state()           # (E, N, 5)
            critic_memory_h = critic_memory_c = None
            baseline_memory_h = baseline_memory_c = None

            # ── baselines: counterfactual b_i — also uses STATE ───
            critic_actions = self._encode_actions_for_critic(all_actions)
            if self.recurrent:
                critic_memory_h = self.critic_memory_h.squeeze(0).clone()
                critic_memory_c = self.critic_memory_c.squeeze(0).clone()
                baseline_memory_h = self.baseline_memory_h.squeeze(0).view(
                    self.num_envs, self.num_agents, -1,
                ).clone()
                baseline_memory_c = self.baseline_memory_c.squeeze(0).view(
                    self.num_envs, self.num_agents, -1,
                ).clone()
                team_val, next_critic_memory = self.critic.critic_pass(
                    critic_state,
                    (self.critic_memory_h, self.critic_memory_c),
                    return_memory=True,
                )
                baselines, next_baseline_memory = self.critic.all_baselines(
                    critic_state,
                    critic_actions,
                    (self.baseline_memory_h, self.baseline_memory_c),
                    return_memory=True,
                )
                self.critic_memory_h = next_critic_memory[0].detach()
                self.critic_memory_c = next_critic_memory[1].detach()
                self.baseline_memory_h = next_baseline_memory[0].detach()
                self.baseline_memory_c = next_baseline_memory[1].detach()
                team_val = team_val.squeeze(-1)
            else:
                team_val = self.critic.critic_pass(critic_state).squeeze(-1)
                baselines = self.critic.all_baselines(critic_state, critic_actions)

            # ── ML-Agents action preprocessing ──────────────────
            # ML-Agents clips continuous actions to [-3, 3] then divides by 3
            # before sending to the env (AgentAction.to_action_tuple with
            # clip=True).  This keeps ~99.7 % of initial samples (std=1)
            # inside the active range, avoiding gradient-killing saturation
            # at the ±1 boundary.  Buffer stores RAW actions for correct
            # log-prob / ratio computation.
            if not self.discrete:
                env_actions = all_actions.clamp(-3, 3) / 3
            else:
                env_actions = all_actions

            # ── step environment decision_period times ────────────
            # Same action for all sub-steps; env applies velocity
            # only on the first sub-step (decision step), then coasts.
            action_dict = {a: env_actions[:, i] for i, a in enumerate(agents)}
            accumulated_reward = torch.zeros(self.num_envs, device=self.device)
            last_done = torch.zeros(self.num_envs, device=self.device)
            last_timeout = torch.zeros(self.num_envs, device=self.device)

            for _dp in range(dp):
                obs_dict, rewards_dict, terminated_dict, truncated_dict, info = (
                    self.env.step(action_dict)
                )
                accumulated_reward += rewards_dict[agents[0]]
                step_done = (terminated_dict[agents[0]] | truncated_dict[agents[0]]).float()
                last_done = torch.max(last_done, step_done)
                last_timeout = torch.max(
                    last_timeout, truncated_dict[agents[0]].float(),
                )

            terminal_state = self.unwrapped.completed_terminal_critic_state
            if self.recurrent:
                timeout_value = self.critic.critic_pass(
                    terminal_state,
                    (self.critic_memory_h, self.critic_memory_c),
                ).squeeze(-1)
            else:
                timeout_value = self.critic.critic_pass(terminal_state).squeeze(-1)
            timeout_value = timeout_value * last_timeout

            # ── store ONE transition per decision ─────────────────
            self.buffer.add(
                obs=obs_stacked,
                critic_states=critic_state,
                actions=all_actions,
                log_probs=all_log_probs,  # per-dim!
                reward=accumulated_reward * self.reward_strength,
                done=last_done,
                timeout=last_timeout,
                timeout_value=timeout_value,
                team_value=team_val,
                baselines=baselines,
                memory_h=memory_h,
                memory_c=memory_c,
                critic_memory_h=critic_memory_h,
                critic_memory_c=critic_memory_c,
                baseline_memory_h=baseline_memory_h,
                baseline_memory_c=baseline_memory_c,
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

            if self.recurrent and done_mask.any():
                done_agents = done_mask[:, None].expand(
                    self.num_envs, self.num_agents,
                ).reshape(-1)
                self.actor_memory_h[:, done_agents, :] = 0.0
                self.actor_memory_c[:, done_agents, :] = 0.0
                self.critic_memory_h[:, done_mask, :] = 0.0
                self.critic_memory_c[:, done_mask, :] = 0.0
                self.baseline_memory_h[:, done_agents, :] = 0.0
                self.baseline_memory_c[:, done_agents, :] = 0.0

            # Count agent-decisions (matching ML-Agents max_steps)
            self.global_step += self.num_envs * self.num_agents

        # ── bootstrap V for lambda-return (uses 5D STATE) ─────────
        last_state = self.unwrapped.get_critic_state()                # (E, N, 5)
        if self.recurrent:
            last_tv = self.critic.critic_pass(
                last_state,
                (self.critic_memory_h, self.critic_memory_c),
            ).squeeze(-1)
        else:
            last_tv = self.critic.critic_pass(last_state).squeeze(-1)
        self.buffer.compute_returns_and_advantages(last_tv)

        return obs_dict

    def _compute_feedforward_losses(
        self,
        batch: dict,
        current_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch["obs"]
        critic_states = batch["critic_states"]
        actions = batch["actions"]
        critic_actions = batch["critic_actions"]
        old_logp = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_tv = batch["old_team_values"]
        old_bl = batch["old_baselines"]

        new_logp, entropy = self.actor.evaluate(obs, actions)
        policy_loss = trust_region_policy_loss(
            advantages.unsqueeze(-1),
            new_logp,
            old_logp,
            current_eps,
        )
        mean_entropy = entropy.mean()

        new_tv = self.critic.critic_pass(critic_states).squeeze(-1)
        critic_act = self._encode_actions_for_critic(critic_actions)
        all_baselines = self.critic.all_baselines(critic_states, critic_act)
        batch_ids = torch.arange(obs.shape[0], device=obs.device)
        new_bl = all_baselines[batch_ids, batch["focal_agent_ids"]]

        value_loss = trust_region_value_loss(new_tv, old_tv, returns, current_eps)
        baseline_loss = trust_region_value_loss(
            new_bl,
            old_bl,
            returns,
            current_eps,
        )
        return policy_loss, value_loss, baseline_loss, mean_entropy

    def _compute_recurrent_losses(
        self,
        batch: dict,
        current_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch["obs"]
        critic_states = batch["critic_states"]
        actions = batch["actions"]
        critic_actions = batch["critic_actions"]
        old_logp = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_tv = batch["old_team_values"]
        old_bl = batch["old_baselines"]
        loss_mask = batch["loss_mask"].bool()

        B, L = obs.shape[:2]
        N = critic_states.shape[2]
        h0 = batch["memory_h"].unsqueeze(0).detach()
        c0 = batch["memory_c"].unsqueeze(0).detach()
        state = (h0, c0)
        logps = []
        ents = []
        for t in range(L):
            logits, state = self.actor.step(obs[:, t], state)
            dist = torch.distributions.Categorical(logits=logits)
            act_t = actions[:, t].squeeze(-1).long()
            logps.append(dist.log_prob(act_t).unsqueeze(-1))
            ents.append(dist.entropy())
            if t < L - 1:
                keep = (1.0 - batch["dones"][:, t]).view(1, B, 1)
                state = (state[0] * keep, state[1] * keep)
        logp_seq = torch.stack(logps, dim=1)
        ent_seq = torch.stack(ents, dim=1)

        policy_loss = trust_region_policy_loss(
            advantages.unsqueeze(-1).reshape(-1, 1),
            logp_seq.reshape(-1, logp_seq.shape[-1]),
            old_logp.reshape(-1, old_logp.shape[-1]),
            current_eps,
            loss_mask.reshape(-1),
        )
        mean_entropy = (
            ent_seq * loss_mask
        ).sum() / loss_mask.sum().clamp_min(1)

        flat_states = critic_states.reshape(B * L, N, critic_states.shape[-1])
        flat_actions = critic_actions.reshape(B * L, N, critic_actions.shape[-1])
        flat_returns = returns.reshape(B * L)
        flat_old_tv = old_tv.reshape(B * L)
        flat_old_bl = old_bl.reshape(B * L)

        critic_act = self._encode_actions_for_critic(flat_actions)
        focal_ids = batch["focal_agent_ids"].unsqueeze(1).expand(B, L).reshape(-1)
        new_tv = self.critic.critic_pass(
            flat_states,
            (
                batch["critic_memory_h"].unsqueeze(0).detach(),
                batch["critic_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=L,
        ).squeeze(-1)
        new_bl = self.critic.focal_baselines(
            flat_states,
            critic_act,
            focal_ids,
            (
                batch["baseline_memory_h"].unsqueeze(0).detach(),
                batch["baseline_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=L,
        ).squeeze(-1)

        flat_mask = loss_mask.reshape(B * L)

        value_loss = trust_region_value_loss(
            new_tv, flat_old_tv, flat_returns, current_eps, flat_mask,
        )
        baseline_loss = trust_region_value_loss(
            new_bl,
            flat_old_bl,
            flat_returns,
            current_eps,
            flat_mask,
        )
        return policy_loss, value_loss, baseline_loss, mean_entropy

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

        # ── Normalize advantages (matching ML-Agents on_policy_trainer._update_policy) ──
        # ML-Agents normalizes advantages to mean=0, std=1 BEFORE the epoch loop.
        # Without this, raw advantage magnitudes vary wildly, destabilizing
        # the policy gradient and causing premature convergence.
        all_adv = self.buffer.advantages[:self.buffer.ptr]
        adv_mean = all_adv.mean()
        adv_std = all_adv.std(unbiased=False)
        self.buffer.advantages[:self.buffer.ptr] = (
            all_adv - adv_mean
        ) / (adv_std + 1e-10)

        for _epoch in range(cfg.num_epochs):
            if self.recurrent:
                batch_iter = self.buffer.get_sequence_batches(
                    cfg.sequence_length, cfg.mini_batch_size,
                )
            else:
                batch_iter = self.buffer.get_batches(cfg.mini_batch_size)

            for batch in batch_iter:
                if self.recurrent:
                    policy_loss, value_loss, baseline_loss, mean_entropy = (
                        self._compute_recurrent_losses(batch, current_eps)
                    )
                else:
                    policy_loss, value_loss, baseline_loss, mean_entropy = (
                        self._compute_feedforward_losses(batch, current_eps)
                    )

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

            # Collect complete ML-Agents-style trajectories. Trajectories end
            # at time_horizon or at the synchronous episode boundary; updates
            # begin only once their focal-agent experiences exceed buffer_size.
            self.buffer.reset()
            while self.global_step < self.cfg.total_timesteps:
                remaining = self.cfg.total_timesteps - self.global_step
                agent_steps_per_env_step = self.num_envs * self.num_agents
                remaining_steps = max(
                    1,
                    (remaining + agent_steps_per_env_step - 1) // agent_steps_per_env_step,
                )
                episode_step = int(self.unwrapped.episode_length_buf.max().item())
                episode_steps_left = max(
                    1,
                    (self.unwrapped.max_episode_length - episode_step + self.decision_period - 1)
                    // self.decision_period,
                )
                rollout_steps = min(
                    self.cfg.horizon,
                    remaining_steps,
                    episode_steps_left,
                )
                obs_dict = self.collect_rollout(
                    obs_dict,
                    rollout_steps,
                    reset_buffer=False,
                )
                experiences = self.buffer.ptr * agent_steps_per_env_step
                if experiences > self.cfg.buffer_size_hint:
                    break

            # 2. update
            metrics = self.update()

            # 3. progress bar
            step_delta = self.global_step - prev_step
            elapsed = time.time() - start_time
            sps = self.global_step / elapsed if elapsed > 0 else 0

            pbar.update(min(step_delta, max(0, self.cfg.total_timesteps - pbar.n)))
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
                self.buffer.rewards[:self.buffer.ptr].sum(dim=0).mean().item()
            )
            self._rollout_reward_history.append(mean_rollout_reward)
            if len(self._rollout_reward_history) > self._max_history:
                self._rollout_reward_history.pop(0)

            # 4b. TensorBoard — gated by summary_freq
            if self.global_step >= self._next_summary_step:
                self._next_summary_step += self.cfg.summary_freq
                s = self.global_step

                # ══════════════════════════════════════════════════
                #  ML-Agents exact tags (same names as Unity TB)
                # ══════════════════════════════════════════════════

                # ── Losses (ML-Agents: Losses/*) ──────────────────
                self.writer.add_scalar(
                    "Losses/Policy Loss",
                    metrics["policy_loss"], s)
                self.writer.add_scalar(
                    "Losses/Value Loss",
                    metrics["value_loss"], s)
                self.writer.add_scalar(
                    "Losses/POCA/Baseline Loss",
                    metrics["baseline_loss"], s)

                # ── Policy (ML-Agents: Policy/*) ──────────────────
                self.writer.add_scalar(
                    "Policy/Entropy", metrics["entropy"], s)
                self.writer.add_scalar(
                    "Policy/Learning Rate", metrics["lr"], s)
                self.writer.add_scalar(
                    "Policy/Epsilon", metrics["eps"], s)
                self.writer.add_scalar(
                    "Policy/Beta", metrics["beta"], s)

                # Actor log_std diagnostic (per-dim)
                if not self.discrete and hasattr(self.actor, "log_std"):
                    log_std = self.actor.log_std.detach()
                    for d in range(log_std.shape[-1]):
                        self.writer.add_scalar(
                            f"Policy/Std dim{d}",
                            log_std[0, d].exp().item(), s)
                    self.writer.add_scalar(
                        "Policy/Log Std Mean",
                        log_std.mean().item(), s)

                # Extrinsic Reward = mean per-step reward over the
                # rollout (ML-Agents logs this as the mean reward
                # received per agent-decision across the buffer)
                mean_step_reward = (
                    self.buffer.rewards[:self.buffer.ptr].mean().item()
                )
                self.writer.add_scalar(
                    "Policy/Extrinsic Reward",
                    mean_step_reward, s)

                # Extrinsic Value Estimate = mean V(s) prediction
                self.writer.add_scalar(
                    "Policy/Extrinsic Value Estimate",
                    self.buffer.team_values[:self.buffer.ptr].mean().item(), s)

                # ── Environment (ML-Agents: Environment/*) ────────
                if self._completed_episode_returns:
                    ep = self._completed_episode_returns
                    self.writer.add_scalar(
                        "Environment/Cumulative Reward",
                        sum(ep) / len(ep), s)
                    self._completed_episode_returns.clear()

                if self._completed_episode_lengths:
                    el = self._completed_episode_lengths
                    self.writer.add_scalar(
                        "Environment/Episode Length",
                        sum(el) / len(el), s)
                    self._completed_episode_lengths.clear()

                # ══════════════════════════════════════════════════
                #  Extra diagnostics (beyond ML-Agents)
                # ══════════════════════════════════════════════════

                self.writer.add_scalar(
                    "Extra/SPS", sps, s)
                self.writer.add_scalar(
                    "Extra/Mean Rollout Reward",
                    mean_rollout_reward, s)
                rolling_avg = (
                    sum(self._rollout_reward_history)
                    / len(self._rollout_reward_history)
                )
                self.writer.add_scalar(
                    "Extra/Rolling Avg Rollout Reward",
                    rolling_avg, s)
                self.writer.add_scalar(
                    "Extra/Mean Abs Advantage",
                    self.buffer.advantages[:self.buffer.ptr].abs().mean().item(), s)

                # Group reward (gate crossings — mission-specific)
                if self._completed_group_rewards:
                    gr = self._completed_group_rewards
                    self.writer.add_scalar(
                        "Extra/Group Reward Mean",
                        sum(gr) / len(gr), s)
                    self._completed_group_rewards.clear()

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
        print(f"[POCA] Done - {self.global_step:,} steps in {elapsed:.0f}s "
              f"({self.global_step / elapsed:.0f} SPS)")

    # ──────────────────────────────────────────────────────────────
    #  Checkpointing
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, path):
        torch.save({
            "paper_parity_version": 3,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "seed": self.cfg.seed,
            # Save architecture for correct restoration in play
            "hidden_dim": getattr(self.cfg, "hidden_dim", 256),
            "num_layers": getattr(self.cfg, "num_layers", 2),
            "recurrent": self.recurrent,
            "memory_size": getattr(self.cfg, "memory_size", 0),
            "memory_size_semantics": "mlagents_total",
            "lstm_hidden_size": self.actor.hidden_size if self.recurrent else 0,
            "sequence_length": getattr(self.cfg, "sequence_length", 0),
            "critic_hidden_dim": self.cfg.critic_hidden_dim,
            "critic_num_layers": self.cfg.critic_num_layers,
            "critic_num_heads": self.cfg.critic_num_heads,
            "decision_period": self.decision_period,
            "discrete": self.discrete,
            "num_actions": self.num_actions if self.discrete else 0,
            "act_dim": self.act_dim,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
        }, path)
        print(f"[POCA] Saved -> {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        parity_version = int(ckpt.get("paper_parity_version", 0))
        if parity_version != 3:
            raise RuntimeError(
                "Refusing to resume a pre-parity checkpoint. Its rollout cadence, "
                "entropy objective, critic state, or recurrent layout may differ. "
                "Use it only for legacy evaluation and start paper-parity training fresh."
            )
        try:
            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match the paper-parity trainer. "
                "Legacy checkpoints can still be evaluated, but training must start "
                "fresh after the fixed-critic revision."
            ) from exc
        self.global_step = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        print(f"[POCA] Loaded <- {path}  (step {self.global_step})")

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
            print(f"[POCA] Removed old checkpoint -> {old.name}")
