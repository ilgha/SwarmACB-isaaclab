# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Learned-option collective Option-Critic trainer for SwarmACB Phase 2.

The local recurrent actor learns a policy over options, one termination model
and one continuous wheel policy per option. Centralized training preserves the
SwarmACB counterfactual construction at two levels:

* primitive-action credit holds every peer option and wheel action fixed;
* option credit holds every peer option fixed.

Only the local actor is needed for decentralized execution.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.utils.tensorboard import SummaryWriter

from .learned_option_critic_buffer import LearnedOptionRolloutBuffer
from .learned_option_critic_networks import (
    LEARNED_OPTION_CRITIC_VERSION,
    LearnedOptionActor,
    termination_objective,
)
from .network_config import PAPER_PARITY_VERSION
from .poca_networks import POCACritic
from .poca_trainer import (
    PolynomialDecay,
    trust_region_value_loss,
)


def _stable_trust_region_policy_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    epsilon: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PPO policy loss with a numerically bounded importance ratio.

    PPO only needs to distinguish ratios inside and outside its narrow clipping
    interval. Bounding the log-ratio before ``exp`` therefore leaves the trust
    region unchanged while preventing an outlying recurrent minibatch from
    overflowing to ``inf`` and poisoning every later optimizer step.
    """
    log_ratio = (log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    objective = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages,
    )
    loss = -objective
    if mask is None:
        return loss.mean()
    active = mask.to(dtype=loss.dtype)
    while active.ndim < loss.ndim:
        active = active.unsqueeze(-1)
    active = active.expand_as(loss)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


@dataclass
class LearnedOptionCriticConfig:
    """Training hyperparameters for learned Option-Critic."""

    trainer_type: str = "learned_option_critic"

    # Rollout and PPO.
    horizon: int = 1000
    num_epochs: int = 3
    mini_batch_size: int = 4096
    clip_eps: float = 0.2
    beta: float = 0.001

    # Objective weights.
    intra_option_coef: float = 1.0
    selector_coef: float = 1.0
    local_option_value_coef: float = 0.1
    option_entropy_coef: float = 0.005
    option_balance_coef: float = 0.001
    option_balance_final_coef: float = 0.0001
    value_coef: float = 0.5
    action_baseline_coef: float = 0.25
    option_value_coef: float = 0.5
    option_baseline_coef: float = 0.25
    termination_coef: float = 1.0
    termination_entropy_coef: float = 0.0
    termination_penalty: float = 0.0
    termination_prior_probability: float = 0.05
    termination_prior_coef: float = 0.001
    termination_prior_final_coef: float = 0.0001
    attention_diversity_coef: float = 0.01
    attention_temporal_coef: float = 0.01

    # GAE.
    gamma: float = 0.99
    lam: float = 0.95

    # Optimizer and schedules.
    lr: float = 3e-4
    actor_lr: float = 1e-4
    adam_eps: float = 1e-8
    lr_schedule: str = "constant"
    eps_schedule: str = "constant"
    beta_schedule: str = "constant"
    max_grad_norm: float = 10.0
    actor_max_grad_norm: float = 1.0
    target_kl: float = 0.01
    adaptive_actor_lr: bool = True
    actor_lr_scale_min: float = 0.05
    actor_lr_decay_factor: float = 1.5
    actor_lr_recovery_factor: float = 1.05
    fused_optimizer: bool = True
    matmul_precision: str = "high"

    # Run control.
    total_timesteps: int = 120_000_000
    checkpoint_interval: int = 120_000
    summary_freq: int = 120_000
    keep_checkpoints: int = 5
    checkpoint_dir: str = "checkpoints/learned_option_critic"
    seed: int = 0

    # Environment stepping.
    decision_period: int = 1
    reward_strength: float = 1.0

    # Networks.
    hidden_dim: int = 128
    num_layers: int = 1
    critic_hidden_dim: int = 128
    critic_num_layers: int = 1
    critic_num_heads: int = 4
    recurrent: bool = True
    memory_size: int = 128
    sequence_length: int = 128
    num_options: int = 6
    option_hidden_dim: int = 512
    option_num_layers: int = 2
    option_memory_size: int = 64
    initial_termination_probability: float = 0.05
    initial_log_std: float = -0.7
    min_log_std: float = -2.5
    max_log_std: float = 0.0
    option_selector_temperature: float = 1.0

    # Logging.
    log_dir: str = "runs/learned_option_critic"
    buffer_size_hint: int = 0


class LearnedOptionCriticTrainer:
    """Train learned continuous options with collective counterfactual credit."""

    CHECKPOINT_VERSION = LEARNED_OPTION_CRITIC_VERSION
    TRAINING_CHECKPOINT_VERSION = 4

    def __init__(
        self,
        env,
        cfg: LearnedOptionCriticConfig | None = None,
    ):
        self.env = env
        self.cfg = cfg or LearnedOptionCriticConfig()
        self.unwrapped = env.unwrapped
        self.device = torch.device(self.unwrapped.device)
        self.num_envs = self.unwrapped.scene.num_envs
        env_cfg = self.unwrapped.cfg
        self.num_agents = getattr(
            env_cfg,
            "num_agents",
            getattr(env_cfg, "num_robots", None),
        )
        self.variant = getattr(env_cfg, "variant", None)
        self.discrete = bool(getattr(env_cfg, "discrete_actions", False))

        if self.variant != "cyclamen":
            raise ValueError(
                "Learned Option-Critic Phase 2 starts from Cyclamen's local "
                f"observation and memory, got variant={self.variant!r}."
            )
        if self.discrete:
            raise ValueError(
                "Learned Option-Critic requires continuous primitive wheel "
                "actions. Configure the Cyclamen observation with continuous "
                "action spaces before constructing the environment."
            )

        agents = env_cfg.possible_agents
        sample_obs = self.env.reset()[0][agents[0]]
        if sample_obs.ndim == 4:
            self.obs_dim = int(
                sample_obs.shape[1]
                * sample_obs.shape[2]
                * sample_obs.shape[3]
            )
        else:
            self.obs_dim = int(sample_obs.shape[1])
        self.act_dim = int(env_cfg.action_spaces[agents[0]])
        self.state_dim = 5
        self.option_state_dim = self.state_dim + self.cfg.num_options
        self.decision_period = int(self.cfg.decision_period)

        if self.obs_dim != 24:
            raise ValueError(
                "Learned Option-Critic Phase 2 requires the full 24-channel "
                "local sensor vector for its learned motor options. Construct "
                "the environment with full_policy_observations enabled."
            )
        if self.act_dim != 2:
            raise ValueError(
                "Phase 2 expects the two normalized e-puck wheel commands, "
                f"got act_dim={self.act_dim}."
            )

        cfg = self.cfg
        if cfg.actor_lr <= 0.0:
            raise ValueError(
                f"actor_lr must be positive, got {cfg.actor_lr}."
            )
        if cfg.actor_max_grad_norm <= 0.0:
            raise ValueError(
                "actor_max_grad_norm must be positive, got "
                f"{cfg.actor_max_grad_norm}."
            )
        if cfg.target_kl < 0.0:
            raise ValueError(
                f"target_kl must be non-negative, got {cfg.target_kl}."
            )
        if not 0.0 < cfg.termination_prior_probability < 1.0:
            raise ValueError(
                "termination_prior_probability must be strictly between "
                f"0 and 1, got {cfg.termination_prior_probability}."
            )
        if min(
            cfg.termination_prior_coef,
            cfg.termination_prior_final_coef,
        ) < 0.0:
            raise ValueError(
                "termination prior coefficients must be non-negative"
            )
        if min(
            cfg.option_balance_coef,
            cfg.option_balance_final_coef,
        ) < 0.0:
            raise ValueError(
                "option balance coefficients must be non-negative"
            )
        if not 0.0 < cfg.actor_lr_scale_min <= 1.0:
            raise ValueError("actor_lr_scale_min must lie in (0, 1]")
        if cfg.actor_lr_decay_factor <= 1.0:
            raise ValueError("actor_lr_decay_factor must be greater than 1")
        if cfg.actor_lr_recovery_factor <= 1.0:
            raise ValueError("actor_lr_recovery_factor must be greater than 1")
        if cfg.matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(
                "matmul_precision must be one of highest, high, or medium"
            )
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        if self.device.type == "cuda":
            allow_tf32 = cfg.matmul_precision != "highest"
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        print(
            f"[LearnedOC] envs={self.num_envs}  agents={self.num_agents}  "
            f"obs={self.obs_dim}  state={self.state_dim}  "
            f"wheel_actions={self.act_dim}  options={cfg.num_options}  "
            f"decision_period={self.decision_period}"
        )
        print(
            "[LearnedOC] Faithful attention paths: selector, termination, and "
            "continuous intra-option policy all consume attended sensors"
        )
        print(
            "[LearnedOC] Counterfactual action credit conditions on active "
            "options and holds all peer controls fixed"
        )
        print(
            "[LearnedOC] Counterfactual option credit holds every peer "
            "option fixed"
        )

        self.actor = LearnedOptionActor(
            self.obs_dim,
            self.act_dim,
            cfg.num_options,
            cfg.hidden_dim,
            cfg.num_layers,
            cfg.memory_size,
            cfg.option_hidden_dim,
            cfg.option_num_layers,
            cfg.option_memory_size,
            cfg.initial_termination_probability,
            cfg.initial_log_std,
            cfg.min_log_std,
            cfg.max_log_std,
            cfg.option_selector_temperature,
        ).to(self.device)

        # V(s): state-only team value used for lambda returns.
        self.team_critic = POCACritic(
            self.state_dim,
            1,
            self.num_agents,
            cfg.critic_hidden_dim,
            cfg.critic_num_heads,
            cfg.critic_num_layers,
            memory_size=cfg.memory_size,
        ).to(self.device)

        # b_i^U((s, omega), a_-i): the focal option is part of the
        # state-only entity, while only the focal wheel action is omitted.
        self.action_critic = POCACritic(
            self.option_state_dim,
            self.act_dim,
            self.num_agents,
            cfg.critic_hidden_dim,
            cfg.critic_num_heads,
            cfg.critic_num_layers,
            memory_size=cfg.memory_size,
        ).to(self.device)

        # Q_Omega(s, omega_vector) and b_i^Omega(s, omega_-i).
        self.option_critic = POCACritic(
            self.state_dim,
            cfg.num_options,
            self.num_agents,
            cfg.critic_hidden_dim,
            cfg.critic_num_heads,
            cfg.critic_num_layers,
            memory_size=cfg.memory_size,
        ).to(self.device)

        self.actor_parameters = list(self.actor.parameters())
        self.critic_parameters = (
            list(self.team_critic.parameters())
            + list(self.action_critic.parameters())
            + list(self.option_critic.parameters())
        )
        self.fused_optimizer_active = False
        optimizer_kwargs = {"eps": cfg.adam_eps}
        if cfg.fused_optimizer and self.device.type == "cuda":
            try:
                self.actor_optimizer = optim.Adam(
                    self.actor_parameters,
                    lr=cfg.actor_lr,
                    fused=True,
                    **optimizer_kwargs,
                )
                self.critic_optimizer = optim.Adam(
                    self.critic_parameters,
                    lr=cfg.lr,
                    fused=True,
                    **optimizer_kwargs,
                )
                self.fused_optimizer_active = True
            except (TypeError, RuntimeError) as error:
                print(
                    "[LearnedOC] Fused Adam unavailable; using standard "
                    f"Adam ({error})"
                )
        if not self.fused_optimizer_active:
            self.actor_optimizer = optim.Adam(
                self.actor_parameters,
                lr=cfg.actor_lr,
                **optimizer_kwargs,
            )
            self.critic_optimizer = optim.Adam(
                self.critic_parameters,
                lr=cfg.lr,
                **optimizer_kwargs,
            )
        # PPO ratios are evaluated against an immutable update-start policy.
        # This also removes one-step versus sequence-mode LSTM round-off from
        # the trust-region reference without changing the behavior weights.
        self.reference_actor = copy.deepcopy(self.actor).eval()
        self.reference_actor.requires_grad_(False)
        self.reference_actor.manager_lstm.flatten_parameters()
        self.reference_actor.option_lstm.flatten_parameters()

        actor_batch = self.num_envs * self.num_agents
        self.actor_memory_h, self.actor_memory_c = self.actor.initial_state(
            actor_batch,
            self.device,
        )
        self.team_memory_h, self.team_memory_c = (
            self.team_critic.initial_state(self.num_envs, self.device)
        )
        (
            self.action_baseline_memory_h,
            self.action_baseline_memory_c,
        ) = self.action_critic.initial_state(actor_batch, self.device)
        (
            self.option_joint_memory_h,
            self.option_joint_memory_c,
        ) = self.option_critic.initial_state(self.num_envs, self.device)
        (
            self.option_baseline_memory_h,
            self.option_baseline_memory_c,
        ) = self.option_critic.initial_state(actor_batch, self.device)
        self.current_options = torch.full(
            (self.num_envs, self.num_agents),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self.lr_schedule = (
            PolynomialDecay(cfg.lr, 1e-10, cfg.total_timesteps)
            if cfg.lr_schedule == "linear"
            else None
        )
        self.actor_lr_schedule = (
            PolynomialDecay(cfg.actor_lr, 1e-10, cfg.total_timesteps)
            if cfg.lr_schedule == "linear"
            else None
        )
        self.eps_schedule = (
            PolynomialDecay(cfg.clip_eps, 0.1, cfg.total_timesteps)
            if cfg.eps_schedule == "linear"
            else None
        )
        self.beta_schedule = (
            PolynomialDecay(cfg.beta, 1e-5, cfg.total_timesteps)
            if cfg.beta_schedule == "linear"
            else None
        )
        self.termination_prior_schedule = PolynomialDecay(
            cfg.termination_prior_coef,
            cfg.termination_prior_final_coef,
            cfg.total_timesteps,
        )
        self.option_balance_schedule = PolynomialDecay(
            cfg.option_balance_coef,
            cfg.option_balance_final_coef,
            cfg.total_timesteps,
        )
        self.current_lr = cfg.lr
        self.current_actor_lr = cfg.actor_lr
        self.current_eps = cfg.clip_eps
        self.current_beta = cfg.beta
        self.current_termination_prior_coef = cfg.termination_prior_coef
        self.current_option_balance_coef = cfg.option_balance_coef
        self.actor_lr_scale = 1.0
        self.reward_strength = cfg.reward_strength
        self._next_checkpoint_step = cfg.checkpoint_interval
        self._next_summary_step = cfg.summary_freq

        steps_to_buffer_target = (
            cfg.buffer_size_hint + actor_batch - 1
        ) // actor_batch
        buffer_capacity = cfg.horizon + steps_to_buffer_target + 1
        self.buffer = LearnedOptionRolloutBuffer(
            horizon=buffer_capacity,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            act_dim=self.act_dim,
            memory_size=self.actor.hidden_size,
            critic_memory_size=self.team_critic.hidden_size,
            gamma=cfg.gamma,
            lam=cfg.lam,
            device=self.device,
        )

        self.global_step = 0
        self.update_count = 0
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.writer.add_text(
            "hyperparameters",
            "\n".join(f"{key}: {value}" for key, value in vars(cfg).items()),
            0,
        )
        self._episode_reward_acc = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._episode_step_count = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self._completed_episode_returns: list[float] = []
        self._completed_episode_lengths: list[float] = []
        self._completed_group_rewards: list[float] = []
        self._rollout_reward_history: list[float] = []
        self._max_history = 100

        print(
            f"[LearnedOC] Actor params: "
            f"{sum(p.numel() for p in self.actor.parameters()):,}"
        )
        print(
            f"[LearnedOC] Manager={cfg.hidden_dim}x{cfg.num_layers}, "
            f"motor={cfg.option_hidden_dim}x{cfg.option_num_layers}, "
            f"packed_memory={self.actor.hidden_size}"
        )
        print(
            "[LearnedOC] Critic params: "
            f"team={sum(p.numel() for p in self.team_critic.parameters()):,}  "
            f"action={sum(p.numel() for p in self.action_critic.parameters()):,}  "
            f"option={sum(p.numel() for p in self.option_critic.parameters()):,}"
        )
        print(
            f"[LearnedOC] CUDA math: precision={cfg.matmul_precision}  "
            f"fused_adam={self.fused_optimizer_active}"
        )
        print(f"[LearnedOC] TensorBoard -> {cfg.log_dir}")

    def _apply_schedules(self):
        if self.lr_schedule is not None:
            self.current_lr = self.lr_schedule.get(self.global_step)
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.current_lr
        if self.actor_lr_schedule is not None:
            base_actor_lr = self.actor_lr_schedule.get(self.global_step)
        else:
            base_actor_lr = self.cfg.actor_lr
        self.current_actor_lr = base_actor_lr * self.actor_lr_scale
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.current_actor_lr
        if self.eps_schedule is not None:
            self.current_eps = self.eps_schedule.get(self.global_step)
        if self.beta_schedule is not None:
            self.current_beta = self.beta_schedule.get(self.global_step)
        self.current_termination_prior_coef = (
            self.termination_prior_schedule.get(self.global_step)
        )
        self.current_option_balance_coef = self.option_balance_schedule.get(
            self.global_step
        )

    def _encode_options(self, options: torch.Tensor) -> torch.Tensor:
        return functional.one_hot(
            options.long(),
            num_classes=self.cfg.num_options,
        ).float()

    def _option_augmented_states(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([states, self._encode_options(options)], dim=-1)

    @torch.no_grad()
    def collect_rollout(
        self,
        obs_dict: dict,
        rollout_steps: int | None = None,
        reset_buffer: bool = True,
    ) -> dict:
        if reset_buffer:
            self.buffer.reset()
        agents = self.unwrapped.cfg.possible_agents
        steps = (
            self.cfg.horizon
            if rollout_steps is None
            else int(rollout_steps)
        )

        for _ in range(steps):
            obs = torch.stack([obs_dict[agent] for agent in agents], dim=1)
            if obs.ndim == 5:
                obs = obs.view(obs.shape[0], obs.shape[1], -1)
            flat_obs = obs.reshape(-1, obs.shape[-1])

            memory_h = self.actor_memory_h.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            memory_c = self.actor_memory_c.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            (
                selector_logits,
                option_values,
                termination_logits,
                action_means,
                action_stds,
                _attentions,
                next_actor_memory,
            ) = self.actor.step(
                flat_obs,
                (self.actor_memory_h, self.actor_memory_c),
            )
            self.actor_memory_h = next_actor_memory[0].detach()
            self.actor_memory_c = next_actor_memory[1].detach()

            option_dist = self.actor.option_dist(selector_logits)
            proposed = option_dist.sample().view(
                self.num_envs,
                self.num_agents,
            )
            proposed_logp = option_dist.log_prob(
                proposed.reshape(-1),
            ).view(self.num_envs, self.num_agents)

            force_new = self.current_options < 0
            prior_options = self.current_options.clamp(min=0)
            beta_logits = self.actor.selected_termination_logits(
                termination_logits,
                prior_options.reshape(-1),
            )
            terminate = Bernoulli(logits=beta_logits).sample().bool().view(
                self.num_envs,
                self.num_agents,
            )
            option_mask = terminate | force_new
            self.current_options = torch.where(
                option_mask,
                proposed,
                self.current_options,
            )
            option_log_probs = torch.where(
                option_mask,
                proposed_logp,
                torch.zeros_like(proposed_logp),
            )
            beta_probs = torch.sigmoid(beta_logits).view(
                self.num_envs,
                self.num_agents,
            )

            flat_options = self.current_options.reshape(-1)
            local_option_values = option_values.gather(
                -1,
                flat_options.unsqueeze(-1),
            ).squeeze(-1).view(self.num_envs, self.num_agents)
            action_dist = self.actor.selected_action_dist(
                action_means,
                action_stds,
                flat_options,
            )
            actions = action_dist.sample().view(
                self.num_envs,
                self.num_agents,
                self.act_dim,
            )
            action_log_probs = action_dist.log_prob(
                actions.reshape(-1, self.act_dim),
            ).view(self.num_envs, self.num_agents, self.act_dim)

            critic_state = self.unwrapped.get_critic_state()
            encoded_options = self._encode_options(self.current_options)
            option_states = torch.cat(
                [critic_state, encoded_options],
                dim=-1,
            )

            team_memory_h = self.team_memory_h.squeeze(0).clone()
            team_memory_c = self.team_memory_c.squeeze(0).clone()
            action_baseline_memory_h = (
                self.action_baseline_memory_h.squeeze(0)
                .view(self.num_envs, self.num_agents, -1)
                .clone()
            )
            action_baseline_memory_c = (
                self.action_baseline_memory_c.squeeze(0)
                .view(self.num_envs, self.num_agents, -1)
                .clone()
            )
            option_joint_memory_h = (
                self.option_joint_memory_h.squeeze(0).clone()
            )
            option_joint_memory_c = (
                self.option_joint_memory_c.squeeze(0).clone()
            )
            option_baseline_memory_h = (
                self.option_baseline_memory_h.squeeze(0)
                .view(self.num_envs, self.num_agents, -1)
                .clone()
            )
            option_baseline_memory_c = (
                self.option_baseline_memory_c.squeeze(0)
                .view(self.num_envs, self.num_agents, -1)
                .clone()
            )

            team_value, next_team_memory = self.team_critic.critic_pass(
                critic_state,
                (self.team_memory_h, self.team_memory_c),
                return_memory=True,
            )
            (
                action_baselines,
                next_action_baseline_memory,
            ) = self.action_critic.all_baselines(
                option_states,
                actions,
                (
                    self.action_baseline_memory_h,
                    self.action_baseline_memory_c,
                ),
                return_memory=True,
            )
            (
                joint_option_value,
                next_option_joint_memory,
            ) = self.option_critic.joint_action_pass(
                critic_state,
                encoded_options,
                (
                    self.option_joint_memory_h,
                    self.option_joint_memory_c,
                ),
                return_memory=True,
            )
            (
                option_baselines,
                next_option_baseline_memory,
            ) = self.option_critic.all_baselines(
                critic_state,
                encoded_options,
                (
                    self.option_baseline_memory_h,
                    self.option_baseline_memory_c,
                ),
                return_memory=True,
            )

            self.team_memory_h = next_team_memory[0].detach()
            self.team_memory_c = next_team_memory[1].detach()
            self.action_baseline_memory_h = (
                next_action_baseline_memory[0].detach()
            )
            self.action_baseline_memory_c = (
                next_action_baseline_memory[1].detach()
            )
            self.option_joint_memory_h = (
                next_option_joint_memory[0].detach()
            )
            self.option_joint_memory_c = (
                next_option_joint_memory[1].detach()
            )
            self.option_baseline_memory_h = (
                next_option_baseline_memory[0].detach()
            )
            self.option_baseline_memory_c = (
                next_option_baseline_memory[1].detach()
            )

            action_dict = {
                agent: actions[:, agent_id]
                for agent_id, agent in enumerate(agents)
            }
            accumulated_reward = torch.zeros(
                self.num_envs,
                device=self.device,
            )
            last_done = torch.zeros_like(accumulated_reward)
            last_timeout = torch.zeros_like(accumulated_reward)
            for _substep in range(self.decision_period):
                (
                    obs_dict,
                    rewards,
                    terminated,
                    truncated,
                    _info,
                ) = self.env.step(action_dict)
                accumulated_reward += rewards[agents[0]]
                step_done = (
                    terminated[agents[0]] | truncated[agents[0]]
                ).float()
                last_done = torch.maximum(last_done, step_done)
                last_timeout = torch.maximum(
                    last_timeout,
                    truncated[agents[0]].float(),
                )

            terminal_state = self.unwrapped.completed_terminal_critic_state
            timeout_value = self.team_critic.critic_pass(
                terminal_state,
                (self.team_memory_h, self.team_memory_c),
            ).squeeze(-1) * last_timeout

            next_obs = torch.stack(
                [obs_dict[agent] for agent in agents],
                dim=1,
            )
            if next_obs.ndim == 5:
                next_obs = next_obs.view(
                    next_obs.shape[0],
                    next_obs.shape[1],
                    -1,
                )
            next_memory_h = self.actor_memory_h.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            next_memory_c = self.actor_memory_c.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            next_critic_state = self.unwrapped.get_critic_state()

            self.buffer.add(
                obs=obs,
                next_obs=next_obs,
                critic_states=critic_state,
                next_critic_states=next_critic_state,
                options=self.current_options,
                option_log_probs=option_log_probs,
                local_option_values=local_option_values,
                option_masks=option_mask.float(),
                beta_probs=beta_probs,
                termination_options=prior_options,
                termination_valid=(~force_new).float(),
                actions=actions,
                action_log_probs=action_log_probs,
                reward=accumulated_reward * self.reward_strength,
                done=last_done,
                timeout=last_timeout,
                timeout_value=timeout_value,
                team_value=team_value.squeeze(-1),
                action_baselines=action_baselines,
                joint_option_value=joint_option_value.squeeze(-1),
                option_baselines=option_baselines,
                memory_h=memory_h,
                memory_c=memory_c,
                next_memory_h=next_memory_h,
                next_memory_c=next_memory_c,
                team_memory_h=team_memory_h,
                team_memory_c=team_memory_c,
                action_baseline_memory_h=action_baseline_memory_h,
                action_baseline_memory_c=action_baseline_memory_c,
                option_joint_memory_h=option_joint_memory_h,
                option_joint_memory_c=option_joint_memory_c,
                next_option_joint_memory_h=(
                    self.option_joint_memory_h.squeeze(0).clone()
                ),
                next_option_joint_memory_c=(
                    self.option_joint_memory_c.squeeze(0).clone()
                ),
                option_baseline_memory_h=option_baseline_memory_h,
                option_baseline_memory_c=option_baseline_memory_c,
            )

            self._episode_reward_acc += accumulated_reward
            self._episode_step_count += self.decision_period
            done_mask = last_done.bool()
            if done_mask.any():
                self._completed_episode_returns.extend(
                    self._episode_reward_acc[done_mask].tolist()
                )
                self._completed_episode_lengths.extend(
                    self._episode_step_count[done_mask].tolist()
                )
                self._completed_group_rewards.extend(
                    self.unwrapped.completed_group_reward[done_mask].tolist()
                )
                self._episode_reward_acc[done_mask] = 0.0
                self._episode_step_count[done_mask] = 0.0
                self.current_options[done_mask] = -1

                done_agents = done_mask[:, None].expand(
                    self.num_envs,
                    self.num_agents,
                ).reshape(-1)
                self.actor_memory_h[:, done_agents, :] = 0.0
                self.actor_memory_c[:, done_agents, :] = 0.0
                self.team_memory_h[:, done_mask, :] = 0.0
                self.team_memory_c[:, done_mask, :] = 0.0
                self.action_baseline_memory_h[:, done_agents, :] = 0.0
                self.action_baseline_memory_c[:, done_agents, :] = 0.0
                self.option_joint_memory_h[:, done_mask, :] = 0.0
                self.option_joint_memory_c[:, done_mask, :] = 0.0
                self.option_baseline_memory_h[:, done_agents, :] = 0.0
                self.option_baseline_memory_c[:, done_agents, :] = 0.0

            self.global_step += self.num_envs * self.num_agents

        last_state = self.unwrapped.get_critic_state()
        last_team_value = self.team_critic.critic_pass(
            last_state,
            (self.team_memory_h, self.team_memory_c),
        ).squeeze(-1)
        self.buffer.compute_returns_and_advantages(last_team_value)
        return obs_dict

    @staticmethod
    def _attention_losses(
        attentions: torch.Tensor,
        loss_mask: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_attention = attentions[loss_mask]
        if active_attention.numel() == 0:
            zero = attentions.sum() * 0.0
            return zero, zero, zero

        normalized = functional.normalize(
            active_attention,
            p=2,
            dim=-1,
            eps=1e-8,
        )
        similarities = torch.matmul(
            normalized,
            normalized.transpose(-1, -2),
        )
        num_options = attentions.shape[-2]
        off_diagonal = ~torch.eye(
            num_options,
            dtype=torch.bool,
            device=attentions.device,
        )
        diversity_loss = similarities[:, off_diagonal].mean()

        pair_mask = (
            loss_mask[:, :-1]
            & loss_mask[:, 1:]
            & (dones[:, :-1] < 0.5)
        )
        if pair_mask.any():
            temporal_delta = (
                attentions[:, 1:] - attentions[:, :-1]
            ).square().mean(dim=(-1, -2))
            temporal_loss = temporal_delta[pair_mask].mean()
        else:
            temporal_loss = attentions.sum() * 0.0
        return diversity_loss, temporal_loss, active_attention.mean()

    def _compute_sequence_losses(
        self,
        batch: dict,
        current_eps: float,
        reference_actor: LearnedOptionActor,
    ) -> dict[str, torch.Tensor]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        states = batch["critic_states"]
        next_states = batch["next_critic_states"]
        options = batch["options"]
        joint_options = batch["critic_options"]
        actions = batch["actions"]
        joint_actions = batch["critic_actions"]
        loss_mask = batch["loss_mask"].bool()
        dones = batch["dones"]
        batch_size, sequence_length = obs.shape[:2]
        num_agents = states.shape[2]

        (
            selector_logits,
            option_values,
            _termination_logits,
            action_means,
            action_stds,
            attentions,
            _next_state,
        ) = self.actor.forward_sequence(
            obs,
            (
                batch["memory_h"].unsqueeze(0).detach(),
                batch["memory_c"].unsqueeze(0).detach(),
            ),
        )
        with torch.no_grad():
            (
                reference_selector_logits,
                _reference_option_values,
                _reference_termination_logits,
                reference_action_means,
                reference_action_stds,
                _reference_attentions,
                _reference_next_state,
            ) = reference_actor.forward_sequence(
                obs,
                (
                    batch["memory_h"].unsqueeze(0),
                    batch["memory_c"].unsqueeze(0),
                ),
            )

        option_dist = self.actor.option_dist(selector_logits)
        new_option_logp = option_dist.log_prob(options)
        with torch.no_grad():
            reference_option_logp = reference_actor.option_dist(
                reference_selector_logits
            ).log_prob(options)
        option_entropy_values = option_dist.entropy()
        boundary_steps = (batch["option_masks"] > 0.5) & loss_mask
        boundary_count = boundary_steps.sum()
        if boundary_count.item() > 0:
            option_entropy = (
                option_entropy_values * boundary_steps
            ).sum() / boundary_count
        else:
            option_entropy = option_entropy_values.sum() * 0.0

        selector_weights = loss_mask.unsqueeze(-1).to(
            dtype=option_dist.probs.dtype
        )
        selector_count = selector_weights.sum().clamp_min(1.0)
        marginal_option_probs = (
            option_dist.probs * selector_weights
        ).sum(dim=(0, 1)) / selector_count
        marginal_option_probs = marginal_option_probs.clamp_min(1e-8)
        option_marginal_entropy = -(
            marginal_option_probs * marginal_option_probs.log()
        ).sum()
        option_balance_loss = (
            marginal_option_probs
            * (
                marginal_option_probs.log()
                + torch.log(torch.tensor(
                    float(self.cfg.num_options),
                    device=marginal_option_probs.device,
                    dtype=marginal_option_probs.dtype,
                ))
            )
        ).sum()
        effective_options = option_marginal_entropy.exp()

        boundary_mask = boundary_steps.reshape(-1)
        flat_new_option_logp = new_option_logp.reshape(-1)
        flat_reference_option_logp = reference_option_logp.reshape(-1)
        flat_behavior_option_logp = batch["old_option_log_probs"].reshape(-1)
        flat_option_advantage = batch["option_advantages"].reshape(
            -1,
        ).detach()
        if boundary_mask.any():
            raw_option_log_ratio = (
                flat_new_option_logp[boundary_mask]
                - flat_reference_option_logp[boundary_mask]
            )
            option_log_ratio = raw_option_log_ratio.clamp(-20.0, 20.0)
            option_ratio = option_log_ratio.exp()
            option_pg_a = (
                option_ratio * flat_option_advantage[boundary_mask]
            )
            option_pg_b = option_ratio.clamp(
                1.0 - current_eps,
                1.0 + current_eps,
            ) * flat_option_advantage[boundary_mask]
            selector_loss = -torch.minimum(
                option_pg_a,
                option_pg_b,
            ).mean()
            option_approx_kl = (
                option_ratio - 1.0 - option_log_ratio
            ).mean()
            behavior_option_logp_error = (
                flat_reference_option_logp[boundary_mask]
                - flat_behavior_option_logp[boundary_mask]
            ).abs().mean()
        else:
            selector_loss = flat_new_option_logp.sum() * 0.0
            option_approx_kl = flat_new_option_logp.sum() * 0.0
            behavior_option_logp_error = flat_new_option_logp.sum() * 0.0

        action_dist = self.actor.selected_action_dist(
            action_means,
            action_stds,
            options,
        )
        new_action_logp = action_dist.log_prob(actions)
        with torch.no_grad():
            reference_action_dist = reference_actor.selected_action_dist(
                reference_action_means,
                reference_action_stds,
                options,
            )
            reference_action_logp = reference_action_dist.log_prob(actions)
        raw_action_log_ratio = (
            new_action_logp - reference_action_logp
        )
        action_ratio_mask = loss_mask.unsqueeze(-1).expand_as(
            raw_action_log_ratio
        )
        bounded_action_log_ratio = raw_action_log_ratio.clamp(-20.0, 20.0)
        action_kl_values = (
            bounded_action_log_ratio.exp()
            - 1.0
            - bounded_action_log_ratio
        )
        action_kl_weights = action_ratio_mask.to(action_kl_values.dtype)
        action_approx_kl = (
            action_kl_values * action_kl_weights
        ).sum() / action_kl_weights.sum().clamp_min(1.0)
        behavior_action_logp_error = (
            (reference_action_logp - batch["old_action_log_probs"]).abs()
            * action_kl_weights
        ).sum() / action_kl_weights.sum().clamp_min(1.0)
        action_entropy_values = action_dist.entropy().mean(dim=-1)
        intra_option_loss = _stable_trust_region_policy_loss(
            batch["action_advantages"].reshape(-1, 1).detach(),
            new_action_logp.reshape(-1, self.act_dim),
            reference_action_logp.reshape(-1, self.act_dim),
            current_eps,
            loss_mask.reshape(-1),
        )
        action_entropy = (
            action_entropy_values * loss_mask
        ).sum() / loss_mask.sum().clamp_min(1)

        flat_next_obs = next_obs.reshape(
            batch_size * sequence_length,
            self.obs_dim,
        )
        (
            next_selector_logits,
            _next_option_values,
            next_termination_logits,
            _next_action_means,
            _next_action_stds,
            _next_attentions,
            _,
        ) = self.actor.step(
            flat_next_obs,
            (
                batch["next_memory_h"].reshape(
                    batch_size * sequence_length,
                    -1,
                ).unsqueeze(0).detach(),
                batch["next_memory_c"].reshape(
                    batch_size * sequence_length,
                    -1,
                ).unsqueeze(0).detach(),
            ),
        )
        next_beta_logits = self.actor.selected_termination_logits(
            next_termination_logits,
            options.reshape(-1),
        ).view(batch_size, sequence_length)
        next_beta = torch.sigmoid(next_beta_logits)

        flat_states = states.reshape(
            batch_size * sequence_length,
            num_agents,
            self.state_dim,
        )
        flat_next_states = next_states.reshape_as(flat_states)
        flat_joint_options = joint_options.reshape(
            batch_size * sequence_length,
            num_agents,
        )
        encoded_joint_options = self._encode_options(flat_joint_options)
        flat_joint_actions = joint_actions.reshape(
            batch_size * sequence_length,
            num_agents,
            self.act_dim,
        )
        option_states = self._option_augmented_states(
            flat_states,
            flat_joint_options,
        )
        focal_ids = batch["focal_agent_ids"].unsqueeze(1).expand(
            batch_size,
            sequence_length,
        ).reshape(-1)
        flat_returns = batch["returns"].reshape(-1)
        flat_loss_mask = loss_mask.reshape(-1)
        selected_local_option_values = option_values.gather(
            -1,
            options.unsqueeze(-1),
        ).squeeze(-1).reshape(-1)
        local_option_value_mean = (
            selected_local_option_values * flat_loss_mask
        ).sum() / flat_loss_mask.sum().clamp_min(1.0)
        option_value_spread = (
            option_values.std(dim=-1, unbiased=False) * loss_mask
        ).sum() / loss_mask.sum().clamp_min(1)

        new_team_values = self.team_critic.critic_pass(
            flat_states,
            (
                batch["team_memory_h"].unsqueeze(0).detach(),
                batch["team_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=sequence_length,
        ).squeeze(-1)
        new_action_baselines = self.action_critic.focal_baselines(
            option_states,
            flat_joint_actions,
            focal_ids,
            (
                batch["action_baseline_memory_h"].unsqueeze(0).detach(),
                batch["action_baseline_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=sequence_length,
        ).squeeze(-1)
        new_joint_option_values = self.option_critic.joint_action_pass(
            flat_states,
            encoded_joint_options,
            (
                batch["option_joint_memory_h"].unsqueeze(0).detach(),
                batch["option_joint_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=sequence_length,
        ).squeeze(-1)
        new_option_baselines = self.option_critic.focal_baselines(
            flat_states,
            encoded_joint_options,
            focal_ids,
            (
                batch["option_baseline_memory_h"].unsqueeze(0).detach(),
                batch["option_baseline_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=sequence_length,
        ).squeeze(-1)

        value_loss = trust_region_value_loss(
            new_team_values,
            batch["old_team_values"].reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        local_option_value_loss = trust_region_value_loss(
            selected_local_option_values,
            batch["old_local_option_values"].reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        action_baseline_loss = trust_region_value_loss(
            new_action_baselines,
            batch["old_action_baselines"].reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        joint_option_value_loss = trust_region_value_loss(
            new_joint_option_values,
            batch["old_joint_option_values"].reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        option_baseline_loss = trust_region_value_loss(
            new_option_baselines,
            batch["old_option_baselines"].reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )

        # Arrival-state termination theorem. For each focal robot, peers keep
        # their active options while only that robot's alternatives are
        # marginalized under the next-state selector.
        with torch.no_grad():
            next_joint_memory = (
                batch["next_option_joint_memory_h"].reshape(
                    batch_size * sequence_length,
                    -1,
                ).unsqueeze(0),
                batch["next_option_joint_memory_c"].reshape(
                    batch_size * sequence_length,
                    -1,
                ).unsqueeze(0),
            )
            next_q_current = self.option_critic.joint_action_pass(
                flat_next_states,
                encoded_joint_options,
                memory=next_joint_memory,
            ).squeeze(-1)
            next_alternatives = (
                self.option_critic.focal_discrete_counterfactual_values(
                    flat_next_states,
                    flat_joint_options,
                    focal_ids,
                    self.cfg.num_options,
                    memory=next_joint_memory,
                )
            )
            next_selector_probs = self.actor.option_dist(
                next_selector_logits,
            ).probs
            next_reselection = (
                next_alternatives * next_selector_probs
            ).sum(dim=-1)
            termination_advantage = (
                next_q_current - next_reselection
            ).view(batch_size, sequence_length)

        termination_mask = (1.0 - dones) * loss_mask
        termination_count = termination_mask.sum()
        if termination_count.item() > 0:
            termination_loss = termination_objective(
                next_beta,
                termination_advantage,
                self.cfg.termination_penalty,
                termination_mask,
            )
            termination_prior_values = (
                functional.binary_cross_entropy_with_logits(
                    next_beta_logits,
                    torch.full_like(
                        next_beta_logits,
                        self.cfg.termination_prior_probability,
                    ),
                    reduction="none",
                )
            )
            termination_prior_loss = (
                termination_prior_values * termination_mask
            ).sum() / termination_count
            termination_entropy_values = Bernoulli(
                logits=next_beta_logits,
            ).entropy()
            termination_entropy = (
                termination_entropy_values * termination_mask
            ).sum() / termination_count
            mean_beta = (
                next_beta * termination_mask
            ).sum() / termination_count
            mean_termination_advantage = (
                termination_advantage * termination_mask
            ).sum() / termination_count
            termination_signal = (
                termination_advantage + self.cfg.termination_penalty
            )
            mean_termination_signal = (
                termination_signal * termination_mask
            ).sum() / termination_count
            termination_low_saturation = (
                (next_beta < 1e-3).to(next_beta.dtype)
                * termination_mask
            ).sum() / termination_count
            termination_high_saturation = (
                (next_beta > 1.0 - 1e-3).to(next_beta.dtype)
                * termination_mask
            ).sum() / termination_count
        else:
            termination_loss = next_beta.sum() * 0.0
            termination_prior_loss = next_beta.sum() * 0.0
            termination_entropy = next_beta.sum() * 0.0
            mean_beta = next_beta.sum() * 0.0
            mean_termination_advantage = next_beta.sum() * 0.0
            mean_termination_signal = next_beta.sum() * 0.0
            termination_low_saturation = next_beta.sum() * 0.0
            termination_high_saturation = next_beta.sum() * 0.0

        (
            attention_diversity_loss,
            attention_temporal_loss,
            mean_attention,
        ) = self._attention_losses(attentions, loss_mask, dones)

        return {
            "intra_option_loss": intra_option_loss,
            "selector_loss": selector_loss,
            "local_option_value_loss": local_option_value_loss,
            "local_option_value_mean": local_option_value_mean,
            "option_value_spread": option_value_spread,
            "value_loss": value_loss,
            "action_baseline_loss": action_baseline_loss,
            "joint_option_value_loss": joint_option_value_loss,
            "option_baseline_loss": option_baseline_loss,
            "termination_loss": termination_loss,
            "action_entropy": action_entropy,
            "option_entropy": option_entropy,
            "option_balance_loss": option_balance_loss,
            "option_marginal_entropy": option_marginal_entropy,
            "effective_options": effective_options,
            "termination_entropy": termination_entropy,
            "termination_prior_loss": termination_prior_loss,
            "attention_diversity_loss": attention_diversity_loss,
            "attention_temporal_loss": attention_temporal_loss,
            "mean_attention": mean_attention,
            "mean_beta": mean_beta,
            "mean_termination_advantage": mean_termination_advantage,
            "mean_termination_signal": mean_termination_signal,
            "termination_low_saturation": termination_low_saturation,
            "termination_high_saturation": termination_high_saturation,
            "action_approx_kl": action_approx_kl,
            "option_approx_kl": option_approx_kl,
            "behavior_action_logp_error": behavior_action_logp_error,
            "behavior_option_logp_error": behavior_option_logp_error,
        }

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return (
            values - values.mean()
        ) / (values.std(unbiased=False) + 1e-10)

    def update(self) -> dict:
        self._apply_schedules()
        active = self.buffer.ptr
        self.buffer.action_advantages[:active] = self._normalize(
            self.buffer.action_advantages[:active]
        )
        boundaries = self.buffer.option_masks[:active] > 0.5
        if boundaries.any():
            boundary_advantages = self.buffer.option_advantages[:active][
                boundaries
            ]
            normalized = self._normalize(boundary_advantages)
            self.buffer.option_advantages[:active][boundaries] = normalized

        metric_names = (
            "intra_option_loss",
            "selector_loss",
            "local_option_value_loss",
            "local_option_value_mean",
            "option_value_spread",
            "value_loss",
            "action_baseline_loss",
            "joint_option_value_loss",
            "option_baseline_loss",
            "termination_loss",
            "action_entropy",
            "option_entropy",
            "option_balance_loss",
            "option_marginal_entropy",
            "effective_options",
            "termination_entropy",
            "termination_prior_loss",
            "attention_diversity_loss",
            "attention_temporal_loss",
            "mean_attention",
            "mean_beta",
            "mean_termination_advantage",
            "mean_termination_signal",
            "termination_low_saturation",
            "termination_high_saturation",
            "action_approx_kl",
            "option_approx_kl",
            "behavior_action_logp_error",
            "behavior_option_logp_error",
        )
        totals = {name: 0.0 for name in metric_names}
        num_batches = 0
        actor_updates = 0
        critic_updates = 0
        optimizer_samples = 0
        actor_gradient_norm_total = 0.0
        critic_gradient_norm_total = 0.0
        max_policy_kl = 0.0
        max_action_kl = 0.0
        max_option_kl = 0.0
        initial_policy_kl = 0.0
        actor_early_stopped = False
        cfg = self.cfg
        self.reference_actor.load_state_dict(self.actor.state_dict())
        self.reference_actor.eval()
        self.reference_actor.manager_lstm.flatten_parameters()
        self.reference_actor.option_lstm.flatten_parameters()

        for _epoch in range(cfg.num_epochs):
            for batch in self.buffer.get_sequence_batches(
                cfg.sequence_length,
                cfg.mini_batch_size,
            ):
                optimizer_samples += int(batch["loss_mask"].sum().item())
                losses = self._compute_sequence_losses(
                    batch,
                    self.current_eps,
                    self.reference_actor,
                )
                action_kl = losses["action_approx_kl"].item()
                option_kl = losses["option_approx_kl"].item()
                policy_kl = max(action_kl, option_kl)
                if num_batches == 0:
                    initial_policy_kl = policy_kl
                    if policy_kl > 1e-6:
                        raise RuntimeError(
                            "OC2 update-start policy does not match its "
                            f"frozen reference (KL={policy_kl:.6g})."
                        )
                max_policy_kl = max(max_policy_kl, policy_kl)
                max_action_kl = max(max_action_kl, action_kl)
                max_option_kl = max(max_option_kl, option_kl)
                apply_actor_update = not actor_early_stopped
                if (
                    apply_actor_update
                    and cfg.target_kl > 0.0
                    and policy_kl > 1.5 * cfg.target_kl
                ):
                    actor_early_stopped = True
                    apply_actor_update = False
                    print(
                        "[LearnedOC] Actor PPO early stop: policy KL "
                        f"{policy_kl:.4f} exceeded "
                        f"{1.5 * cfg.target_kl:.4f}; centralized critics "
                        "continue"
                    )
                actor_loss = (
                    cfg.intra_option_coef * losses["intra_option_loss"]
                    + cfg.selector_coef * losses["selector_loss"]
                    + cfg.local_option_value_coef
                    * losses["local_option_value_loss"]
                    + cfg.termination_coef * losses["termination_loss"]
                    + self.current_termination_prior_coef
                    * losses["termination_prior_loss"]
                    + self.current_option_balance_coef
                    * losses["option_balance_loss"]
                    + cfg.attention_diversity_coef
                    * losses["attention_diversity_loss"]
                    + cfg.attention_temporal_coef
                    * losses["attention_temporal_loss"]
                    - self.current_beta * losses["action_entropy"]
                    - cfg.option_entropy_coef * losses["option_entropy"]
                    - cfg.termination_entropy_coef
                    * losses["termination_entropy"]
                )
                critic_loss = (
                    cfg.value_coef * losses["value_loss"]
                    + cfg.action_baseline_coef
                    * losses["action_baseline_loss"]
                    + cfg.option_value_coef
                    * losses["joint_option_value_loss"]
                    + cfg.option_baseline_coef
                    * losses["option_baseline_loss"]
                )

                if not torch.isfinite(actor_loss):
                    bad_losses = {
                        name: float(value.detach().cpu())
                        for name, value in losses.items()
                        if not torch.isfinite(value)
                    }
                    raise FloatingPointError(
                        "LearnedOC produced a non-finite actor loss before "
                        f"backward: {bad_losses}"
                    )
                if not torch.isfinite(critic_loss):
                    bad_losses = {
                        name: float(value.detach().cpu())
                        for name, value in losses.items()
                        if not torch.isfinite(value)
                    }
                    raise FloatingPointError(
                        "LearnedOC produced a non-finite critic loss before "
                        f"backward: {bad_losses}"
                    )

                if apply_actor_update:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.actor_parameters,
                        cfg.actor_max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    self.actor_optimizer.step()
                    actor_gradient_norm_total += float(actor_gradient_norm)
                    actor_updates += 1

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic_parameters,
                    cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.critic_optimizer.step()

                for name in metric_names:
                    totals[name] += losses[name].item()
                critic_gradient_norm_total += float(critic_gradient_norm)
                critic_updates += 1
                num_batches += 1

        if actor_updates == 0:
            raise RuntimeError(
                "OC2 applied no actor updates for this rollout. The frozen "
                "reference invariant should guarantee at least one safe "
                "policy minibatch."
            )

        bad_parameters: list[str] = []
        for module_name, module in (
            ("actor", self.actor),
            ("team_critic", self.team_critic),
            ("action_critic", self.action_critic),
            ("option_critic", self.option_critic),
        ):
            bad_parameters.extend(
                f"{module_name}.{parameter_name}"
                for parameter_name, parameter in module.named_parameters()
                if not torch.isfinite(parameter).all()
            )
        if bad_parameters:
            raise FloatingPointError(
                "LearnedOC optimizer produced non-finite parameters: "
                + ", ".join(bad_parameters[:10])
            )

        self.update_count += 1
        divisor = max(num_batches, 1)
        metrics = {
            name: total / divisor
            for name, total in totals.items()
        }
        option_counts = torch.bincount(
            self.buffer.options[:active].reshape(-1),
            minlength=cfg.num_options,
        ).float()
        switch_rate = self.buffer.option_masks[:active].mean().item()
        active_actions = self.buffer.actions[:active]
        option_stds = self.actor.option_log_stds().detach().exp().mean(dim=-1)
        termination_options = self.buffer.termination_options[:active]
        termination_valid = self.buffer.termination_valid[:active] > 0.5
        per_option_beta = []
        per_option_switch = []
        per_option_termination_count = []
        for option_id in range(cfg.num_options):
            option_mask = termination_valid & (
                termination_options == option_id
            )
            count = int(option_mask.sum().item())
            per_option_termination_count.append(count)
            if count > 0:
                per_option_beta.append(
                    self.buffer.beta_probs[:active][option_mask].mean().item()
                )
                per_option_switch.append(
                    self.buffer.option_masks[:active][option_mask].mean().item()
                )
            else:
                per_option_beta.append(0.0)
                per_option_switch.append(0.0)
        applied_actor_lr = self.current_actor_lr
        if cfg.adaptive_actor_lr and cfg.target_kl > 0.0:
            if actor_early_stopped or max_policy_kl > cfg.target_kl:
                self.actor_lr_scale /= cfg.actor_lr_decay_factor
            elif max_policy_kl < cfg.target_kl / 3.0:
                self.actor_lr_scale *= cfg.actor_lr_recovery_factor
            self.actor_lr_scale = min(
                1.0,
                max(cfg.actor_lr_scale_min, self.actor_lr_scale),
            )

        metrics.update({
            "lr": self.current_lr,
            "actor_lr": applied_actor_lr,
            "actor_lr_scale": self.actor_lr_scale,
            "eps": self.current_eps,
            "beta": self.current_beta,
            "termination_prior_coef": (
                self.current_termination_prior_coef
            ),
            "option_balance_coef": self.current_option_balance_coef,
            "gradient_norm": (
                actor_gradient_norm_total / max(actor_updates, 1)
            ),
            "critic_gradient_norm": (
                critic_gradient_norm_total / max(critic_updates, 1)
            ),
            "max_policy_kl": max_policy_kl,
            "max_action_kl": max_action_kl,
            "max_option_kl": max_option_kl,
            "initial_policy_kl": initial_policy_kl,
            "kl_early_stop": float(actor_early_stopped),
            "actor_updates": float(actor_updates),
            "critic_updates": float(critic_updates),
            "optimizer_samples": float(optimizer_samples),
            "actor_update_fraction": (
                actor_updates / max(num_batches, 1)
            ),
            "switch_rate": switch_rate,
            "mean_option_duration": 1.0 / max(switch_rate, 1e-8),
            "mean_abs_action": active_actions.abs().mean().item(),
            "action_saturation": (
                active_actions.abs() > 0.95
            ).float().mean().item(),
            "option_stds": option_stds.tolist(),
            "option_betas": per_option_beta,
            "option_switch_rates": per_option_switch,
            "option_termination_counts": per_option_termination_count,
            "option_usage": (
                option_counts
                / option_counts.sum().clamp_min(1.0)
            ).tolist(),
        })
        return metrics

    def _zero_memories(self):
        self.current_options.fill_(-1)
        for tensor in (
            self.actor_memory_h,
            self.actor_memory_c,
            self.team_memory_h,
            self.team_memory_c,
            self.action_baseline_memory_h,
            self.action_baseline_memory_c,
            self.option_joint_memory_h,
            self.option_joint_memory_c,
            self.option_baseline_memory_h,
            self.option_baseline_memory_c,
        ):
            tensor.zero_()

    def train(self):
        start_time = time.time()
        start_step = self.global_step
        obs_dict, _ = self.env.reset()
        self._zero_memories()
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        progress = tqdm(
            total=self.cfg.total_timesteps,
            initial=self.global_step,
            desc="LearnedOC Training",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )

        while self.global_step < self.cfg.total_timesteps:
            previous_step = self.global_step
            self.buffer.reset()
            rollout_seconds = 0.0
            while self.global_step < self.cfg.total_timesteps:
                remaining = self.cfg.total_timesteps - self.global_step
                agent_steps = self.num_envs * self.num_agents
                remaining_steps = max(
                    1,
                    (remaining + agent_steps - 1) // agent_steps,
                )
                episode_step = int(
                    self.unwrapped.episode_length_buf.max().item()
                )
                episode_steps_left = max(
                    1,
                    (
                        self.unwrapped.max_episode_length
                        - episode_step
                        + self.decision_period
                        - 1
                    ) // self.decision_period,
                )
                rollout_steps = min(
                    self.cfg.horizon,
                    remaining_steps,
                    episode_steps_left,
                )
                rollout_start = time.perf_counter()
                obs_dict = self.collect_rollout(
                    obs_dict,
                    rollout_steps,
                    reset_buffer=False,
                )
                rollout_seconds += time.perf_counter() - rollout_start
                experiences = self.buffer.ptr * agent_steps
                if experiences > self.cfg.buffer_size_hint:
                    break

            update_start = time.perf_counter()
            metrics = self.update()
            update_seconds = time.perf_counter() - update_start
            step_delta = self.global_step - previous_step
            metrics.update({
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "rollout_sps": step_delta / max(rollout_seconds, 1e-9),
                "optimizer_samples_per_second": (
                    metrics["optimizer_samples"]
                    / max(update_seconds, 1e-9)
                ),
            })
            elapsed = time.time() - start_time
            steps_per_second = (
                (self.global_step - start_step) / elapsed
                if elapsed > 0
                else 0.0
            )
            progress.update(min(
                step_delta,
                max(0, self.cfg.total_timesteps - progress.n),
            ))
            progress.set_postfix(
                act=f"{metrics['intra_option_loss']:.3f}",
                opt=f"{metrics['selector_loss']:.3f}",
                term=f"{metrics['termination_loss']:.3f}",
                sw=f"{metrics['switch_rate']:.2f}",
                kl=f"{metrics['max_policy_kl']:.3f}",
                kl_stop=int(metrics["kl_early_stop"]),
                actor_frac=f"{metrics['actor_update_fraction']:.2f}",
                SPS=f"{steps_per_second:.0f}",
            )
            self._write_update_diagnostics(metrics)

            active_rewards = self.buffer.rewards[:self.buffer.ptr]
            mean_rollout_reward = active_rewards.sum(dim=0).mean().item()
            self._rollout_reward_history.append(mean_rollout_reward)
            if len(self._rollout_reward_history) > self._max_history:
                self._rollout_reward_history.pop(0)

            if self.global_step >= self._next_summary_step:
                self._next_summary_step += self.cfg.summary_freq
                self._write_summary(
                    metrics,
                    steps_per_second,
                    mean_rollout_reward,
                )

            if self.global_step >= self._next_checkpoint_step:
                self.save_checkpoint(
                    checkpoint_dir
                    / f"option_critic_2_{self.global_step}.pt"
                )
                self._next_checkpoint_step += self.cfg.checkpoint_interval
                self._manage_checkpoints(checkpoint_dir)

        progress.close()
        self.writer.close()
        self.save_checkpoint(
            checkpoint_dir / "option_critic_2_final.pt"
        )
        elapsed = time.time() - start_time
        trained_steps = self.global_step - start_step
        print(
            f"[LearnedOC] Done - {self.global_step:,} total steps in "
            f"{elapsed:.0f}s ({trained_steps / max(elapsed, 1e-9):.0f} SPS "
            "for this session)"
        )

    def _write_update_diagnostics(self, metrics: dict):
        """Log stability signals for every optimizer update.

        Summary intervals need not align with rollout boundaries, so logging
        these per-update values prevents a short collapse or KL intervention
        from disappearing between periodic mission summaries.
        """
        step = self.global_step
        update_scalars = {
            "Update/Max Policy KL": "max_policy_kl",
            "Update/KL Early Stop": "kl_early_stop",
            "Update/Actor Update Fraction": "actor_update_fraction",
            "Update/Actor Learning Rate": "actor_lr",
            "Update/Actor Learning Rate Scale": "actor_lr_scale",
            "Update/Mean Termination Probability": "mean_beta",
            "Update/Mean Termination Signal": "mean_termination_signal",
            "Update/Termination Low Saturation": (
                "termination_low_saturation"
            ),
            "Update/Termination Prior Loss": "termination_prior_loss",
            "Update/Effective Options": "effective_options",
            "Update/Option Balance Loss": "option_balance_loss",
            "Update/Switch Rate": "switch_rate",
            "Performance/Rollout Seconds": "rollout_seconds",
            "Performance/Update Seconds": "update_seconds",
            "Performance/Rollout SPS": "rollout_sps",
            "Performance/Optimizer Samples Per Second": (
                "optimizer_samples_per_second"
            ),
        }
        for tag, metric_name in update_scalars.items():
            self.writer.add_scalar(tag, metrics[metric_name], step)

    def _write_summary(
        self,
        metrics: dict,
        steps_per_second: float,
        mean_rollout_reward: float,
    ):
        step = self.global_step
        scalar_names = {
            "Losses/Intra-Option Policy Loss": "intra_option_loss",
            "Losses/Option Selector Loss": "selector_loss",
            "Losses/Local Attended Option Value": (
                "local_option_value_loss"
            ),
            "Losses/Value Loss": "value_loss",
            "Losses/Counterfactual Action Baseline Loss": (
                "action_baseline_loss"
            ),
            "Losses/Collective Option Value Loss": (
                "joint_option_value_loss"
            ),
            "Losses/Counterfactual Option Baseline Loss": (
                "option_baseline_loss"
            ),
            "Losses/Termination Loss": "termination_loss",
            "Losses/Termination Prior": "termination_prior_loss",
            "Losses/Option Balance": "option_balance_loss",
            "Losses/Attention Diversity": "attention_diversity_loss",
            "Losses/Attention Temporal": "attention_temporal_loss",
            "Policy/Pre-Squash Intra-Option Entropy": "action_entropy",
            "Policy/Option Entropy At Boundaries": "option_entropy",
            "Policy/Option Marginal Entropy": "option_marginal_entropy",
            "Policy/Effective Options": "effective_options",
            "Policy/Termination Entropy": "termination_entropy",
            "Policy/Mean Attention": "mean_attention",
            "Policy/Local Option Value Mean": "local_option_value_mean",
            "Policy/Local Option Value Spread": "option_value_spread",
            "Policy/Mean Termination Probability": "mean_beta",
            "Policy/Mean Termination Advantage": (
                "mean_termination_advantage"
            ),
            "Policy/Mean Termination Signal": "mean_termination_signal",
            "Policy/Termination Low Saturation": (
                "termination_low_saturation"
            ),
            "Policy/Termination High Saturation": (
                "termination_high_saturation"
            ),
            "Policy/Switch Rate": "switch_rate",
            "Policy/Mean Option Duration Decisions": "mean_option_duration",
            "Policy/Mean Absolute Wheel Action": "mean_abs_action",
            "Policy/Wheel Action Saturation": "action_saturation",
            "Policy/Learning Rate": "lr",
            "Policy/Actor Learning Rate": "actor_lr",
            "Policy/Actor Learning Rate Scale": "actor_lr_scale",
            "Policy/Termination Prior Coef": "termination_prior_coef",
            "Policy/Option Balance Coef": "option_balance_coef",
            "Policy/Epsilon": "eps",
            "Policy/Beta": "beta",
            "Diagnostics/Gradient Norm Before Clip": "gradient_norm",
            "Diagnostics/Actor Gradient Norm Before Clip": "gradient_norm",
            "Diagnostics/Critic Gradient Norm Before Clip": (
                "critic_gradient_norm"
            ),
            "Diagnostics/Max Policy KL": "max_policy_kl",
            "Diagnostics/Max Action KL": "max_action_kl",
            "Diagnostics/Max Option KL": "max_option_kl",
            "Diagnostics/Initial Policy KL": "initial_policy_kl",
            "Diagnostics/KL Early Stop": "kl_early_stop",
            "Diagnostics/Actor Updates Applied": "actor_updates",
            "Diagnostics/Critic Updates Applied": "critic_updates",
            "Diagnostics/Optimizer Samples": "optimizer_samples",
            "Diagnostics/Actor Update Fraction": "actor_update_fraction",
            "Diagnostics/Behavior Action Log Prob Error": (
                "behavior_action_logp_error"
            ),
            "Diagnostics/Behavior Option Log Prob Error": (
                "behavior_option_logp_error"
            ),
        }
        for tag, metric_name in scalar_names.items():
            self.writer.add_scalar(tag, metrics[metric_name], step)
        for option_id, usage in enumerate(metrics["option_usage"]):
            self.writer.add_scalar(
                f"Policy/Option Usage/{option_id}",
                usage,
                step,
            )
        for option_id, std in enumerate(metrics["option_stds"]):
            self.writer.add_scalar(
                f"Policy/Intra-Option Std/{option_id}",
                std,
                step,
            )
        for option_id, count in enumerate(
            metrics["option_termination_counts"]
        ):
            if count <= 0:
                continue
            self.writer.add_scalar(
                f"Policy/Termination Probability/{option_id}",
                metrics["option_betas"][option_id],
                step,
            )
            self.writer.add_scalar(
                f"Policy/Option Switch Rate/{option_id}",
                metrics["option_switch_rates"][option_id],
                step,
            )

        active = self.buffer.ptr
        self.writer.add_scalar(
            "Policy/Extrinsic Reward",
            self.buffer.rewards[:active].mean().item(),
            step,
        )
        self.writer.add_scalar(
            "Policy/Extrinsic Value Estimate",
            self.buffer.team_values[:active].mean().item(),
            step,
        )
        self.writer.add_scalar("Extra/SPS", steps_per_second, step)
        self.writer.add_scalar(
            "Extra/Mean Rollout Reward",
            mean_rollout_reward,
            step,
        )
        self.writer.add_scalar(
            "Extra/Rolling Avg Rollout Reward",
            sum(self._rollout_reward_history)
            / len(self._rollout_reward_history),
            step,
        )

        if self._completed_episode_returns:
            values = self._completed_episode_returns
            self.writer.add_scalar(
                "Environment/Cumulative Reward",
                sum(values) / len(values),
                step,
            )
            values.clear()
        if self._completed_episode_lengths:
            values = self._completed_episode_lengths
            self.writer.add_scalar(
                "Environment/Episode Length",
                sum(values) / len(values),
                step,
            )
            values.clear()
        if self._completed_group_rewards:
            values = self._completed_group_rewards
            self.writer.add_scalar(
                "Extra/Group Reward Mean",
                sum(values) / len(values),
                step,
            )
            values.clear()

    def save_checkpoint(self, path: str | Path):
        torch.save({
            "trainer_type": "learned_option_critic",
            "option_critic_phase": 2,
            "learned_option_critic_version": self.CHECKPOINT_VERSION,
            "training_checkpoint_version": self.TRAINING_CHECKPOINT_VERSION,
            "paper_parity_version": PAPER_PARITY_VERSION,
            "fixed_options": False,
            "learned_options": True,
            "collective_counterfactual": True,
            "attention_options": True,
            "attention_conditioned_outputs": [
                "option_selector",
                "local_option_value",
                "intra_option_policy",
                "termination",
            ],
            "option_selection": "categorical_attended_selector",
            "separate_selector_value_heads": True,
            "primitive_action_space": "continuous_wheels",
            "action_distribution": "tanh_squashed_normal",
            "action_transform": "identity_normalized",
            "variant": self.variant,
            "actor": self.actor.state_dict(),
            "team_critic": self.team_critic.state_dict(),
            "action_critic": self.action_critic.state_dict(),
            "option_critic": self.option_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "actor_lr_scale": self.actor_lr_scale,
            "seed": self.cfg.seed,
            "hidden_dim": self.cfg.hidden_dim,
            "num_layers": self.cfg.num_layers,
            "recurrent": True,
            "memory_size": self.cfg.memory_size,
            "memory_size_semantics": "mlagents_total",
            "lstm_hidden_size": self.actor.manager_hidden_size,
            "actor_packed_memory_size": self.actor.hidden_size,
            "sequence_length": self.cfg.sequence_length,
            "option_hidden_dim": self.cfg.option_hidden_dim,
            "option_num_layers": self.cfg.option_num_layers,
            "option_memory_size": self.cfg.option_memory_size,
            "option_recurrent_size": self.actor.option_recurrent_size,
            "initial_termination_probability": (
                self.cfg.initial_termination_probability
            ),
            "initial_log_std": self.cfg.initial_log_std,
            "min_log_std": self.cfg.min_log_std,
            "max_log_std": self.cfg.max_log_std,
            "option_selector_temperature": (
                self.cfg.option_selector_temperature
            ),
            "actor_learning_rate": self.cfg.actor_lr,
            "critic_learning_rate": self.cfg.lr,
            "actor_max_grad_norm": self.cfg.actor_max_grad_norm,
            "max_grad_norm": self.cfg.max_grad_norm,
            "target_kl": self.cfg.target_kl,
            "adaptive_actor_lr": self.cfg.adaptive_actor_lr,
            "fused_optimizer": self.fused_optimizer_active,
            "matmul_precision": self.cfg.matmul_precision,
            "termination_prior_probability": (
                self.cfg.termination_prior_probability
            ),
            "termination_prior_coef": self.cfg.termination_prior_coef,
            "termination_prior_final_coef": (
                self.cfg.termination_prior_final_coef
            ),
            "option_balance_coef": self.cfg.option_balance_coef,
            "option_balance_final_coef": (
                self.cfg.option_balance_final_coef
            ),
            "critic_hidden_dim": self.cfg.critic_hidden_dim,
            "critic_num_layers": self.cfg.critic_num_layers,
            "critic_num_heads": self.cfg.critic_num_heads,
            "decision_period": self.decision_period,
            "discrete": False,
            "num_actions": self.act_dim,
            "num_options": self.cfg.num_options,
            "act_dim": self.act_dim,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
        }, path)
        print(f"[LearnedOC] Saved -> {path}")

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("trainer_type") != "learned_option_critic":
            raise RuntimeError(
                "Checkpoint is not a learned Option-Critic Phase 2 model."
            )
        version = int(
            checkpoint.get("learned_option_critic_version", 0)
        )
        if version != self.CHECKPOINT_VERSION:
            raise RuntimeError(
                f"Checkpoint uses learned Option-Critic version {version}; "
                f"this trainer expects version {self.CHECKPOINT_VERSION}. "
                "OC2 v3 separates selector policy logits from local option "
                "values and requires fresh training."
            )
        training_version = int(
            checkpoint.get("training_checkpoint_version", 0)
        )
        if training_version != self.TRAINING_CHECKPOINT_VERSION:
            raise RuntimeError(
                "Checkpoint uses a legacy OC2 training "
                f"layout (version {training_version}); this trainer expects "
                f"version {self.TRAINING_CHECKPOINT_VERSION}. The corrected "
                "anti-collapse regularization and adaptive PPO schedule "
                "require a fresh training run."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.team_critic.load_state_dict(checkpoint["team_critic"])
        self.action_critic.load_state_dict(checkpoint["action_critic"])
        self.option_critic.load_state_dict(checkpoint["option_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )
        self.global_step = int(checkpoint["global_step"])
        self.update_count = int(checkpoint["update_count"])
        self.actor_lr_scale = float(checkpoint.get("actor_lr_scale", 1.0))
        # Resume periodic work at the first boundary strictly after the
        # restored step. Without this, long-running jobs save and summarize
        # every update while counters catch up from their initial interval.
        if self.cfg.checkpoint_interval > 0:
            self._next_checkpoint_step = (
                self.global_step // self.cfg.checkpoint_interval + 1
            ) * self.cfg.checkpoint_interval
        if self.cfg.summary_freq > 0:
            self._next_summary_step = (
                self.global_step // self.cfg.summary_freq + 1
            ) * self.cfg.summary_freq
        print(
            f"[LearnedOC] Loaded <- {path} "
            f"(step {self.global_step})"
        )

    def _manage_checkpoints(self, checkpoint_dir: Path):
        keep = self.cfg.keep_checkpoints
        if keep <= 0:
            return
        numbered = sorted(
            checkpoint_dir.glob("option_critic_2_*.pt"),
            key=lambda candidate: candidate.stat().st_mtime,
        )
        numbered = [
            candidate
            for candidate in numbered
            if candidate.stem != "option_critic_2_final"
        ]
        while len(numbered) > keep:
            old = numbered.pop(0)
            old.unlink()
            print(f"[LearnedOC] Removed old checkpoint -> {old.name}")
