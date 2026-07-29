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

import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter

from .learned_option_critic_buffer import LearnedOptionRolloutBuffer
from .learned_option_critic_networks import LearnedOptionActor
from .network_config import PAPER_PARITY_VERSION
from .poca_networks import POCACritic
from .poca_trainer import (
    PolynomialDecay,
    trust_region_policy_loss,
    trust_region_value_loss,
)


@dataclass
class LearnedOptionCriticConfig:
    """Training hyperparameters for learned Option-Critic."""

    trainer_type: str = "learned_option_critic"

    # Rollout and PPO.
    horizon: int = 1000
    num_epochs: int = 3
    mini_batch_size: int = 2048
    clip_eps: float = 0.2
    beta: float = 0.005

    # Objective weights.
    intra_option_coef: float = 1.0
    selector_coef: float = 1.0
    option_entropy_coef: float = 0.005
    value_coef: float = 0.5
    action_baseline_coef: float = 0.25
    option_value_coef: float = 0.5
    option_baseline_coef: float = 0.25
    termination_coef: float = 0.1
    termination_entropy_coef: float = 0.001
    termination_penalty: float = 0.01
    attention_diversity_coef: float = 0.01
    attention_temporal_coef: float = 0.01

    # GAE.
    gamma: float = 0.99
    lam: float = 0.95

    # Optimizer and schedules.
    lr: float = 3e-4
    adam_eps: float = 1e-8
    lr_schedule: str = "constant"
    eps_schedule: str = "constant"
    beta_schedule: str = "constant"

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

    # Logging.
    log_dir: str = "runs/learned_option_critic"
    buffer_size_hint: int = 0


class LearnedOptionCriticTrainer:
    """Train learned continuous options with collective counterfactual credit."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        env,
        cfg: LearnedOptionCriticConfig | None = None,
    ):
        self.env = env
        self.cfg = cfg or LearnedOptionCriticConfig()
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
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
        print(
            f"[LearnedOC] envs={self.num_envs}  agents={self.num_agents}  "
            f"obs={self.obs_dim}  state={self.state_dim}  "
            f"wheel_actions={self.act_dim}  options={cfg.num_options}  "
            f"decision_period={self.decision_period}"
        )
        print(
            "[LearnedOC] Learning recurrent policy-over-options, "
            "terminations, attentions, and continuous intra-option policies"
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

        parameters = (
            list(self.actor.parameters())
            + list(self.team_critic.parameters())
            + list(self.action_critic.parameters())
            + list(self.option_critic.parameters())
        )
        self.optimizer = optim.Adam(
            parameters,
            lr=cfg.lr,
            eps=cfg.adam_eps,
        )

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
        self.current_lr = cfg.lr
        self.current_eps = cfg.clip_eps
        self.current_beta = cfg.beta
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
            "[LearnedOC] Critic params: "
            f"team={sum(p.numel() for p in self.team_critic.parameters()):,}  "
            f"action={sum(p.numel() for p in self.action_critic.parameters()):,}  "
            f"option={sum(p.numel() for p in self.option_critic.parameters()):,}"
        )
        print(f"[LearnedOC] TensorBoard -> {cfg.log_dir}")

    def _apply_schedules(self):
        if self.lr_schedule is not None:
            self.current_lr = self.lr_schedule.get(self.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr
        if self.eps_schedule is not None:
            self.current_eps = self.eps_schedule.get(self.global_step)
        if self.beta_schedule is not None:
            self.current_beta = self.beta_schedule.get(self.global_step)

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
                option_logits,
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

            option_dist = Categorical(logits=option_logits)
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
            action_dist = self.actor.selected_action_dist(
                action_means,
                action_stds,
                flat_options,
            )
            raw_actions = action_dist.sample().view(
                self.num_envs,
                self.num_agents,
                self.act_dim,
            )
            action_log_probs = action_dist.log_prob(
                raw_actions.reshape(-1, self.act_dim),
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
                raw_actions,
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

            env_actions = raw_actions.clamp(-3.0, 3.0) / 3.0
            action_dict = {
                agent: env_actions[:, agent_id]
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
                option_masks=option_mask.float(),
                beta_probs=beta_probs,
                actions=raw_actions,
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
            option_logits,
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

        flat_option_dist = Categorical(
            logits=option_logits.reshape(
                batch_size * sequence_length,
                self.cfg.num_options,
            )
        )
        new_option_logp = flat_option_dist.log_prob(
            options.reshape(-1),
        ).view(batch_size, sequence_length)
        option_entropy_values = flat_option_dist.entropy().view(
            batch_size,
            sequence_length,
        )
        option_entropy = (
            option_entropy_values * loss_mask
        ).sum() / loss_mask.sum().clamp_min(1)

        boundary_mask = (
            (batch["option_masks"] > 0.5) & loss_mask
        ).reshape(-1)
        flat_new_option_logp = new_option_logp.reshape(-1)
        flat_old_option_logp = batch["old_option_log_probs"].reshape(-1)
        flat_option_advantage = batch["option_advantages"].reshape(
            -1,
        ).detach()
        if boundary_mask.any():
            option_ratio = (
                flat_new_option_logp[boundary_mask]
                - flat_old_option_logp[boundary_mask]
            ).exp()
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
        else:
            selector_loss = flat_new_option_logp.sum() * 0.0

        action_dist = self.actor.selected_action_dist(
            action_means,
            action_stds,
            options,
        )
        new_action_logp = action_dist.log_prob(actions)
        action_entropy_values = action_dist.entropy().mean(dim=-1)
        intra_option_loss = trust_region_policy_loss(
            batch["action_advantages"].reshape(-1, 1).detach(),
            new_action_logp.reshape(-1, self.act_dim),
            batch["old_action_log_probs"].reshape(-1, self.act_dim),
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
            next_option_logits,
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
            next_selector_probs = torch.softmax(
                next_option_logits,
                dim=-1,
            )
            next_reselection = (
                next_alternatives * next_selector_probs
            ).sum(dim=-1)
            termination_advantage = (
                next_q_current - next_reselection
            ).view(batch_size, sequence_length)

        termination_mask = (1.0 - dones) * loss_mask
        termination_count = termination_mask.sum()
        if termination_count.item() > 0:
            termination_signal = (
                termination_advantage + self.cfg.termination_penalty
            )
            termination_loss = (
                next_beta * termination_signal * termination_mask
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
        else:
            termination_loss = next_beta.sum() * 0.0
            termination_entropy = next_beta.sum() * 0.0
            mean_beta = next_beta.sum() * 0.0
            mean_termination_advantage = next_beta.sum() * 0.0

        (
            attention_diversity_loss,
            attention_temporal_loss,
            mean_attention,
        ) = self._attention_losses(attentions, loss_mask, dones)

        return {
            "intra_option_loss": intra_option_loss,
            "selector_loss": selector_loss,
            "value_loss": value_loss,
            "action_baseline_loss": action_baseline_loss,
            "joint_option_value_loss": joint_option_value_loss,
            "option_baseline_loss": option_baseline_loss,
            "termination_loss": termination_loss,
            "action_entropy": action_entropy,
            "option_entropy": option_entropy,
            "termination_entropy": termination_entropy,
            "attention_diversity_loss": attention_diversity_loss,
            "attention_temporal_loss": attention_temporal_loss,
            "mean_attention": mean_attention,
            "mean_beta": mean_beta,
            "mean_termination_advantage": mean_termination_advantage,
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
            "value_loss",
            "action_baseline_loss",
            "joint_option_value_loss",
            "option_baseline_loss",
            "termination_loss",
            "action_entropy",
            "option_entropy",
            "termination_entropy",
            "attention_diversity_loss",
            "attention_temporal_loss",
            "mean_attention",
            "mean_beta",
            "mean_termination_advantage",
        )
        totals = {name: 0.0 for name in metric_names}
        num_updates = 0
        cfg = self.cfg

        for _epoch in range(cfg.num_epochs):
            for batch in self.buffer.get_sequence_batches(
                cfg.sequence_length,
                cfg.mini_batch_size,
            ):
                losses = self._compute_sequence_losses(
                    batch,
                    self.current_eps,
                )
                total_loss = (
                    cfg.intra_option_coef * losses["intra_option_loss"]
                    + cfg.selector_coef * losses["selector_loss"]
                    + cfg.value_coef * losses["value_loss"]
                    + cfg.action_baseline_coef
                    * losses["action_baseline_loss"]
                    + cfg.option_value_coef
                    * losses["joint_option_value_loss"]
                    + cfg.option_baseline_coef
                    * losses["option_baseline_loss"]
                    + cfg.termination_coef * losses["termination_loss"]
                    + cfg.attention_diversity_coef
                    * losses["attention_diversity_loss"]
                    + cfg.attention_temporal_coef
                    * losses["attention_temporal_loss"]
                    - self.current_beta * losses["action_entropy"]
                    - cfg.option_entropy_coef * losses["option_entropy"]
                    - cfg.termination_entropy_coef
                    * losses["termination_entropy"]
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                for name in metric_names:
                    totals[name] += losses[name].item()
                num_updates += 1

        self.update_count += 1
        divisor = max(num_updates, 1)
        metrics = {
            name: total / divisor
            for name, total in totals.items()
        }
        option_counts = torch.bincount(
            self.buffer.options[:active].reshape(-1),
            minlength=cfg.num_options,
        ).float()
        metrics.update({
            "lr": self.current_lr,
            "eps": self.current_eps,
            "beta": self.current_beta,
            "switch_rate": (
                self.buffer.option_masks[:active].mean().item()
            ),
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
                obs_dict = self.collect_rollout(
                    obs_dict,
                    rollout_steps,
                    reset_buffer=False,
                )
                experiences = self.buffer.ptr * agent_steps
                if experiences > self.cfg.buffer_size_hint:
                    break

            metrics = self.update()
            step_delta = self.global_step - previous_step
            elapsed = time.time() - start_time
            steps_per_second = (
                self.global_step / elapsed if elapsed > 0 else 0.0
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
                SPS=f"{steps_per_second:.0f}",
            )

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
        print(
            f"[LearnedOC] Done - {self.global_step:,} steps in "
            f"{elapsed:.0f}s ({self.global_step / elapsed:.0f} SPS)"
        )

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
            "Losses/Attention Diversity": "attention_diversity_loss",
            "Losses/Attention Temporal": "attention_temporal_loss",
            "Policy/Intra-Option Entropy": "action_entropy",
            "Policy/Option Entropy": "option_entropy",
            "Policy/Termination Entropy": "termination_entropy",
            "Policy/Mean Attention": "mean_attention",
            "Policy/Mean Termination Probability": "mean_beta",
            "Policy/Mean Termination Advantage": (
                "mean_termination_advantage"
            ),
            "Policy/Switch Rate": "switch_rate",
            "Policy/Learning Rate": "lr",
            "Policy/Epsilon": "eps",
            "Policy/Beta": "beta",
        }
        for tag, metric_name in scalar_names.items():
            self.writer.add_scalar(tag, metrics[metric_name], step)
        for option_id, usage in enumerate(metrics["option_usage"]):
            self.writer.add_scalar(
                f"Policy/Option Usage/{option_id}",
                usage,
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
            "paper_parity_version": PAPER_PARITY_VERSION,
            "fixed_options": False,
            "learned_options": True,
            "collective_counterfactual": True,
            "attention_options": True,
            "primitive_action_space": "continuous_wheels",
            "variant": self.variant,
            "actor": self.actor.state_dict(),
            "team_critic": self.team_critic.state_dict(),
            "action_critic": self.action_critic.state_dict(),
            "option_critic": self.option_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "seed": self.cfg.seed,
            "hidden_dim": self.cfg.hidden_dim,
            "num_layers": self.cfg.num_layers,
            "recurrent": True,
            "memory_size": self.cfg.memory_size,
            "memory_size_semantics": "mlagents_total",
            "lstm_hidden_size": self.actor.hidden_size,
            "sequence_length": self.cfg.sequence_length,
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
                f"this trainer expects version {self.CHECKPOINT_VERSION}."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.team_critic.load_state_dict(checkpoint["team_critic"])
        self.action_critic.load_state_dict(checkpoint["action_critic"])
        self.option_critic.load_state_dict(checkpoint["option_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = int(checkpoint["global_step"])
        self.update_count = int(checkpoint["update_count"])
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
