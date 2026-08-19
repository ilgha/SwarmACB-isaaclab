# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-module collective Option-Critic trainer for SwarmACB phase 1.

This trainer starts from the cyclamen controller shape: local 4D observations,
recurrent memory, and the six predefined ACB behavior modules.  Unlike POCA,
the policy does not choose a fresh module at every decision. It chooses an
option and learns a termination model that decides when to keep or switch it.
Centralized RSA critics train the shared local manager with collective
counterfactual signals and are discarded during decentralized execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

import torch
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical
from torch.utils.tensorboard import SummaryWriter

from .option_critic_buffer import FixedOptionRolloutBuffer
from .option_critic_networks import FixedOptionManager
from .poca_networks import POCACritic
from .poca_trainer import PolynomialDecay, trust_region_value_loss
from .network_config import PAPER_PARITY_VERSION


@dataclass
class FixedOptionCriticConfig:
    """Training hyper-parameters for fixed-module Option-Critic."""

    trainer_type: str = "option_critic"

    # Rollout
    horizon: int = 1000
    num_epochs: int = 3
    mini_batch_size: int = 2048

    # PPO-style option selection
    clip_eps: float = 0.2
    beta: float = 0.005
    value_coef: float = 0.5
    option_value_coef: float = 0.5
    baseline_coef: float = 0.25

    # Option termination
    termination_coef: float = 0.1
    termination_entropy_coef: float = 0.001
    termination_penalty: float = 0.01

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # Optimizer
    lr: float = 3e-4
    adam_eps: float = 1e-8

    # Schedules
    lr_schedule: str = "constant"
    eps_schedule: str = "constant"
    beta_schedule: str = "constant"

    # Run control
    total_timesteps: int = 120_000_000
    checkpoint_interval: int = 120_000
    summary_freq: int = 120_000
    keep_checkpoints: int = 5
    checkpoint_dir: str = "checkpoints/option_critic"
    seed: int = 0

    # Environment stepping
    # Preserve the five-update action period of the Unity Epuck prefab.
    decision_period: int = 5
    reward_strength: float = 1.0

    # Network
    hidden_dim: int = 128
    num_layers: int = 1
    critic_hidden_dim: int = 128
    critic_num_layers: int = 2
    critic_num_heads: int = 4
    recurrent: bool = True
    memory_size: int = 128
    sequence_length: int = 128
    num_options: int = 6

    # TensorBoard
    log_dir: str = "runs/option_critic"
    buffer_size_hint: int = 0


class FixedOptionCriticTrainer:
    """Learn decentralized option control from collective critic signals."""

    def __init__(self, env, cfg: FixedOptionCriticConfig | None = None):
        self.env = env
        self.cfg = cfg or FixedOptionCriticConfig()
        self.unwrapped = env.unwrapped
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.scene.num_envs
        cfg_env = self.unwrapped.cfg
        self.num_agents = getattr(cfg_env, "num_agents", getattr(cfg_env, "num_robots", None))
        self.discrete = getattr(cfg_env, "discrete_actions", False)
        self.num_actions = getattr(cfg_env, "num_actions", 6)
        self.variant = getattr(cfg_env, "variant", None)

        if self.variant != "cyclamen":
            raise ValueError(
                "Fixed-module Option-Critic phase 1 is defined from the cyclamen "
                f"SwarmACB controller, got variant={self.variant!r}."
            )
        if not self.discrete:
            raise ValueError(
                "Fixed-module Option-Critic phase 1 requires a discrete CASA variant. "
                "Use cyclamen for the intended SwarmACB baseline."
            )
        if self.num_actions != self.cfg.num_options:
            raise ValueError(
                f"Expected {self.cfg.num_options} fixed modules, env exposes {self.num_actions}."
            )

        sample_obs = self.env.reset()[0][self.unwrapped.cfg.possible_agents[0]]
        if sample_obs.ndim == 4:
            self.obs_dim = int(sample_obs.shape[1] * sample_obs.shape[2] * sample_obs.shape[3])
        else:
            self.obs_dim = sample_obs.shape[1]

        self.state_dim = 5
        c = self.cfg
        self.decision_period = c.decision_period

        print(
            f"[FixedOC] envs={self.num_envs}  agents={self.num_agents}  "
            f"obs={self.obs_dim}  state={self.state_dim}  options={c.num_options}  "
            f"decision_period={self.decision_period}"
        )
        print("[FixedOC] Options are fixed ACB modules; learning shared local selector + termination")
        print("[FixedOC] Centralized RSA critic: collective Q_Omega + CASA counterfactual baselines")
        print(f"[FixedOC] Recurrent manager: LSTM units={c.memory_size // 2}  "
              f"memory_vector={c.memory_size}  sequence_length={c.sequence_length}")
        print(f"[FixedOC] Recurrent RSA critic: hidden={c.critic_hidden_dim}  "
              f"layers={c.critic_num_layers}  heads={c.critic_num_heads}")

        self.manager = FixedOptionManager(
            self.obs_dim,
            c.num_options,
            c.hidden_dim,
            c.num_layers,
            c.memory_size,
        ).to(self.device)
        self.critic = POCACritic(
            self.state_dim,
            c.num_options,
            self.num_agents,
            c.critic_hidden_dim,
            c.critic_num_heads,
            c.critic_num_layers,
            memory_size=c.memory_size,
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.manager.parameters()) + list(self.critic.parameters()),
            lr=c.lr,
            eps=c.adam_eps,
        )

        memory_batch = self.num_envs * self.num_agents
        self.manager_memory_h, self.manager_memory_c = self.manager.initial_state(
            memory_batch,
            self.device,
        )
        self.value_memory_h, self.value_memory_c = self.critic.initial_state(
            self.num_envs, self.device,
        )
        self.joint_memory_h, self.joint_memory_c = self.critic.initial_state(
            self.num_envs, self.device,
        )
        self.baseline_memory_h, self.baseline_memory_c = self.critic.initial_state(
            self.num_envs * self.num_agents, self.device,
        )
        self.current_options = torch.full(
            (self.num_envs, self.num_agents),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        self._init_lr = c.lr
        self._init_eps = c.clip_eps
        self._init_beta = c.beta
        self.lr_schedule = PolynomialDecay(c.lr, 1e-10, c.total_timesteps) if c.lr_schedule == "linear" else None
        self.eps_schedule = PolynomialDecay(c.clip_eps, 0.1, c.total_timesteps) if c.eps_schedule == "linear" else None
        self.beta_schedule = PolynomialDecay(c.beta, 1e-5, c.total_timesteps) if c.beta_schedule == "linear" else None
        self.current_lr = c.lr
        self.current_eps = c.clip_eps
        self.current_beta = c.beta
        self.reward_strength = c.reward_strength
        self._next_checkpoint_step = c.checkpoint_interval
        self._next_summary_step = c.summary_freq

        steps_to_buffer_target = (
            c.buffer_size_hint + self.num_envs * self.num_agents - 1
        ) // (self.num_envs * self.num_agents)
        buffer_capacity = c.horizon + steps_to_buffer_target + 1
        self.buffer = FixedOptionRolloutBuffer(
            horizon=buffer_capacity,
            num_envs=self.num_envs,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            memory_size=self.manager.hidden_size,
            critic_memory_size=self.critic.hidden_size,
            gamma=c.gamma,
            lam=c.lam,
            device=self.device,
        )

        self.global_step = 0
        self.update_count = 0
        self.writer = SummaryWriter(log_dir=c.log_dir)
        hp_text = "\n".join(f"{k}: {v}" for k, v in vars(c).items())
        self.writer.add_text("hyperparameters", hp_text, 0)

        self._episode_reward_acc = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._completed_episode_returns: list[float] = []
        self._completed_episode_lengths: list[float] = []
        self._completed_group_rewards: list[float] = []
        self._rollout_reward_history: list[float] = []
        self._max_history = 100

        manager_params = sum(p.numel() for p in self.manager.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"[FixedOC] Manager params: {manager_params:,}  Critic params: {critic_params:,}")
        print(f"[FixedOC] TensorBoard -> {c.log_dir}")

    def _apply_schedules(self):
        step = self.global_step
        if self.lr_schedule is not None:
            self.current_lr = self.lr_schedule.get(step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.current_lr
        if self.eps_schedule is not None:
            self.current_eps = self.eps_schedule.get(step)
        if self.beta_schedule is not None:
            self.current_beta = self.beta_schedule.get(step)

    def _encode_options_for_critic(self, options: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            options.long(),
            num_classes=self.cfg.num_options,
        ).float()

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
        dp = self.decision_period
        steps = self.cfg.horizon if rollout_steps is None else int(rollout_steps)

        for _ in range(steps):
            obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
            if obs_stacked.ndim == 5:
                obs_stacked = obs_stacked.view(obs_stacked.shape[0], obs_stacked.shape[1], -1)
            flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])

            memory_h = self.manager_memory_h.squeeze(0).view(self.num_envs, self.num_agents, -1).clone()
            memory_c = self.manager_memory_c.squeeze(0).view(self.num_envs, self.num_agents, -1).clone()
            option_logits, termination_logits, next_memory = self.manager.step(
                flat_obs,
                (self.manager_memory_h, self.manager_memory_c),
            )
            self.manager_memory_h = next_memory[0].detach()
            self.manager_memory_c = next_memory[1].detach()

            option_dist = Categorical(logits=option_logits)
            proposed_options = option_dist.sample().view(self.num_envs, self.num_agents)
            proposed_logp = option_dist.log_prob(proposed_options.reshape(-1)).view(
                self.num_envs,
                self.num_agents,
            )

            force_new = self.current_options < 0
            termination_options = self.current_options.clamp(min=0)
            safe_current = termination_options.reshape(-1)
            beta_logits_selected = termination_logits.gather(
                -1,
                safe_current.unsqueeze(-1),
            ).squeeze(-1)
            beta_dist = Bernoulli(logits=beta_logits_selected)
            sampled_terminate = beta_dist.sample().bool().view(self.num_envs, self.num_agents)
            option_mask = (sampled_terminate | force_new).float()
            self.current_options = torch.where(
                option_mask.bool(),
                proposed_options,
                self.current_options,
            )

            option_logp = torch.where(
                option_mask.bool(),
                proposed_logp,
                torch.zeros_like(proposed_logp),
            )
            beta_probs = torch.sigmoid(beta_logits_selected).view(self.num_envs, self.num_agents)

            critic_state = self.unwrapped.get_critic_state()
            critic_options = self._encode_options_for_critic(self.current_options)
            value_memory_h = self.value_memory_h.squeeze(0).clone()
            value_memory_c = self.value_memory_c.squeeze(0).clone()
            joint_memory_h = self.joint_memory_h.squeeze(0).clone()
            joint_memory_c = self.joint_memory_c.squeeze(0).clone()
            baseline_memory_h = self.baseline_memory_h.squeeze(0).view(
                self.num_envs, self.num_agents, -1,
            ).clone()
            baseline_memory_c = self.baseline_memory_c.squeeze(0).view(
                self.num_envs, self.num_agents, -1,
            ).clone()
            team_value, next_value_memory = self.critic.critic_pass(
                critic_state,
                (self.value_memory_h, self.value_memory_c),
                return_memory=True,
            )
            joint_option_value, next_joint_memory = self.critic.joint_action_pass(
                critic_state,
                critic_options,
                (self.joint_memory_h, self.joint_memory_c),
                return_memory=True,
            )
            baselines, next_baseline_memory = self.critic.all_baselines(
                critic_state,
                critic_options,
                (self.baseline_memory_h, self.baseline_memory_c),
                return_memory=True,
            )
            self.value_memory_h = next_value_memory[0].detach()
            self.value_memory_c = next_value_memory[1].detach()
            self.joint_memory_h = next_joint_memory[0].detach()
            self.joint_memory_c = next_joint_memory[1].detach()
            self.baseline_memory_h = next_baseline_memory[0].detach()
            self.baseline_memory_c = next_baseline_memory[1].detach()
            team_value = team_value.squeeze(-1)
            joint_option_value = joint_option_value.squeeze(-1)

            env_actions = self.current_options.unsqueeze(-1)
            action_dict = {a: env_actions[:, i] for i, a in enumerate(agents)}
            accumulated_reward = torch.zeros(self.num_envs, device=self.device)
            last_done = torch.zeros(self.num_envs, device=self.device)
            last_timeout = torch.zeros(self.num_envs, device=self.device)

            for _dp in range(dp):
                obs_dict, rewards_dict, terminated_dict, truncated_dict, _info = self.env.step(action_dict)
                accumulated_reward += rewards_dict[agents[0]]
                step_done = (terminated_dict[agents[0]] | truncated_dict[agents[0]]).float()
                last_done = torch.max(last_done, step_done)
                last_timeout = torch.max(
                    last_timeout, truncated_dict[agents[0]].float(),
                )

            terminal_state = self.unwrapped.completed_terminal_critic_state
            timeout_value = self.critic.critic_pass(
                terminal_state,
                (self.value_memory_h, self.value_memory_c),
            ).squeeze(-1) * last_timeout

            next_obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
            if next_obs_stacked.ndim == 5:
                next_obs_stacked = next_obs_stacked.view(
                    next_obs_stacked.shape[0],
                    next_obs_stacked.shape[1],
                    -1,
                )
            next_memory_h = self.manager_memory_h.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            next_memory_c = self.manager_memory_c.squeeze(0).view(
                self.num_envs,
                self.num_agents,
                -1,
            ).clone()
            next_critic_state = self.unwrapped.get_critic_state()

            self.buffer.add(
                obs=obs_stacked,
                next_obs=next_obs_stacked,
                critic_states=critic_state,
                next_critic_states=next_critic_state,
                options=self.current_options,
                option_log_probs=option_logp,
                option_masks=option_mask,
                beta_probs=beta_probs,
                reward=accumulated_reward * self.reward_strength,
                done=last_done,
                timeout=last_timeout,
                timeout_value=timeout_value,
                team_value=team_value,
                joint_option_value=joint_option_value,
                baselines=baselines,
                memory_h=memory_h,
                memory_c=memory_c,
                next_memory_h=next_memory_h,
                next_memory_c=next_memory_c,
                value_memory_h=value_memory_h,
                value_memory_c=value_memory_c,
                joint_memory_h=joint_memory_h,
                joint_memory_c=joint_memory_c,
                next_joint_memory_h=self.joint_memory_h.squeeze(0).clone(),
                next_joint_memory_c=self.joint_memory_c.squeeze(0).clone(),
                baseline_memory_h=baseline_memory_h,
                baseline_memory_c=baseline_memory_c,
            )

            self._episode_reward_acc += accumulated_reward
            self._episode_step_count += dp
            done_mask = last_done.bool()
            if done_mask.any():
                self._completed_episode_returns.extend(self._episode_reward_acc[done_mask].tolist())
                self._completed_episode_lengths.extend(self._episode_step_count[done_mask].tolist())
                self._completed_group_rewards.extend(
                    self.unwrapped.completed_group_reward[done_mask].tolist()
                )
                self._episode_reward_acc[done_mask] = 0.0
                self._episode_step_count[done_mask] = 0.0
                self.current_options[done_mask] = -1

                done_agents = done_mask[:, None].expand(self.num_envs, self.num_agents).reshape(-1)
                self.manager_memory_h[:, done_agents, :] = 0.0
                self.manager_memory_c[:, done_agents, :] = 0.0
                self.value_memory_h[:, done_mask, :] = 0.0
                self.value_memory_c[:, done_mask, :] = 0.0
                self.joint_memory_h[:, done_mask, :] = 0.0
                self.joint_memory_c[:, done_mask, :] = 0.0
                self.baseline_memory_h[:, done_agents, :] = 0.0
                self.baseline_memory_c[:, done_agents, :] = 0.0

            self.global_step += self.num_envs * self.num_agents

        last_state = self.unwrapped.get_critic_state()
        last_value = self.critic.critic_pass(
            last_state,
            (self.value_memory_h, self.value_memory_c),
        ).squeeze(-1)
        self.buffer.compute_returns_and_advantages(last_value)
        return obs_dict

    def _compute_sequence_losses(
        self,
        batch: dict,
        current_eps: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        critic_states = batch["critic_states"]
        next_critic_states = batch["next_critic_states"]
        options = batch["options"]
        critic_options_ids = batch["critic_options"]
        old_logp = batch["old_option_log_probs"]
        masks = batch["option_masks"]
        advantages = batch["advantages"]
        returns = batch["returns"]
        old_team_values = batch["old_team_values"]
        old_joint_option_values = batch["old_joint_option_values"]
        old_baselines = batch["old_baselines"]
        dones = batch["dones"]
        loss_mask = batch["loss_mask"].bool()

        B, L = obs.shape[:2]
        N = critic_states.shape[2]
        h0 = batch["memory_h"].unsqueeze(0).detach()
        c0 = batch["memory_c"].unsqueeze(0).detach()
        state = (h0, c0)

        option_logits = []
        for t in range(L):
            opt_logits_t, _term_logits_t, state = self.manager.step(
                obs[:, t],
                state,
            )
            option_logits.append(opt_logits_t)
            if t < L - 1:
                keep = (1.0 - dones[:, t]).view(1, B, 1)
                state = (state[0] * keep, state[1] * keep)

        option_logits = torch.stack(option_logits, dim=1)
        opt_dist = Categorical(logits=option_logits.reshape(B * L, self.cfg.num_options))
        new_logp = opt_dist.log_prob(options.reshape(-1)).view(B, L)
        option_entropy_values = opt_dist.entropy().view(B, L)
        option_entropy = (
            option_entropy_values * loss_mask
        ).sum() / loss_mask.sum().clamp_min(1)

        new_logp_flat = new_logp.reshape(-1)
        old_logp_flat = old_logp.reshape(-1)
        adv_flat = advantages.reshape(-1).detach()
        mask_flat = (masks.reshape(-1) > 0.5) & loss_mask.reshape(-1)
        if mask_flat.any():
            ratio = (new_logp_flat[mask_flat] - old_logp_flat[mask_flat]).exp()
            pg_a = ratio * adv_flat[mask_flat]
            pg_b = ratio.clamp(1.0 - current_eps, 1.0 + current_eps) * adv_flat[mask_flat]
            policy_loss = -torch.min(pg_a, pg_b).mean()
        else:
            policy_loss = new_logp_flat.sum() * 0.0

        next_obs_seq = next_obs.reshape(B * L, next_obs.shape[-1])
        next_h = batch["next_memory_h"].reshape(B * L, -1)
        next_c = batch["next_memory_c"].reshape(B * L, -1)
        (
            next_option_logits,
            next_termination_logits,
            _next_state,
        ) = self.manager.step(
            next_obs_seq,
            (next_h.unsqueeze(0).detach(), next_c.unsqueeze(0).detach()),
        )
        next_option_logits = next_option_logits.view(B, L, self.cfg.num_options)
        next_termination_logits = next_termination_logits.view(B, L, self.cfg.num_options)

        next_beta_logits = next_termination_logits.gather(
            -1,
            options.unsqueeze(-1),
        ).squeeze(-1)
        next_beta = torch.sigmoid(next_beta_logits)
        nonterminal = 1.0 - dones

        flat_states = critic_states.reshape(B * L, N, critic_states.shape[-1])
        flat_next_states = next_critic_states.reshape(
            B * L,
            N,
            next_critic_states.shape[-1],
        )
        flat_critic_option_ids = critic_options_ids.reshape(B * L, N)
        critic_options = self._encode_options_for_critic(flat_critic_option_ids)
        flat_returns = returns.reshape(B * L)
        new_team_values = self.critic.critic_pass(
            flat_states,
            (
                batch["value_memory_h"].unsqueeze(0).detach(),
                batch["value_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=L,
        ).squeeze(-1)
        new_joint_option_values = self.critic.joint_action_pass(
            flat_states,
            critic_options,
            (
                batch["joint_memory_h"].unsqueeze(0).detach(),
                batch["joint_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=L,
        ).squeeze(-1)
        focal_ids = batch["focal_agent_ids"].unsqueeze(1).expand(B, L).reshape(-1)
        new_baselines = self.critic.focal_baselines(
            flat_states,
            critic_options,
            focal_ids,
            (
                batch["baseline_memory_h"].unsqueeze(0).detach(),
                batch["baseline_memory_c"].unsqueeze(0).detach(),
            ),
            sequence_length=L,
        ).squeeze(-1)

        flat_loss_mask = loss_mask.reshape(B * L)

        value_loss = trust_region_value_loss(
            new_team_values,
            old_team_values.reshape(B * L),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        joint_option_value_loss = trust_region_value_loss(
            new_joint_option_values,
            old_joint_option_values.reshape(B * L),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )
        baseline_loss = trust_region_value_loss(
            new_baselines,
            old_baselines.reshape(-1),
            flat_returns,
            current_eps,
            flat_loss_mask,
        )

        # The termination theorem is evaluated after entering s'. For robot i,
        # continuation uses the current collective joint-option value while
        # reselection marginalizes only i's alternatives and holds peers fixed.
        with torch.no_grad():
            next_joint_memory = (
                batch["next_joint_memory_h"].reshape(B * L, -1).unsqueeze(0),
                batch["next_joint_memory_c"].reshape(B * L, -1).unsqueeze(0),
            )
            next_q_current = self.critic.joint_action_pass(
                flat_next_states,
                critic_options,
                memory=next_joint_memory,
            ).squeeze(-1)
            next_counterfactual_values = self.critic.focal_discrete_counterfactual_values(
                flat_next_states,
                flat_critic_option_ids,
                focal_ids,
                self.cfg.num_options,
                memory=next_joint_memory,
            )
            next_selector_probs = torch.softmax(next_option_logits, dim=-1).reshape(
                B * L,
                self.cfg.num_options,
            )
            next_reselection_values = (
                next_counterfactual_values * next_selector_probs
            ).sum(dim=-1)
            option_advantage = (
                next_q_current - next_reselection_values
            ).reshape(B, L)

        term_signal = option_advantage + self.cfg.termination_penalty
        term_mask = nonterminal * loss_mask
        term_count = term_mask.sum()
        if term_count.item() > 0:
            termination_loss = (next_beta * term_signal * term_mask).sum() / term_count
            term_entropy = Bernoulli(logits=next_beta_logits).entropy()
            termination_entropy = (term_entropy * term_mask).sum() / term_count
            mean_beta = (next_beta * term_mask).sum() / term_count
            mean_option_advantage = (option_advantage * term_mask).sum() / term_count
        else:
            termination_loss = next_beta.sum() * 0.0
            termination_entropy = next_beta.sum() * 0.0
            mean_beta = next_beta.sum() * 0.0
            mean_option_advantage = next_beta.sum() * 0.0

        return (
            policy_loss,
            value_loss,
            joint_option_value_loss,
            baseline_loss,
            termination_loss,
            option_entropy,
            termination_entropy,
            mean_beta,
            mean_option_advantage,
        )

    def update(self) -> dict:
        cfg = self.cfg
        self._apply_schedules()
        current_eps = self.current_eps
        current_beta = self.current_beta

        active = self.buffer.ptr
        all_adv = self.buffer.advantages[:active]
        self.buffer.advantages[:active] = (
            all_adv - all_adv.mean()
        ) / (all_adv.std(unbiased=False) + 1e-10)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "joint_option_value_loss": 0.0,
            "baseline_loss": 0.0,
            "termination_loss": 0.0,
            "option_entropy": 0.0,
            "termination_entropy": 0.0,
            "mean_beta": 0.0,
            "mean_option_advantage": 0.0,
        }
        n_updates = 0

        for _epoch in range(cfg.num_epochs):
            for batch in self.buffer.get_sequence_batches(
                cfg.sequence_length,
                cfg.mini_batch_size,
            ):
                (
                    policy_loss,
                    value_loss,
                    joint_option_value_loss,
                    baseline_loss,
                    termination_loss,
                    option_entropy,
                    termination_entropy,
                    mean_beta,
                    mean_option_advantage,
                ) = self._compute_sequence_losses(batch, current_eps)

                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    + cfg.option_value_coef * joint_option_value_loss
                    + cfg.baseline_coef * baseline_loss
                    + cfg.termination_coef * termination_loss
                    - current_beta * option_entropy
                    - cfg.termination_entropy_coef * termination_entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["joint_option_value_loss"] += joint_option_value_loss.item()
                totals["baseline_loss"] += baseline_loss.item()
                totals["termination_loss"] += termination_loss.item()
                totals["option_entropy"] += option_entropy.item()
                totals["termination_entropy"] += termination_entropy.item()
                totals["mean_beta"] += mean_beta.item()
                totals["mean_option_advantage"] += mean_option_advantage.item()
                n_updates += 1

        self.update_count += 1
        n = max(n_updates, 1)
        option_counts = torch.bincount(
            self.buffer.options[:active].reshape(-1),
            minlength=cfg.num_options,
        ).float()
        option_usage = (option_counts / option_counts.sum().clamp(min=1.0)).tolist()
        return {
            "policy_loss": totals["policy_loss"] / n,
            "value_loss": totals["value_loss"] / n,
            "joint_option_value_loss": totals["joint_option_value_loss"] / n,
            "baseline_loss": totals["baseline_loss"] / n,
            "termination_loss": totals["termination_loss"] / n,
            "option_entropy": totals["option_entropy"] / n,
            "termination_entropy": totals["termination_entropy"] / n,
            "mean_beta": totals["mean_beta"] / n,
            "mean_option_advantage": totals["mean_option_advantage"] / n,
            "lr": self.current_lr,
            "eps": self.current_eps,
            "beta": self.current_beta,
            "switch_rate": self.buffer.option_masks[:active].mean().item(),
            "option_usage": option_usage,
        }

    def train(self):
        start_time = time.time()
        obs_dict, _ = self.env.reset()
        self.current_options.fill_(-1)
        self.manager_memory_h.zero_()
        self.manager_memory_c.zero_()
        self.value_memory_h.zero_()
        self.value_memory_c.zero_()
        self.joint_memory_h.zero_()
        self.joint_memory_c.zero_()
        self.baseline_memory_h.zero_()
        self.baseline_memory_c.zero_()

        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(
            total=self.cfg.total_timesteps,
            initial=self.global_step,
            desc="FixedOC Training",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        while self.global_step < self.cfg.total_timesteps:
            prev_step = self.global_step
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
            metrics = self.update()

            step_delta = self.global_step - prev_step
            elapsed = time.time() - start_time
            sps = self.global_step / elapsed if elapsed > 0 else 0
            remaining_bar = max(0, self.cfg.total_timesteps - pbar.n)
            pbar.update(min(step_delta, remaining_bar))
            pbar.set_postfix(
                upd=self.update_count,
                pg=f"{metrics['policy_loss']:.3f}",
                vf=f"{metrics['value_loss']:.3f}",
                term=f"{metrics['termination_loss']:.3f}",
                sw=f"{metrics['switch_rate']:.2f}",
                SPS=f"{sps:.0f}",
            )

            mean_rollout_reward = self.buffer.rewards[:self.buffer.ptr].sum(dim=0).mean().item()
            self._rollout_reward_history.append(mean_rollout_reward)
            if len(self._rollout_reward_history) > self._max_history:
                self._rollout_reward_history.pop(0)

            if self.global_step >= self._next_summary_step:
                self._next_summary_step += self.cfg.summary_freq
                s = self.global_step
                self.writer.add_scalar("Losses/Policy Loss", metrics["policy_loss"], s)
                self.writer.add_scalar("Losses/Value Loss", metrics["value_loss"], s)
                self.writer.add_scalar("Losses/OptionCritic/Collective Option Value Loss", metrics["joint_option_value_loss"], s)
                self.writer.add_scalar("Losses/OptionCritic/Counterfactual Baseline Loss", metrics["baseline_loss"], s)
                self.writer.add_scalar("Losses/OptionCritic/Termination Loss", metrics["termination_loss"], s)
                self.writer.add_scalar("Policy/Option Entropy", metrics["option_entropy"], s)
                self.writer.add_scalar("Policy/Termination Entropy", metrics["termination_entropy"], s)
                self.writer.add_scalar("Policy/Mean Termination Probability", metrics["mean_beta"], s)
                self.writer.add_scalar("Policy/Mean Option Advantage", metrics["mean_option_advantage"], s)
                self.writer.add_scalar("Policy/Switch Rate", metrics["switch_rate"], s)
                self.writer.add_scalar("Policy/Learning Rate", metrics["lr"], s)
                self.writer.add_scalar("Policy/Epsilon", metrics["eps"], s)
                self.writer.add_scalar("Policy/Beta", metrics["beta"], s)
                for option_id, usage in enumerate(metrics["option_usage"]):
                    self.writer.add_scalar(f"Policy/Option Usage/{option_id}", usage, s)
                active_rewards = self.buffer.rewards[:self.buffer.ptr]
                active_values = self.buffer.team_values[:self.buffer.ptr]
                self.writer.add_scalar("Policy/Extrinsic Reward", active_rewards.mean().item(), s)
                self.writer.add_scalar("Policy/Extrinsic Value Estimate", active_values.mean().item(), s)
                self.writer.add_scalar("Extra/SPS", sps, s)
                self.writer.add_scalar("Extra/Mean Rollout Reward", mean_rollout_reward, s)
                rolling_avg = sum(self._rollout_reward_history) / len(self._rollout_reward_history)
                self.writer.add_scalar("Extra/Rolling Avg Rollout Reward", rolling_avg, s)

                if self._completed_episode_returns:
                    ep = self._completed_episode_returns
                    self.writer.add_scalar("Environment/Cumulative Reward", sum(ep) / len(ep), s)
                    self._completed_episode_returns.clear()
                if self._completed_episode_lengths:
                    el = self._completed_episode_lengths
                    self.writer.add_scalar("Environment/Episode Length", sum(el) / len(el), s)
                    self._completed_episode_lengths.clear()
                if self._completed_group_rewards:
                    gr = self._completed_group_rewards
                    self.writer.add_scalar("Extra/Group Reward Mean", sum(gr) / len(gr), s)
                    self._completed_group_rewards.clear()

            if self.global_step >= self._next_checkpoint_step:
                self.save_checkpoint(ckpt_dir / f"option_critic_{self.global_step}.pt")
                self._next_checkpoint_step += self.cfg.checkpoint_interval
                self._manage_checkpoints(ckpt_dir)

        pbar.close()
        self.writer.close()
        self.save_checkpoint(ckpt_dir / "option_critic_final.pt")
        elapsed = time.time() - start_time
        print(
            f"[FixedOC] Done - {self.global_step:,} steps in {elapsed:.0f}s "
            f"({self.global_step / elapsed:.0f} SPS)"
        )

    def save_checkpoint(self, path):
        torch.save({
            "trainer_type": "option_critic",
            "option_critic_version": 7,
            "paper_parity_version": PAPER_PARITY_VERSION,
            "fixed_options": True,
            "collective_counterfactual": True,
            "variant": self.variant,
            "manager": self.manager.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "update_count": self.update_count,
            "seed": self.cfg.seed,
            "hidden_dim": getattr(self.cfg, "hidden_dim", 128),
            "num_layers": getattr(self.cfg, "num_layers", 1),
            "recurrent": True,
            "memory_size": getattr(self.cfg, "memory_size", 128),
            "memory_size_semantics": "mlagents_total",
            "lstm_hidden_size": self.manager.hidden_size,
            "sequence_length": getattr(self.cfg, "sequence_length", 128),
            "critic_hidden_dim": self.cfg.critic_hidden_dim,
            "critic_num_layers": self.cfg.critic_num_layers,
            "critic_num_heads": self.cfg.critic_num_heads,
            "decision_period": self.decision_period,
            "discrete": True,
            "num_actions": self.cfg.num_options,
            "num_options": self.cfg.num_options,
            "act_dim": 1,
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
        }, path)
        print(f"[FixedOC] Saved -> {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        parity_version = int(ckpt.get("paper_parity_version", 0))
        if parity_version != PAPER_PARITY_VERSION:
            raise RuntimeError(
                f"Refusing to resume a parity-v{parity_version} Option-Critic "
                f"checkpoint with the parity-v{PAPER_PARITY_VERSION} trainer. "
                "Use it only for legacy evaluation and start training fresh."
            )
        try:
            self.manager.load_state_dict(ckpt["manager"])
            self.critic.load_state_dict(ckpt["critic"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match Option-Critic version 7. "
                "Legacy checkpoints remain available for evaluation; retraining "
                "must start fresh."
            ) from exc
        self.global_step = ckpt["global_step"]
        self.update_count = ckpt["update_count"]
        print(f"[FixedOC] Loaded <- {path}  (step {self.global_step})")

    def _manage_checkpoints(self, ckpt_dir: Path):
        keep = self.cfg.keep_checkpoints
        if keep <= 0:
            return
        numbered = sorted(
            ckpt_dir.glob("option_critic_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        numbered = [p for p in numbered if p.stem != "option_critic_final"]
        while len(numbered) > keep:
            old = numbered.pop(0)
            old.unlink()
            print(f"[FixedOC] Removed old checkpoint -> {old.name}")
