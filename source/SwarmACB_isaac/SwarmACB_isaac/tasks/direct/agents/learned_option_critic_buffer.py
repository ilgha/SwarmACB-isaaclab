# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for learned-option collective Option-Critic."""

from __future__ import annotations

import torch


class LearnedOptionRolloutBuffer:
    """Recurrent rollout storage for primitive and option-level objectives."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        act_dim: int,
        memory_size: int,
        critic_memory_size: int,
        gamma: float,
        lam: float,
        device: torch.device | str,
    ):
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)
        self.memory_size = int(memory_size)
        self.critic_memory_size = int(critic_memory_size)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.device = device

        time, envs, agents = self.horizon, self.num_envs, self.num_agents
        self.obs = torch.zeros(time, envs, agents, obs_dim, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.critic_states = torch.zeros(
            time, envs, agents, state_dim, device=device,
        )
        self.next_critic_states = torch.zeros_like(self.critic_states)

        self.options = torch.zeros(
            time, envs, agents, dtype=torch.long, device=device,
        )
        self.option_log_probs = torch.zeros(time, envs, agents, device=device)
        self.local_option_values = torch.zeros(time, envs, agents, device=device)
        self.option_masks = torch.zeros(time, envs, agents, device=device)
        self.beta_probs = torch.zeros(time, envs, agents, device=device)
        self.termination_options = torch.zeros(
            time, envs, agents, dtype=torch.long, device=device,
        )
        self.termination_valid = torch.zeros(time, envs, agents, device=device)

        self.actions = torch.zeros(
            time, envs, agents, act_dim, device=device,
        )
        self.action_log_probs = torch.zeros_like(self.actions)

        self.rewards = torch.zeros(time, envs, device=device)
        self.dones = torch.zeros(time, envs, device=device)
        self.timeouts = torch.zeros(time, envs, device=device)
        self.timeout_values = torch.zeros(time, envs, device=device)
        self.team_values = torch.zeros(time, envs, device=device)
        self.action_baselines = torch.zeros(time, envs, agents, device=device)
        self.joint_option_values = torch.zeros(time, envs, device=device)
        self.option_baselines = torch.zeros(time, envs, agents, device=device)

        self.memory_h = torch.zeros(
            time, envs, agents, memory_size, device=device,
        )
        self.memory_c = torch.zeros_like(self.memory_h)
        self.next_memory_h = torch.zeros_like(self.memory_h)
        self.next_memory_c = torch.zeros_like(self.memory_h)

        critic_size = self.critic_memory_size
        self.team_memory_h = torch.zeros(
            time, envs, critic_size, device=device,
        )
        self.team_memory_c = torch.zeros_like(self.team_memory_h)
        self.action_baseline_memory_h = torch.zeros(
            time, envs, agents, critic_size, device=device,
        )
        self.action_baseline_memory_c = torch.zeros_like(
            self.action_baseline_memory_h,
        )
        self.option_joint_memory_h = torch.zeros(
            time, envs, critic_size, device=device,
        )
        self.option_joint_memory_c = torch.zeros_like(
            self.option_joint_memory_h,
        )
        self.next_option_joint_memory_h = torch.zeros_like(
            self.option_joint_memory_h,
        )
        self.next_option_joint_memory_c = torch.zeros_like(
            self.option_joint_memory_h,
        )
        self.option_baseline_memory_h = torch.zeros(
            time, envs, agents, critic_size, device=device,
        )
        self.option_baseline_memory_c = torch.zeros_like(
            self.option_baseline_memory_h,
        )

        self.returns = torch.zeros(time, envs, device=device)
        self.action_advantages = torch.zeros(
            time, envs, agents, device=device,
        )
        self.option_advantages = torch.zeros_like(self.action_advantages)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(
        self,
        *,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        critic_states: torch.Tensor,
        next_critic_states: torch.Tensor,
        options: torch.Tensor,
        option_log_probs: torch.Tensor,
        local_option_values: torch.Tensor,
        option_masks: torch.Tensor,
        beta_probs: torch.Tensor,
        termination_options: torch.Tensor,
        termination_valid: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        timeout: torch.Tensor,
        timeout_value: torch.Tensor,
        team_value: torch.Tensor,
        action_baselines: torch.Tensor,
        joint_option_value: torch.Tensor,
        option_baselines: torch.Tensor,
        memory_h: torch.Tensor,
        memory_c: torch.Tensor,
        next_memory_h: torch.Tensor,
        next_memory_c: torch.Tensor,
        team_memory_h: torch.Tensor,
        team_memory_c: torch.Tensor,
        action_baseline_memory_h: torch.Tensor,
        action_baseline_memory_c: torch.Tensor,
        option_joint_memory_h: torch.Tensor,
        option_joint_memory_c: torch.Tensor,
        next_option_joint_memory_h: torch.Tensor,
        next_option_joint_memory_c: torch.Tensor,
        option_baseline_memory_h: torch.Tensor,
        option_baseline_memory_c: torch.Tensor,
    ):
        if self.ptr >= self.horizon:
            raise RuntimeError("Learned Option-Critic rollout buffer is full")
        t = self.ptr
        self.obs[t] = obs
        self.next_obs[t] = next_obs
        self.critic_states[t] = critic_states
        self.next_critic_states[t] = next_critic_states
        self.options[t] = options.long()
        self.option_log_probs[t] = option_log_probs
        self.local_option_values[t] = local_option_values
        self.option_masks[t] = option_masks
        self.beta_probs[t] = beta_probs
        self.termination_options[t] = termination_options.long()
        self.termination_valid[t] = termination_valid
        self.actions[t] = actions
        self.action_log_probs[t] = action_log_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.timeouts[t] = timeout
        self.timeout_values[t] = timeout_value
        self.team_values[t] = team_value
        self.action_baselines[t] = action_baselines
        self.joint_option_values[t] = joint_option_value
        self.option_baselines[t] = option_baselines
        self.memory_h[t] = memory_h
        self.memory_c[t] = memory_c
        self.next_memory_h[t] = next_memory_h
        self.next_memory_c[t] = next_memory_c
        self.team_memory_h[t] = team_memory_h
        self.team_memory_c[t] = team_memory_c
        self.action_baseline_memory_h[t] = action_baseline_memory_h
        self.action_baseline_memory_c[t] = action_baseline_memory_c
        self.option_joint_memory_h[t] = option_joint_memory_h
        self.option_joint_memory_c[t] = option_joint_memory_c
        self.next_option_joint_memory_h[t] = next_option_joint_memory_h
        self.next_option_joint_memory_c[t] = next_option_joint_memory_c
        self.option_baseline_memory_h[t] = option_baseline_memory_h
        self.option_baseline_memory_c[t] = option_baseline_memory_c
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_team_value: torch.Tensor,
    ):
        if self.ptr <= 0:
            return

        last = self.ptr - 1
        done_last = self.dones[last]
        boundary_last = self.timeouts[last] * self.timeout_values[last]
        bootstrap_last = torch.where(
            done_last.bool(),
            boundary_last,
            last_team_value,
        )
        self.returns[last] = (
            self.rewards[last] + self.gamma * bootstrap_last
        )
        for t in reversed(range(last)):
            done = self.dones[t]
            continuation = (
                (1.0 - self.lam) * self.team_values[t + 1]
                + self.lam * self.returns[t + 1]
            )
            boundary = self.timeouts[t] * self.timeout_values[t]
            self.returns[t] = self.rewards[t] + self.gamma * (
                (1.0 - done) * continuation + done * boundary
            )

        targets = self.returns[:self.ptr].unsqueeze(-1)
        self.action_advantages[:self.ptr] = (
            targets - self.action_baselines[:self.ptr]
        )
        self.option_advantages[:self.ptr] = (
            targets - self.option_baselines[:self.ptr]
        )

    def get_sequence_batches(
        self,
        sequence_length: int,
        mini_batch_size: int,
    ):
        """Yield padded focal-agent recurrent minibatches."""
        time = self.ptr
        if time <= 0:
            return

        envs, agents = self.num_envs, self.num_agents
        length = max(1, min(int(sequence_length), time))
        chunks: list[tuple[int, int, int, int]] = []
        for env_id in range(envs):
            segment_start = 0
            segment_ends = [
                t + 1 for t in range(time)
                if self.dones[t, env_id] > 0.5
            ]
            if not segment_ends or segment_ends[-1] != time:
                segment_ends.append(time)
            for segment_end in segment_ends:
                for start_t in range(segment_start, segment_end, length):
                    end_t = min(start_t + length, segment_end)
                    for agent_id in range(agents):
                        chunks.append((env_id, agent_id, start_t, end_t))
                segment_start = segment_end

        order = torch.randperm(len(chunks), device=self.device).tolist()
        sequences_per_batch = max(1, int(mini_batch_size) // length)

        def padded(value: torch.Tensor, active: int) -> torch.Tensor:
            result = torch.zeros(
                (length, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            result[:active] = value
            return result

        for offset in range(0, len(order), sequences_per_batch):
            selected = [
                chunks[index]
                for index in order[offset:offset + sequences_per_batch]
            ]
            if (
                len(selected) < sequences_per_batch
                and len(chunks) >= sequences_per_batch
            ):
                continue

            def stack_group(name: str):
                values = getattr(self, name)
                return torch.stack([
                    padded(values[start:end, env_id], end - start)
                    for env_id, _agent_id, start, end in selected
                ])

            def stack_focal(name: str):
                values = getattr(self, name)
                return torch.stack([
                    padded(
                        values[start:end, env_id, agent_id],
                        end - start,
                    )
                    for env_id, agent_id, start, end in selected
                ])

            loss_mask = torch.zeros(
                len(selected),
                length,
                device=self.device,
            )
            for row, (_env_id, _agent_id, start, end) in enumerate(selected):
                loss_mask[row, :end - start] = 1.0

            yield {
                "obs": stack_focal("obs"),
                "next_obs": stack_focal("next_obs"),
                "critic_states": stack_group("critic_states"),
                "next_critic_states": stack_group("next_critic_states"),
                "options": stack_focal("options"),
                "critic_options": stack_group("options"),
                "old_option_log_probs": stack_focal("option_log_probs"),
                "old_local_option_values": stack_focal(
                    "local_option_values"
                ),
                "option_masks": stack_focal("option_masks"),
                "actions": stack_focal("actions"),
                "critic_actions": stack_group("actions"),
                "old_action_log_probs": stack_focal("action_log_probs"),
                "action_advantages": stack_focal("action_advantages"),
                "option_advantages": stack_focal("option_advantages"),
                "returns": stack_group("returns"),
                "old_team_values": stack_group("team_values"),
                "old_action_baselines": stack_focal("action_baselines"),
                "old_joint_option_values": stack_group(
                    "joint_option_values",
                ),
                "old_option_baselines": stack_focal("option_baselines"),
                "dones": stack_group("dones"),
                "memory_h": torch.stack([
                    self.memory_h[start, env_id, agent_id]
                    for env_id, agent_id, start, _end in selected
                ]),
                "memory_c": torch.stack([
                    self.memory_c[start, env_id, agent_id]
                    for env_id, agent_id, start, _end in selected
                ]),
                "next_memory_h": stack_focal("next_memory_h"),
                "next_memory_c": stack_focal("next_memory_c"),
                "team_memory_h": torch.stack([
                    self.team_memory_h[start, env_id]
                    for env_id, _agent_id, start, _end in selected
                ]),
                "team_memory_c": torch.stack([
                    self.team_memory_c[start, env_id]
                    for env_id, _agent_id, start, _end in selected
                ]),
                "action_baseline_memory_h": torch.stack([
                    self.action_baseline_memory_h[
                        start, env_id, agent_id
                    ]
                    for env_id, agent_id, start, _end in selected
                ]),
                "action_baseline_memory_c": torch.stack([
                    self.action_baseline_memory_c[
                        start, env_id, agent_id
                    ]
                    for env_id, agent_id, start, _end in selected
                ]),
                "option_joint_memory_h": torch.stack([
                    self.option_joint_memory_h[start, env_id]
                    for env_id, _agent_id, start, _end in selected
                ]),
                "option_joint_memory_c": torch.stack([
                    self.option_joint_memory_c[start, env_id]
                    for env_id, _agent_id, start, _end in selected
                ]),
                "next_option_joint_memory_h": stack_group(
                    "next_option_joint_memory_h",
                ),
                "next_option_joint_memory_c": stack_group(
                    "next_option_joint_memory_c",
                ),
                "option_baseline_memory_h": torch.stack([
                    self.option_baseline_memory_h[
                        start, env_id, agent_id
                    ]
                    for env_id, agent_id, start, _end in selected
                ]),
                "option_baseline_memory_c": torch.stack([
                    self.option_baseline_memory_c[
                        start, env_id, agent_id
                    ]
                    for env_id, agent_id, start, _end in selected
                ]),
                "focal_agent_ids": torch.tensor(
                    [
                        agent_id
                        for _env_id, agent_id, _start, _end in selected
                    ],
                    dtype=torch.long,
                    device=self.device,
                ),
                "loss_mask": loss_mask,
            }
