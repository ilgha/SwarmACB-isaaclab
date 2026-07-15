# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for fixed-module collective Option-Critic training."""

from __future__ import annotations

import torch


class FixedOptionRolloutBuffer:
    """Fixed-horizon storage shaped around SwarmACB vectorized MARL.

    Tensors use ``(T, E, N, ...)`` for time, environments, and agents. Rewards,
    returns, and centralized values are shared team quantities. Counterfactual
    advantages retain one entry per interchangeable robot.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        memory_size: int,
        critic_memory_size: int,
        gamma: float,
        lam: float,
        device: torch.device | str,
    ):
        self.horizon = horizon
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.critic_memory_size = critic_memory_size
        self.gamma = gamma
        self.lam = lam
        self.device = device

        T, E, N = horizon, num_envs, num_agents
        self.obs = torch.zeros(T, E, N, obs_dim, device=device)
        self.next_obs = torch.zeros(T, E, N, obs_dim, device=device)
        self.critic_states = torch.zeros(T, E, N, state_dim, device=device)
        self.next_critic_states = torch.zeros(T, E, N, state_dim, device=device)
        self.options = torch.zeros(T, E, N, dtype=torch.long, device=device)
        self.option_log_probs = torch.zeros(T, E, N, device=device)
        self.option_masks = torch.zeros(T, E, N, device=device)
        self.beta_probs = torch.zeros(T, E, N, device=device)
        self.rewards = torch.zeros(T, E, device=device)
        self.dones = torch.zeros(T, E, device=device)
        self.timeouts = torch.zeros(T, E, device=device)
        self.timeout_values = torch.zeros(T, E, device=device)
        self.team_values = torch.zeros(T, E, device=device)
        self.joint_option_values = torch.zeros(T, E, device=device)
        self.baselines = torch.zeros(T, E, N, device=device)
        self.memory_h = torch.zeros(T, E, N, memory_size, device=device)
        self.memory_c = torch.zeros(T, E, N, memory_size, device=device)
        self.next_memory_h = torch.zeros(T, E, N, memory_size, device=device)
        self.next_memory_c = torch.zeros(T, E, N, memory_size, device=device)
        H = critic_memory_size
        self.value_memory_h = torch.zeros(T, E, H, device=device)
        self.value_memory_c = torch.zeros(T, E, H, device=device)
        self.joint_memory_h = torch.zeros(T, E, H, device=device)
        self.joint_memory_c = torch.zeros(T, E, H, device=device)
        self.next_joint_memory_h = torch.zeros(T, E, H, device=device)
        self.next_joint_memory_c = torch.zeros(T, E, H, device=device)
        self.baseline_memory_h = torch.zeros(T, E, N, H, device=device)
        self.baseline_memory_c = torch.zeros(T, E, N, H, device=device)

        self.returns = torch.zeros(T, E, device=device)
        self.advantages = torch.zeros(T, E, N, device=device)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        critic_states: torch.Tensor,
        next_critic_states: torch.Tensor,
        options: torch.Tensor,
        option_log_probs: torch.Tensor,
        option_masks: torch.Tensor,
        beta_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        timeout: torch.Tensor,
        timeout_value: torch.Tensor,
        team_value: torch.Tensor,
        joint_option_value: torch.Tensor,
        baselines: torch.Tensor,
        memory_h: torch.Tensor,
        memory_c: torch.Tensor,
        next_memory_h: torch.Tensor,
        next_memory_c: torch.Tensor,
        value_memory_h: torch.Tensor,
        value_memory_c: torch.Tensor,
        joint_memory_h: torch.Tensor,
        joint_memory_c: torch.Tensor,
        next_joint_memory_h: torch.Tensor,
        next_joint_memory_c: torch.Tensor,
        baseline_memory_h: torch.Tensor,
        baseline_memory_c: torch.Tensor,
    ):
        if self.ptr >= self.horizon:
            raise RuntimeError("Fixed Option-Critic rollout buffer is full")
        t = self.ptr
        self.obs[t] = obs
        self.next_obs[t] = next_obs
        self.critic_states[t] = critic_states
        self.next_critic_states[t] = next_critic_states
        self.options[t] = options.long()
        self.option_log_probs[t] = option_log_probs
        self.option_masks[t] = option_masks
        self.beta_probs[t] = beta_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.timeouts[t] = timeout
        self.timeout_values[t] = timeout_value
        self.team_values[t] = team_value
        self.joint_option_values[t] = joint_option_value
        self.baselines[t] = baselines
        self.memory_h[t] = memory_h
        self.memory_c[t] = memory_c
        self.next_memory_h[t] = next_memory_h
        self.next_memory_c[t] = next_memory_c
        self.value_memory_h[t] = value_memory_h
        self.value_memory_c[t] = value_memory_c
        self.joint_memory_h[t] = joint_memory_h
        self.joint_memory_c[t] = joint_memory_c
        self.next_joint_memory_h[t] = next_joint_memory_h
        self.next_joint_memory_c[t] = next_joint_memory_c
        self.baseline_memory_h[t] = baseline_memory_h
        self.baseline_memory_c[t] = baseline_memory_c
        self.ptr += 1

    def compute_returns_and_advantages(self, last_team_value: torch.Tensor):
        """Compute lambda returns and per-robot counterfactual advantages."""
        if self.ptr <= 0:
            return

        last = self.ptr - 1
        done_last = self.dones[last]
        boundary_last = self.timeouts[last] * self.timeout_values[last]
        bootstrap_last = torch.where(done_last.bool(), boundary_last, last_team_value)
        self.returns[last] = self.rewards[last] + self.gamma * bootstrap_last
        for t in reversed(range(last)):
            done = self.dones[t]
            continuation = (
                (1.0 - self.lam) * self.team_values[t + 1]
                + self.lam * self.returns[t + 1]
            )
            boundary = self.timeouts[t] * self.timeout_values[t]
            self.returns[t] = (
                self.rewards[t]
                + self.gamma
                * ((1.0 - done) * continuation + done * boundary)
            )

        self.advantages[: self.ptr] = (
            self.returns[: self.ptr].unsqueeze(-1) - self.baselines[: self.ptr]
        )

    def get_sequence_batches(self, sequence_length: int, mini_batch_size: int):
        """Yield padded recurrent minibatches matching ML-Agents masks."""
        T, E, N = self.ptr, self.num_envs, self.num_agents
        if T <= 0:
            return
        L = max(1, min(int(sequence_length), T))
        chunks: list[tuple[int, int, int, int]] = []
        for env_id in range(E):
            segment_start = 0
            ends = [t + 1 for t in range(T) if self.dones[t, env_id] > 0.5]
            if not ends or ends[-1] != T:
                ends.append(T)
            for segment_end in ends:
                for start_t in range(segment_start, segment_end, L):
                    end_t = min(start_t + L, segment_end)
                    for agent_id in range(N):
                        chunks.append((env_id, agent_id, start_t, end_t))
                segment_start = segment_end

        order = torch.randperm(len(chunks), device=self.device).tolist()
        sequences_per_batch = max(1, int(mini_batch_size) // L)

        def padded(value: torch.Tensor, length: int) -> torch.Tensor:
            result = torch.zeros(
                (L, *value.shape[1:]), dtype=value.dtype, device=value.device,
            )
            result[:length] = value
            return result

        for start in range(0, len(order), sequences_per_batch):
            selected = [chunks[i] for i in order[start:start + sequences_per_batch]]
            if len(selected) < sequences_per_batch and len(chunks) >= sequences_per_batch:
                continue

            def stack_group(name: str):
                values = getattr(self, name)
                return torch.stack([
                    padded(values[s:e, env_id], e - s)
                    for env_id, _agent_id, s, e in selected
                ])

            def stack_focal(name: str):
                values = getattr(self, name)
                return torch.stack([
                    padded(values[s:e, env_id, agent_id], e - s)
                    for env_id, agent_id, s, e in selected
                ])

            loss_mask = torch.zeros(len(selected), L, device=self.device)
            for row, (_env_id, _agent_id, s, e) in enumerate(selected):
                loss_mask[row, :e - s] = 1.0

            yield {
                "obs": stack_focal("obs"),
                "next_obs": stack_focal("next_obs"),
                "critic_states": stack_group("critic_states"),
                "next_critic_states": stack_group("next_critic_states"),
                "options": stack_focal("options"),
                "critic_options": stack_group("options"),
                "old_option_log_probs": stack_focal("option_log_probs"),
                "option_masks": stack_focal("option_masks"),
                "advantages": stack_focal("advantages"),
                "returns": stack_group("returns"),
                "old_team_values": stack_group("team_values"),
                "old_joint_option_values": stack_group("joint_option_values"),
                "old_baselines": stack_focal("baselines"),
                "dones": stack_group("dones"),
                "memory_h": torch.stack([
                    self.memory_h[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "memory_c": torch.stack([
                    self.memory_c[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "next_memory_h": stack_focal("next_memory_h"),
                "next_memory_c": stack_focal("next_memory_c"),
                "value_memory_h": torch.stack([
                    self.value_memory_h[s, env_id]
                    for env_id, _agent_id, s, _e in selected
                ]),
                "value_memory_c": torch.stack([
                    self.value_memory_c[s, env_id]
                    for env_id, _agent_id, s, _e in selected
                ]),
                "joint_memory_h": torch.stack([
                    self.joint_memory_h[s, env_id]
                    for env_id, _agent_id, s, _e in selected
                ]),
                "joint_memory_c": torch.stack([
                    self.joint_memory_c[s, env_id]
                    for env_id, _agent_id, s, _e in selected
                ]),
                "next_joint_memory_h": stack_group("next_joint_memory_h"),
                "next_joint_memory_c": stack_group("next_joint_memory_c"),
                "baseline_memory_h": torch.stack([
                    self.baseline_memory_h[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "baseline_memory_c": torch.stack([
                    self.baseline_memory_c[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "focal_agent_ids": torch.tensor(
                    [agent_id for _env_id, agent_id, _s, _e in selected],
                    dtype=torch.long, device=self.device,
                ),
                "loss_mask": loss_mask,
            }

    def _get_sequence_batches_unpadded(self, sequence_length: int, mini_batch_size: int):
        """Yield shuffled BPTT windows."""
        T, E, N = self.ptr, self.num_envs, self.num_agents
        if T <= 0:
            return
        seq_len = max(1, min(int(sequence_length), T))
        grouped: dict[int, list[tuple[int, int, int, int]]] = {}
        for env_id in range(E):
            segment_start = 0
            segment_ends = [
                t + 1 for t in range(T) if self.dones[t, env_id].item() > 0.5
            ]
            if not segment_ends or segment_ends[-1] != T:
                segment_ends.append(T)
            for segment_end in segment_ends:
                for start_t in range(segment_start, segment_end, seq_len):
                    end_t = min(start_t + seq_len, segment_end)
                    for agent_id in range(N):
                        grouped.setdefault(end_t - start_t, []).append(
                            (env_id, agent_id, start_t, end_t)
                        )
                segment_start = segment_end

        lengths = list(grouped.keys())
        for length_idx in torch.randperm(len(lengths), device=self.device).tolist():
            chunks = grouped[lengths[length_idx]]
            order = torch.randperm(len(chunks), device=self.device).tolist()
            seq_batch_size = max(1, int(mini_batch_size) // max(lengths[length_idx], 1))

            for start in range(0, len(order), seq_batch_size):
                selected = [chunks[i] for i in order[start:start + seq_batch_size]]

                def stack_group(name: str):
                    values = getattr(self, name)
                    return torch.stack(
                        [values[s:e, env_id]
                         for env_id, _agent_id, s, e in selected],
                        dim=0,
                    )

                def stack_focal(name: str):
                    values = getattr(self, name)
                    return torch.stack(
                        [values[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected],
                        dim=0,
                    )

                yield {
                    "obs": stack_focal("obs"),
                    "next_obs": stack_focal("next_obs"),
                    "critic_states": stack_group("critic_states"),
                    "next_critic_states": stack_group("next_critic_states"),
                    "options": stack_focal("options"),
                    "critic_options": stack_group("options"),
                    "old_option_log_probs": stack_focal("option_log_probs"),
                    "option_masks": stack_focal("option_masks"),
                    "advantages": stack_focal("advantages"),
                    "returns": stack_group("returns"),
                    "old_team_values": stack_group("team_values"),
                    "old_joint_option_values": stack_group("joint_option_values"),
                    "old_baselines": stack_focal("baselines"),
                    "dones": stack_group("dones"),
                    "memory_h": torch.stack(
                        [self.memory_h[s, env_id, agent_id]
                         for env_id, agent_id, s, _e in selected],
                        dim=0,
                    ),
                    "memory_c": torch.stack(
                        [self.memory_c[s, env_id, agent_id]
                         for env_id, agent_id, s, _e in selected],
                        dim=0,
                    ),
                    "next_memory_h": stack_focal("next_memory_h"),
                    "next_memory_c": stack_focal("next_memory_c"),
                    "focal_agent_ids": torch.tensor(
                        [agent_id for _env_id, agent_id, _s, _e in selected],
                        dtype=torch.long,
                        device=self.device,
                    ),
                }
