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
        self.team_values = torch.zeros(T, E, device=device)
        self.joint_option_values = torch.zeros(T, E, device=device)
        self.baselines = torch.zeros(T, E, N, device=device)
        self.memory_h = torch.zeros(T, E, N, memory_size, device=device)
        self.memory_c = torch.zeros(T, E, N, memory_size, device=device)
        self.next_memory_h = torch.zeros(T, E, N, memory_size, device=device)
        self.next_memory_c = torch.zeros(T, E, N, memory_size, device=device)

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
        team_value: torch.Tensor,
        joint_option_value: torch.Tensor,
        baselines: torch.Tensor,
        memory_h: torch.Tensor,
        memory_c: torch.Tensor,
        next_memory_h: torch.Tensor,
        next_memory_c: torch.Tensor,
    ):
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
        self.team_values[t] = team_value
        self.joint_option_values[t] = joint_option_value
        self.baselines[t] = baselines
        self.memory_h[t] = memory_h
        self.memory_c[t] = memory_c
        self.next_memory_h[t] = next_memory_h
        self.next_memory_c[t] = next_memory_c
        self.ptr += 1

    def compute_returns_and_advantages(self, last_team_value: torch.Tensor):
        """Compute lambda returns and per-robot counterfactual advantages."""
        if self.ptr <= 0:
            return

        last = self.ptr - 1
        self.returns[last] = (
            self.rewards[last]
            + self.gamma * (1.0 - self.dones[last]) * last_team_value
        )
        for t in reversed(range(last)):
            self.returns[t] = (
                self.rewards[t]
                + self.gamma
                * (1.0 - self.dones[t])
                * (
                    (1.0 - self.lam) * self.team_values[t + 1]
                    + self.lam * self.returns[t + 1]
                )
            )

        self.advantages[: self.ptr] = (
            self.returns[: self.ptr].unsqueeze(-1) - self.baselines[: self.ptr]
        )

    def get_sequence_batches(self, sequence_length: int, mini_batch_size: int):
        """Yield shuffled BPTT windows."""
        T, E = self.ptr, self.num_envs
        if T <= 0:
            return
        seq_len = max(1, min(int(sequence_length), T))
        grouped: dict[int, list[tuple[int, int, int]]] = {}
        for env_id in range(E):
            for start_t in range(0, T, seq_len):
                end_t = min(start_t + seq_len, T)
                grouped.setdefault(end_t - start_t, []).append((env_id, start_t, end_t))

        lengths = list(grouped.keys())
        for length_idx in torch.randperm(len(lengths), device=self.device).tolist():
            chunks = grouped[lengths[length_idx]]
            order = torch.randperm(len(chunks), device=self.device).tolist()
            seq_batch_size = max(1, int(mini_batch_size) // max(lengths[length_idx], 1))

            for start in range(0, len(order), seq_batch_size):
                selected = [chunks[i] for i in order[start:start + seq_batch_size]]

                def stack_time(name: str):
                    values = getattr(self, name)
                    return torch.stack(
                        [values[s:e, env_id] for env_id, s, e in selected],
                        dim=0,
                    )

                yield {
                    "obs": stack_time("obs"),
                    "next_obs": stack_time("next_obs"),
                    "critic_states": stack_time("critic_states"),
                    "next_critic_states": stack_time("next_critic_states"),
                    "options": stack_time("options"),
                    "old_option_log_probs": stack_time("option_log_probs"),
                    "option_masks": stack_time("option_masks"),
                    "advantages": stack_time("advantages"),
                    "returns": stack_time("returns"),
                    "old_team_values": stack_time("team_values"),
                    "old_joint_option_values": stack_time("joint_option_values"),
                    "old_baselines": stack_time("baselines"),
                    "dones": stack_time("dones"),
                    "memory_h": torch.stack(
                        [self.memory_h[s, env_id] for env_id, s, _e in selected],
                        dim=0,
                    ),
                    "memory_c": torch.stack(
                        [self.memory_c[s, env_id] for env_id, s, _e in selected],
                        dim=0,
                    ),
                    "next_memory_h": stack_time("next_memory_h"),
                    "next_memory_c": stack_time("next_memory_c"),
                }
