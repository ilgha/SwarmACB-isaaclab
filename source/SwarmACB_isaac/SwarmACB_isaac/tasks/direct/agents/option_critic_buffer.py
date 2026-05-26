# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for fixed-module Option-Critic training."""

from __future__ import annotations

import torch


class FixedOptionRolloutBuffer:
    """Fixed-horizon storage shaped around SwarmACB vectorized MARL.

    Tensors use ``(T, E, N, ...)`` for time, environments, and agents.  Rewards
    and values are shared team quantities with shape ``(T, E)``.
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
        self.critic_states = torch.zeros(T, E, N, state_dim, device=device)
        self.options = torch.zeros(T, E, N, dtype=torch.long, device=device)
        self.termination_options = torch.zeros(T, E, N, dtype=torch.long, device=device)
        self.termination_masks = torch.zeros(T, E, N, device=device)
        self.option_log_probs = torch.zeros(T, E, N, device=device)
        self.option_masks = torch.zeros(T, E, N, device=device)
        self.beta_probs = torch.zeros(T, E, N, device=device)
        self.rewards = torch.zeros(T, E, device=device)
        self.dones = torch.zeros(T, E, device=device)
        self.values = torch.zeros(T, E, device=device)
        self.memory_h = torch.zeros(T, E, N, memory_size, device=device)
        self.memory_c = torch.zeros(T, E, N, memory_size, device=device)

        self.returns = torch.zeros(T, E, device=device)
        self.advantages = torch.zeros(T, E, device=device)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        critic_states: torch.Tensor,
        options: torch.Tensor,
        termination_options: torch.Tensor,
        termination_masks: torch.Tensor,
        option_log_probs: torch.Tensor,
        option_masks: torch.Tensor,
        beta_probs: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        memory_h: torch.Tensor,
        memory_c: torch.Tensor,
    ):
        t = self.ptr
        self.obs[t] = obs
        self.critic_states[t] = critic_states
        self.options[t] = options.long()
        self.termination_options[t] = termination_options.long()
        self.termination_masks[t] = termination_masks
        self.option_log_probs[t] = option_log_probs
        self.option_masks[t] = option_masks
        self.beta_probs[t] = beta_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.values[t] = value
        self.memory_h[t] = memory_h
        self.memory_c[t] = memory_c
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor):
        """Compute standard GAE returns for the shared team value."""
        last_adv = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * mask * next_value - self.values[t]
            last_adv = delta + self.gamma * self.lam * mask * last_adv
            self.advantages[t] = last_adv
        self.returns = self.advantages + self.values

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
            L = lengths[length_idx]
            chunks = grouped[L]
            order = torch.randperm(len(chunks), device=self.device).tolist()
            seq_batch_size = max(1, int(mini_batch_size) // max(L, 1))

            for start in range(0, len(order), seq_batch_size):
                selected = [chunks[i] for i in order[start:start + seq_batch_size]]
                yield {
                    "obs": torch.stack(
                        [self.obs[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "critic_states": torch.stack(
                        [self.critic_states[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "options": torch.stack(
                        [self.options[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "termination_options": torch.stack(
                        [self.termination_options[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "termination_masks": torch.stack(
                        [self.termination_masks[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "old_option_log_probs": torch.stack(
                        [self.option_log_probs[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "option_masks": torch.stack(
                        [self.option_masks[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "advantages": torch.stack(
                        [self.advantages[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "returns": torch.stack(
                        [self.returns[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "old_values": torch.stack(
                        [self.values[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "dones": torch.stack(
                        [self.dones[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "memory_h": torch.stack(
                        [self.memory_h[s, env_id] for env_id, s, _e in selected], dim=0
                    ),
                    "memory_c": torch.stack(
                        [self.memory_c[s, env_id] for env_id, s, _e in selected], dim=0
                    ),
                }
