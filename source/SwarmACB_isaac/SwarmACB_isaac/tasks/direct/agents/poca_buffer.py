# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""POCA rollout buffer with lambda-return and counterfactual advantages.

Matches ML-Agents advantage computation exactly:

    lambda_returns[T-1] = r[T-1] + γ (1-d[T-1]) V_next
    lambda_returns[t]   = γ λ (1-d[t]) lambda_returns[t+1]
                          + r[t]
                          + (1-λ) γ (1-d[t]) V[t+1]

    advantage_i[t] = lambda_returns[t] − baseline_i[t]

Key change from previous version:
    log_probs are stored **per action dimension** (T, E, N, act_dim)
    to support ML-Agents' per-dimension ratio and PPO clipping.

Reference: ml-agents/mlagents/trainers/trainer/trainer_utils.py  lambda_return()
           ml-agents/mlagents/trainers/poca/trainer.py            _process_trajectory()
"""

from __future__ import annotations

import torch


class POCARolloutBuffer:
    """Fixed-horizon rollout storage for POCA.

    All tensors are shaped ``(T, E, …)`` where T = horizon, E = num_envs.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: int = 5,
        memory_size: int = 0,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: torch.device | str = "cuda",
    ):
        self.horizon = horizon
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.memory_size = int(memory_size or 0)
        self.gamma = gamma
        self.lam = lam
        self.device = device

        T, E, N = horizon, num_envs, num_agents

        # ── Per-step storage ──────────────────────────────────────
        self.obs = torch.zeros(T, E, N, obs_dim, device=device)
        # Critic state: 5D polar (ρ, cos α, sin α, cos β, sin β) — separate from obs
        # The critic uses this instead of agent observations (SwarmACB modification).
        self.critic_states = torch.zeros(T, E, N, state_dim, device=device)
        self.actions = torch.zeros(T, E, N, act_dim, device=device)
        # Per-dim log_probs (NOT summed!) — needed for ML-Agents per-dim PPO clipping
        self.log_probs = torch.zeros(T, E, N, act_dim, device=device)

        self.rewards = torch.zeros(T, E, device=device)          # shared team reward
        self.dones = torch.zeros(T, E, device=device)

        self.team_values = torch.zeros(T, E, device=device)      # V(s_t)
        self.baselines = torch.zeros(T, E, N, device=device)     # b_i(s_t, a_{-i,t})
        if self.memory_size > 0:
            self.memory_h = torch.zeros(T, E, N, self.memory_size, device=device)
            self.memory_c = torch.zeros(T, E, N, self.memory_size, device=device)
        else:
            self.memory_h = None
            self.memory_c = None

        # ── Computed after rollout ────────────────────────────────
        self.returns = torch.zeros(T, E, device=device)          # λ-return (same for all agents)
        self.advantages = torch.zeros(T, E, N, device=device)    # λ-return − baseline_i

        self.ptr = 0

    # ──────────────────────────────────────────────────────────────

    def reset(self):
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,            # (E, N, obs_dim)
        critic_states: torch.Tensor,   # (E, N, state_dim)  — 5D polar state for critic
        actions: torch.Tensor,         # (E, N, act_dim)
        log_probs: torch.Tensor,       # (E, N, act_dim)  — per-dim!
        reward: torch.Tensor,          # (E,)
        done: torch.Tensor,            # (E,)
        team_value: torch.Tensor,      # (E,)
        baselines: torch.Tensor,       # (E, N)
        memory_h: torch.Tensor | None = None,  # (E, N, memory_size)
        memory_c: torch.Tensor | None = None,  # (E, N, memory_size)
    ):
        t = self.ptr
        self.obs[t] = obs
        self.critic_states[t] = critic_states
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.team_values[t] = team_value
        self.baselines[t] = baselines
        if self.memory_size > 0:
            if memory_h is None or memory_c is None:
                raise ValueError("Recurrent rollout buffer requires memory_h and memory_c")
            self.memory_h[t] = memory_h
            self.memory_c[t] = memory_c
        self.ptr += 1

    # ──────────────────────────────────────────────────────────────
    #  Lambda-return & counterfactual advantage
    # ──────────────────────────────────────────────────────────────

    def compute_returns_and_advantages(
        self,
        last_team_value: torch.Tensor,   # (E,) — bootstrap V
    ):
        """Compute λ-returns and POCA counterfactual advantages.

        Matches ML-Agents ``lambda_return`` exactly, extended with
        done-masking for vectorized envs that auto-reset mid-rollout.
        """
        T = self.horizon
        gamma, lam = self.gamma, self.lam

        # λ-return for the last step
        mask_last = 1.0 - self.dones[T - 1]
        self.returns[T - 1] = (
            self.rewards[T - 1] + gamma * mask_last * last_team_value
        )

        for t in reversed(range(T - 1)):
            mask = 1.0 - self.dones[t]
            v_next = self.team_values[t + 1]           # V(s_{t+1})
            self.returns[t] = (
                gamma * lam * mask * self.returns[t + 1]
                + self.rewards[t]
                + (1.0 - lam) * gamma * mask * v_next
            )

        # POCA advantage:  λ-return − baseline_i  (NOT value!)
        # returns: (T, E)  baselines: (T, E, N)  → broadcast
        self.advantages = self.returns.unsqueeze(-1) - self.baselines

    # ──────────────────────────────────────────────────────────────
    #  Mini-batch iteration
    # ──────────────────────────────────────────────────────────────

    def get_batches(self, mini_batch_size: int):
        """Yield shuffled mini-batches flattened across T × E."""
        T, E, N = self.horizon, self.num_envs, self.num_agents
        total = T * E

        flat_obs = self.obs.view(total, N, self.obs_dim)
        flat_cs = self.critic_states.view(total, N, self.state_dim)
        flat_act = self.actions.view(total, N, self.act_dim)
        flat_logp = self.log_probs.view(total, N, self.act_dim)    # per-dim!
        flat_adv = self.advantages.view(total, N)
        flat_ret = self.returns.view(total)                 # same for all agents
        flat_tv = self.team_values.view(total)
        flat_bl = self.baselines.view(total, N)

        indices = torch.randperm(total, device=self.device)

        for start in range(0, total, mini_batch_size):
            end = min(start + mini_batch_size, total)
            idx = indices[start:end]
            yield {
                "obs": flat_obs[idx],                # (MB, N, obs_dim)
                "critic_states": flat_cs[idx],       # (MB, N, state_dim)
                "actions": flat_act[idx],            # (MB, N, act_dim)
                "old_log_probs": flat_logp[idx],     # (MB, N, act_dim) — per-dim!
                "advantages": flat_adv[idx],         # (MB, N)
                "returns": flat_ret[idx],            # (MB,)
                "old_team_values": flat_tv[idx],     # (MB,)
                "old_baselines": flat_bl[idx],       # (MB, N)
            }

    def get_sequence_batches(self, sequence_length: int, mini_batch_size: int):
        """Yield shuffled BPTT windows for recurrent actor updates."""
        if self.memory_size <= 0 or self.memory_h is None or self.memory_c is None:
            raise RuntimeError("get_sequence_batches requires recurrent memory storage")

        T, E = self.horizon, self.num_envs
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
                    "actions": torch.stack(
                        [self.actions[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "old_log_probs": torch.stack(
                        [self.log_probs[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "advantages": torch.stack(
                        [self.advantages[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "dones": torch.stack(
                        [self.dones[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "returns": torch.stack(
                        [self.returns[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "old_team_values": torch.stack(
                        [self.team_values[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "old_baselines": torch.stack(
                        [self.baselines[s:e, env_id] for env_id, s, e in selected], dim=0
                    ),
                    "memory_h": torch.stack(
                        [self.memory_h[s, env_id] for env_id, s, _e in selected], dim=0
                    ),
                    "memory_c": torch.stack(
                        [self.memory_c[s, env_id] for env_id, s, _e in selected], dim=0
                    ),
                }
