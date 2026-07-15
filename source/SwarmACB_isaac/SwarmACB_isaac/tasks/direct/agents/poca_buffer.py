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
        critic_memory_size: int = 0,
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
        self.critic_memory_size = int(critic_memory_size or 0)
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
        self.timeouts = torch.zeros(T, E, device=device)
        self.timeout_values = torch.zeros(T, E, device=device)

        self.team_values = torch.zeros(T, E, device=device)      # V(s_t)
        self.baselines = torch.zeros(T, E, N, device=device)     # b_i(s_t, a_{-i,t})
        if self.memory_size > 0:
            self.memory_h = torch.zeros(T, E, N, self.memory_size, device=device)
            self.memory_c = torch.zeros(T, E, N, self.memory_size, device=device)
        else:
            self.memory_h = None
            self.memory_c = None
        if self.critic_memory_size > 0:
            H = self.critic_memory_size
            self.critic_memory_h = torch.zeros(T, E, H, device=device)
            self.critic_memory_c = torch.zeros(T, E, H, device=device)
            self.baseline_memory_h = torch.zeros(T, E, N, H, device=device)
            self.baseline_memory_c = torch.zeros(T, E, N, H, device=device)
        else:
            self.critic_memory_h = None
            self.critic_memory_c = None
            self.baseline_memory_h = None
            self.baseline_memory_c = None

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
        timeout: torch.Tensor,         # (E,)
        timeout_value: torch.Tensor,   # (E,)
        team_value: torch.Tensor,      # (E,)
        baselines: torch.Tensor,       # (E, N)
        memory_h: torch.Tensor | None = None,  # (E, N, memory_size)
        memory_c: torch.Tensor | None = None,  # (E, N, memory_size)
        critic_memory_h: torch.Tensor | None = None,  # (E, critic_memory_size)
        critic_memory_c: torch.Tensor | None = None,
        baseline_memory_h: torch.Tensor | None = None,  # (E, N, critic_memory_size)
        baseline_memory_c: torch.Tensor | None = None,
    ):
        if self.ptr >= self.horizon:
            raise RuntimeError("POCA rollout buffer is full")
        t = self.ptr
        self.obs[t] = obs
        self.critic_states[t] = critic_states
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.timeouts[t] = timeout
        self.timeout_values[t] = timeout_value
        self.team_values[t] = team_value
        self.baselines[t] = baselines
        if self.memory_size > 0:
            if memory_h is None or memory_c is None:
                raise ValueError("Recurrent rollout buffer requires memory_h and memory_c")
            self.memory_h[t] = memory_h
            self.memory_c[t] = memory_c
        if self.critic_memory_size > 0:
            critic_memories = (
                critic_memory_h, critic_memory_c,
                baseline_memory_h, baseline_memory_c,
            )
            if any(value is None for value in critic_memories):
                raise ValueError("Recurrent critic buffer requires all critic memories")
            self.critic_memory_h[t] = critic_memory_h
            self.critic_memory_c[t] = critic_memory_c
            self.baseline_memory_h[t] = baseline_memory_h
            self.baseline_memory_c[t] = baseline_memory_c
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
        T = self.ptr
        if T <= 0:
            return
        gamma, lam = self.gamma, self.lam

        # λ-return for the last step
        done_last = self.dones[T - 1]
        boundary_last = self.timeouts[T - 1] * self.timeout_values[T - 1]
        bootstrap_last = torch.where(
            done_last.bool(), boundary_last, last_team_value,
        )
        self.returns[T - 1] = self.rewards[T - 1] + gamma * bootstrap_last

        for t in reversed(range(T - 1)):
            done = self.dones[t]
            mask = 1.0 - done
            v_next = self.team_values[t + 1]           # V(s_{t+1})
            continuation = (1.0 - lam) * v_next + lam * self.returns[t + 1]
            boundary = self.timeouts[t] * self.timeout_values[t]
            bootstrap = mask * continuation + done * boundary
            self.returns[t] = (
                self.rewards[t] + gamma * bootstrap
            )

        # POCA advantage:  λ-return − baseline_i  (NOT value!)
        # returns: (T, E)  baselines: (T, E, N)  → broadcast
        self.advantages = self.returns.unsqueeze(-1) - self.baselines

    # ──────────────────────────────────────────────────────────────
    #  Mini-batch iteration
    # ──────────────────────────────────────────────────────────────

    def get_batches(self, mini_batch_size: int):
        """Yield focal-agent minibatches matching ML-Agents buffer semantics."""
        T, E, N = self.ptr, self.num_envs, self.num_agents
        total_groups = T * E
        total_agents = total_groups * N

        flat_obs = self.obs[:T].reshape(total_groups, N, self.obs_dim)
        flat_cs = self.critic_states[:T].reshape(total_groups, N, self.state_dim)
        flat_act = self.actions[:T].reshape(total_groups, N, self.act_dim)
        flat_logp = self.log_probs[:T].reshape(total_groups, N, self.act_dim)
        flat_adv = self.advantages[:T].reshape(total_groups, N)
        flat_ret = self.returns[:T].reshape(total_groups)
        flat_tv = self.team_values[:T].reshape(total_groups)
        flat_bl = self.baselines[:T].reshape(total_groups, N)

        indices = torch.randperm(total_agents, device=self.device)

        usable = (
            total_agents if total_agents < mini_batch_size
            else total_agents - total_agents % mini_batch_size
        )
        for start in range(0, usable, mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            group_idx = torch.div(idx, N, rounding_mode="floor")
            agent_idx = idx.remainder(N)
            yield {
                "obs": flat_obs[group_idx, agent_idx],
                "critic_states": flat_cs[group_idx],
                "actions": flat_act[group_idx, agent_idx],
                "critic_actions": flat_act[group_idx],
                "old_log_probs": flat_logp[group_idx, agent_idx],
                "advantages": flat_adv[group_idx, agent_idx],
                "returns": flat_ret[group_idx],
                "old_team_values": flat_tv[group_idx],
                "old_baselines": flat_bl[group_idx, agent_idx],
                "focal_agent_ids": agent_idx,
            }

    def get_sequence_batches(self, sequence_length: int, mini_batch_size: int):
        """Yield ML-Agents-style padded recurrent minibatches."""
        if self.memory_size <= 0 or self.memory_h is None or self.memory_c is None:
            raise RuntimeError("get_sequence_batches requires recurrent memory storage")

        T, E, N = self.ptr, self.num_envs, self.num_agents
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

            batch = {
                "obs": stack_focal("obs"),
                "critic_states": stack_group("critic_states"),
                "actions": stack_focal("actions"),
                "critic_actions": stack_group("actions"),
                "old_log_probs": stack_focal("log_probs"),
                "advantages": stack_focal("advantages"),
                "dones": stack_group("dones"),
                "returns": stack_group("returns"),
                "old_team_values": stack_group("team_values"),
                "old_baselines": stack_focal("baselines"),
                "memory_h": torch.stack([
                    self.memory_h[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "memory_c": torch.stack([
                    self.memory_c[s, env_id, agent_id]
                    for env_id, agent_id, s, _e in selected
                ]),
                "focal_agent_ids": torch.tensor(
                    [agent_id for _env_id, agent_id, _s, _e in selected],
                    dtype=torch.long, device=self.device,
                ),
                "loss_mask": loss_mask,
            }
            if self.critic_memory_size > 0:
                batch.update({
                    "critic_memory_h": torch.stack([
                        self.critic_memory_h[s, env_id]
                        for env_id, _agent_id, s, _e in selected
                    ]),
                    "critic_memory_c": torch.stack([
                        self.critic_memory_c[s, env_id]
                        for env_id, _agent_id, s, _e in selected
                    ]),
                    "baseline_memory_h": torch.stack([
                        self.baseline_memory_h[s, env_id, agent_id]
                        for env_id, agent_id, s, _e in selected
                    ]),
                    "baseline_memory_c": torch.stack([
                        self.baseline_memory_c[s, env_id, agent_id]
                        for env_id, agent_id, s, _e in selected
                    ]),
                })
            yield batch

    def _get_sequence_batches_unpadded(self, sequence_length: int, mini_batch_size: int):
        """Yield shuffled BPTT windows for recurrent actor updates."""
        if self.memory_size <= 0 or self.memory_h is None or self.memory_c is None:
            raise RuntimeError("get_sequence_batches requires recurrent memory storage")

        T, E, N = self.ptr, self.num_envs, self.num_agents
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
            L = lengths[length_idx]
            chunks = grouped[L]
            order = torch.randperm(len(chunks), device=self.device).tolist()
            seq_batch_size = max(1, int(mini_batch_size) // max(L, 1))

            for start in range(0, len(order), seq_batch_size):
                selected = [chunks[i] for i in order[start:start + seq_batch_size]]
                yield {
                    "obs": torch.stack(
                        [self.obs[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected], dim=0
                    ),
                    "critic_states": torch.stack(
                        [self.critic_states[s:e, env_id]
                         for env_id, _agent_id, s, e in selected], dim=0
                    ),
                    "actions": torch.stack(
                        [self.actions[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected], dim=0
                    ),
                    "critic_actions": torch.stack(
                        [self.actions[s:e, env_id]
                         for env_id, _agent_id, s, e in selected], dim=0
                    ),
                    "old_log_probs": torch.stack(
                        [self.log_probs[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected], dim=0
                    ),
                    "advantages": torch.stack(
                        [self.advantages[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected], dim=0
                    ),
                    "dones": torch.stack(
                        [self.dones[s:e, env_id]
                         for env_id, _agent_id, s, e in selected], dim=0
                    ),
                    "returns": torch.stack(
                        [self.returns[s:e, env_id]
                         for env_id, _agent_id, s, e in selected], dim=0
                    ),
                    "old_team_values": torch.stack(
                        [self.team_values[s:e, env_id]
                         for env_id, _agent_id, s, e in selected], dim=0
                    ),
                    "old_baselines": torch.stack(
                        [self.baselines[s:e, env_id, agent_id]
                         for env_id, agent_id, s, e in selected], dim=0
                    ),
                    "memory_h": torch.stack(
                        [self.memory_h[s, env_id, agent_id]
                         for env_id, agent_id, s, _e in selected], dim=0
                    ),
                    "memory_c": torch.stack(
                        [self.memory_c[s, env_id, agent_id]
                         for env_id, agent_id, s, _e in selected], dim=0
                    ),
                    "focal_agent_ids": torch.tensor(
                        [agent_id for _env_id, agent_id, _s, _e in selected],
                        dtype=torch.long,
                        device=self.device,
                    ),
                }
