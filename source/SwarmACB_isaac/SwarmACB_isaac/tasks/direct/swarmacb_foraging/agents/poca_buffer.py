# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout buffer for POCA multi-agent PPO.

Stores per-step data for all agents across all environments, then computes
GAE (Generalised Advantage Estimation) advantages per agent using the POCA
counterfactual baseline.
"""

from __future__ import annotations

import torch


class POCARolloutBuffer:
    """Fixed-length rollout storage with GAE computation.

    All tensors are sized ``(T, E, …)`` where *T* = horizon length and
    *E* = number of parallel environments.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
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
        self.gamma = gamma
        self.lam = lam
        self.device = device

        # Per-agent data  — shape (T, E, N, *)
        self.obs = torch.zeros(horizon, num_envs, num_agents, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, num_agents, act_dim, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, num_agents, device=device)

        # Shared team reward — shape (T, E)
        self.rewards = torch.zeros(horizon, num_envs, device=device)

        # Dones/truncated — shape (T, E)
        self.dones = torch.zeros(horizon, num_envs, device=device)

        # Critic outputs stored at collection time — shape (T, E) / (T, E, N)
        self.team_values = torch.zeros(horizon, num_envs, device=device)
        self.agent_baselines = torch.zeros(horizon, num_envs, num_agents, device=device)

        # Global state for critic — shape (T, E, state_dim)
        self.states = torch.zeros(horizon, num_envs, state_dim, device=device)

        # Computed after rollout — shape (T, E, N)
        self.advantages = torch.zeros(horizon, num_envs, num_agents, device=device)
        self.returns = torch.zeros(horizon, num_envs, num_agents, device=device)

        self.ptr = 0  # current write index

    def reset(self):
        """Reset pointer for a new rollout (doesn't zero memory)."""
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,            # (E, N, obs_dim)
        actions: torch.Tensor,         # (E, N, act_dim)
        log_probs: torch.Tensor,       # (E, N)
        reward: torch.Tensor,          # (E,)
        done: torch.Tensor,            # (E,)
        state: torch.Tensor,           # (E, state_dim)
        team_value: torch.Tensor,      # (E,)
        agent_baselines: torch.Tensor, # (E, N)
    ):
        """Store one time-step of experience."""
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = reward
        self.dones[t] = done
        self.states[t] = state
        self.team_values[t] = team_value
        self.agent_baselines[t] = agent_baselines
        self.ptr += 1

    def compute_advantages(
        self,
        last_team_value: torch.Tensor,      # (E,)
        last_agent_baselines: torch.Tensor,  # (E, N)
    ):
        """Compute POCA-style GAE advantages.

        The POCA advantage for agent *i* at step *t* is derived from the
        *counterfactual baseline* b_i and the *team value* V:

            δ_t = r_t + γ V(s_{t+1}) − b_i(s_t, a_{-i,t})

        We then apply standard GAE λ-returns over these δ's.
        """
        T, E, N = self.horizon, self.num_envs, self.num_agents
        gae = torch.zeros(E, N, device=self.device)

        next_team_val = last_team_value      # (E,)
        next_baselines = last_agent_baselines  # (E, N)

        for t in reversed(range(T)):
            mask = 1.0 - self.dones[t]  # (E,)

            # TD error per agent:  δ_i = r + γ V(s') − b_i(s, a_{-i})
            # On termination the bootstrap is 0 (handled by mask).
            td_target = self.rewards[t].unsqueeze(1) + self.gamma * mask.unsqueeze(1) * next_team_val.unsqueeze(1)
            delta = td_target - self.agent_baselines[t]  # (E, N)

            gae = delta + self.gamma * self.lam * mask.unsqueeze(1) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.agent_baselines[t]

            next_team_val = self.team_values[t]
            next_baselines = self.agent_baselines[t]

    # ────────────────────────────────────
    #  Mini-batch iteration
    # ────────────────────────────────────

    def get_batches(self, mini_batch_size: int):
        """Yield randomised mini-batches (flattened across T, E).

        Each yielded dict maps key → tensor of shape ``(MB, …)``.
        """
        T, E, N = self.horizon, self.num_envs, self.num_agents
        total = T * E

        # Flatten time & env dims
        flat_obs = self.obs.view(total, N, self.obs_dim)
        flat_act = self.actions.view(total, N, self.act_dim)
        flat_logp = self.log_probs.view(total, N)
        flat_adv = self.advantages.view(total, N)
        flat_ret = self.returns.view(total, N)
        flat_state = self.states.view(total, self.state_dim)
        flat_tv = self.team_values.view(total)

        # Shuffle indices
        indices = torch.randperm(total, device=self.device)

        for start in range(0, total, mini_batch_size):
            end = min(start + mini_batch_size, total)
            idx = indices[start:end]
            yield {
                "obs": flat_obs[idx],        # (MB, N, obs_dim)
                "actions": flat_act[idx],     # (MB, N, act_dim)
                "old_log_probs": flat_logp[idx],  # (MB, N)
                "advantages": flat_adv[idx],  # (MB, N)
                "returns": flat_ret[idx],     # (MB, N)
                "states": flat_state[idx],    # (MB, state_dim)
                "old_team_values": flat_tv[idx],  # (MB,)
            }
