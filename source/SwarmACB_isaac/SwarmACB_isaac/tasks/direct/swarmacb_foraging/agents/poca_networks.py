# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""POCA (Proximal-policy Optimization with Cooperative Agents) neural networks.

Architecture overview
─────────────────────
• **Actor** — shared MLP policy that maps each agent's *local* observation (47-D)
  to a Gaussian action distribution (2-D: left / right wheel velocity).
  All robots share the same actor weights (parameter sharing).

• **Critic** — attention-based centralized value function.  Receives the
  *global state* (42-D) plus every agent's local observation, runs an
  entity-attention block, and outputs a *team baseline* V(s) **and** a
  per-agent *counterfactual baseline* Q_i(s, a_{-i}) used to compute
  POCA-style advantages.

Reference: Cohen, M. et al., "On the Use and Misuse of Absorbing States …"
(Unity ML-Agents POCA implementation).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


# ──────────────────────────────────────────────────────────────────────
#  Actor  (shared weights — local obs → action distribution)
# ──────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Simple MLP Gaussian actor.

    Inputs:  obs  (B, obs_dim)
    Outputs: mu   (B, act_dim)
             std  (B, act_dim)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable per-dim

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = torch.tanh(self.mu_head(h))        # actions in [-1, 1]
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return log_prob (summed over action dims) and entropy."""
        dist = self.get_dist(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)            # (B,)
        return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────
#  Entity-attention block (used inside the centralised critic)
# ──────────────────────────────────────────────────────────────────────

class EntityAttention(nn.Module):
    """Multi-head dot-product attention over agent embeddings.

    Each head performs:  Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)  — N agent embeddings.

        Returns:
            (B, N, embed_dim)  — attention-updated embeddings.
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale          # (B, H, N, N)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)
        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────
#  POCA Centralised Critic
# ──────────────────────────────────────────────────────────────────────

class POCACritic(nn.Module):
    """Attention-based centralised critic for POCA.

    For each agent *i* the critic computes:

    1. **Team value** V(s) — single scalar summarising team return.
    2. **Counterfactual baseline** b_i(s, a_{−i}) — what agent *i* "would
       get" if it were removed.  The POCA advantage is:

           A_i = R − b_i(s, a_{−i})

       where *R* is the actual team return.

    Architecture:
        • Each agent's local obs is projected into an embedding.
        • Global state is projected and appended as an extra "entity".
        • Entity-attention mixes information.
        • A value head reads off the team V(s) from the global entity.
        • Per-agent heads read off counterfactual baselines.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        num_agents: int,
        embed_dim: int = 128,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_agents = num_agents

        # Per-agent observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )

        # Global state encoder (becomes the "global entity")
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )

        # Entity attention
        self.attention = EntityAttention(embed_dim, num_heads)

        # Post-attention residual MLP
        self.post_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ELU(),
        )

        # Team value head (reads from global entity token)
        self.value_head = nn.Linear(embed_dim, 1)

        # Counterfactual baseline head (reads from each agent token)
        self.baseline_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        agent_obs: torch.Tensor,   # (B, N, obs_dim)
        global_state: torch.Tensor, # (B, state_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            team_value:       (B, 1)   — V(s)
            agent_baselines:  (B, N)   — b_i(s, a_{-i})
        """
        B = agent_obs.shape[0]
        N = self.num_agents

        # Encode
        agent_emb = self.obs_encoder(agent_obs)               # (B, N, D)
        state_emb = self.state_encoder(global_state).unsqueeze(1)  # (B, 1, D)

        # Concatenate: [agent_0, …, agent_{N-1}, global]
        entities = torch.cat([agent_emb, state_emb], dim=1)   # (B, N+1, D)

        # Attention
        entities = entities + self.attention(entities)         # residual
        entities = entities + self.post_attn(entities)         # residual MLP

        # Split back
        agent_tokens = entities[:, :N, :]    # (B, N, D)
        global_token = entities[:, N, :]     # (B, D)

        team_value = self.value_head(global_token)             # (B, 1)
        baselines = self.baseline_head(agent_tokens).squeeze(-1)  # (B, N)

        return team_value, baselines
