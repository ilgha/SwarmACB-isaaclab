# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""POCA neural networks — faithful reproduction of ML-Agents architecture.

Architecture (matching ML-Agents exactly):
──────────────────────────────────────────
• **Activation**: Swish (SiLU).

• **Actor** — SimpleActor with `conditional_sigma=False` (state-independent std)
  and `tanh_squash=False` (raw linear mean, no squashing).
  mu_head initialized with KaimingHeNormal, gain=0.2.
  Log-prob returned **per action dimension** (not summed) for per-dim PPO clipping.
  Entropy: mean across action dims.

• **Critic** — POCAValueNetwork wrapping MultiAgentNetworkBody:
  - Raw observations go directly into EntityEmbedding (no separate obs MLP).
  - EntityEmbedding = LinearEncoder(1 layer, Swish, T-Fixup init).
  - Two entity encoders: obs_entity_enc (obs only) and obs_act_entity_enc (obs+action).
  - ResidualSelfAttention with masked average pooling.
  - Post-attention LinearEncoder (num_layers, Swish, T-Fixup init).
  - Append normalized agent count → value head.

• **Weight init**: T-Fixup scheme — Normal(std = (0.125 / embed_dim)^0.5)
  used for entity embeddings, attention projections, and post-attention encoder.

Reference: Unity ML-Agents POCA
  ml-agents/mlagents/trainers/poca/optimizer_torch.py
  ml-agents/mlagents/trainers/torch_entities/networks.py
  ml-agents/mlagents/trainers/torch_entities/attention.py
  ml-agents/mlagents/trainers/torch_entities/layers.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal


# ──────────────────────────────────────────────────────────────────────
#  Swish / SiLU activation  (ML-Agents uses this, NOT ELU)
# ──────────────────────────────────────────────────────────────────────

Swish = nn.SiLU  # PyTorch's built-in SiLU is identical to Swish


# ──────────────────────────────────────────────────────────────────────
#  Weight init helpers  (matching ML-Agents layers.py)
# ──────────────────────────────────────────────────────────────────────

def _linear_layer(
    input_size: int,
    output_size: int,
    kernel_init: str = "xavier_uniform",
    kernel_gain: float = 1.0,
    bias_init: str = "zeros",
) -> nn.Linear:
    """Create nn.Linear with ML-Agents-style initialization.

    kernel_init options: "xavier_uniform", "kaiming_normal", "normal"
    """
    layer = nn.Linear(input_size, output_size)
    if kernel_init == "kaiming_normal":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    elif kernel_init == "normal":
        nn.init.normal_(layer.weight)
    elif kernel_init == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    else:
        raise ValueError(f"Unknown kernel_init: {kernel_init}")
    layer.weight.data *= kernel_gain

    if bias_init == "zeros":
        nn.init.zeros_(layer.bias)
    return layer


# ──────────────────────────────────────────────────────────────────────
#  LinearEncoder  (ML-Agents layers.py — Linear + Swish per layer)
# ──────────────────────────────────────────────────────────────────────

class LinearEncoder(nn.Module):
    """MLP encoder matching ML-Agents LinearEncoder.

    Uses Swish activation (NOT ELU) and configurable weight init.
    Default init is KaimingHeNormal (matching ML-Agents default).
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        hidden_size: int,
        kernel_init: str = "kaiming_normal",
        kernel_gain: float = 1.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            _linear_layer(input_size, hidden_size,
                          kernel_init=kernel_init, kernel_gain=kernel_gain),
            Swish(),
        ]
        for _ in range(num_layers - 1):
            layers += [
                _linear_layer(hidden_size, hidden_size,
                              kernel_init=kernel_init, kernel_gain=kernel_gain),
                Swish(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
#  EntityEmbedding  (ML-Agents attention.py)
#
#  Key: Uses LinearEncoder (1 layer, Swish) with T-Fixup init, NOT
#  a plain nn.Linear.
# ──────────────────────────────────────────────────────────────────────

class EntityEmbedding(nn.Module):
    """Embed raw entities before self-attention.

    Matches ML-Agents EntityEmbedding — uses a 1-layer LinearEncoder
    with T-Fixup initialization (Normal, gain = (0.125 / embed_size)^0.5).
    """

    def __init__(self, entity_size: int, embedding_size: int):
        super().__init__()
        gain = (0.125 / embedding_size) ** 0.5
        self.encoder = LinearEncoder(
            entity_size, 1, embedding_size,
            kernel_init="normal", kernel_gain=gain,
        )

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """entities: (B, num_entities, entity_size) → (B, num_entities, embed)"""
        return self.encoder(entities)


# ──────────────────────────────────────────────────────────────────────
#  Actor  (shared weights — local obs → action distribution)
# ──────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Gaussian actor matching ML-Agents SimpleActor.

    - conditional_sigma=False  → log_std is a learned parameter, not state-dependent
    - tanh_squash=False        → mean is the raw linear output (no tanh)
    - mu_head init: KaimingHeNormal, gain=0.2 (ML-Agents GaussianDistribution)
    - log_prob returns PER-DIMENSION values (ML-Agents computes per-dim ratio)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, num_layers: int = 2):
        super().__init__()
        # Body MLP — Swish activation, KaimingHeNormal (ML-Agents default)
        layers: list[nn.Module] = [
            _linear_layer(obs_dim, hidden, kernel_init="kaiming_normal"),
            Swish(),
        ]
        for _ in range(num_layers - 1):
            layers += [
                _linear_layer(hidden, hidden, kernel_init="kaiming_normal"),
                Swish(),
            ]
        self.net = nn.Sequential(*layers)

        # Mean head — KaimingHeNormal with gain=0.2 (ML-Agents GaussianDistribution)
        self.mu_head = _linear_layer(
            hidden, act_dim,
            kernel_init="kaiming_normal", kernel_gain=0.2,
        )

        # State-independent log_std, shape (1, act_dim) matching ML-Agents
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = self.mu_head(h)       # NO tanh — tanh_squash=False
        log_sigma = mu * 0 + self.log_std   # broadcast to (B, act_dim)
        std = log_sigma.exp()
        return mu, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return per-dim log_prob and mean entropy.

        Returns:
            log_prob_per_dim: (B, act_dim) — NOT summed across dims.
                ML-Agents computes PPO ratio per action dimension.
            entropy: (B,) — mean entropy across action dims.
        """
        dist = self.get_dist(obs)
        log_prob_per_dim = dist.log_prob(actions)      # (B, act_dim)
        entropy = dist.entropy().mean(dim=-1)           # (B,)
        return log_prob_per_dim, entropy


# ──────────────────────────────────────────────────────────────────────
#  Discrete Actor  (Categorical policy for discrete action spaces)
# ──────────────────────────────────────────────────────────────────────

class DiscreteActor(nn.Module):
    """Categorical actor for single-branch discrete action spaces.

    Matches ML-Agents MultiCategoricalDistribution with a single branch.
    Architecture mirrors the continuous Actor (same MLP body) but outputs
    logits → Categorical distribution instead of Gaussian.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256, num_layers: int = 2):
        super().__init__()
        self.num_actions = num_actions

        # Body MLP — same architecture as continuous Actor
        layers: list[nn.Module] = [
            _linear_layer(obs_dim, hidden, kernel_init="kaiming_normal"),
            Swish(),
        ]
        for _ in range(num_layers - 1):
            layers += [
                _linear_layer(hidden, hidden, kernel_init="kaiming_normal"),
                Swish(),
            ]
        self.net = nn.Sequential(*layers)

        # Logits head — outputs raw logits for each discrete action
        self.logits_head = _linear_layer(
            hidden, num_actions,
            kernel_init="kaiming_normal", kernel_gain=0.2,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return logits (B, num_actions)."""
        return self.logits_head(self.net(obs))

    def get_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        logits = self(obs)
        return torch.distributions.Categorical(logits=logits)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Return log_prob (B, 1) and entropy (B,).

        Args:
            obs: (B, obs_dim)
            actions: (B,) or (B, 1) — integer action indices.

        Returns:
            log_prob: (B, 1) — one value per branch (single branch).
            entropy: (B,) — categorical entropy.
        """
        dist = self.get_dist(obs)
        act = actions.squeeze(-1).long()
        log_prob = dist.log_prob(act).unsqueeze(-1)   # (B, 1)
        entropy = dist.entropy()                       # (B,)
        return log_prob, entropy


# ──────────────────────────────────────────────────────────────────────
#  Residual Self-Attention  (ML-Agents attention.py)
# ──────────────────────────────────────────────────────────────────────

class ResidualSelfAttention(nn.Module):
    """Pre-norm residual multi-head self-attention with masked average pooling.

    Flow:  LayerNorm → Q/K/V projections → MultiHead Attention (with optional mask)
           → fc_out + residual → LayerNorm → masked average pool over entities.

    Weight init follows ML-Agents T-Fixup:
        Normal(std=1) then multiply by gain = (0.125 / embed_dim)^0.5.
    """

    NEG_INF = -1e6
    EPSILON = 1e-6

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        gain = (0.125 / embed_dim) ** 0.5
        self.fc_q = _linear_layer(embed_dim, embed_dim, kernel_init="normal", kernel_gain=gain)
        self.fc_k = _linear_layer(embed_dim, embed_dim, kernel_init="normal", kernel_gain=gain)
        self.fc_v = _linear_layer(embed_dim, embed_dim, kernel_init="normal", kernel_gain=gain)
        self.fc_out = _linear_layer(embed_dim, embed_dim, kernel_init="normal", kernel_gain=gain)

        # ML-Agents uses a custom LayerNorm WITHOUT learnable parameters
        # (no gamma/beta — just (x - mean) / sqrt(var + 1e-5))
        self.embedding_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.residual_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, inp: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            inp      — (B, N, embed_dim)
            key_mask — (B, N), 1.0 = masked out, 0.0 = valid.
                       None means all entities are valid.
        Returns:
            (B, embed_dim) — masked-average-pooled attention output.
        """
        B, N, D = inp.shape
        H, d = self.num_heads, self.head_dim

        # Pre-norm
        x = self.embedding_norm(inp)

        # Q, K, V projections
        q = self.fc_q(x).view(B, N, H, d).transpose(1, 2)   # (B, H, N, d)
        k = self.fc_k(x).view(B, N, H, d).transpose(1, 2)
        v = self.fc_v(x).view(B, N, H, d).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)      # (B, H, N, N)

        # Apply attention mask (if provided)
        if key_mask is not None:
            mask_4d = key_mask.view(B, 1, 1, N)               # broadcast to (B, H, N, N)
            attn = attn + mask_4d * self.NEG_INF

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)

        # Residual connection + output projection
        output = self.fc_out(out) + x
        output = self.residual_norm(output)

        # Masked average pooling (ML-Agents: only average over valid entities)
        if key_mask is not None:
            valid = (1.0 - key_mask).unsqueeze(-1)            # (B, N, 1)
            numerator = (output * valid).sum(dim=1)           # (B, D)
            denominator = valid.sum(dim=1) + self.EPSILON     # (B, 1)
            return numerator / denominator
        else:
            return output.mean(dim=1)                         # (B, D)


# ──────────────────────────────────────────────────────────────────────
#  POCA Centralised Critic
#  (matches POCAValueNetwork + MultiAgentNetworkBody exactly)
#
#  Architecture:
#    raw obs → EntityEmbedding (1-layer LinearEncoder + Swish + T-Fixup)
#    → ResidualSelfAttention → LinearEncoder (num_layers, Swish, T-Fixup)
#    → cat(encoding, norm_agent_count) → value_head
#
#  NO separate obs_encoder MLP — entity embeddings take raw obs directly.
# ──────────────────────────────────────────────────────────────────────

class POCACritic(nn.Module):
    """Attention-based centralized critic with counterfactual baselines.

    Two evaluation modes (matching ML-Agents POCAValueNetwork):
    ────────────────────
    **critic_pass(all_obs)**
        All agents → obs-only entity embedding → self-attention → V(s).

    **baseline(agent_i_obs, other_obs, other_actions)**
        Agent i → obs-only embedding.
        Others  → obs+action embedding.
        Together through self-attention → counterfactual b_i.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        h_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.h_size = h_size

        # Entity embeddings (raw obs → h_size embedding)
        # ML-Agents: EntityEmbedding = LinearEncoder(1 layer, Swish, T-Fixup init)
        self.obs_entity_enc = EntityEmbedding(obs_dim, h_size)
        self.obs_act_entity_enc = EntityEmbedding(obs_dim + act_dim, h_size)

        # Self-attention over entity embeddings
        self.self_attn = ResidualSelfAttention(h_size, num_heads)

        # Post-attention encoder (ML-Agents: LinearEncoder with KaimingHeNormal + T-Fixup gain)
        # NOTE: ML-Agents LinearEncoder defaults to KaimingHeNormal init, NOT Normal.
        # Only EntityEmbeddings and RSA projections use Normal init.
        t_fixup_gain = (0.125 / h_size) ** 0.5
        self.linear_encoder = LinearEncoder(
            h_size, num_layers, h_size,
            kernel_init="kaiming_normal", kernel_gain=t_fixup_gain,
        )

        # Value head: h_size + 1 → 1  (the +1 is for normalized agent count)
        self.value_head = nn.Linear(h_size + 1, 1)

        # Max-agent tracker for normalization (non-trainable)
        self._current_max_agents = nn.Parameter(
            torch.tensor(1.0), requires_grad=False
        )

    # ── helpers ───────────────────────────────────────────────────

    def _norm_agent_count(self, n: int, B: int, device: torch.device) -> torch.Tensor:
        """Normalize agent count to [-1, +1] matching ML-Agents."""
        if n > self._current_max_agents.item():
            self._current_max_agents.data.fill_(float(n))
        val = n * 2.0 / self._current_max_agents.item() - 1.0
        return torch.full((B, 1), val, device=device)

    def _encode_and_value(self, entities: torch.Tensor, n_agents: int) -> torch.Tensor:
        """Shared tail: RSA → linear encoder → value head."""
        B = entities.shape[0]
        pooled = self.self_attn(entities)                          # (B, h)
        encoding = self.linear_encoder(pooled)                     # (B, h)
        nc = self._norm_agent_count(n_agents, B, encoding.device)
        encoding = torch.cat([encoding, nc], dim=-1)               # (B, h+1)
        return self.value_head(encoding)                           # (B, 1)

    # ── public API ────────────────────────────────────────────────

    def critic_pass(self, all_agent_obs: torch.Tensor) -> torch.Tensor:
        """Team value V(s).  All agents go through obs-only entity encoding.

        Args:   all_agent_obs — (B, N, obs_dim)
        Returns: (B, 1)
        """
        B, N, _ = all_agent_obs.shape
        entities = self.obs_entity_enc(all_agent_obs)      # (B, N, h)
        return self._encode_and_value(entities, N)

    def baseline(
        self,
        agent_i_obs: torch.Tensor,     # (B, obs_dim)
        other_obs: torch.Tensor,       # (B, M, obs_dim)
        other_actions: torch.Tensor,   # (B, M, act_dim)
    ) -> torch.Tensor:
        """Counterfactual baseline b_i.

        Agent i → obs-only entity embedding (no action).
        Others  → obs+action entity embedding.

        Returns: (B, 1)
        """
        M = other_obs.shape[1]

        # Agent i: obs-only entity
        ent_i = self.obs_entity_enc(agent_i_obs.unsqueeze(1))     # (B, 1, h)

        # Others: obs+action entities
        obs_act = torch.cat([other_obs, other_actions], dim=-1)   # (B, M, obs+act)
        ent_o = self.obs_act_entity_enc(obs_act)                  # (B, M, h)

        entities = torch.cat([ent_i, ent_o], dim=1)               # (B, 1+M, h)
        return self._encode_and_value(entities, 1 + M)

    def all_baselines(
        self,
        all_obs: torch.Tensor,       # (B, N, obs_dim)
        all_actions: torch.Tensor,   # (B, N, act_dim)
    ) -> torch.Tensor:
        """Compute baselines for every agent (batched — single forward pass).

        Instead of N separate forward passes, we construct all N counterfactual
        entity sets at once by stacking along the batch dimension:
          (B*N, N, h_size) where for each "virtual batch" entry b*N+i,
          entity 0 = obs-only embedding of agent i,
          entities 1..N-1 = obs+act embedding of all others.

        Returns: (B, N)
        """
        B, N, _ = all_obs.shape

        # Embed all obs-only: (B, N, h)
        obs_emb = self.obs_entity_enc(all_obs)                     # (B, N, h)

        # Embed all obs+act: (B, N, h)
        obs_act = torch.cat([all_obs, all_actions], dim=-1)        # (B, N, obs+act)
        obs_act_emb = self.obs_act_entity_enc(obs_act)             # (B, N, h)

        # For agent i's baseline:
        #   entity 0 = obs_emb[:, i]  (obs-only)
        #   entities 1..N-1 = obs_act_emb[:, other_indices]
        # We build this for ALL i simultaneously by repeating and masking.

        # Expand obs_emb for the "self" slot: (B, N, 1, h) — agent i's obs embedding
        self_ent = obs_emb.unsqueeze(2)                            # (B, N, 1, h)

        # Build "others" obs_act_emb for each agent i:
        # For each i, we need all j != i. We can construct this with a rolled gather.
        # Create index for "others": for agent i, others = [0..i-1, i+1..N-1]
        # Efficient: tile obs_act_emb to (B, N, N, h), then remove diagonal
        tiled = obs_act_emb.unsqueeze(1).expand(B, N, N, self.h_size)  # (B, N, N, h)
        # Mask to remove self (diagonal)
        mask = ~torch.eye(N, dtype=torch.bool, device=all_obs.device)  # (N, N)
        # Gather others: (B, N, N-1, h)
        others_ent = tiled[:, :, :, :].reshape(B * N, N, self.h_size)
        mask_flat = mask.unsqueeze(0).expand(B, -1, -1).reshape(B * N, N)
        others_flat = others_ent[mask_flat].reshape(B, N, N - 1, self.h_size)

        # Concatenate self + others: (B, N, N, h)
        entities = torch.cat([self_ent, others_flat], dim=2)       # (B, N, N, h)

        # Reshape to (B*N, N, h) for a single RSA forward pass
        entities_flat = entities.reshape(B * N, N, self.h_size)

        # Shared tail: RSA → encoder → value head
        values = self._encode_and_value(entities_flat, N)          # (B*N, 1)
        return values.squeeze(-1).reshape(B, N)                    # (B, N)
