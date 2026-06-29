# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Networks for the fixed-module Option-Critic phase.

Phase 1 treats the six existing ACB behavior modules as fixed options. The
local network therefore learns only the shared option selector and per-option
termination model; the collective critic remains centralized during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from .poca_networks import LinearEncoder


class FixedOptionManager(nn.Module):
    """Shared recurrent option selector and termination model.

    Given a local robot observation, the module emits:

    - logits for the policy over fixed options, pi_O(o | h_t)
    - logits for option termination probabilities, beta_o(h_t)
    The hidden state lets the phase-1 implementation start from the cyclamen
    setting, where memory is part of the final SwarmACB controller.
    """

    def __init__(
        self,
        obs_dim: int,
        num_options: int,
        hidden: int = 128,
        num_layers: int = 1,
        memory_size: int = 64,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_options = num_options
        self.memory_size = memory_size

        self.encoder = LinearEncoder(
            obs_dim,
            num_layers,
            hidden,
            kernel_init="kaiming_normal",
        )
        self.lstm = nn.LSTM(hidden, memory_size, batch_first=True)
        self.option_head = nn.Linear(memory_size, num_options)
        self.termination_head = nn.Linear(memory_size, num_options)

        nn.init.kaiming_normal_(self.option_head.weight, nonlinearity="linear")
        self.option_head.weight.data *= 0.2
        nn.init.zeros_(self.option_head.bias)
        nn.init.kaiming_normal_(self.termination_head.weight, nonlinearity="linear")
        self.termination_head.weight.data *= 0.2
        # Slightly negative bias encourages options to persist at the start.
        nn.init.constant_(self.termination_head.bias, -1.0)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.memory_size, device=device)
        c = torch.zeros(1, batch_size, self.memory_size, device=device)
        return h, c

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return local selector logits and termination logits."""
        B, T = obs_seq.shape[:2]
        enc = self.encoder(obs_seq.reshape(B * T, self.obs_dim)).view(B, T, -1)
        if state is None:
            state = self.initial_state(B, obs_seq.device)
        out, next_state = self.lstm(enc, state)
        return self.option_head(out), self.termination_head(out), next_state

    def step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        option_logits, termination_logits, next_state = self.forward_sequence(
            obs.unsqueeze(1),
            state,
        )
        return option_logits[:, 0], termination_logits[:, 0], next_state

    def get_option_dist(self, option_logits: torch.Tensor) -> Categorical:
        return Categorical(logits=option_logits)

    def get_termination_dist(
        self,
        termination_logits: torch.Tensor,
        options: torch.Tensor,
    ) -> Bernoulli:
        selected_logits = termination_logits.gather(-1, options.long().unsqueeze(-1))
        return Bernoulli(logits=selected_logits.squeeze(-1))
