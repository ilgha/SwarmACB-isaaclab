# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Local actor networks for learned-option collective Option-Critic.

Phase 2 retains Cyclamen's compact local observation and recurrent memory, but
learns the intra-option motor policies instead of treating the six ACB modules
as fixed options. Centralized critics are defined in ``poca_networks.py`` and
are used only during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .poca_networks import LinearEncoder, _linear_layer, _mlagents_lstm


class LearnedOptionActor(nn.Module):
    """Shared recurrent policy-over-options, terminations, and motor options.

    The recurrent manager receives Cyclamen's four inputs. Each motor option
    applies a state-dependent attention mask to the full local sensor vector,
    then combines the attended encoding with the recurrent manager state. This
    follows feature-level Attention Option-Critic while keeping execution local
    to each robot.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_options: int,
        hidden: int = 128,
        num_layers: int = 1,
        memory_size: int = 128,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_options = int(num_options)
        self.memory_size = int(memory_size)
        self.manager_obs_dim = 4 if self.obs_dim == 24 else self.obs_dim
        if self.obs_dim not in (4, 24):
            raise ValueError(
                "LearnedOptionActor expects either the 4D Cyclamen input or "
                f"the full 24D local sensor input, got {self.obs_dim}."
            )

        self.encoder = LinearEncoder(
            self.manager_obs_dim,
            num_layers,
            hidden,
            kernel_init="kaiming_normal",
        )
        self.lstm, self.hidden_size = _mlagents_lstm(hidden, memory_size)
        self.attention_encoder = LinearEncoder(
            self.obs_dim,
            num_layers,
            self.hidden_size,
            kernel_init="kaiming_normal",
        )
        self.option_encoder = LinearEncoder(
            self.obs_dim,
            num_layers,
            self.hidden_size,
            kernel_init="kaiming_normal",
        )

        self.option_head = _linear_layer(
            self.hidden_size,
            self.num_options,
            kernel_init="kaiming_normal",
            kernel_gain=0.1,
        )
        self.attention_head = _linear_layer(
            self.hidden_size,
            self.num_options * self.obs_dim,
            kernel_init="kaiming_normal",
            kernel_gain=0.1,
        )
        self.action_heads = nn.ModuleList([
            _linear_layer(
                self.hidden_size,
                self.act_dim,
                kernel_init="kaiming_normal",
                kernel_gain=0.2,
            )
            for _ in range(self.num_options)
        ])
        self.termination_heads = nn.ModuleList([
            _linear_layer(
                self.hidden_size,
                1,
                kernel_init="kaiming_normal",
                kernel_gain=0.2,
            )
            for _ in range(self.num_options)
        ])
        for head in self.termination_heads:
            nn.init.constant_(head.bias, -1.0)

        # ML-Agents-style state-independent Gaussian scale, now per option.
        self.log_std = nn.Parameter(torch.zeros(self.num_options, self.act_dim))

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h, c

    def _option_outputs(
        self,
        features: torch.Tensor,
        observations: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        leading_shape = features.shape[:-1]
        sensor_context = self.attention_encoder(
            observations.reshape(-1, self.obs_dim)
        ).view(*leading_shape, self.hidden_size)
        attentions = torch.sigmoid(
            self.attention_head(features + sensor_context).view(
                *leading_shape,
                self.num_options,
                self.obs_dim,
            )
        )
        attended_obs = observations.unsqueeze(-2) * attentions
        attended = self.option_encoder(
            attended_obs.reshape(-1, self.obs_dim)
        ).view(
            *leading_shape,
            self.num_options,
            -1,
        )
        attended = attended + features.unsqueeze(-2)

        action_means = torch.stack([
            head(attended[..., option_id, :])
            for option_id, head in enumerate(self.action_heads)
        ], dim=-2)
        termination_logits = torch.cat([
            head(attended[..., option_id, :])
            for option_id, head in enumerate(self.termination_heads)
        ], dim=-1)
        action_stds = self.log_std.exp().view(
            *((1,) * len(leading_shape)),
            self.num_options,
            self.act_dim,
        ).expand_as(action_means)
        option_logits = self.option_head(features)
        return (
            option_logits,
            termination_logits,
            action_means,
            action_stds,
            attentions,
        )

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Evaluate a recurrent observation sequence.

        Args:
            obs_seq: ``(batch, time, obs_dim)``.

        Returns:
            Option logits, termination logits, action means, action standard
            deviations, attention masks, and the next recurrent state.
        """
        batch_size, sequence_length = obs_seq.shape[:2]
        manager_obs = (
            obs_seq[..., 16:20]
            if self.obs_dim == 24
            else obs_seq
        )
        encoded = self.encoder(
            manager_obs.reshape(
                batch_size * sequence_length,
                self.manager_obs_dim,
            )
        ).view(batch_size, sequence_length, -1)
        if state is None:
            state = self.initial_state(batch_size, obs_seq.device)
        features, next_state = self.lstm(encoded, state)
        outputs = self._option_outputs(features, obs_seq)
        return (*outputs, next_state)

    def step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        outputs = self.forward_sequence(obs.unsqueeze(1), state)
        return (
            outputs[0][:, 0],
            outputs[1][:, 0],
            outputs[2][:, 0],
            outputs[3][:, 0],
            outputs[4][:, 0],
            outputs[5],
        )

    @staticmethod
    def _gather_options(
        values: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the active option from ``(..., options, features)``."""
        gather_index = options.long().unsqueeze(-1).unsqueeze(-1)
        gather_index = gather_index.expand(*options.shape, 1, values.shape[-1])
        return values.gather(-2, gather_index).squeeze(-2)

    def selected_action_dist(
        self,
        action_means: torch.Tensor,
        action_stds: torch.Tensor,
        options: torch.Tensor,
    ) -> Normal:
        means = self._gather_options(action_means, options)
        stds = self._gather_options(action_stds, options)
        return Normal(means, stds)

    @staticmethod
    def selected_termination_logits(
        termination_logits: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        return termination_logits.gather(
            -1,
            options.long().unsqueeze(-1),
        ).squeeze(-1)
