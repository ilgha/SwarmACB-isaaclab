# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Decentralized actor for collective Attention Option-Critic.

Each learned option receives its own attended local sensor observation. Its
option value, continuous two-wheel intra-option policy, and termination
function are computed from that attended representation. New checkpoints
follow Attention Option-Critic and select options epsilon-softly from the
attended option values. There is no unmasked shortcut to any option output.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Categorical, Normal

from .poca_networks import LinearEncoder, _linear_layer, _mlagents_lstm


LEARNED_OPTION_CRITIC_VERSION = 4
SUPPORTED_LEARNED_OPTION_CRITIC_VERSIONS = (2, 3, 4)


def termination_objective(
    termination_probability: torch.Tensor,
    option_advantage: torch.Tensor,
    deliberation_cost: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Option-Critic termination-gradient objective.

    Minimizing ``beta * (Q_omega - V_omega + xi)`` decreases termination when
    the current option is better than reselection and increases it when the
    current option is sufficiently worse. ``xi`` is the deliberation cost.
    """
    active = mask.to(dtype=termination_probability.dtype)
    count = active.sum()
    if count.item() <= 0:
        return termination_probability.sum() * 0.0
    signal = option_advantage + float(deliberation_cost)
    return (termination_probability * signal * active).sum() / count


class SquashedNormal:
    """Diagonal Normal followed by ``tanh``, with corrected log probability.

    The environment consumes actions in ``[-1, 1]``. Using a squashed policy
    keeps sampled and evaluated actions identical, unlike clipping an
    unbounded Gaussian after its log probability has already been computed.
    ``entropy()`` returns the base-Normal entropy, which is the stable quantity
    used as the exploration regularizer.
    """

    _EPS = 1e-6

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.loc)

    @property
    def stddev(self) -> torch.Tensor:
        return self.scale

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.rsample(sample_shape))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        bounded = value.clamp(-1.0 + self._EPS, 1.0 - self._EPS)
        pre_tanh = 0.5 * (
            torch.log1p(bounded) - torch.log1p(-bounded)
        )
        log_det_jacobian = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - functional.softplus(-2.0 * pre_tanh)
        )
        return self.base_dist.log_prob(pre_tanh) - log_det_jacobian

    def entropy(self) -> torch.Tensor:
        return self.base_dist.entropy()


class LearnedOptionActor(nn.Module):
    """Shared recurrent Attention Option-Critic policy for every robot.

    A compact Cyclamen manager memory helps generate state-dependent attention
    masks. Each option then processes only ``h_omega(x) * x`` through a shared
    sensor encoder and recurrent cell with option-specific memory. Separate
    heads implement the local option value, continuous intra-option wheel
    policy, and termination function for every option.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_options: int,
        hidden: int = 128,
        num_layers: int = 1,
        memory_size: int = 128,
        option_hidden: int = 512,
        option_num_layers: int = 2,
        option_memory_size: int = 64,
        initial_termination_probability: float = 0.27,
        initial_log_std: float = -0.7,
        min_log_std: float = -2.5,
        max_log_std: float = 0.0,
        option_selector_temperature: float = 1.0,
        separate_selector: bool = False,
        epsilon_greedy_selector: bool = True,
        squash_actions: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_options = int(num_options)
        self.memory_size = int(memory_size)
        self.option_memory_size = int(option_memory_size)
        self.option_hidden = int(option_hidden)
        self.option_num_layers = int(option_num_layers)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.option_selector_temperature = float(
            option_selector_temperature
        )
        self.separate_selector = bool(separate_selector)
        self.epsilon_greedy_selector = bool(epsilon_greedy_selector)
        self.squash_actions = bool(squash_actions)
        self.manager_obs_dim = 4 if self.obs_dim == 24 else self.obs_dim

        if self.obs_dim not in (4, 24):
            raise ValueError(
                "LearnedOptionActor expects either the 4D Cyclamen input or "
                f"the full 24D local sensor input, got {self.obs_dim}."
            )
        if self.num_options <= 0:
            raise ValueError("num_options must be positive")
        if not 0.0 < initial_termination_probability < 1.0:
            raise ValueError(
                "initial_termination_probability must be strictly between 0 and 1"
            )
        if self.squash_actions:
            if not self.min_log_std < self.max_log_std:
                raise ValueError("min_log_std must be smaller than max_log_std")
            if not self.min_log_std <= initial_log_std <= self.max_log_std:
                raise ValueError(
                    "initial_log_std must lie inside [min_log_std, max_log_std]"
                )
        if self.option_selector_temperature <= 0.0:
            raise ValueError("option_selector_temperature must be positive")

        self.manager_encoder = LinearEncoder(
            self.manager_obs_dim,
            num_layers,
            hidden,
            kernel_init="kaiming_normal",
        )
        self.manager_lstm, self.manager_hidden_size = _mlagents_lstm(
            hidden,
            memory_size,
        )
        self.attention_encoder = LinearEncoder(
            self.obs_dim,
            num_layers,
            self.manager_hidden_size,
            kernel_init="kaiming_normal",
        )
        self.attention_head = _linear_layer(
            self.manager_hidden_size,
            self.num_options * self.obs_dim,
            kernel_init="kaiming_normal",
            kernel_gain=0.1,
        )

        self.option_sensor_encoder = LinearEncoder(
            self.obs_dim,
            self.option_num_layers,
            self.option_hidden,
            kernel_init="kaiming_normal",
        )
        self.option_lstm, self.option_recurrent_size = _mlagents_lstm(
            self.option_hidden,
            self.option_memory_size,
        )
        self.option_output_encoder = LinearEncoder(
            self.option_hidden + self.option_recurrent_size,
            self.option_num_layers,
            self.option_hidden,
            kernel_init="kaiming_normal",
        )

        self.option_value_heads = nn.ModuleList([
            _linear_layer(
                self.option_hidden,
                1,
                kernel_init="kaiming_normal",
                kernel_gain=0.1,
            )
            for _ in range(self.num_options)
        ])
        # Architecture version 3 used a separate PPO selector. It is retained
        # only so completed legacy checkpoints remain playable; version 4
        # follows AOC and selects epsilon-soft from option values directly.
        if self.separate_selector:
            self.selector_heads = nn.ModuleList([
                _linear_layer(
                    self.option_hidden,
                    1,
                    kernel_init="kaiming_normal",
                    kernel_gain=0.01,
                )
                for _ in range(self.num_options)
            ])
        self.action_heads = nn.ModuleList([
            _linear_layer(
                self.option_hidden,
                self.act_dim,
                kernel_init="kaiming_normal",
                kernel_gain=0.1,
            )
            for _ in range(self.num_options)
        ])
        self.termination_heads = nn.ModuleList([
            _linear_layer(
                self.option_hidden,
                1,
                kernel_init="kaiming_normal",
                kernel_gain=0.1,
            )
            for _ in range(self.num_options)
        ])
        termination_bias = math.log(
            initial_termination_probability
            / (1.0 - initial_termination_probability)
        )
        for head in self.termination_heads:
            nn.init.constant_(head.bias, termination_bias)

        if self.squash_actions:
            std_fraction = (
                (float(initial_log_std) - self.min_log_std)
                / (self.max_log_std - self.min_log_std)
            )
            std_fraction = min(max(std_fraction, 1e-6), 1.0 - 1e-6)
            initial_std_logit = math.log(std_fraction / (1.0 - std_fraction))
            self.log_std_logits = nn.Parameter(torch.full(
                (self.num_options, self.act_dim),
                initial_std_logit,
            ))
        else:
            # Match ML-Agents' continuous actor: state-independent log sigma,
            # with one independently learned pair for every option.
            self.log_std = nn.Parameter(torch.full(
                (self.num_options, self.act_dim),
                float(initial_log_std),
            ))

        # Packed public memory: one manager state plus one state per option.
        self.hidden_size = (
            self.manager_hidden_size
            + self.num_options * self.option_recurrent_size
        )

    def option_log_stds(self) -> torch.Tensor:
        if not self.squash_actions:
            return self.log_std
        fraction = torch.sigmoid(self.log_std_logits)
        return self.min_log_std + (
            self.max_log_std - self.min_log_std
        ) * fraction

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: dict,
        device: torch.device | str,
    ) -> "LearnedOptionActor":
        version = int(checkpoint.get("learned_option_critic_version", 0))
        if version not in SUPPORTED_LEARNED_OPTION_CRITIC_VERSIONS:
            raise RuntimeError(
                f"Checkpoint uses learned Option-Critic version {version}; "
                "the current actor supports versions "
                f"{SUPPORTED_LEARNED_OPTION_CRITIC_VERSIONS}."
            )
        if bool(checkpoint.get("discrete", False)):
            raise RuntimeError(
                "This checkpoint selects predefined behavior modules. "
                "OC2 is defined as six learned continuous intra-option "
                "wheel policies, so that experimental checkpoint is not "
                "compatible with LearnedOptionActor."
            )
        action_distribution = checkpoint.get(
            "action_distribution",
            "tanh_squashed_normal" if version <= 3 else "mlagents_normal",
        )
        squash_actions = action_distribution == "tanh_squashed_normal"
        actor = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            act_dim=int(checkpoint.get("act_dim", 2)),
            num_options=int(checkpoint["num_options"]),
            hidden=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            memory_size=int(checkpoint["memory_size"]),
            option_hidden=int(checkpoint["option_hidden_dim"]),
            option_num_layers=int(checkpoint["option_num_layers"]),
            option_memory_size=int(checkpoint["option_memory_size"]),
            initial_termination_probability=float(
                checkpoint["initial_termination_probability"]
            ),
            initial_log_std=float(checkpoint["initial_log_std"]),
            min_log_std=float(checkpoint["min_log_std"]),
            max_log_std=float(checkpoint["max_log_std"]),
            option_selector_temperature=float(
                checkpoint.get(
                    "option_selector_temperature",
                    checkpoint.get("option_value_temperature", 1.0),
                )
            ),
            separate_selector=(version == 3),
            epsilon_greedy_selector=(version >= 4),
            squash_actions=squash_actions,
        ).to(device)
        actor.load_state_dict(checkpoint["actor"])
        return actor

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h, c

    def _unpack_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        h, c = state
        expected = (1, batch_size, self.hidden_size)
        if tuple(h.shape) != expected or tuple(c.shape) != expected:
            raise ValueError(
                f"Expected packed recurrent state {expected}, got "
                f"h={tuple(h.shape)} c={tuple(c.shape)}"
            )
        manager_state = (
            h[..., :self.manager_hidden_size].contiguous(),
            c[..., :self.manager_hidden_size].contiguous(),
        )
        option_h = h[..., self.manager_hidden_size:].reshape(
            1,
            batch_size * self.num_options,
            self.option_recurrent_size,
        )
        option_c = c[..., self.manager_hidden_size:].reshape_as(option_h)
        return manager_state, (option_h.contiguous(), option_c.contiguous())

    def _pack_state(
        self,
        manager_state: tuple[torch.Tensor, torch.Tensor],
        option_state: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        option_h = option_state[0].reshape(
            1,
            batch_size,
            self.num_options * self.option_recurrent_size,
        )
        option_c = option_state[1].reshape_as(option_h)
        return (
            torch.cat([manager_state[0], option_h], dim=-1),
            torch.cat([manager_state[1], option_c], dim=-1),
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
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Evaluate ``(batch, time, obs_dim)`` recurrent observations."""
        if obs_seq.ndim != 3 or obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observations (batch, time, {self.obs_dim}), "
                f"got {tuple(obs_seq.shape)}"
            )
        batch_size, sequence_length = obs_seq.shape[:2]
        if state is None:
            state = self.initial_state(batch_size, obs_seq.device)
        manager_state, option_state = self._unpack_state(state, batch_size)

        manager_obs = (
            obs_seq[..., 16:20]
            if self.obs_dim == 24
            else obs_seq
        )
        manager_encoded = self.manager_encoder(
            manager_obs.reshape(-1, self.manager_obs_dim)
        ).view(batch_size, sequence_length, -1)
        manager_features, next_manager_state = self.manager_lstm(
            manager_encoded,
            manager_state,
        )
        sensor_context = self.attention_encoder(
            obs_seq.reshape(-1, self.obs_dim)
        ).view(batch_size, sequence_length, self.manager_hidden_size)
        attentions = torch.sigmoid(
            self.attention_head(manager_features + sensor_context).view(
                batch_size,
                sequence_length,
                self.num_options,
                self.obs_dim,
            )
        )

        attended_obs = obs_seq.unsqueeze(-2) * attentions
        option_sequences = attended_obs.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_options,
            sequence_length,
            self.obs_dim,
        )
        option_encoded = self.option_sensor_encoder(
            option_sequences.reshape(-1, self.obs_dim)
        ).view(
            batch_size * self.num_options,
            sequence_length,
            self.option_hidden,
        )
        option_recurrent, next_option_state = self.option_lstm(
            option_encoded,
            option_state,
        )
        option_context = torch.cat(
            [option_encoded, option_recurrent],
            dim=-1,
        )
        option_features = self.option_output_encoder(
            option_context.reshape(
                -1,
                self.option_hidden + self.option_recurrent_size,
            )
        ).view(
            batch_size,
            self.num_options,
            sequence_length,
            self.option_hidden,
        ).permute(0, 2, 1, 3).contiguous()

        option_values = torch.cat([
            head(option_features[..., option_id, :])
            for option_id, head in enumerate(self.option_value_heads)
        ], dim=-1)
        if self.separate_selector:
            selector_logits = torch.cat([
                head(option_features[..., option_id, :])
                for option_id, head in enumerate(self.selector_heads)
            ], dim=-1)
        else:
            # AOC uses attended option values directly for epsilon-soft option
            # selection. Architecture-v2 checkpoints also shared this output,
            # but interpreted it through a softmax distribution.
            selector_logits = option_values
        action_means = torch.stack([
            head(option_features[..., option_id, :])
            for option_id, head in enumerate(self.action_heads)
        ], dim=-2)
        termination_logits = torch.cat([
            head(option_features[..., option_id, :])
            for option_id, head in enumerate(self.termination_heads)
        ], dim=-1)
        action_stds = self.option_log_stds().exp().view(
            1,
            1,
            self.num_options,
            self.act_dim,
        ).expand_as(action_means)
        next_state = self._pack_state(
            next_manager_state,
            next_option_state,
            batch_size,
        )
        return (
            selector_logits,
            option_values,
            termination_logits,
            action_means,
            action_stds,
            attentions,
            next_state,
        )

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
            outputs[5][:, 0],
            outputs[6],
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
    ) -> Normal | SquashedNormal:
        means = self._gather_options(action_means, options)
        stds = self._gather_options(action_stds, options)
        if self.squash_actions:
            return SquashedNormal(means, stds)
        return Normal(means, stds)

    def option_dist(
        self,
        option_scores: torch.Tensor,
        epsilon: float = 0.0,
    ) -> Categorical:
        """Return the call-and-return policy over options.

        Attention Option-Critic uses an epsilon-soft policy over learned
        ``Q_Omega`` values. Versions 2 and 3 used softmax logits; retaining
        that path keeps their completed checkpoints playable without using it
        for new training.
        """
        if not self.epsilon_greedy_selector:
            return Categorical(
                logits=option_scores / self.option_selector_temperature,
            )
        epsilon = float(epsilon)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("option epsilon must lie in [0, 1]")
        num_options = option_scores.shape[-1]
        probs = torch.full_like(option_scores, epsilon / num_options)
        greedy = option_scores.argmax(dim=-1, keepdim=True)
        probs.scatter_add_(
            -1,
            greedy,
            torch.full_like(greedy, 1.0 - epsilon, dtype=probs.dtype),
        )
        return Categorical(probs=probs)

    def option_state_value(
        self,
        option_scores: torch.Tensor,
        option_values: torch.Tensor,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Evaluate ``V_Omega`` under the epsilon-soft option policy.

        The termination-gradient theorem compares continuation with the value
        of reselecting through ``pi_Omega``. The policy probabilities come from
        the decentralized attended scores, while ``option_values`` may be the
        centralized counterfactual values used only during training.
        """
        if option_scores.shape != option_values.shape:
            raise ValueError(
                "option scores and values must have the same shape, got "
                f"{tuple(option_scores.shape)} and {tuple(option_values.shape)}"
            )
        return (
            self.option_dist(option_scores, epsilon=epsilon).probs
            * option_values
        ).sum(dim=-1)

    @staticmethod
    def selected_termination_logits(
        termination_logits: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        return termination_logits.gather(
            -1,
            options.long().unsqueeze(-1),
        ).squeeze(-1)
