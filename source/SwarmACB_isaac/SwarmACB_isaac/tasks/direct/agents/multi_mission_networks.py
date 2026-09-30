"""Mission-independent reactive skills and mission-specific OC controllers.

The actor's legacy module names are retained so exported mission actors can be
played by the OC2 viewers. Binding shares parameters, not recurrent state.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from .learned_option_critic_networks import LearnedOptionActor


LIBRARY_MODULES = (
    "attention_encoder", "attention_head", "option_sensor_encoder", "action_heads",
)
LIBRARY_NAMES = (*LIBRARY_MODULES, "log_std")
OBSERVATION_CONTRACT = {
    "channels": 24,
    "order": "8_proximity,8_light,3_ground,ztilde,4_rab",
    "preprocessing": "environment_full_policy_observations_v1",
    "mission_conditioned": False,
    "action_transform": "clip_minus3_3_divide3",
}


def make_actor(cfg):
    return LearnedOptionActor(
        obs_dim=24, act_dim=2, num_options=cfg.num_options,
        hidden=cfg.hidden_dim, num_layers=cfg.num_layers,
        memory_size=cfg.memory_size, option_hidden=cfg.option_hidden_dim,
        option_num_layers=cfg.option_num_layers,
        option_memory_size=cfg.option_memory_size,
        initial_termination_probability=cfg.initial_termination_probability,
        initial_log_std=cfg.initial_log_std,
        reactive_intra_options=True, linear_intra_options=False,
        separate_selector=False, epsilon_greedy_selector=True, squash_actions=False,
    )


def is_library_key(key):
    return key.split(".", 1)[0] in LIBRARY_NAMES


class SharedOptionLibrary(nn.Module):
    """All trainable parameters on the reactive sensor-to-wheel path."""

    def __init__(self, actor: LearnedOptionActor):
        super().__init__()
        if not actor.reactive_intra_options or actor.linear_intra_options:
            raise ValueError("OC3 requires OC2-mini reactive motor policies")
        for name in LIBRARY_MODULES:
            setattr(self, name, copy.deepcopy(getattr(actor, name)))
        self.log_std = nn.Parameter(actor.log_std.detach().clone())

    def bind(self, actor: LearnedOptionActor):
        for name in LIBRARY_NAMES:
            setattr(actor, name, getattr(self, name))

    def freeze(self):
        self.requires_grad_(False)
        return self

    def gradient_payload(self):
        return {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in self.named_parameters()
        }


class MissionOptionController:
    """Partition an existing mini actor without changing its forward pass."""

    def __init__(self, actor, library):
        self.actor = actor
        self.library = library
        library.bind(actor)

    def private_parameters(self):
        return [p for name, p in self.actor.named_parameters() if not is_library_key(name)]

    def private_state_dict(self):
        return {k: v for k, v in self.actor.state_dict().items() if not is_library_key(k)}

    def load_private_state_dict(self, state):
        expected = self.private_state_dict()
        if set(state) != set(expected):
            raise ValueError("Mission-controller checkpoint keys do not match")
        merged = self.actor.state_dict()
        merged.update(state)
        self.actor.load_state_dict(merged, strict=True)


def average_gradients(library, payloads):
    """Equal mission weights, including zero contribution for unused skills."""
    if not payloads:
        raise ValueError("At least one mission gradient is required")
    names = set(dict(library.named_parameters()))
    if any(set(payload) != names for payload in payloads):
        raise ValueError("Shared gradient keys do not match the library")
    for name, parameter in library.named_parameters():
        gradients = [payload[name] for payload in payloads if payload[name] is not None]
        if not gradients:
            parameter.grad = None
            continue
        if any(g.shape != parameter.shape or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError(f"Invalid shared gradient for {name}")
        parameter.grad = sum(g.to(parameter.device) for g in gradients) / len(payloads)
