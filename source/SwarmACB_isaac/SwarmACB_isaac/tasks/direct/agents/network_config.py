# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free network configuration shared by training and audits."""

from __future__ import annotations

from typing import Any, Mapping


PAPER_PARITY_VERSION = 4


def apply_network_settings(
    cfg: Any,
    network: Mapping[str, Any],
    critic: Mapping[str, Any],
    variant: str,
    block: Mapping[str, Any],
) -> Any:
    """Apply ML-Agents network settings to actor and centralized critic.

    Unity constructs both networks from the same ``NetworkSettings`` object.
    ``critic_settings`` is an Isaac-only escape hatch; omitted fields must
    therefore inherit the corresponding actor setting.
    """
    cfg.hidden_dim = network.get("hidden_units", cfg.hidden_dim)
    cfg.num_layers = network.get("num_layers", cfg.num_layers)
    cfg.critic_hidden_dim = critic.get("hidden_units", cfg.hidden_dim)
    cfg.critic_num_layers = critic.get("num_layers", cfg.num_layers)
    cfg.critic_num_heads = critic.get("num_heads", cfg.critic_num_heads)

    if hasattr(cfg, "num_options"):
        cfg.num_options = network.get(
            "num_options",
            block.get("num_options", cfg.num_options),
        )
    if hasattr(cfg, "option_hidden_dim"):
        cfg.option_hidden_dim = network.get(
            "option_hidden_units",
            cfg.option_hidden_dim,
        )
        cfg.option_num_layers = network.get(
            "option_num_layers",
            cfg.option_num_layers,
        )

    memory = network.get("memory", {})
    cfg.recurrent = bool(memory) or variant == "cyclamen"
    if cfg.recurrent:
        cfg.memory_size = memory.get("memory_size", cfg.memory_size)
        cfg.sequence_length = memory.get("sequence_length", cfg.sequence_length)
        if hasattr(cfg, "option_memory_size"):
            cfg.option_memory_size = memory.get(
                "option_memory_size",
                cfg.option_memory_size,
            )

    return cfg
