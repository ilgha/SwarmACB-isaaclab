# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Directional Gate (DGT) mission — gym registration."""

import gymnasium as gym

gym.register(
    id="SwarmACB-DirectionalGate-v0",
    entry_point=f"{__name__}.directional_gate_env:DirectionalGateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.directional_gate_env_cfg:DirectionalGateEnvCfg",
    },
)
