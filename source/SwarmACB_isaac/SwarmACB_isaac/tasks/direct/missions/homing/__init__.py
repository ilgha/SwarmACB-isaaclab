# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Homing mission gym registration."""

import gymnasium as gym

gym.register(
    id="SwarmACB-Homing-v0",
    entry_point=f"{__name__}.homing_env:HomingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.homing_env_cfg:HomingEnvCfg",
    },
)
