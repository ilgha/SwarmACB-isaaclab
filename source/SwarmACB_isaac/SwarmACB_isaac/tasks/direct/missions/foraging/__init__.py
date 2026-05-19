# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Foraging mission gym registration."""

import gymnasium as gym

gym.register(
    id="SwarmACB-Foraging-v0",
    entry_point=f"{__name__}.foraging_env:ForagingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.foraging_env_cfg:ForagingEnvCfg",
    },
)
