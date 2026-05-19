# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Sheltering-with-constrained-access mission gym registration."""

import gymnasium as gym

for _id in ("SwarmACB-Sheltering-v0", "SwarmACB-SCA-v0", "SwarmACB-SHL-v0"):
    gym.register(
        id=_id,
        entry_point=f"{__name__}.sheltering_env:ShelteringEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.sheltering_env_cfg:ShelteringEnvCfg",
        },
    )
