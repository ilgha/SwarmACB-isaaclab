# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SwarmACB Foraging Task Registration."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Template-Swarmacb-Foraging-Direct-v0",
    entry_point=f"{__name__}.swarmacb_foraging_env:SwarmACBForagingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.swarmacb_foraging_env_cfg:SwarmACBForagingEnvCfg",
    },
)
