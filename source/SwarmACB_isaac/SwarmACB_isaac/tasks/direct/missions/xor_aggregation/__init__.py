# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""XOR-Aggregation mission gym registration."""

import gymnasium as gym

gym.register(
    id="SwarmACB-XOR-v0",
    entry_point=f"{__name__}.xor_aggregation_env:XorAggregationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xor_aggregation_env_cfg:XorAggregationEnvCfg",
    },
)
