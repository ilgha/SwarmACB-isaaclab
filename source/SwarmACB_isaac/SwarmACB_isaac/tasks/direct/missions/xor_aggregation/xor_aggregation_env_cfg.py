# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the XOR-Aggregation mission."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..directional_gate.directional_gate_env_cfg import DirectionalGateEnvCfg


@configclass
class XorAggregationEnvCfg(DirectionalGateEnvCfg):
    """XOR-Aggregation mission in the shared 4.91 m^2 dodecagonal arena."""

    episode_length_s: float = 180.0
    has_light: bool = False
    spawn_area_size: tuple = (2.4, 2.4)
    spawn_circle_radius: float = 1.2

    # Two identical black circular aggregation areas from the mission diagram:
    # diameter 0.60 m, centers 0.50 m left/right of the arena center.
    target_radius: float = 0.30
    target_centers: tuple = ((-0.50, 0.0), (0.50, 0.0))
