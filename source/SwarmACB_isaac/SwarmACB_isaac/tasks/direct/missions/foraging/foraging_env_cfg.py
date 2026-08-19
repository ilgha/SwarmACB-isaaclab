# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Foraging mission."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..directional_gate.directional_gate_env_cfg import DirectionalGateEnvCfg


@configclass
class ForagingEnvCfg(DirectionalGateEnvCfg):
    """Foraging mission in the shared 4.91 m^2 dodecagonal arena."""

    episode_length_s: float = 180.0
    has_light: bool = True
    light_position: tuple = (0.0, -1.5, 0.0)
    spawn_area_size: tuple = (1.8, 1.8)
    spawn_circle_radius: float = 0.0

    # Two black food sources and one white nest near the red light.
    food_radius: float = 0.15
    food_centers: tuple = ((-0.75, 0.0), (0.75, 0.0))
    # Unity foraging.prefab: centre=-8.95 and local z scale=0.63. With the
    # 10x e-puck scene scale, the northern edge is -0.895 + 0.63 / 2 = -0.58 m.
    nest_top_y: float = -0.58
