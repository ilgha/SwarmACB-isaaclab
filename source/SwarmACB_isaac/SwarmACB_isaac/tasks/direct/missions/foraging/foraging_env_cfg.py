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
    light_position: tuple = (0.0, -1.4, 0.0)
    spawn_area_size: tuple = (1.8, 1.8)

    # Two black food sources and one white nest near the red light.
    food_radius: float = 0.15
    food_centers: tuple = ((-0.75, 0.0), (0.75, 0.0))
    # The nest is the southern white band, whose northern edge is 0.63 m
    # below the arena centre in the mission diagram.
    nest_top_y: float = -0.63
