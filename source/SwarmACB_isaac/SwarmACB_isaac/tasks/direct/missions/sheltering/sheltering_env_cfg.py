# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Sheltering with constrained access."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..directional_gate.directional_gate_env_cfg import DirectionalGateEnvCfg


@configclass
class ShelteringEnvCfg(DirectionalGateEnvCfg):
    """Sheltering with constrained access in the shared arena."""

    episode_length_s: float = 180.0
    has_light: bool = True
    light_position: tuple = (0.0, -1.5, 0.0)
    spawn_area_size: tuple = (1.8, 1.8)
    spawn_circle_radius: float = 0.0

    # Three-walled 0.50 m x 0.30 m shelter centered in the arena.
    # The south side is open and faces the red light.
    shelter_center: tuple = (0.0, 0.0)
    shelter_size: tuple = (0.50, 0.30)
    shelter_wall_thickness: float = 0.03

    # Two black circular areas flanking the shelter.
    black_area_radius: float = 0.30
    black_area_centers: tuple = ((-0.80, 0.0), (0.80, 0.0))
