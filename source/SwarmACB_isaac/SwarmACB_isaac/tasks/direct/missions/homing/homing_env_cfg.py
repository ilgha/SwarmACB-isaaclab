# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Homing mission."""

from __future__ import annotations

from isaaclab.utils import configclass

from ..directional_gate.directional_gate_env_cfg import DirectionalGateEnvCfg


@configclass
class HomingEnvCfg(DirectionalGateEnvCfg):
    """Homing mission in the shared 4.91 m^2 dodecagonal arena."""

    episode_length_s: float = 120.0
    has_light: bool = False

    # Single black goal area in the southern half.
    goal_radius: float = 0.30
    goal_center: tuple = (0.0, -0.70)
