# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""E-puck robot model and sensor suite for Isaac Lab.

Provides vectorized implementations of the e-puck's sensor suite (proximity,
light, ground, range-and-bearing) and differential-drive kinematics, matching
the reference model RM 1.1 from the paper.  Also provides the 6 predefined
behaviour modules used by the ACB variants (daisy through cyclamen).
"""

from .epuck_sensors import EpuckSensors
from .behavior_modules import BehaviorModules, BehaviorID, compute_wheels_from_vector
