# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Mission environments for SwarmACB.

Each sub-package registers a Gymnasium environment for one mission:
  - directional_gate  (DGT)
  - xor_aggregation  (XOR)
  - homing           (HOM)
  - foraging         (FOR)
  - sheltering       (SHL/SCA)
"""

import gymnasium as gym  # noqa: F401
