# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized behaviour modules for the ACB (Actor-Critic with Behavior modules) variants.

Implements the 6 predefined behaviour modules used by daisy, lily, tulip, and cyclamen:
  0 — Exploration  (ballistic + obstacle avoidance)
  1 — Stop         (halt both wheels)
  2 — Phototaxis   (drive toward light)
  3 — Anti-phototaxis (drive away from light)
  4 — Attraction   (drive toward neighbors)
  5 — Repulsion    (drive away from neighbors)

All modules are reactive and memoryless.  Each converts aggregated sensor values
(angle + magnitude) into a 2D direction vector, then calls the shared helper
``compute_wheels_from_vector`` to produce differential-drive wheel velocities.

Reference: Epuck.cs ComputeWheelsVelocityFromVector + individual module implementations.
Module parameters are fixed across all ACB variants and all missions (Table 1 in paper):
  exploration τ = 5,  phototaxis k = 5,  anti-phototaxis k = 5,
  attraction α = 5,  repulsion α = 5.
"""

from __future__ import annotations

import math
import torch
from enum import IntEnum


class BehaviorID(IntEnum):
    """Identifiers for the 6 behaviour modules.  Matches the 6-way softmax output."""
    EXPLORATION = 0
    STOP = 1
    PHOTOTAXIS = 2
    ANTI_PHOTOTAXIS = 3
    ATTRACTION = 4
    REPULSION = 5


# ──────────────────────────────────────────────────────────────────────
#  Shared helper: 2D direction → (left_vel, right_vel)
# ──────────────────────────────────────────────────────────────────────

def compute_wheels_from_vector(
    dx: torch.Tensor,          # (E, N)  forward component in body frame
    dy: torch.Tensor,          # (E, N)  lateral component in body frame
    max_speed: float = 0.12,   # m/s  (paper: [-0.12, 0.12])
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D direction in the robot body frame to wheel velocities.

    When the target is in front (dx > 0), uses the standard rule:
      left  = dx − dy,  right = dx + dy
    When the target is behind (dx ≤ 0), the robot turns in place
    toward the target direction (since wheels never reverse).

    All outputs clamped to [0, max_speed].
    """
    # Bearing angle to target in body frame
    bearing = torch.atan2(dy, dx)  # (-π, π]

    # Target in front half: drive + steer
    left_fwd = (dx - dy).clamp(0, max_speed)
    right_fwd = (dx + dy).clamp(0, max_speed)

    # Target behind: pure in-place turn toward it
    # bearing > 0 → target is to the left → turn left
    #   (left slow, right fast)
    # bearing < 0 → target is to the right → turn right
    #   (left fast, right slow)
    turn_speed = max_speed * 0.6
    left_turn = torch.where(
        bearing >= 0,
        torch.zeros_like(dx),
        torch.full_like(dx, turn_speed),
    )
    right_turn = torch.where(
        bearing >= 0,
        torch.full_like(dx, turn_speed),
        torch.zeros_like(dx),
    )

    in_front = dx > 0
    left = torch.where(in_front, left_fwd, left_turn)
    right = torch.where(in_front, right_fwd, right_turn)
    return left, right
    return left, right


# ──────────────────────────────────────────────────────────────────────
#  Individual modules
# ──────────────────────────────────────────────────────────────────────

class BehaviorModules:
    """Vectorized execution of all 6 behaviour modules.

    All methods accept (E, N) batch dimensions and produce
    left_vel, right_vel: (E, N) tensors.
    """

    def __init__(
        self,
        max_speed: float = 0.12,   # paper: wheel velocity in [-0.12, 0.12] m/s
        obstacle_gain: float = 5.0,  # k for phototaxis / anti-phototaxis
        social_gain: float = 5.0,    # α for attraction / repulsion
        explore_tau: int = 5,        # τ for exploration (turn duration in ticks)
        device: str | torch.device = "cuda",
    ):
        self.max_speed = max_speed
        self.obstacle_gain = obstacle_gain
        self.social_gain = social_gain
        self.explore_tau = explore_tau
        self.device = torch.device(device)

        # Exploration internal state (random walk timer)
        self._explore_timer: torch.Tensor | None = None
        self._explore_turn_dir: torch.Tensor | None = None

    def init_exploration_state(self, E: int, N: int):
        """Initialize exploration state machine for E envs × N agents.

        States: 0 = RANDOM_WALK, 1 = OBSTACLE_AVOIDANCE
        Matches Unity AutoMoDeControlStepExploration.
        """
        dev = self.device
        self._explore_state = torch.zeros(
            E, N, device=dev, dtype=torch.long,
        )  # 0 = random walk, 1 = avoiding
        self._avoid_steps = torch.zeros(
            E, N, device=dev, dtype=torch.long,
        )
        self._avoid_dir = torch.zeros(
            E, N, device=dev,
        )  # +1 = turn right, -1 = turn left

    def reset_exploration_state(self, env_mask: torch.Tensor):
        """Reset exploration state for reset environments."""
        if self._explore_state is not None:
            self._explore_state[env_mask] = 0
            self._avoid_steps[env_mask] = 0
            self._avoid_dir[env_mask] = 0.0

    # ── Dispatch by module index ──────────────────────────────────

    def dispatch(
        self,
        module_ids: torch.Tensor,          # (E, N) int ∈ {0..5}
        prox_value: torch.Tensor,          # (E, N)  aggregated proximity magnitude
        prox_angle: torch.Tensor,          # (E, N)  aggregated proximity angle (body frame)
        light_value: torch.Tensor,         # (E, N)  aggregated light magnitude
        light_angle: torch.Tensor,         # (E, N)  aggregated light angle (body frame)
        rab_value: torch.Tensor,           # (E, N)  neighbor presence
        rab_angle: torch.Tensor,           # (E, N)  aggregated neighbor angle (body frame)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the selected behaviour module for each agent (vectorized dispatch).

        Returns:
            left_vel:  (E, N)
            right_vel: (E, N)
        """
        E, N = module_ids.shape
        device = module_ids.device

        left = torch.zeros(E, N, device=device)
        right = torch.zeros(E, N, device=device)

        # Compute wheel velocities for all modules, then scatter by mask
        for mod_id in range(6):
            mask = (module_ids == mod_id)
            if not mask.any():
                continue

            if mod_id == BehaviorID.EXPLORATION:
                lv, rv = self._exploration(prox_value, prox_angle, mask)
            elif mod_id == BehaviorID.STOP:
                lv = torch.zeros_like(left)
                rv = torch.zeros_like(right)
            elif mod_id == BehaviorID.PHOTOTAXIS:
                lv, rv = self._phototaxis(
                    light_value, light_angle, prox_value, prox_angle,
                )
            elif mod_id == BehaviorID.ANTI_PHOTOTAXIS:
                lv, rv = self._anti_phototaxis(
                    light_value, light_angle, prox_value, prox_angle,
                )
            elif mod_id == BehaviorID.ATTRACTION:
                lv, rv = self._attraction(
                    rab_value, rab_angle,
                    prox_value, prox_angle, mask,
                )
            elif mod_id == BehaviorID.REPULSION:
                lv, rv = self._repulsion(
                    rab_value, rab_angle,
                    prox_value, prox_angle, mask,
                )

            left = torch.where(mask, lv, left)
            right = torch.where(mask, rv, right)

        return left, right

    # ── Individual module implementations ─────────────────────────

    def _exploration(
        self,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exploration with ballistic obstacle avoidance.

        State machine (matching Unity AutoMoDe):
          RANDOM_WALK (state 0):
            - Drive forward at max speed.
            - If obstacle in front half (|angle| < π/2 and
              value > threshold): latch turn direction from
              sign(prox_angle), pick random turn duration
              [1..τ], switch to OBSTACLE_AVOIDANCE.
          OBSTACLE_AVOIDANCE (state 1):
            - Pure in-place rotation (locked direction).
            - Decrement step counter.
            - When counter reaches 0 → back to RANDOM_WALK.
        """
        E, N = prox_value.shape
        device = prox_value.device

        if self._explore_state is None:
            self.init_exploration_state(E, N)

        ms = self.max_speed
        state = self._explore_state    # (E, N)
        steps = self._avoid_steps      # (E, N)
        adir = self._avoid_dir         # (E, N)

        # --- RANDOM_WALK agents: check for obstacles ---
        walking = (state == 0) & active_mask
        front_obs = (
            (prox_value > 0.1)
            & (prox_angle.abs() < math.pi * 0.5)
        )
        trigger = walking & front_obs

        if trigger.any():
            # Latch turn direction: obstacle on right → turn
            # left (-1), obstacle on left → turn right (+1)
            new_dir = torch.where(
                prox_angle >= 0,
                -torch.ones(E, N, device=device),
                torch.ones(E, N, device=device),
            )
            adir = torch.where(trigger, new_dir, adir)
            # Random turn duration in [3, τ+3] — minimum 3
            # steps ensures enough rotation to clear wall
            min_turn = 3
            dur = torch.randint(
                min_turn, self.explore_tau + min_turn,
                (E, N),
                device=device, dtype=torch.long,
            )
            steps = torch.where(trigger, dur, steps)
            state = torch.where(
                trigger,
                torch.ones(E, N, device=device, dtype=torch.long),
                state,
            )

        # --- OBSTACLE_AVOIDANCE agents: rotate in place ---
        avoiding = (state == 1) & active_mask
        steps = torch.where(avoiding, steps - 1, steps)
        done_avoiding = avoiding & (steps <= 0)
        state = torch.where(
            done_avoiding,
            torch.zeros(E, N, device=device, dtype=torch.long),
            state,
        )

        # --- Compute wheel velocities ---
        # Random walk: forward at max speed
        lv_walk = torch.full((E, N), ms, device=device)
        rv_walk = torch.full((E, N), ms, device=device)

        # Avoidance: pure in-place turn
        # dir = +1 → turn right → left forward, right backward
        # dir = -1 → turn left  → left backward, right forward
        lv_avoid = adir * ms
        rv_avoid = -adir * ms

        is_avoiding = (state == 1) & active_mask
        lv = torch.where(is_avoiding, lv_avoid, lv_walk)
        rv = torch.where(is_avoiding, rv_avoid, rv_walk)

        # Save state back
        self._explore_state = state
        self._avoid_steps = steps
        self._avoid_dir = adir

        return lv, rv

    def _phototaxis(
        self,
        light_value: torch.Tensor,
        light_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move toward light, with obstacle avoidance (gain k)."""
        # Target direction: toward light (body frame)
        dx = torch.cos(light_angle) * light_value
        dy = torch.sin(light_angle) * light_value

        # Obstacle avoidance
        dx, dy = self._add_obstacle_avoidance(dx, dy, prox_value, prox_angle)

        return compute_wheels_from_vector(dx, dy, self.max_speed)

    def _anti_phototaxis(
        self,
        light_value: torch.Tensor,
        light_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move away from light, with obstacle avoidance (gain k)."""
        # Target direction: AWAY from light
        dx = -torch.cos(light_angle) * light_value
        dy = -torch.sin(light_angle) * light_value

        # Obstacle avoidance
        dx, dy = self._add_obstacle_avoidance(dx, dy, prox_value, prox_angle)

        return compute_wheels_from_vector(dx, dy, self.max_speed)

    def _attraction(
        self,
        rab_value: torch.Tensor,
        rab_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move toward neighbors (gain α), with obstacle avoidance.

        Falls back to exploration when no neighbor is detected.
        """
        has_neighbor = rab_value.abs() > 1e-4

        # Direction toward neighbors
        dx = torch.cos(rab_angle) * rab_value * self.social_gain
        dy = torch.sin(rab_angle) * rab_value * self.social_gain
        dx, dy = self._add_obstacle_avoidance(
            dx, dy, prox_value, prox_angle,
        )
        lv_a, rv_a = compute_wheels_from_vector(
            dx, dy, self.max_speed,
        )

        # Fallback: exploration
        lv_e, rv_e = self._exploration(
            prox_value, prox_angle, active_mask & ~has_neighbor,
        )

        lv = torch.where(has_neighbor, lv_a, lv_e)
        rv = torch.where(has_neighbor, rv_a, rv_e)
        return lv, rv

    def _repulsion(
        self,
        rab_value: torch.Tensor,
        rab_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move away from neighbors (gain α), with obstacle avoidance.

        Falls back to exploration when no neighbor is detected.
        """
        has_neighbor = rab_value.abs() > 1e-4

        # Direction AWAY from neighbors
        dx = -torch.cos(rab_angle) * rab_value * self.social_gain
        dy = -torch.sin(rab_angle) * rab_value * self.social_gain
        dx, dy = self._add_obstacle_avoidance(
            dx, dy, prox_value, prox_angle,
        )
        lv_r, rv_r = compute_wheels_from_vector(
            dx, dy, self.max_speed,
        )

        # Fallback: exploration
        lv_e, rv_e = self._exploration(
            prox_value, prox_angle, active_mask & ~has_neighbor,
        )

        lv = torch.where(has_neighbor, lv_r, lv_e)
        rv = torch.where(has_neighbor, rv_r, rv_e)
        return lv, rv

    def _add_obstacle_avoidance(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add obstacle-avoidance component (gain k) to direction vector.

        From Unity Epuck.cs: resultant = target + k * (-prox_direction)
        """
        k = self.obstacle_gain
        avoid_dx = -torch.cos(prox_angle) * prox_value * k
        avoid_dy = -torch.sin(prox_angle) * prox_value * k
        return dx + avoid_dx, dy + avoid_dy
