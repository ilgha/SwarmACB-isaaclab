# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized behaviour modules for the ACB (Actor-Critic with Behavior modules) variants.

Implements the 6 predefined behaviour modules used by daisy, lily, tulip, and cyclamen:
  0 — Exploration  (ballistic + obstacle avoidance state machine)
  1 — Stop         (halt both wheels)
  2 — Phototaxis   (drive toward light, obstacle avoidance state machine)
  3 — Anti-phototaxis (drive away from light, obstacle avoidance state machine)
  4 — Attraction   (drive toward neighbors, vector obstacle avoidance)
  5 — Repulsion    (drive away from neighbors, vector obstacle avoidance)

All modules are reactive.  Exploration, phototaxis, and anti-phototaxis maintain
per-robot state machines for obstacle avoidance turns.  Attraction and repulsion
use purely vectorial obstacle avoidance (no state machine).

``compute_wheels_from_vector`` is an exact replication of Unity Epuck.cs
``ComputeWheelsVelocityFromVector`` (angle-based, wheels in [-maxSpeed, +maxSpeed]).

Reference: Unity Epuck.cs AutoMoDe behaviour implementations.
Module gains match Unity exactly:
  phototaxis     obstacle gain = 0.5
  anti-phototaxis obstacle gain = 0.5
  attraction     obstacle gain = 0.6
  repulsion      obstacle gain = 0.5,  repulsion alpha = alphaParameter
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
    dx: torch.Tensor,          # (E, N)  x component in body frame (forward)
    dy: torch.Tensor,          # (E, N)  y component in body frame (lateral)
    max_speed: float = 0.12,   # m/s  (paper: [-0.12, 0.12])
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D direction in the robot body frame to wheel velocities.

    Exact replication of Unity Epuck.cs ``ComputeWheelsVelocityFromVector``:
      1. If (x, y) ≈ (0, 0) → wheels = (0, 0)
      2. angle = atan2(y, x) forced into [0, 2π)
      3. If angle ∈ [0, π) → right = 1, left = cos(angle)
         Else              → right = cos(angle), left = 1
      4. Scale so max(|left|, |right|) maps to maxSpeed

    Wheels can be in [-maxSpeed, +maxSpeed] (negative = reverse).
    """
    near_zero = (dx.abs() < 1e-5) & (dy.abs() < 1e-5)

    # Angle in [0, 2π)
    angle = torch.atan2(dy, dx)                              # (-π, π]
    angle = torch.where(angle < 0, angle + 2.0 * math.pi, angle)  # [0, 2π)

    cos_a = torch.cos(angle)

    # angle ∈ [0, π) → left hemisphere: right = 1, left = cos(angle)
    # angle ∈ [π, 2π) → right hemisphere: right = cos(angle), left = 1
    front = angle < math.pi
    left = torch.where(front, cos_a, torch.ones_like(cos_a))
    right = torch.where(front, torch.ones_like(cos_a), cos_a)

    # Scale so max(|left|, |right|) = max_speed
    max_val = torch.max(left.abs(), right.abs()).clamp(min=1e-5)
    scale = max_speed / max_val
    left = left * scale
    right = right * scale

    # Near-zero input → zero output
    left = torch.where(near_zero, torch.zeros_like(left), left)
    right = torch.where(near_zero, torch.zeros_like(right), right)

    return left, right


# ──────────────────────────────────────────────────────────────────────
#  Individual modules
# ──────────────────────────────────────────────────────────────────────

class BehaviorModules:
    """Vectorized execution of all 6 behaviour modules.

    Matches Unity Epuck.cs exactly:
      - Exploration: state-machine (RANDOM_WALK / OBSTACLE_AVOIDANCE)
      - Stop: wheels = 0
      - Phototaxis: state-machine obstacle avoidance + vector steering
      - Anti-phototaxis: state-machine obstacle avoidance + vector steering
      - Attraction: pure vector arithmetic (rab - 0.6*prox), forward fallback
      - Repulsion: pure vector arithmetic (-alpha*rab - 0.5*prox), forward fallback

    All methods accept (E, N) batch dimensions and produce
    left_vel, right_vel: (E, N) tensors.
    """

    def __init__(
        self,
        max_speed: float = 0.12,       # paper: wheel velocity in [-0.12, 0.12] m/s
        alpha_parameter: float = 5.0,  # alphaParameter for repulsion (and inside GetAttraction)
        prox_threshold: float = 0.1,   # m_fProximityThreshold from ARGoS
        device: str | torch.device = "cuda",
    ):
        self.max_speed = max_speed
        self.alpha_parameter = alpha_parameter
        self.prox_threshold = prox_threshold
        self.device = torch.device(device)

        # Random turn step range matching Unity: Vector2Int(1, 5)
        # Random.Range(1, 5) → 1, 2, 3, or 4 steps
        self._turn_range_lo = 1
        self._turn_range_hi = 5  # exclusive

        # State will be initialized on first use
        self._initialized = False

    def init_state(self, E: int, N: int):
        """Initialize all state machines for E envs × N agents.

        Matches Unity: exploration state, phototaxis BehaviorState,
        anti-phototaxis BehaviorState.
        """
        dev = self.device

        # Exploration state machine
        self._explore_state = torch.zeros(E, N, device=dev, dtype=torch.long)  # 0=walk, 1=avoid
        self._explore_steps = torch.zeros(E, N, device=dev, dtype=torch.long)
        self._explore_dir = torch.zeros(E, N, device=dev)  # +1=turn right, -1=turn left

        # Phototaxis BehaviorState (matching Unity m_sPhototaxisState)
        self._photo_avoiding = torch.zeros(E, N, device=dev, dtype=torch.bool)
        self._photo_steps = torch.zeros(E, N, device=dev, dtype=torch.long)
        self._photo_dir = torch.zeros(E, N, device=dev)  # +1 or -1

        # Anti-phototaxis BehaviorState (matching Unity m_sAntiPhotoState)
        self._antiphoto_avoiding = torch.zeros(E, N, device=dev, dtype=torch.bool)
        self._antiphoto_steps = torch.zeros(E, N, device=dev, dtype=torch.long)
        self._antiphoto_dir = torch.zeros(E, N, device=dev)

        self._initialized = True

    # Keep old API for backward compat — delegates to init_state
    def init_exploration_state(self, E: int, N: int):
        self.init_state(E, N)

    def reset_exploration_state(self, env_mask: torch.Tensor):
        """Reset all state machines for reset environments."""
        if not self._initialized:
            return
        self._explore_state[env_mask] = 0
        self._explore_steps[env_mask] = 0
        self._explore_dir[env_mask] = 0.0
        self._photo_avoiding[env_mask] = False
        self._photo_steps[env_mask] = 0
        self._photo_dir[env_mask] = 0.0
        self._antiphoto_avoiding[env_mask] = False
        self._antiphoto_steps[env_mask] = 0
        self._antiphoto_dir[env_mask] = 0.0

    # ── Dispatch by module index ──────────────────────────────────

    def dispatch(
        self,
        module_ids: torch.Tensor,          # (E, N) int ∈ {0..5}
        prox_value: torch.Tensor,          # (E, N)  aggregated proximity magnitude
        prox_angle: torch.Tensor,          # (E, N)  aggregated proximity angle (body frame)
        light_value: torch.Tensor,         # (E, N)  aggregated light magnitude
        light_angle: torch.Tensor,         # (E, N)  aggregated light angle (body frame)
        rab_vec_x: torch.Tensor,           # (E, N)  alpha-weighted RAB vector x (body frame)
        rab_vec_y: torch.Tensor,           # (E, N)  alpha-weighted RAB vector y (body frame)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute the selected behaviour module for each agent (vectorized dispatch).

        Returns:
            left_vel:  (E, N)
            right_vel: (E, N)
        """
        E, N = module_ids.shape
        device = module_ids.device

        if not self._initialized:
            self.init_state(E, N)

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
                    light_value, light_angle, prox_value, prox_angle, mask,
                )
            elif mod_id == BehaviorID.ANTI_PHOTOTAXIS:
                lv, rv = self._anti_phototaxis(
                    light_value, light_angle, prox_value, prox_angle, mask,
                )
            elif mod_id == BehaviorID.ATTRACTION:
                lv, rv = self._attraction(
                    rab_vec_x, rab_vec_y, prox_value, prox_angle,
                )
            elif mod_id == BehaviorID.REPULSION:
                lv, rv = self._repulsion(
                    rab_vec_x, rab_vec_y, prox_value, prox_angle,
                )

            left = torch.where(mask, lv, left)
            right = torch.where(mask, rv, right)

        return left, right

    # ── Individual module implementations ─────────────────────────

    def _is_obstacle_in_front(
        self,
        prox_value: torch.Tensor,   # (E, N)
        prox_angle: torch.Tensor,   # (E, N)
    ) -> torch.Tensor:
        """Matches Unity IsObstacleInFront: value >= threshold AND |angle| <= π/2."""
        return (prox_value >= self.prox_threshold) & (prox_angle.abs() <= math.pi * 0.5)

    def _get_turn_direction(self, prox_angle: torch.Tensor) -> torch.Tensor:
        """Latch turn direction from proximity angle.

        Unity: (ProximityAngle < 0) ? LEFT : RIGHT
          LEFT  → L = -maxSpeed, R = +maxSpeed  (dir = -1)
          RIGHT → L = +maxSpeed, R = -maxSpeed  (dir = +1)
        """
        return torch.where(
            prox_angle < 0,
            -torch.ones_like(prox_angle),   # LEFT
            torch.ones_like(prox_angle),    # RIGHT
        )

    def _exploration(
        self,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exploration with ballistic obstacle avoidance.

        State machine matching Unity AutoMoDeControlStepExploration exactly:
          RANDOM_WALK (state 0):
            - Drive forward at max speed.
            - If IsObstacleInFront → latch turn direction,
              pick random turn duration [1..5) (=1-4 steps), switch to AVOIDANCE.
          OBSTACLE_AVOIDANCE (state 1):
            - Pure in-place rotation: L=-maxSpeed, R=+maxSpeed (LEFT) or opposite.
            - Decrement step counter. When 0 → back to RANDOM_WALK.
        """
        E, N = prox_value.shape
        device = prox_value.device

        if not self._initialized:
            self.init_state(E, N)

        ms = self.max_speed
        state = self._explore_state
        steps = self._explore_steps
        adir = self._explore_dir

        # --- RANDOM_WALK agents: check for obstacles ---
        walking = (state == 0) & active_mask
        trigger = walking & self._is_obstacle_in_front(prox_value, prox_angle)

        if trigger.any():
            new_dir = self._get_turn_direction(prox_angle)
            adir = torch.where(trigger, new_dir, adir)
            dur = torch.randint(
                self._turn_range_lo, self._turn_range_hi,
                (E, N), device=device, dtype=torch.long,
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
        # Random walk: both wheels at max speed (forward)
        lv_walk = torch.full((E, N), ms, device=device)
        rv_walk = torch.full((E, N), ms, device=device)

        # Avoidance: pure in-place turn matching Unity exactly:
        # LEFT (dir=-1):  L = -maxSpeed, R = +maxSpeed
        # RIGHT (dir=+1): L = +maxSpeed, R = -maxSpeed
        lv_avoid = adir * ms
        rv_avoid = -adir * ms

        is_avoiding = (state == 1) & active_mask
        lv = torch.where(is_avoiding, lv_avoid, lv_walk)
        rv = torch.where(is_avoiding, rv_avoid, rv_walk)

        # Save state
        self._explore_state = state
        self._explore_steps = steps
        self._explore_dir = adir

        return lv, rv

    def _behavior_state_avoidance(
        self,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
        avoiding: torch.Tensor,      # (E, N) bool — current avoidance state
        turn_steps: torch.Tensor,    # (E, N) long — remaining turn steps
        turn_dir: torch.Tensor,      # (E, N) float — +1 or -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """State-machine obstacle avoidance used by phototaxis & anti-phototaxis.

        Matches Unity BehaviorState pattern (m_sPhototaxisState, m_sAntiPhotoState).

        Returns:
            avoiding:   updated bool mask
            turn_steps: updated step counter
            turn_dir:   updated direction
            is_turning: (E, N) bool — True if this agent is currently doing avoidance turn
        """
        E, N = prox_value.shape
        device = prox_value.device

        # --- Currently avoiding: decrement steps ---
        currently_avoiding = avoiding & active_mask
        turn_steps = torch.where(currently_avoiding, turn_steps - 1, turn_steps)
        done = currently_avoiding & (turn_steps <= 0)
        avoiding = torch.where(done, torch.zeros_like(avoiding), avoiding)

        # --- Not avoiding: check for new obstacle ---
        not_avoiding = ~avoiding & active_mask
        obstacle = self._is_obstacle_in_front(prox_value, prox_angle)
        trigger = not_avoiding & obstacle

        if trigger.any():
            new_dir = self._get_turn_direction(prox_angle)
            turn_dir = torch.where(trigger, new_dir, turn_dir)
            dur = torch.randint(
                self._turn_range_lo, self._turn_range_hi,
                (E, N), device=device, dtype=torch.long,
            )
            turn_steps = torch.where(trigger, dur, turn_steps)
            avoiding = torch.where(trigger, torch.ones_like(avoiding), avoiding)

        is_turning = avoiding & active_mask
        return avoiding, turn_steps, turn_dir, is_turning

    def _phototaxis(
        self,
        light_value: torch.Tensor,
        light_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move toward light, with state-machine obstacle avoidance.

        Matches Unity AutoMoDeControlStepPhototaxis exactly:
          - If in avoidance state: pure in-place turn
          - If obstacle detected: enter avoidance state
          - Otherwise: compute vector = light - 0.5*prox, forward fallback if mag < 0.1
        """
        E, N = prox_value.shape
        device = prox_value.device
        ms = self.max_speed

        # Update phototaxis avoidance state machine
        self._photo_avoiding, self._photo_steps, self._photo_dir, is_turning = \
            self._behavior_state_avoidance(
                prox_value, prox_angle, active_mask,
                self._photo_avoiding, self._photo_steps, self._photo_dir,
            )

        # --- Avoidance turn wheels ---
        lv_turn = self._photo_dir * ms       # LEFT: L=-ms, R=+ms; RIGHT: L=+ms, R=-ms
        rv_turn = -self._photo_dir * ms

        # --- Phototaxis vector computation (for non-avoiding agents) ---
        # Unity: lx = LightValue * cos(LightAngle), ly = LightValue * sin(LightAngle)
        #        px = ProximityValue * cos(ProximityAngle), py = ...
        #        rx = lx - 0.5*px, ry = ly - 0.5*py
        lx = light_value * torch.cos(light_angle)
        ly = light_value * torch.sin(light_angle)
        px = prox_value * torch.cos(prox_angle)
        py = prox_value * torch.sin(prox_angle)
        rx = lx - 0.5 * px
        ry = ly - 0.5 * py

        # Forward fallback if magnitude < 0.1
        mag = torch.sqrt(rx * rx + ry * ry)
        small = mag < 0.1
        rx = torch.where(small, torch.ones_like(rx), rx)
        ry = torch.where(small, torch.zeros_like(ry), ry)

        lv_steer, rv_steer = compute_wheels_from_vector(rx, ry, ms)

        # Select: turning agents use turn wheels, others use steering
        lv = torch.where(is_turning, lv_turn, lv_steer)
        rv = torch.where(is_turning, rv_turn, rv_steer)
        return lv, rv

    def _anti_phototaxis(
        self,
        light_value: torch.Tensor,
        light_angle: torch.Tensor,
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move away from light, with state-machine obstacle avoidance.

        Matches Unity AutoMoDeControlStepAntiPhototaxis exactly:
          - rx = -lx - 0.5*px, ry = -ly - 0.5*py
        """
        E, N = prox_value.shape
        device = prox_value.device
        ms = self.max_speed

        # Update anti-phototaxis avoidance state machine
        self._antiphoto_avoiding, self._antiphoto_steps, self._antiphoto_dir, is_turning = \
            self._behavior_state_avoidance(
                prox_value, prox_angle, active_mask,
                self._antiphoto_avoiding, self._antiphoto_steps, self._antiphoto_dir,
            )

        # Avoidance turn wheels
        lv_turn = self._antiphoto_dir * ms
        rv_turn = -self._antiphoto_dir * ms

        # Anti-phototaxis: target = AWAY from light
        # Unity: rx = -lx - 0.5*px, ry = -ly - 0.5*py
        lx = light_value * torch.cos(light_angle)
        ly = light_value * torch.sin(light_angle)
        px = prox_value * torch.cos(prox_angle)
        py = prox_value * torch.sin(prox_angle)
        rx = -lx - 0.5 * px
        ry = -ly - 0.5 * py

        # Forward fallback
        mag = torch.sqrt(rx * rx + ry * ry)
        small = mag < 0.1
        rx = torch.where(small, torch.ones_like(rx), rx)
        ry = torch.where(small, torch.zeros_like(ry), ry)

        lv_steer, rv_steer = compute_wheels_from_vector(rx, ry, ms)

        lv = torch.where(is_turning, lv_turn, lv_steer)
        rv = torch.where(is_turning, rv_turn, rv_steer)
        return lv, rv

    def _attraction(
        self,
        rab_vec_x: torch.Tensor,     # (E, N) alpha-weighted RAB vector x component
        rab_vec_y: torch.Tensor,     # (E, N) alpha-weighted RAB vector y component
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move toward neighbors, with vectorial obstacle avoidance.

        Matches Unity AutoMoDeControlStepAttraction exactly:
          - rab_vec already includes alpha/(1+dist) weighting from GetAttractionVectorToNeighbors
          - rx = rab_x - 0.6*prox_x, ry = rab_y - 0.6*prox_y
          - Forward fallback if magnitude < 0.1
          - NO exploration fallback, NO state-machine avoidance
        """
        px = prox_value * torch.cos(prox_angle)
        py = prox_value * torch.sin(prox_angle)

        rx = rab_vec_x - 0.6 * px
        ry = rab_vec_y - 0.6 * py

        # Forward fallback
        mag = torch.sqrt(rx * rx + ry * ry)
        small = mag < 0.1
        rx = torch.where(small, torch.ones_like(rx), rx)
        ry = torch.where(small, torch.zeros_like(ry), ry)

        return compute_wheels_from_vector(rx, ry, self.max_speed)

    def _repulsion(
        self,
        rab_vec_x: torch.Tensor,     # (E, N) alpha-weighted RAB vector x component
        rab_vec_y: torch.Tensor,     # (E, N) alpha-weighted RAB vector y component
        prox_value: torch.Tensor,
        prox_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move away from neighbors, with vectorial obstacle avoidance.

        Matches Unity AutoMoDeControlStepRepulsion exactly:
          - rab_vec already includes alpha/(1+dist) from GetAttractionVectorToNeighbors
          - rx = -alpha * rab_x - 0.5*prox_x, ry = -alpha * rab_y - 0.5*prox_y
          - Forward fallback if magnitude < 0.1
          - NO exploration fallback
        """
        px = prox_value * torch.cos(prox_angle)
        py = prox_value * torch.sin(prox_angle)

        rx = -self.alpha_parameter * rab_vec_x - 0.5 * px
        ry = -self.alpha_parameter * rab_vec_y - 0.5 * py

        # Forward fallback
        mag = torch.sqrt(rx * rx + ry * ry)
        small = mag < 0.1
        rx = torch.where(small, torch.ones_like(rx), rx)
        ry = torch.where(small, torch.zeros_like(ry), ry)

        return compute_wheels_from_vector(rx, ry, self.max_speed)
