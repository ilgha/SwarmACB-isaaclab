# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Directional Gate (DGT) mission environment.

All constants match the paper: dodecagonal arena, 4.91 m², 20 e-puck robots,
120 s episodes at 10 Hz, reward = K⁺ − K⁻ (correct − incorrect gate crossings).

The ``variant`` field selects among the five CASA methods:
  dandelion  → 24-dim obs, 2-dim continuous (wheel velocities)
  daisy      → 24-dim obs, 6-way discrete (behaviour module selector)
  lily       → 4-dim obs, 6-way discrete
  tulip      → 4-dim obs, 6-way discrete  (smaller network — handled by trainer)
  cyclamen   → 4-dim obs, 6-way discrete  (LSTM — handled by trainer)
"""

from __future__ import annotations

import math

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# ── Arena geometry ────────────────────────────────────────────────────
#  Regular dodecagon (12 sides) of area 4.91 m²
#  Area = (1/2) * n * R² * sin(2π/n)  →  R = √(2A / (n sin(2π/n)))
#  n=12: Area = 3R²  →  R = √(4.91/3) ≈ 1.279 m

_ARENA_N_SIDES = 12
_ARENA_AREA = 4.91                                       # m²
_ARENA_CIRCUMRADIUS = math.sqrt(
    2 * _ARENA_AREA / (_ARENA_N_SIDES * math.sin(2 * math.pi / _ARENA_N_SIDES))
)  # ≈ 1.279 m

# ── Number of agents ──────────────────────────────────────────────────
_NUM_AGENTS = 20

# ── Variant-dependent spaces ─────────────────────────────────────────
#  These are convenience look-ups used by the @configclass below.
_OBS_DIM = {
    "dandelion": 24,  # 8 prox + 8 light + 3 ground + 1 ztilde + 4 RAB
    "daisy":     24,
    "lily":       4,  # 3 ground + 1 ztilde
    "tulip":      4,
    "cyclamen":   4,
}
_ACT_DIM = {
    "dandelion": 2,   # continuous (left_vel, right_vel)
    "daisy":     1,   # discrete module index 0..5  (stored as int, space=1)
    "lily":      1,
    "tulip":     1,
    "cyclamen":  1,
}
_NUM_BEHAVIOR_MODULES = 6  # exploration, stop, phototaxis, anti-phototaxis, attraction, repulsion


def _agent_names(n: int = _NUM_AGENTS) -> list[str]:
    return [f"epuck_{i}" for i in range(n)]


def _obs_spaces(variant: str, n: int = _NUM_AGENTS) -> dict[str, int]:
    d = _OBS_DIM[variant]
    return {f"epuck_{i}": d for i in range(n)}


def _act_spaces(variant: str, n: int = _NUM_AGENTS) -> dict[str, int]:
    d = _ACT_DIM[variant]
    return {f"epuck_{i}": d for i in range(n)}


# ──────────────────────────────────────────────────────────────────────

@configclass
class DirectionalGateEnvCfg(DirectMARLEnvCfg):
    """Environment config for the Directional Gate (DGT) mission."""

    # ── CASA variant ──────────────────────────────────────────────
    variant: str = "dandelion"  # "dandelion" | "daisy" | "lily" | "tulip" | "cyclamen"

    # ── Multi-agent spaces (initialised for dandelion; call update_variant()
    #    after changing ``variant`` to refresh these) ──────────────
    num_agents: int = _NUM_AGENTS
    possible_agents: list = _agent_names()
    observation_spaces: dict = _obs_spaces("dandelion")
    action_spaces: dict = _act_spaces("dandelion")
    state_space: int = -1  # critic state is computed separately by the trainer

    # Whether actions are discrete (for POCA trainer)
    discrete_actions: bool = False
    num_actions: int = _NUM_BEHAVIOR_MODULES  # only used when discrete_actions=True

    # ── Simulation ────────────────────────────────────────────────
    decimation: int = 1            # physics steps per control step
    episode_length_s: float = 120.0  # 120 s (T=120s in paper)
    sim: SimulationCfg = SimulationCfg(
        dt=0.1,              # 10 Hz control frequency (paper: f=10 Hz)
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
    )

    # ── Scene ─────────────────────────────────────────────────────
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=5,           # paper: 5 parallel envs during training
        env_spacing=4.0,      # enough for arena diameter ~2.56 m
        replicate_physics=True,
    )

    # ── Arena ─────────────────────────────────────────────────────
    arena_num_sides: int = _ARENA_N_SIDES
    arena_area: float = _ARENA_AREA
    arena_circumradius: float = _ARENA_CIRCUMRADIUS   # ≈ 1.279 m
    arena_wall_height: float = 0.08                    # visual wall height

    # ── E-puck robot ──────────────────────────────────────────────
    robot_radius: float = 0.035     # m (e-puck body radius)
    robot_height: float = 0.05      # m (visual cylinder height)
    robot_mass: float = 0.190       # kg (~190 g with battery)
    max_wheel_speed: float = 0.12   # m/s  (paper: [-0.12, 0.12])
    wheelbase: float = 0.053        # m  (e-puck inter-wheel distance)

    # ── Sensor parameters ─────────────────────────────────────────
    prox_range: float = 0.10        # IR proximity sensor range (m)
    rab_range: float = 0.20         # range-and-bearing detection radius (m)
    light_threshold: float = 0.2    # min reading to register light

    # ── Ground zones (Directional Gate) ───────────────────────────
    #  Arena coordinates: origin at arena center, +Y = north.
    #  Positions are computed from the north arena wall inward.
    #  North inradius ≈ R*cos(π/12) ≈ 1.236 m.
    #
    #  Black corridor: 0.50 m wide × 1.06 m long, touching N wall
    #    x ∈ [−0.25, 0.25], y ∈ [north_inradius−1.06, north_inradius]
    #  White gate: 0.45 m wide × 0.33 m long, directly south
    #    x ∈ [−0.225, 0.225], y ∈ [corr_south−0.33, corr_south]
    #  Two side walls: 0.50 m long vertical barriers at x = ± 0.25
    #    from gate_south to gate_south + 0.50
    #  Everything else: grey (default arena floor)
    corridor_width: float = 0.50
    corridor_length: float = 1.06
    gate_width: float = 0.45
    gate_length: float = 0.33
    side_wall_length: float = 0.50

    # ── Light source ──────────────────────────────────────────────
    #  At south edge of arena, just outside the wall.
    light_position: tuple = (0.0, -1.4, 0.0)

    # ── Reward ────────────────────────────────────────────────────
    #  r(t) = K⁺(t) − K⁻(t)  (paper)
    #  No reward shaping.

    # ── Behaviour module parameters (ACB variants) ────────────────
    explore_tau: int = 5            # exploration turn duration (ticks)
    obstacle_gain: float = 5.0     # k for phototaxis / anti-phototaxis
    social_gain: float = 5.0       # α for attraction / repulsion

    # ──────────────────────────────────────────────────────────────

    def update_variant(self, variant: str):
        """Reconfigure spaces after changing the CASA variant.

        Call this after setting ``self.variant = "lily"`` (etc.) to
        keep observation_spaces and action_spaces in sync.
        """
        self.variant = variant
        self.observation_spaces = _obs_spaces(variant, self.num_agents)
        self.action_spaces = _act_spaces(variant, self.num_agents)
        self.discrete_actions = (variant != "dandelion")
