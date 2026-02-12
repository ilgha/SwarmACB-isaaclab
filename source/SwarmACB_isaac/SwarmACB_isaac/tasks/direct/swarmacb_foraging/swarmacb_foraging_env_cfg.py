# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import field
from typing import ClassVar

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


##
# Scene Constants
##

# Semantic labels for raycasting
SEMANTIC_LABELS = {
    "empty": 0,
    "wall": 1,
    "food": 2,
    "robot": 3,
    "nest": 4,
}


##
# Environment Configuration
##

@configclass
class SwarmACBForagingEnvCfg(DirectMARLEnvCfg):
    """Configuration for SwarmACB Foraging Environment."""

    # ==================
    # Environment Setup
    # ==================
    decimation = 2
    episode_length_s = 60.0  # 60 seconds per episode
    
    # Multi-agent specification
    num_robots = 4  # Number of robots per environment
    possible_agents: list[str] = field(default_factory=lambda: [f"robot_{i}" for i in range(4)])
    action_spaces: dict[str, int] = field(default_factory=lambda: {f"robot_{i}": 2 for i in range(4)})
    # Actor obs: raycaster (36) + light_sensor (8) + in_nest (1) + carrying_food (1) + neighbor_count (1) = 47
    observation_spaces: dict[str, int] = field(default_factory=lambda: {f"robot_{i}": 47 for i in range(4)})
    # Critic state: per-robot (pos_xy 2 + vel_xy 2 + yaw 1 + carrying 1) * N + food_positions 2*num_food + nest_pos 2
    # = 6*N + 2*num_food + 2.  For N=4, food=8: 6*4 + 16 + 2 = 42
    state_space = 42

    # ==================
    # Simulation
    # ==================
    sim: SimulationCfg = SimulationCfg(dt=1 / 120.0, render_interval=decimation)

    # ==================
    # Scene
    # ==================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=12.0, replicate_physics=True)

    # ==================
    # Arena Configuration
    # ==================
    arena_diameter = 8.0  # Diameter of dodecagonal arena (meters)
    arena_wall_height = 0.5  # Height of perimeter walls (meters)
    arena_wall_thickness = 0.1  # Thickness of walls (meters)
    num_arena_sides = 12  # Dodecagon (12 sides)

    # ==================
    # Nest Configuration
    # ==================
    nest_radius = 0.8  # Radius of nest circular patch (meters)
    nest_distance_from_center = 2.0  # Distance from arena center to nest center (meters)
    nest_angle = 0.0  # Angle on arena perimeter where nest is placed (radians, 0 = +X side)
    nest_floor_height = 0.001  # Height of nest floor patch above ground (meters) - flush with ground

    # ==================
    # Food Configuration
    # ==================
    num_food_units = 8  # Number of cube/food units to spawn
    food_size = 0.1  # Side length of food cubes (meters)
    food_mass = 0.05  # Mass of food cubes (kg)
    
    # ==================
    # Sensor Configuration
    # ==================
    # Raycaster (proximity/discrimination sensor)
    raycaster_num_rays = 36  # Number of rays around robot (circle pattern)
    raycaster_max_distance = 2.0  # Max range of raycaster (meters)
    raycaster_height_offset = 0.1  # Sensor mounted at this height on robot

    # Neighbor counting
    neighbor_detection_range = 1.0  # Range to count nearby robots (meters)

    # Light sensor
    light_sensor_num_rays = 8  # Number of light detection directions around robot
    light_intensity_scale = 1.0  # Scaling for light values

    # ==================
    # Action Configuration
    # ==================
    # Continuous wheel velocity control
    wheel_max_velocity = 1.0  # Max linear velocity per wheel (m/s)
    
    # ==================
    # Reward Configuration
    # ==================
    reward_food_in_nest = 1.0  # Reward per food unit in nest
    reward_step_penalty = 0  # Penalty per step (encourage efficiency)
    reward_collision_penalty = 0  # Penalty for robot-robot collision
    
    # ==================
    # Robot Configuration
    # ==================
    robot_wheel_base = 0.2  # Distance between left and right wheels (meters)
    robot_radius = 0.15     # Approximate collision/detection radius of robot body (meters)
    robot_spawn_radius = 1.5  # Max distance from center for random robot spawn (meters)
    
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.5)),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )


##
# Observation and Reward Utilities
##

# Observation size calculation
def get_observation_size(cfg: SwarmACBForagingEnvCfg) -> int:
    """Calculate total ACTOR observation size per agent (local perception only).
    
    Breakdown:
    - Raycaster: num_rays (normalized distance, detects walls + food + robots)
    - Light sensors: num_light_rays
    - In nest: 1
    - Carrying food: 1
    - Neighbor count: 1
    """
    return cfg.raycaster_num_rays + cfg.light_sensor_num_rays + 1 + 1 + 1


def get_state_size(cfg: SwarmACBForagingEnvCfg) -> int:
    """Calculate total CRITIC state size (privileged global information).
    
    Breakdown per robot: pos_xy(2) + vel_xy(2) + yaw(1) + carrying(1) = 6
    Global: food_positions(2 * num_food) + nest_pos(2)
    Total: 6 * N + 2 * num_food + 2
    """
    per_robot = 6
    global_info = 2 * cfg.num_food_units + 2  # food positions + nest pos
    return per_robot * cfg.num_robots + global_info
