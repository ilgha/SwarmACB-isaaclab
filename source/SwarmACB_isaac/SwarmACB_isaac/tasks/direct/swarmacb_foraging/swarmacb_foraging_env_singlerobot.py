# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swarm Collaborative Foraging Environment."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .swarmacb_foraging_env_cfg import SwarmACBForagingEnvCfg, SEMANTIC_LABELS


class SwarmACBForagingEnv(DirectMARLEnv):
    """Multi-agent foraging environment with semantic raycasting and light navigation."""

    cfg: SwarmACBForagingEnvCfg

    def __init__(self, cfg: SwarmACBForagingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_robots = self.cfg.num_robots
        
        # Get robot rigid object from scene
        self.robot: RigidObject = self.scene.rigid_objects["robot"]
        
        # Get food rigid objects from scene
        self.food_objects: list[RigidObject] = [
            self.scene.rigid_objects[f"food_{i}"] for i in range(self.cfg.num_food_units)
        ]
        
        # Ray count from config (analytical raycaster, no warp sensor)
        self.num_raycaster_rays = self.cfg.raycaster_num_rays
        self.obs_size = self.num_raycaster_rays + self.cfg.light_sensor_num_rays + 1 + 1  # rays + light + nest + neighbor
        print(f"[SwarmACB] Raycaster rays (analytical): {self.num_raycaster_rays}, Total obs size: {self.obs_size}")

        # Pre-compute ray angle offsets (local frame, evenly spaced 360 degrees)
        self.ray_angle_offsets = torch.linspace(
            0, 2 * math.pi * (self.num_raycaster_rays - 1) / self.num_raycaster_rays,
            self.num_raycaster_rays, device=self.device
        )

        # Debug storage for visualization (filled each step by _compute_ray_distances)
        self.debug_ray_hits_w = torch.zeros(
            (self.scene.num_envs, self.num_raycaster_rays, 3), device=self.device
        )
        self.debug_ray_origins_w = torch.zeros(
            (self.scene.num_envs, 3), device=self.device
        )
        
        # Track food collection for rewards
        # Shape: (num_envs, num_food_units) - True if food has been collected
        self.food_collected = torch.zeros(
            (self.scene.num_envs, self.cfg.num_food_units), device=self.device, dtype=torch.bool
        )

    def _setup_scene(self):
        """Setup the simulation scene with arena, food, robots, and sensors."""
        
        # ==================
        # Ground Plane
        # ==================
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ==================
        # Instantiate Robot RigidObject (spawned from config)
        # ==================
        self._robot = RigidObject(self.cfg.robot)

        # ==================
        # Spawn Arena (Dodecagonal Walls)
        # ==================
        self._spawn_arena()

        # ==================
        # Spawn Nest Area
        # ==================
        self._spawn_nest()

        # ==================
        # Spawn Food Units
        # ==================
        self._spawn_food()

        # ==================
        # Spawn Light Beacon
        # ==================
        self._spawn_light_beacon()

        # ==================
        # Clone Environments
        # ==================
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # ==================
        # Register Assets in Scene (AFTER cloning)
        # ==================
        self.scene.rigid_objects["robot"] = self._robot
        
        # Register food items as RigidObjects (prims already exist from _spawn_food)
        self._food_rigid_objects = []
        for food_idx in range(self.cfg.num_food_units):
            food_rigid = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Food_{food_idx}",
            ))
            self.scene.rigid_objects[f"food_{food_idx}"] = food_rigid
            self._food_rigid_objects.append(food_rigid)

        # ==================
        # Pre-compute Wall Geometry for Analytical Raycasting
        # ==================
        self._precompute_wall_segments()
        
        # ==================
        # Add Lighting
        # ==================
        # Dome light for ambient illumination
        dome_light_cfg = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.9, 0.9, 1.0)  # Slightly cool white
        )
        dome_light_cfg.func("/World/DomeLight", dome_light_cfg)
        
        # Directional light for shadows and depth
        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 0.95, 0.8),  # Warm sunlight
            angle=0.5,  # Slight softness
        )
        distant_light_cfg.func("/World/SunLight", distant_light_cfg)

    def _spawn_carter_robots(self):
        """Placeholder - Robot spawning handled by Articulation from config."""
        pass

    def _spawn_arena(self):
        """Spawn dodecagonal walled arena."""
        arena_radius = self.cfg.arena_diameter / 2
        num_sides = self.cfg.num_arena_sides
        wall_length = 2 * arena_radius * math.sin(math.pi / num_sides)

        for side in range(num_sides):
            angle = 2 * math.pi * side / num_sides + math.pi / num_sides
            wall_center_x = arena_radius * math.cos(angle)
            wall_center_y = arena_radius * math.sin(angle)
            wall_yaw = angle + math.pi / 2

            wall_prim_path = f"/World/envs/env_.*/Arena_Wall_{side}"
            
            # Wall as a thin box
            wall_cfg = sim_utils.CuboidCfg(
                size=(wall_length, self.cfg.arena_wall_thickness, self.cfg.arena_wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            )
            # Convert yaw to quaternion for rotation
            quat = [
                math.cos(wall_yaw / 2), 0, 0, math.sin(wall_yaw / 2)
            ]  # Quaternion [w, x, y, z]
            wall_cfg.func(
                prim_path=wall_prim_path,
                cfg=wall_cfg,
                translation=(wall_center_x, wall_center_y, self.cfg.arena_wall_height / 2),
                orientation=quat,
            )

    def _spawn_nest(self):
        """Spawn the nest area (flat circular floor marker, no collision)."""
        nest_x = self.cfg.nest_distance_from_center * math.cos(self.cfg.nest_angle)
        nest_y = self.cfg.nest_distance_from_center * math.sin(self.cfg.nest_angle)

        nest_prim_path = "/World/envs/env_.*/Nest"
        # Very thin disc flush with ground — purely visual, no collision
        nest_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.nest_radius,
            height=0.002,  # 2mm thick disc
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            # No collision — blocks slide freely over the nest marker
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5)),
        )
        nest_cfg.func(
            prim_path=nest_prim_path,
            cfg=nest_cfg,
            translation=(nest_x, nest_y, 0.001),  # Flush with ground
        )

        # Store nest LOCAL position for reward/observation computation
        self.nest_pos_local = torch.tensor(
            [nest_x, nest_y], device=self.device, dtype=torch.float32
        )

    def _spawn_food(self):
        """Spawn food cubes randomly in arena (outside nest)."""
        arena_radius = self.cfg.arena_diameter / 2

        for food_idx in range(self.cfg.num_food_units):
            # Rejection sampling: spawn inside arena but outside nest
            while True:
                # Random position in arena
                angle = torch.rand(1, device=self.device) * 2 * math.pi
                radius = torch.rand(1, device=self.device) * (arena_radius - self.cfg.food_size)
                food_x = float(radius * torch.cos(angle))
                food_y = float(radius * torch.sin(angle))

                # Check if outside nest
                dist_to_nest = math.sqrt(
                    (food_x - self.nest_pos_local[0]) ** 2 + (food_y - self.nest_pos_local[1]) ** 2
                )
                if dist_to_nest > self.cfg.nest_radius + self.cfg.food_size:
                    break

            food_prim_path = f"/World/envs/env_.*/Food_{food_idx}"
            food_cfg = sim_utils.CuboidCfg(
                size=(self.cfg.food_size, self.cfg.food_size, self.cfg.food_size),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.food_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            )
            food_cfg.func(
                prim_path=food_prim_path,
                cfg=food_cfg,
                translation=(food_x, food_y, self.cfg.food_size / 2),
            )

    def _spawn_light_beacon(self):
        """Spawn light beacon outside arena near nest."""
        beacon_distance = self.cfg.arena_diameter / 2 + 1.0
        beacon_x = beacon_distance * math.cos(self.cfg.nest_angle)
        beacon_y = beacon_distance * math.sin(self.cfg.nest_angle)
        beacon_z = 1.0

        # Create a light sphere (visual cue only, no physics)
        beacon_prim_path = "/World/Light_Beacon"
        beacon_cfg = sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        )
        beacon_cfg.func(
            prim_path=beacon_prim_path,
            cfg=beacon_cfg,
            translation=(beacon_x, beacon_y, beacon_z),
        )

        # Store beacon position
        self.light_beacon_pos = torch.tensor(
            [beacon_x, beacon_y, beacon_z], device=self.device, dtype=torch.float32
        )

    def _precompute_wall_segments(self):
        """Pre-compute dodecagonal arena wall segment endpoints for analytical raycasting.

        Stores wall_segments tensor of shape (num_walls, 2, 2) where:
          wall_segments[i, 0] = endpoint A (x, y)
          wall_segments[i, 1] = endpoint B (x, y)
        """
        arena_radius = self.cfg.arena_diameter / 2
        num_sides = self.cfg.num_arena_sides
        wall_length = 2 * arena_radius * math.sin(math.pi / num_sides)

        wall_segments = torch.zeros((num_sides, 2, 2), device=self.device)
        for side in range(num_sides):
            angle = 2 * math.pi * side / num_sides + math.pi / num_sides
            cx = arena_radius * math.cos(angle)
            cy = arena_radius * math.sin(angle)
            wall_yaw = angle + math.pi / 2
            half_len = wall_length / 2

            wall_segments[side, 0, 0] = cx + half_len * math.cos(wall_yaw)  # Ax
            wall_segments[side, 0, 1] = cy + half_len * math.sin(wall_yaw)  # Ay
            wall_segments[side, 1, 0] = cx - half_len * math.cos(wall_yaw)  # Bx
            wall_segments[side, 1, 1] = cy - half_len * math.sin(wall_yaw)  # By

        self.wall_segments = wall_segments  # (num_walls, 2, 2)

    def _compute_ray_distances(self):
        """Compute analytical 2D ray-segment distances for walls and ray-circle for food.

        For each ray from the robot, computes the nearest intersection distance
        with dodecagonal arena walls and food cubes. Results are clamped to
        [0, raycaster_max_distance] and stored in self.debug_ray_hits_w for
        visualization.

        Returns:
            Normalized distances tensor of shape (num_envs, num_rays) in [0, 1].
        """
        num_envs = self.scene.num_envs
        num_rays = self.num_raycaster_rays
        max_dist = self.cfg.raycaster_max_distance

        # --- Robot pose in env-local frame ---
        robot_pos_w = self.robot.data.root_pos_w  # (num_envs, 3)
        env_origins = self.scene.env_origins  # (num_envs, 3)
        ox = (robot_pos_w[:, 0] - env_origins[:, 0]).unsqueeze(1)  # (num_envs, 1)
        oy = (robot_pos_w[:, 1] - env_origins[:, 1]).unsqueeze(1)

        # Yaw from quaternion
        quat = self.robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Ray directions in world frame (robot yaw + local offsets)
        world_angles = self.ray_angle_offsets.unsqueeze(0) + yaw.unsqueeze(1)  # (E, R)
        dx = torch.cos(world_angles)  # (E, R)
        dy = torch.sin(world_angles)

        distances = torch.full((num_envs, num_rays), max_dist, device=self.device)

        # ---- Wall intersection (ray–line-segment) ----
        for i in range(self.wall_segments.shape[0]):
            ax = self.wall_segments[i, 0, 0]
            ay = self.wall_segments[i, 0, 1]
            bx = self.wall_segments[i, 1, 0]
            by = self.wall_segments[i, 1, 1]

            # Vectors
            aox = ax - ox  # (E, 1)
            aoy = ay - oy
            bax = bx - ax  # scalar
            bay = by - ay

            # 2D cross products
            d_cross_ba = dx * bay - dy * bax  # (E, R)
            ao_cross_ba = aox * bay - aoy * bax
            ao_cross_d = aox * dy - aoy * dx

            # Avoid div-by-zero for parallel rays
            valid = d_cross_ba.abs() > 1e-8
            safe = torch.where(valid, d_cross_ba, torch.ones_like(d_cross_ba))

            t = ao_cross_ba / safe  # distance along ray
            s = ao_cross_d / safe   # parameter along segment [0..1]

            hit = valid & (t > 0) & (s >= 0) & (s <= 1)
            distances = torch.where(hit & (t < distances), t, distances)

        # ---- Food intersection (ray–circle) ----
        food_radius = self.cfg.food_size * 0.7  # approximate cube as circle
        for food_idx in range(self.cfg.num_food_units):
            food_pos_w = self.food_objects[food_idx].data.root_pos_w  # (E, 3)
            fx = (food_pos_w[:, 0:1] - env_origins[:, 0:1])  # (E, 1)
            fy = (food_pos_w[:, 1:2] - env_origins[:, 1:2])

            # Vector from ray origin to food center
            fcx = fx - ox  # (E, 1)  broadcasts with (E, R)
            fcy = fy - oy

            # Projection onto ray direction (distance along ray to closest approach)
            proj = fcx * dx + fcy * dy  # (E, R)
            # Perpendicular distance from food center to ray line
            perp = (fcx * dy - fcy * dx).abs()  # (E, R)

            # Hit if perp < radius and proj > 0 (food is ahead)
            inside_sq = (food_radius ** 2 - perp ** 2).clamp(min=0)
            t_food = proj - torch.sqrt(inside_sq)
            t_food = t_food.clamp(min=0)

            hit_food = (perp < food_radius) & (proj > 0)
            distances = torch.where(hit_food & (t_food > 0) & (t_food < distances), t_food, distances)

        # Clamp to max distance
        distances = distances.clamp(max=max_dist)

        # ---- Store debug hit points in WORLD frame for visualization ----
        hit_x_local = ox + distances * dx
        hit_y_local = oy + distances * dy
        sensor_z = robot_pos_w[:, 2:3] + self.cfg.raycaster_height_offset  # (E, 1)

        self.debug_ray_origins_w = robot_pos_w.clone()
        self.debug_ray_origins_w[:, 2] = self.debug_ray_origins_w[:, 2] + self.cfg.raycaster_height_offset
        self.debug_ray_hits_w = torch.stack([
            hit_x_local + env_origins[:, 0:1],
            hit_y_local + env_origins[:, 1:2],
            sensor_z.expand_as(hit_x_local),
        ], dim=-1)  # (E, R, 3)

        # Normalize to [0, 1]
        return distances / max_dist

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Store actions before physics step."""
        self.actions = actions

    def _apply_action(self) -> None:
        """Apply differential drive velocity commands to robots.
        
        Converts [v_left, v_right] wheel velocities to body-frame linear + angular velocity,
        then transforms to world frame and writes to the rigid body.
        """
        for idx, agent_name in enumerate(self.cfg.possible_agents):
            if agent_name in self.actions:
                # Actions: [v_left, v_right]
                action = self.actions[agent_name]
                v_left = action[:, 0] * self.cfg.wheel_max_velocity
                v_right = action[:, 1] * self.cfg.wheel_max_velocity

                # Differential drive kinematics
                v_linear = (v_left + v_right) / 2.0  # (num_envs,)
                omega = (v_right - v_left) / self.cfg.robot_wheel_base  # (num_envs,)

                # Get current robot orientation (quaternion w, x, y, z)
                quat = self.robot.data.root_quat_w  # (num_envs, 4)
                # Extract yaw from quaternion (rotation around z-axis)
                # For quat [w, x, y, z], yaw = atan2(2*(wz + xy), 1 - 2*(yy + zz))
                w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

                # Convert body-frame velocity to world frame
                vx_world = v_linear * torch.cos(yaw)  # (num_envs,)
                vy_world = v_linear * torch.sin(yaw)  # (num_envs,)

                # Build root velocity tensor: [vx, vy, vz, wx, wy, wz]
                root_vel = torch.zeros((self.scene.num_envs, 6), device=self.device)
                root_vel[:, 0] = vx_world  # Linear X (world)
                root_vel[:, 1] = vy_world  # Linear Y (world)
                root_vel[:, 5] = omega     # Angular Z (yaw rate)

                # Write velocity to simulation
                self.robot.write_root_velocity_to_sim(root_vel)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute local observations for each agent from sensor data.
        
        ACTOR observations (NO privileged info):
        - Raycaster distances: proximity to obstacles
        - Light sensor: beacon detection  
        - In-nest flag: are we in the target zone
        - Neighbor count: nearby robots
        
        Note: Position and velocity are NOT included (only critic has access to full state).
        """
        num_envs = self.scene.num_envs
        
        # ==================
        # Analytical Raycaster (Local Perception)
        # ==================
        # Compute 2D ray-wall and ray-food intersection distances analytically.
        # Returns normalized [0, 1] distances; also populates self.debug_ray_hits_w.
        raycaster_normalized = self._compute_ray_distances()  # (num_envs, num_rays)
        
        # ==================
        # Convert robot world positions to env-local positions
        # ==================
        robot_pos_w = self.robot.data.root_pos_w[:, :3]  # (num_envs, 3)
        env_origins = self.scene.env_origins  # (num_envs, 3)
        robot_pos_local = robot_pos_w[:, :2] - env_origins[:, :2]  # (num_envs, 2)
        
        robot_quat = self.robot.data.root_quat_w  # (num_envs, 4)
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # ==================
        # Light Sensor (Local Perception)
        # ==================
        # 8 rays detecting beacon direction based on actual robot position
        # beacon_pos is in local frame (spawned at global level but relative to env)
        beacon_local = self.light_beacon_pos[:2].unsqueeze(0)  # (1, 2)
        beacon_dir = beacon_local - robot_pos_local  # (num_envs, 2)
        beacon_dist = torch.linalg.norm(beacon_dir, dim=-1, keepdim=True).clamp(min=0.01)
        beacon_angle = torch.atan2(beacon_dir[:, 1], beacon_dir[:, 0]) - robot_yaw  # relative angle
        
        # 8 sensor directions evenly spaced around robot
        sensor_angles = torch.linspace(0, 2 * math.pi * 7 / 8, 8, device=self.device)  # (8,)
        # Compute activation: cosine similarity between sensor direction and beacon direction
        angle_diff = beacon_angle.unsqueeze(1) - sensor_angles.unsqueeze(0)  # (num_envs, 8)
        # Light intensity decreases with distance, peaks when sensor points at beacon
        light_intensity = torch.cos(angle_diff).clamp(min=0.0) / beacon_dist  # (num_envs, 8)
        light_sensor_obs = light_intensity.clamp(0.0, 1.0)
        
        # ==================
        # In-Nest Detection (Local Perception)
        # ==================
        dist_to_nest = torch.linalg.norm(robot_pos_local - self.nest_pos_local.unsqueeze(0), dim=-1)
        in_nest = (dist_to_nest < self.cfg.nest_radius).unsqueeze(1).float()  # (num_envs, 1)
        
        # ==================
        # Neighbor Counting (Local Perception)
        # ==================
        # For single-robot setup, no neighbors
        neighbor_count = torch.zeros((num_envs, 1), device=self.device, dtype=torch.float32)
        
        # ==================
        # Concatenate Actor Observations (46D)
        # ==================
        # NO position/velocity - only local sensors
        obs_dict = {}
        for agent_name in self.cfg.possible_agents:
            obs = torch.cat(
                [
                    raycaster_normalized,  # (num_envs, 36)
                    light_sensor_obs,      # (num_envs, 8)
                    in_nest,               # (num_envs, 1)
                    neighbor_count,        # (num_envs, 1)
                ],
                dim=-1,
            )  # Total: 46D (raycaster + light + nest + neighbors)
            obs_dict[agent_name] = obs
        
        return obs_dict

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute rewards for each agent based on food collection and step penalties."""
        num_envs = self.scene.num_envs
        rewards = {}

        # Initialize rewards with step penalty
        step_reward = torch.full((num_envs,), self.cfg.reward_step_penalty, device=self.device, dtype=torch.float32)
        
        # Get env origins for world-to-local conversion
        env_origins = self.scene.env_origins  # (num_envs, 3)
        
        # ==================
        # Food Collection Reward
        # ==================
        # For each food unit, check if it's in the nest
        # Award +1.0 for each freshly collected food (wasn't collected before)
        food_reward = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        
        for food_idx in range(self.cfg.num_food_units):
            # Get actual food position in env-local coordinates
            food_pos_w = self.food_objects[food_idx].data.root_pos_w  # (num_envs, 3)
            food_pos_local = food_pos_w[:, :2] - env_origins[:, :2]  # (num_envs, 2)
            
            # Check if food is inside the nest area
            dist_food_to_nest = torch.linalg.norm(
                food_pos_local - self.nest_pos_local.unsqueeze(0), dim=-1
            )  # (num_envs,)
            food_in_nest = dist_food_to_nest < self.cfg.nest_radius
            
            # Check if this is a new collection (wasn't collected before)
            newly_collected = food_in_nest & ~self.food_collected[:, food_idx]
            food_reward[newly_collected] += self.cfg.reward_food_in_nest
            
            # Update tracking
            self.food_collected[:, food_idx] = self.food_collected[:, food_idx] | food_in_nest
        
        # ==================
        # Total Reward
        # ==================
        for agent_name in self.cfg.possible_agents:
            reward = step_reward + food_reward
            rewards[agent_name] = reward

        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Check episode termination."""
        num_envs = self.scene.num_envs

        # Episode ends after max time
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: torch.zeros(num_envs, dtype=torch.bool, device=self.device) for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environment instances."""
        super()._reset_idx(env_ids)
        
        # Reset food collection tracking for these environments
        if env_ids is None:
            self.food_collected[:] = False
        else:
            self.food_collected[env_ids] = False
        
        # Reset robot to initial position with some randomization
        if env_ids is not None and len(env_ids) > 0:
            num_reset = len(env_ids)
            # Random position near center
            rand_angle = torch.rand(num_reset, device=self.device) * 2 * math.pi
            rand_radius = torch.rand(num_reset, device=self.device) * 1.0  # within 1m of center
            
            default_state = self.robot.data.default_root_state[env_ids].clone()
            default_state[:, 0] = rand_radius * torch.cos(rand_angle)  # x
            default_state[:, 1] = rand_radius * torch.sin(rand_angle)  # y
            default_state[:, 2] = 0.1  # z
            # Zero velocities
            default_state[:, 7:] = 0.0
            
            self.robot.write_root_state_to_sim(default_state, env_ids)
            
            # Reset food positions randomly in arena (outside nest)
            arena_radius = self.cfg.arena_diameter / 2
            for food_idx in range(self.cfg.num_food_units):
                food_state = self.food_objects[food_idx].data.default_root_state[env_ids].clone()
                # Random position in arena, outside nest
                for attempt in range(100):
                    f_angle = torch.rand(num_reset, device=self.device) * 2 * math.pi
                    f_radius = torch.rand(num_reset, device=self.device) * (arena_radius - 0.5)
                    fx = f_radius * torch.cos(f_angle)
                    fy = f_radius * torch.sin(f_angle)
                    dist_to_nest = torch.sqrt(
                        (fx - self.nest_pos_local[0]) ** 2 + (fy - self.nest_pos_local[1]) ** 2
                    )
                    if (dist_to_nest > self.cfg.nest_radius + self.cfg.food_size).all():
                        break
                food_state[:, 0] = fx
                food_state[:, 1] = fy
                food_state[:, 2] = self.cfg.food_size / 2
                food_state[:, 7:] = 0.0  # Zero velocities
                self.food_objects[food_idx].write_root_state_to_sim(food_state, env_ids)
