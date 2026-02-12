# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swarm Collaborative Foraging Environment — Multi-Robot."""

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


# Distinct colours per robot so they are easy to tell apart in the viewport.
_ROBOT_COLORS = [
    (0.5, 0.0, 0.5),   # purple
    (0.0, 0.4, 0.8),   # blue
    (0.8, 0.2, 0.0),   # red-orange
    (0.0, 0.7, 0.3),   # green
    (0.9, 0.7, 0.0),   # gold
    (0.0, 0.8, 0.8),   # cyan
    (0.8, 0.0, 0.4),   # magenta
    (0.4, 0.4, 0.4),   # grey
]


class SwarmACBForagingEnv(DirectMARLEnv):
    """Multi-agent cooperative foraging environment.

    Each environment contains *N* robots that share a common arena with food
    cubes and a nest.  Robots can detect walls, food, and other robots via an
    analytical 2-D raycaster, and sense the direction of a light beacon placed
    near the nest.  The shared reward signal fires when any food cube enters
    the nest area.
    """

    cfg: SwarmACBForagingEnvCfg

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------

    def __init__(self, cfg: SwarmACBForagingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.cfg.num_robots
        E = self.scene.num_envs

        # ---- Robot handles (one RigidObject per robot index) ----
        self.robots: list[RigidObject] = [
            self.scene.rigid_objects[f"robot_{i}"] for i in range(N)
        ]

        # ---- Food handles ----
        self.food_objects: list[RigidObject] = [
            self.scene.rigid_objects[f"food_{i}"] for i in range(self.cfg.num_food_units)
        ]

        # ---- Analytical raycaster setup ----
        self.num_raycaster_rays = self.cfg.raycaster_num_rays
        self.obs_size = (
            self.num_raycaster_rays
            + self.cfg.light_sensor_num_rays
            + 1   # in_nest
            + 1   # carrying_food (placeholder — always 0 for now)
            + 1   # neighbor_count
        )
        print(f"[SwarmACB] {N} robots, {self.num_raycaster_rays} analytical rays, obs={self.obs_size}D")

        # Pre-compute ray angle offsets once
        self.ray_angle_offsets = torch.linspace(
            0,
            2 * math.pi * (self.num_raycaster_rays - 1) / self.num_raycaster_rays,
            self.num_raycaster_rays,
            device=self.device,
        )

        # Debug storage for first robot in env 0 (used by manual_control.py).
        self.debug_ray_hits_w = torch.zeros(
            (E, self.num_raycaster_rays, 3), device=self.device
        )
        self.debug_ray_origins_w = torch.zeros((E, 3), device=self.device)

        # Track food collection — shared across all robots in each env.
        self.food_collected = torch.zeros(
            (E, self.cfg.num_food_units), device=self.device, dtype=torch.bool
        )

    # ------------------------------------------------------------------
    #  Scene construction
    # ------------------------------------------------------------------

    def _setup_scene(self):
        """Build simulation scene: ground, N robots, arena, food, beacon."""

        # Ground
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Robots — spawn N distinct prims (Robot_0 … Robot_{N-1})
        self._robot_handles: list[RigidObject] = []
        for i in range(self.cfg.num_robots):
            color = _ROBOT_COLORS[i % len(_ROBOT_COLORS)]
            robot_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Robot_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 0.15),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        linear_damping=2.0,
                        angular_damping=2.0,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
            )
            robot_obj = RigidObject(robot_cfg)
            self._robot_handles.append(robot_obj)

        # Arena / Nest / Food / Beacon
        self._spawn_arena()
        self._spawn_nest()
        self._spawn_food()
        self._spawn_light_beacon()

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register robots AFTER cloning
        for i, rh in enumerate(self._robot_handles):
            self.scene.rigid_objects[f"robot_{i}"] = rh

        # Register food
        self._food_rigid_objects: list[RigidObject] = []
        for food_idx in range(self.cfg.num_food_units):
            food_rigid = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Food_{food_idx}",
            ))
            self.scene.rigid_objects[f"food_{food_idx}"] = food_rigid
            self._food_rigid_objects.append(food_rigid)

        # Pre-compute wall geometry
        self._precompute_wall_segments()

        # Lighting
        dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 1.0))
        dome_light_cfg.func("/World/DomeLight", dome_light_cfg)
        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=2000.0, color=(1.0, 0.95, 0.8), angle=0.5
        )
        distant_light_cfg.func("/World/SunLight", distant_light_cfg)

    # ------------------------------------------------------------------
    #  Arena / Nest / Food / Beacon spawning
    # ------------------------------------------------------------------

    def _spawn_arena(self):
        """Spawn dodecagonal walled arena."""
        arena_radius = self.cfg.arena_diameter / 2
        num_sides = self.cfg.num_arena_sides
        wall_length = 2 * arena_radius * math.sin(math.pi / num_sides)

        for side in range(num_sides):
            angle = 2 * math.pi * side / num_sides + math.pi / num_sides
            cx = arena_radius * math.cos(angle)
            cy = arena_radius * math.sin(angle)
            wall_yaw = angle + math.pi / 2

            wall_cfg = sim_utils.CuboidCfg(
                size=(wall_length, self.cfg.arena_wall_thickness, self.cfg.arena_wall_height),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.8)),
            )
            quat = [math.cos(wall_yaw / 2), 0, 0, math.sin(wall_yaw / 2)]
            wall_cfg.func(
                prim_path=f"/World/envs/env_.*/Arena_Wall_{side}",
                cfg=wall_cfg,
                translation=(cx, cy, self.cfg.arena_wall_height / 2),
                orientation=quat,
            )

    def _spawn_nest(self):
        """Spawn nest (flat visual disc, no collision)."""
        nest_x = self.cfg.nest_distance_from_center * math.cos(self.cfg.nest_angle)
        nest_y = self.cfg.nest_distance_from_center * math.sin(self.cfg.nest_angle)

        nest_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.nest_radius,
            height=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.5)),
        )
        nest_cfg.func(
            prim_path="/World/envs/env_.*/Nest",
            cfg=nest_cfg,
            translation=(nest_x, nest_y, 0.001),
        )
        self.nest_pos_local = torch.tensor([nest_x, nest_y], device=self.device, dtype=torch.float32)

    def _spawn_food(self):
        """Spawn food cubes randomly inside arena, outside nest."""
        arena_radius = self.cfg.arena_diameter / 2
        for food_idx in range(self.cfg.num_food_units):
            while True:
                angle = torch.rand(1, device=self.device) * 2 * math.pi
                radius = torch.rand(1, device=self.device) * (arena_radius - self.cfg.food_size)
                fx = float(radius * torch.cos(angle))
                fy = float(radius * torch.sin(angle))
                if math.sqrt((fx - float(self.nest_pos_local[0])) ** 2
                             + (fy - float(self.nest_pos_local[1])) ** 2) > self.cfg.nest_radius + self.cfg.food_size:
                    break

            food_cfg = sim_utils.CuboidCfg(
                size=(self.cfg.food_size, self.cfg.food_size, self.cfg.food_size),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.food_mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            )
            food_cfg.func(
                prim_path=f"/World/envs/env_.*/Food_{food_idx}",
                cfg=food_cfg,
                translation=(fx, fy, self.cfg.food_size / 2),
            )

    def _spawn_light_beacon(self):
        """Spawn directional light beacon outside arena near nest."""
        d = self.cfg.arena_diameter / 2 + 1.0
        bx = d * math.cos(self.cfg.nest_angle)
        by = d * math.sin(self.cfg.nest_angle)
        bz = 1.0

        beacon_cfg = sim_utils.SphereCfg(
            radius=0.1,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        )
        beacon_cfg.func(prim_path="/World/Light_Beacon", cfg=beacon_cfg, translation=(bx, by, bz))
        self.light_beacon_pos = torch.tensor([bx, by, bz], device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    #  Analytical raycaster helpers
    # ------------------------------------------------------------------

    def _precompute_wall_segments(self):
        """Store wall segment endpoints as (num_walls, 2, 2) tensor."""
        arena_radius = self.cfg.arena_diameter / 2
        num_sides = self.cfg.num_arena_sides
        wall_length = 2 * arena_radius * math.sin(math.pi / num_sides)
        segs = torch.zeros((num_sides, 2, 2), device=self.device)
        for s in range(num_sides):
            a = 2 * math.pi * s / num_sides + math.pi / num_sides
            cx, cy = arena_radius * math.cos(a), arena_radius * math.sin(a)
            yaw = a + math.pi / 2
            hl = wall_length / 2
            segs[s, 0, 0] = cx + hl * math.cos(yaw)
            segs[s, 0, 1] = cy + hl * math.sin(yaw)
            segs[s, 1, 0] = cx - hl * math.cos(yaw)
            segs[s, 1, 1] = cy - hl * math.sin(yaw)
        self.wall_segments = segs

    def _compute_ray_distances(self, robot_idx: int) -> torch.Tensor:
        """Analytical 2-D ray cast for one robot across all envs.

        Detects: walls, food cubes, and *other* robots.

        Args:
            robot_idx: index into ``self.robots``.

        Returns:
            Normalized distance tensor (E, R) in [0, 1].
        """
        E = self.scene.num_envs
        R = self.num_raycaster_rays
        max_d = self.cfg.raycaster_max_distance
        env_origins = self.scene.env_origins

        # --- This robot's pose (env-local) ---
        pos_w = self.robots[robot_idx].data.root_pos_w
        ox = (pos_w[:, 0] - env_origins[:, 0]).unsqueeze(1)
        oy = (pos_w[:, 1] - env_origins[:, 1]).unsqueeze(1)

        quat = self.robots[robot_idx].data.root_quat_w
        wq, xq, yq, zq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))

        angles = self.ray_angle_offsets.unsqueeze(0) + yaw.unsqueeze(1)
        dx = torch.cos(angles)
        dy = torch.sin(angles)

        dists = torch.full((E, R), max_d, device=self.device)

        # ---- Walls (ray–line-segment intersection) ----
        for i in range(self.wall_segments.shape[0]):
            ax, ay = self.wall_segments[i, 0, 0], self.wall_segments[i, 0, 1]
            bx, by = self.wall_segments[i, 1, 0], self.wall_segments[i, 1, 1]
            aox, aoy = ax - ox, ay - oy
            bax, bay = bx - ax, by - ay
            d_cross = dx * bay - dy * bax
            valid = d_cross.abs() > 1e-8
            safe = torch.where(valid, d_cross, torch.ones_like(d_cross))
            t = (aox * bay - aoy * bax) / safe
            s = (aox * dy - aoy * dx) / safe
            hit = valid & (t > 0) & (s >= 0) & (s <= 1)
            dists = torch.where(hit & (t < dists), t, dists)

        # ---- Food (ray–circle approximation) ----
        fr = self.cfg.food_size * 0.7
        for fi in range(self.cfg.num_food_units):
            fp = self.food_objects[fi].data.root_pos_w
            fx = (fp[:, 0:1] - env_origins[:, 0:1])
            fy = (fp[:, 1:2] - env_origins[:, 1:2])
            fcx, fcy = fx - ox, fy - oy
            proj = fcx * dx + fcy * dy
            perp = (fcx * dy - fcy * dx).abs()
            t_food = (proj - torch.sqrt((fr ** 2 - perp ** 2).clamp(min=0))).clamp(min=0)
            hit_f = (perp < fr) & (proj > 0) & (t_food > 0)
            dists = torch.where(hit_f & (t_food < dists), t_food, dists)

        # ---- Other robots (ray–circle approximation) ----
        rr = self.cfg.robot_radius
        for ri in range(self.cfg.num_robots):
            if ri == robot_idx:
                continue
            rp = self.robots[ri].data.root_pos_w
            rx = (rp[:, 0:1] - env_origins[:, 0:1])
            ry = (rp[:, 1:2] - env_origins[:, 1:2])
            rcx, rcy = rx - ox, ry - oy
            proj = rcx * dx + rcy * dy
            perp = (rcx * dy - rcy * dx).abs()
            t_rob = (proj - torch.sqrt((rr ** 2 - perp ** 2).clamp(min=0))).clamp(min=0)
            hit_r = (perp < rr) & (proj > 0) & (t_rob > 0)
            dists = torch.where(hit_r & (t_rob < dists), t_rob, dists)

        dists = dists.clamp(max=max_d)

        # ---- Debug visualisation for robot 0 ----
        if robot_idx == 0:
            hit_x = ox + dists * dx
            hit_y = oy + dists * dy
            sz = pos_w[:, 2:3] + self.cfg.raycaster_height_offset
            self.debug_ray_origins_w = pos_w.clone()
            self.debug_ray_origins_w[:, 2] += self.cfg.raycaster_height_offset
            self.debug_ray_hits_w = torch.stack([
                hit_x + env_origins[:, 0:1],
                hit_y + env_origins[:, 1:2],
                sz.expand_as(hit_x),
            ], dim=-1)

        return dists / max_d

    # ------------------------------------------------------------------
    #  Physics step
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        """Apply differential-drive velocity to *each* robot."""
        for idx in range(self.cfg.num_robots):
            agent_name = self.cfg.possible_agents[idx]
            if agent_name not in self.actions:
                continue
            act = self.actions[agent_name]
            vl = act[:, 0] * self.cfg.wheel_max_velocity
            vr = act[:, 1] * self.cfg.wheel_max_velocity

            v_lin = (vl + vr) / 2.0
            omega = (vr - vl) / self.cfg.robot_wheel_base

            quat = self.robots[idx].data.root_quat_w
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            vel = torch.zeros((self.scene.num_envs, 6), device=self.device)
            vel[:, 0] = v_lin * torch.cos(yaw)
            vel[:, 1] = v_lin * torch.sin(yaw)
            vel[:, 5] = omega
            self.robots[idx].write_root_velocity_to_sim(vel)

    # ------------------------------------------------------------------
    #  Observations (actor) & State (critic)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Per-agent local observations (47-D).

        Breakdown: raycaster(36) + light(8) + in_nest(1) + carrying(1) + neighbors(1).
        """
        E = self.scene.num_envs
        N = self.cfg.num_robots
        env_origins = self.scene.env_origins

        # Gather all robots' local positions once (needed for neighbor counting).
        all_pos_local = torch.zeros((E, N, 2), device=self.device)
        all_yaw = torch.zeros((E, N), device=self.device)
        for i in range(N):
            pw = self.robots[i].data.root_pos_w
            all_pos_local[:, i, 0] = pw[:, 0] - env_origins[:, 0]
            all_pos_local[:, i, 1] = pw[:, 1] - env_origins[:, 1]
            q = self.robots[i].data.root_quat_w
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            all_yaw[:, i] = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        obs_dict: dict[str, torch.Tensor] = {}
        for idx in range(N):
            agent_name = self.cfg.possible_agents[idx]

            # -- Raycaster --
            ray_norm = self._compute_ray_distances(idx)  # (E, 36)

            pos_local = all_pos_local[:, idx]  # (E, 2)
            yaw = all_yaw[:, idx]              # (E,)

            # -- Light sensor (8-D) --
            beacon_xy = self.light_beacon_pos[:2].unsqueeze(0)
            b_dir = beacon_xy - pos_local
            b_dist = torch.linalg.norm(b_dir, dim=-1, keepdim=True).clamp(min=0.01)
            b_angle = torch.atan2(b_dir[:, 1], b_dir[:, 0]) - yaw
            s_angles = torch.linspace(0, 2 * math.pi * 7 / 8, 8, device=self.device)
            diff = b_angle.unsqueeze(1) - s_angles.unsqueeze(0)
            light_obs = (torch.cos(diff).clamp(min=0.0) / b_dist).clamp(0.0, 1.0)  # (E, 8)

            # -- In-nest --
            d_nest = torch.linalg.norm(pos_local - self.nest_pos_local.unsqueeze(0), dim=-1)
            in_nest = (d_nest < self.cfg.nest_radius).unsqueeze(1).float()

            # -- Carrying food (placeholder — future: explicit pick-up mechanic) --
            carrying = torch.zeros((E, 1), device=self.device)

            # -- Neighbor count --
            nr = self.cfg.neighbor_detection_range
            count = torch.zeros((E, 1), device=self.device)
            for j in range(N):
                if j == idx:
                    continue
                d = torch.linalg.norm(pos_local - all_pos_local[:, j], dim=-1)
                count[:, 0] += (d < nr).float()

            obs_dict[agent_name] = torch.cat([ray_norm, light_obs, in_nest, carrying, count], dim=-1)

        return obs_dict

    def _get_states(self) -> torch.Tensor:
        """Privileged critic state (global view).

        Layout: [robot_0_state … robot_{N-1}_state | food_positions | nest_pos]
        Per-robot: pos_xy(2) + vel_xy(2) + yaw(1) + carrying(1) = 6
        """
        E = self.scene.num_envs
        N = self.cfg.num_robots
        eo = self.scene.env_origins

        parts: list[torch.Tensor] = []
        for i in range(N):
            pw = self.robots[i].data.root_pos_w
            vw = self.robots[i].data.root_lin_vel_w
            q = self.robots[i].data.root_quat_w
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            pos_xy = pw[:, :2] - eo[:, :2]
            vel_xy = vw[:, :2]
            carrying = torch.zeros((E, 1), device=self.device)
            parts.append(torch.cat([pos_xy, vel_xy, yaw.unsqueeze(1), carrying], dim=-1))

        # Food positions (env-local)
        for fi in range(self.cfg.num_food_units):
            fp = self.food_objects[fi].data.root_pos_w
            parts.append(fp[:, :2] - eo[:, :2])

        # Nest position (broadcast)
        nest = self.nest_pos_local.unsqueeze(0).expand(E, -1)
        parts.append(nest)

        return torch.cat(parts, dim=-1)  # (E, state_space)

    # ------------------------------------------------------------------
    #  Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Shared team reward: +1.0 per newly collected food."""
        E = self.scene.num_envs
        eo = self.scene.env_origins

        food_reward = torch.zeros(E, device=self.device, dtype=torch.float32)
        for fi in range(self.cfg.num_food_units):
            fp = self.food_objects[fi].data.root_pos_w[:, :2] - eo[:, :2]
            d = torch.linalg.norm(fp - self.nest_pos_local.unsqueeze(0), dim=-1)
            in_nest = d < self.cfg.nest_radius
            newly = in_nest & ~self.food_collected[:, fi]
            food_reward[newly] += self.cfg.reward_food_in_nest
            self.food_collected[:, fi] |= in_nest

        step_pen = self.cfg.reward_step_penalty
        total = food_reward + step_pen

        # All agents share the same team reward
        return {a: total for a in self.cfg.possible_agents}

    # ------------------------------------------------------------------
    #  Termination / Reset
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        E = self.scene.num_envs
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {a: torch.zeros(E, dtype=torch.bool, device=self.device) for a in self.cfg.possible_agents}
        truncated = {a: time_out for a in self.cfg.possible_agents}
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            self.food_collected[:] = False
        else:
            self.food_collected[env_ids] = False

        if env_ids is not None and len(env_ids) > 0:
            n = len(env_ids)
            spawn_r = self.cfg.robot_spawn_radius

            # Reset each robot to a random position / heading
            for i in range(self.cfg.num_robots):
                ang = torch.rand(n, device=self.device) * 2 * math.pi
                rad = torch.rand(n, device=self.device) * spawn_r
                ds = self.robots[i].data.default_root_state[env_ids].clone()
                ds[:, 0] = rad * torch.cos(ang)
                ds[:, 1] = rad * torch.sin(ang)
                ds[:, 2] = 0.1
                # Random heading (quaternion w, x, y, z — rotation about z)
                heading = torch.rand(n, device=self.device) * 2 * math.pi
                ds[:, 3] = torch.cos(heading / 2)   # qw
                ds[:, 4] = 0.0                        # qx
                ds[:, 5] = 0.0                        # qy
                ds[:, 6] = torch.sin(heading / 2)   # qz
                ds[:, 7:] = 0.0  # zero velocities
                self.robots[i].write_root_state_to_sim(ds, env_ids)

            # Reset food positions
            arena_r = self.cfg.arena_diameter / 2
            for fi in range(self.cfg.num_food_units):
                fs = self.food_objects[fi].data.default_root_state[env_ids].clone()
                for _ in range(100):
                    fa = torch.rand(n, device=self.device) * 2 * math.pi
                    fr = torch.rand(n, device=self.device) * (arena_r - 0.5)
                    fx = fr * torch.cos(fa)
                    fy = fr * torch.sin(fa)
                    d2n = torch.sqrt((fx - self.nest_pos_local[0]) ** 2 + (fy - self.nest_pos_local[1]) ** 2)
                    if (d2n > self.cfg.nest_radius + self.cfg.food_size).all():
                        break
                fs[:, 0] = fx
                fs[:, 1] = fy
                fs[:, 2] = self.cfg.food_size / 2
                fs[:, 7:] = 0.0
                self.food_objects[fi].write_root_state_to_sim(fs, env_ids)
