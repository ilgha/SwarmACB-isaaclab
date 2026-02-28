# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Directional Gate (DGT) mission — DirectMARLEnv implementation.

Faithful to the paper:
  - Dodecagonal arena (4.91 m²), 20 e-puck cylinder robots
  - White gate strip mid-arena, black corridor north of gate
  - Light source at south edge
  - r(t) = K⁺(t) − K⁻(t): correct crossings (north→south over white gate)
    minus incorrect crossings (south→north over white→black transition)
  - T = 120 s at 10 Hz = 1200 steps
  - Supports all 5 CASA variants (dandelion through cyclamen)

Physics are *kinematic*: e-pucks are modelled as 2-D circles with differential
drive; no USD articulation is needed.  Wall collisions and inter-robot collisions
are resolved analytically (elastic push-out).  This keeps the env pure-PyTorch and
massively parallelisable on GPU, matching the Unity implementation style.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .directional_gate_env_cfg import DirectionalGateEnvCfg

# Relative imports for shared epuck utilities
from ...epuck.epuck_sensors import EpuckSensors
from ...epuck.behavior_modules import BehaviorModules


class DirectionalGateEnv(DirectMARLEnv):
    """Directional Gate mission environment for SwarmACB."""

    cfg: DirectionalGateEnvCfg

    def __init__(self, cfg: DirectionalGateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        E = self.num_envs
        N = self.cfg.num_agents
        dev = self.device

        # ── Robot kinematic state ─────────────────────────────────
        self.agent_pos = torch.zeros(E, N, 2, device=dev)   # XY
        self.agent_yaw = torch.zeros(E, N, device=dev)      # heading (rad)

        # ── Gate-crossing detection ──────────────────────────────
        # Trigger line: middle of the white gate
        self._gate_trigger_y = (self._gate_south_y()
                                + cfg.gate_length / 2.0)
        self._gate_x_halfwidth = cfg.gate_width / 2.0
        # Track previous y-position to detect crossings
        self.prev_y = torch.zeros(E, N, device=dev)

        # ── Episode reward accumulator (for trainer compatibility) ─
        self.completed_group_reward = torch.zeros(E, device=dev)
        self._episode_group_reward = torch.zeros(E, device=dev)

        # ── Precompute arena wall segments ────────────────────────
        self.arena_wall_segments = self._build_wall_segments()

        # ── Gate walls (physical barriers flanking the gate opening) ──
        #  Two vertical side-walls at x = ±corridor_hw,
        #  from gate_south_y to gate_south_y + side_wall_length
        self.gate_wall_segments = self._build_gate_wall_segments()

        # Combined list for proximity sensor raycasts
        self.wall_segments = self.arena_wall_segments + self.gate_wall_segments

        # ── Sensor suite ──────────────────────────────────────────
        self.sensors = EpuckSensors(
            prox_range=cfg.prox_range,
            rab_range=cfg.rab_range,
            light_threshold=cfg.light_threshold,
            device=dev,
        )
        self.light_pos = torch.tensor(
            cfg.light_position[:2], dtype=torch.float32, device=dev,
        )

        # ── Behaviour modules (for ACB discrete variants) ─────────
        self.behavior_modules = BehaviorModules(
            max_speed=cfg.max_wheel_speed,
            obstacle_gain=cfg.obstacle_gain,
            social_gain=cfg.social_gain,
            explore_tau=cfg.explore_tau,
            device=dev,
        )
        self.behavior_modules.init_exploration_state(E, N)

        # ── Arena center / light direction for critic state ───────
        self.arena_center = torch.zeros(2, device=dev)
        light_vec = self.light_pos - self.arena_center
        self.light_dir = light_vec / (light_vec.norm() + 1e-8)

        # ── Sensor cache (avoids double computation for discrete variants) ──
        self._sensor_cache = None

        # ── Precompute wall face normals/points for vectorized collision ──
        self._wall_normals, self._wall_points = self._precompute_wall_faces()

    # ──────────────────────────────────────────────────────────────
    #  Scene setup (visual only — physics are kinematic)
    # ──────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """Spawn visual primitives: ground plane, arena, zones, robot markers."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Dome light so we can see
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ── Visual arena geometry ─────────────────────────────────
        self._spawn_arena_visuals()

        # ── Robot instanced markers ───────────────────────────────
        self._robot_markers = self._create_robot_markers()
        self._heading_markers = self._create_heading_markers()

        # Pre-build marker index arrays
        N = self.cfg.num_agents
        self._robot_proto_idx = np.zeros(N, dtype=np.int32)  # single prototype
        self._heading_proto_idx = np.zeros(N, dtype=np.int32)

        # Clone environments (even though we don't use articulations,
        # DirectMARLEnv expects this call)
        self.scene.clone_environments(copy_from_source=False)

    def _spawn_arena_visuals(self):
        """Spawn static visual geometry for the dodecagonal arena."""
        cfg = self.cfg
        R = cfg.arena_circumradius
        n = cfg.arena_num_sides
        wall_h = cfg.arena_wall_height
        wall_thick = 0.01

        inradius = R * math.cos(math.pi / n)

        # Grey arena floor (large rectangle extending beyond arena)
        floor_side = R * 3.0
        floor_cfg = sim_utils.CuboidCfg(
            size=(floor_side, floor_side, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.45, 0.45),
            ),
        )
        floor_cfg.func(
            "/World/Arena/Floor", floor_cfg,
            translation=(0.0, 0.0, 0.001),
        )

        # White gate zone
        gate_w = cfg.gate_width
        gate_south = self._gate_south_y()
        corr_south = self._corridor_south_y()
        gate_l = corr_south - gate_south
        gate_cy = (gate_south + corr_south) / 2.0
        gate_cfg = sim_utils.CuboidCfg(
            size=(gate_w, gate_l, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        )
        gate_cfg.func("/World/Arena/GateZone", gate_cfg, translation=(0.0, gate_cy, 0.002))

        # Black corridor zone
        corr_w = cfg.corridor_width
        ni = self._north_inradius()
        corr_l = ni - corr_south
        corr_cy = (corr_south + ni) / 2.0
        corr_cfg = sim_utils.CuboidCfg(
            size=(corr_w, corr_l, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
        )
        corr_cfg.func("/World/Arena/CorridorZone", corr_cfg, translation=(0.0, corr_cy, 0.003))

        # Dodecagonal wall segments
        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.70, 0.40))
        for i in range(n):
            a1 = 2 * math.pi * i / n + math.pi / n
            a2 = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
            ax, ay = R * math.cos(a1), R * math.sin(a1)
            bx, by = R * math.cos(a2), R * math.sin(a2)
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            seg_angle = math.atan2(by - ay, bx - ax)
            w_cfg = sim_utils.CuboidCfg(
                size=(seg_len, wall_thick, wall_h),
                visual_material=wall_mat,
            )
            qw = math.cos(seg_angle / 2)
            qz = math.sin(seg_angle / 2)
            w_cfg.func(
                f"/World/Arena/Wall_{i}", w_cfg,
                translation=(mx, my, wall_h / 2),
                orientation=(qw, 0.0, 0.0, qz),
            )

        # Gate side walls
        gate_wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.39, 0.20))
        hw = cfg.corridor_width / 2.0
        wl = cfg.side_wall_length
        wcy = gate_south + wl / 2.0
        for side_i, sx in enumerate([-hw, hw]):
            gw_cfg = sim_utils.CuboidCfg(
                size=(wall_thick, wl, wall_h),
                visual_material=gate_wall_mat,
            )
            gw_cfg.func(
                f"/World/Arena/GateWall_{side_i}", gw_cfg,
                translation=(sx, wcy, wall_h / 2),
            )

        # Light source indicator (red sphere)
        lx, ly, lz = cfg.light_position
        li_cfg = sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15)),
        )
        li_cfg.func("/World/Arena/LightIndicator", li_cfg, translation=(lx, ly, 0.04))

    def _create_robot_markers(self) -> VisualizationMarkers:
        """Create instanced cylinder markers for all robots."""
        r = self.cfg.robot_radius
        h = self.cfg.robot_height
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Robots",
            markers={
                "robot": sim_utils.CylinderCfg(
                    radius=r,
                    height=h,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.31, 0.55, 0.86),
                    ),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _create_heading_markers(self) -> VisualizationMarkers:
        """Create small sphere markers for heading indication."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Headings",
            markers={
                "heading": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 1.0, 0.3),
                    ),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _update_visual_markers(self):
        """Update robot and heading marker positions from kinematic state.

        Skipped when running headless (no viewport) to avoid costly GPU→CPU
        transfers every step.  Also throttled to every 5th step when visible.
        """
        # Skip entirely in headless mode (no viewer)
        if not hasattr(self, '_render_mode') or self._render_mode is None:
            # Check if sim has a running viewer
            try:
                if not self.sim.has_gui():
                    return
            except (AttributeError, RuntimeError):
                pass

        # Throttle to every 5th step to reduce CPU overhead
        step = getattr(self, '_marker_counter', 0)
        self._marker_counter = step + 1
        if step % 5 != 0:
            return

        N = self.cfg.num_agents

        # For now, visualise env 0 only (markers are shared across all envs
        # in the USD stage; we show the first env's state)
        pos_2d = self.agent_pos[0].detach().cpu().numpy()  # (N, 2)
        yaws = self.agent_yaw[0].detach().cpu().numpy()     # (N,)

        robot_z = self.cfg.robot_height / 2.0

        # Robot body positions
        robot_pos = np.zeros((N, 3), dtype=np.float32)
        robot_pos[:, 0] = pos_2d[:, 0]
        robot_pos[:, 1] = pos_2d[:, 1]
        robot_pos[:, 2] = robot_z

        # Robot orientations (yaw → quaternion w,x,y,z)
        robot_orient = np.zeros((N, 4), dtype=np.float32)
        robot_orient[:, 0] = np.cos(yaws / 2)
        robot_orient[:, 3] = np.sin(yaws / 2)

        self._robot_markers.visualize(
            translations=robot_pos,
            orientations=robot_orient,
            marker_indices=self._robot_proto_idx,
        )

        # Heading indicators (small sphere in front)
        arrow_len = self.cfg.robot_radius * 1.8
        head_pos = np.zeros((N, 3), dtype=np.float32)
        head_pos[:, 0] = pos_2d[:, 0] + arrow_len * np.cos(yaws)
        head_pos[:, 1] = pos_2d[:, 1] + arrow_len * np.sin(yaws)
        head_pos[:, 2] = robot_z + 0.01

        self._heading_markers.visualize(
            translations=head_pos,
            marker_indices=self._heading_proto_idx,
        )

    # ──────────────────────────────────────────────────────────────
    #  Arena geometry helpers
    # ──────────────────────────────────────────────────────────────

    def _build_wall_segments(self) -> list[tuple[float, float, float, float]]:
        """Return list of (ax, ay, bx, by) line segments for the dodecagonal wall."""
        R = self.cfg.arena_circumradius
        n = self.cfg.arena_num_sides
        verts = []
        for i in range(n):
            angle = 2 * math.pi * i / n + math.pi / n  # offset so flat side is south
            verts.append((R * math.cos(angle), R * math.sin(angle)))
        segments = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            segments.append((ax, ay, bx, by))
        return segments

    def _build_gate_wall_segments(self):
        """Build two vertical side-wall segments flanking
        the corridor+gate structure.

        Each wall is 0.50 m long, at x = ±(corridor_width/2).
        They run from gate_south to gate_south + wall_length,
        stopping at the south corners of the white gate.
        """
        cfg = self.cfg
        hw = cfg.corridor_width / 2.0
        gate_south = self._gate_south_y()
        wl = cfg.side_wall_length
        return [
            (-hw, gate_south, -hw, gate_south + wl),
            (hw, gate_south, hw, gate_south + wl),
        ]

    # ── Derived Y-coordinate helpers ───────────────────────────

    def _north_inradius(self) -> float:
        R = self.cfg.arena_circumradius
        return R * math.cos(math.pi / self.cfg.arena_num_sides)

    def _corridor_south_y(self) -> float:
        return self._north_inradius() - self.cfg.corridor_length

    def _gate_south_y(self) -> float:
        return self._corridor_south_y() - self.cfg.gate_length

    def _resolve_gate_wall_collisions(self):
        """Push robots out of the two vertical side walls.

        The walls are at x = ±(corridor_width / 2), spanning
        from gate_south to gate_south + side_wall_length.
        """
        cfg = self.cfg
        r = cfg.robot_radius
        hw = cfg.corridor_width / 2.0
        gate_south = self._gate_south_y()
        wall_top = gate_south + cfg.side_wall_length

        px = self.agent_pos[:, :, 0]
        py = self.agent_pos[:, :, 1]

        # Only apply in the Y range of the walls
        in_wall_y = (py > gate_south) & (py < wall_top)

        # Left wall at x = -hw
        dx_left = px - (-hw)
        pen_left = r - dx_left.abs()
        near_left = (
            (pen_left > 0) & in_wall_y & (px < 0)
        )
        sign_l = torch.sign(dx_left)
        sign_l = torch.where(
            sign_l == 0, -torch.ones_like(sign_l), sign_l
        )
        self.agent_pos[:, :, 0] = torch.where(
            near_left, -hw + sign_l * r,
            self.agent_pos[:, :, 0],
        )

        # Right wall at x = +hw
        px = self.agent_pos[:, :, 0]  # re-read
        dx_right = px - hw
        pen_right = r - dx_right.abs()
        near_right = (
            (pen_right > 0) & in_wall_y & (px > 0)
        )
        sign_r = torch.sign(dx_right)
        sign_r = torch.where(
            sign_r == 0, torch.ones_like(sign_r), sign_r
        )
        self.agent_pos[:, :, 0] = torch.where(
            near_right, hw + sign_r * r,
            self.agent_pos[:, :, 0],
        )

    def _ground_color(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute ground colour: 0=black, 0.5=grey, 1=white.

        Args:
            pos: (E, N, 2) agent XY positions

        Returns:
            ground: (E, N, 3)
        """
        cfg = self.cfg
        x = pos[:, :, 0]  # (E, N)
        y = pos[:, :, 1]

        # Derived Y boundaries
        ni = self._north_inradius()
        corr_south = ni - cfg.corridor_length
        gate_south = corr_south - cfg.gate_length
        corr_hw = cfg.corridor_width / 2.0
        gate_hw = cfg.gate_width / 2.0

        # Default: grey
        color = torch.full_like(x, 0.5)

        # White gate: centered, gate_width wide
        in_gate = (
            (x.abs() < gate_hw)
            & (y > gate_south)
            & (y < corr_south)
        )
        color = torch.where(
            in_gate, torch.ones_like(color), color
        )

        # Black corridor: corridor_width wide, above gate
        in_corridor = (
            (x.abs() < corr_hw)
            & (y >= corr_south)
            & (y < ni)
        )
        color = torch.where(
            in_corridor, torch.zeros_like(color), color
        )

        return color.unsqueeze(-1).expand(-1, -1, 3)

    # ──────────────────────────────────────────────────────────────
    #  Actions
    # ──────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        """Store raw actions from the policy.  (E, act_dim) per agent."""
        self._raw_actions = actions

    def _apply_action(self) -> None:
        """Convert actions → wheel velocities → kinematic integration.

        Dandelion: actions are directly (left_vel, right_vel).
        ACB variants: action is a module index; run the selected behaviour module.
        """
        cfg = self.cfg
        E, N = self.num_envs, cfg.num_agents
        dev = self.device

        if cfg.discrete_actions:
            # ── ACB discrete dispatch ─────────────────────────────
            # Stack module indices: (E, N)
            module_ids = torch.stack(
                [self._raw_actions[a].squeeze(-1).long() for a in cfg.possible_agents],
                dim=1,
            )  # (E, N)

            # Compute needed sensor aggregates for behaviour modules
            prox_vals, prox_value, prox_angle = self.sensors.compute_proximity(
                self.agent_pos, self.agent_yaw,
                obstacle_segments=self.wall_segments,
                all_agent_pos=self.agent_pos,
                robot_radius=cfg.robot_radius,
            )
            light_vals, light_value, light_angle = self.sensors.compute_light(
                self.agent_pos, self.agent_yaw, self.light_pos,
            )
            ztilde, rab_proj = self.sensors.compute_rab(
                self.agent_pos, self.agent_yaw,
            )

            # Cache sensor results so _get_observations can reuse them
            self._sensor_cache = {
                "prox_vals": prox_vals,
                "prox_value": prox_value,
                "prox_angle": prox_angle,
                "light_vals": light_vals,
                "light_value": light_value,
                "light_angle": light_angle,
                "ztilde": ztilde,
                "rab_proj": rab_proj,
            }

            # RAB aggregate angle in body frame for attraction / repulsion
            rab_sum_angle = torch.atan2(
                rab_proj[:, :, 1] + rab_proj[:, :, 3],  # sin components
                rab_proj[:, :, 0] + rab_proj[:, :, 2],  # cos components
            )
            rab_magnitude = ztilde  # use neighbour presence as magnitude

            left_vel, right_vel = self.behavior_modules.dispatch(
                module_ids,
                prox_value, prox_angle,
                light_value, light_angle,
                rab_magnitude, rab_sum_angle,
            )
        else:
            # ── Dandelion continuous ──────────────────────────────
            # Stack (E, N, 2): [left_vel, right_vel]
            actions_stacked = torch.stack(
                [self._raw_actions[a] for a in cfg.possible_agents],
                dim=1,
            )  # (E, N, 2)
            left_vel = actions_stacked[:, :, 0].clamp(-cfg.max_wheel_speed, cfg.max_wheel_speed)
            right_vel = actions_stacked[:, :, 1].clamp(-cfg.max_wheel_speed, cfg.max_wheel_speed)

        # ── Differential-drive kinematic integration ──────────────
        dt = cfg.sim.dt
        dx, dy, d_yaw = EpuckSensors.differential_drive(
            left_vel, right_vel, self.agent_yaw, cfg.wheelbase, dt,
        )
        self.agent_pos[:, :, 0] += dx
        self.agent_pos[:, :, 1] += dy
        self.agent_yaw += d_yaw
        # Wrap yaw to [-π, π]
        self.agent_yaw = torch.atan2(torch.sin(self.agent_yaw), torch.cos(self.agent_yaw))

        # ── Wall collision (clamp to arena interior) ──────────────
        self._resolve_wall_collisions()

        # ── Gate wall collision ───────────────────────────────────
        self._resolve_gate_wall_collisions()

        # ── Inter-robot collision ─────────────────────────────────
        self._resolve_robot_collisions()

        # ── Update visual markers in the viewport ─────────────────
        self._update_visual_markers()

    # ──────────────────────────────────────────────────────────────
    #  Collision resolution
    # ──────────────────────────────────────────────────────────────

    def _precompute_wall_faces(self):
        """Precompute wall face normals and reference points as tensors.

        Returns:
            normals: (n, 2) — inward normal for each face
            points:  (n, 2) — point on each face (at inradius)
        """
        R = self.cfg.arena_circumradius
        n = self.cfg.arena_num_sides
        inradius = R * math.cos(math.pi / n)

        normals_list = []
        points_list = []
        for i in range(n):
            angle = 2 * math.pi * i / n + math.pi / n
            next_angle = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
            mid_angle = (angle + next_angle) / 2.0
            # Inward normal (toward center)
            nx = -math.cos(mid_angle)
            ny = -math.sin(mid_angle)
            # Point on the wall face
            wx = inradius * math.cos(mid_angle)
            wy = inradius * math.sin(mid_angle)
            normals_list.append([nx, ny])
            points_list.append([wx, wy])

        normals = torch.tensor(normals_list, dtype=torch.float32, device=self.device)  # (n, 2)
        points = torch.tensor(points_list, dtype=torch.float32, device=self.device)    # (n, 2)
        return normals, points

    def _resolve_wall_collisions(self):
        """Push robots inside the dodecagonal arena boundary (fully vectorized)."""
        r = self.cfg.robot_radius
        normals = self._wall_normals   # (n, 2)
        points = self._wall_points     # (n, 2)

        # Agent positions: (E, N, 2)
        # Broadcast: pos (E, N, 1, 2) - points (1, 1, n, 2) → (E, N, n, 2)
        diff = self.agent_pos.unsqueeze(2) - points.view(1, 1, -1, 2)   # (E, N, n, 2)

        # Signed distance to each face: dot(diff, normal) → (E, N, n)
        n_vec = normals.view(1, 1, -1, 2)                                # (1, 1, n, 2)
        signed_dist = (diff * n_vec).sum(dim=-1)                         # (E, N, n)

        # Penetration = robot_radius - signed_dist
        penetration = r - signed_dist                                     # (E, N, n)

        # Only push where penetrating (pen > 0)
        push_mask = penetration > 0                                       # (E, N, n)
        penetration = penetration * push_mask.float()                     # zero out non-penetrating

        # Push displacement per face: pen * normal → (E, N, n, 2)
        push = penetration.unsqueeze(-1) * n_vec                         # (E, N, n, 2)

        # Sum pushes from all penetrating faces
        total_push = push.sum(dim=2)                                      # (E, N, 2)
        self.agent_pos = self.agent_pos + total_push

    def _resolve_robot_collisions(self):
        """Elastic push-out between robot pairs (one pass)."""
        r = self.cfg.robot_radius
        min_dist = 2 * r
        N = self.cfg.num_agents

        # Pairwise distances (E, N, N)
        dx = self.agent_pos[:, :, 0].unsqueeze(2) - self.agent_pos[:, :, 0].unsqueeze(1)
        dy = self.agent_pos[:, :, 1].unsqueeze(2) - self.agent_pos[:, :, 1].unsqueeze(1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Only resolve i < j pairs (upper triangle)
        mask = torch.triu(torch.ones(N, N, device=self.device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).expand(self.num_envs, -1, -1)

        overlap = (min_dist - dist).clamp(min=0) * mask.float()  # (E, N, N)

        if overlap.sum() == 0:
            return

        # Separation direction (i→j)
        nx = dx / (dist + 1e-8)
        ny = dy / (dist + 1e-8)

        # Push each robot half the overlap
        push_x = (overlap * nx * 0.5).sum(dim=2)  # effect on robot i
        push_y = (overlap * ny * 0.5).sum(dim=2)

        self.agent_pos[:, :, 0] += push_x
        # Apply reverse push on j (sum over dim 1 of transposed)
        self.agent_pos[:, :, 0] -= (overlap * nx * 0.5).sum(dim=1)
        self.agent_pos[:, :, 1] += push_y
        self.agent_pos[:, :, 1] -= (overlap * ny * 0.5).sum(dim=1)

    # ──────────────────────────────────────────────────────────────
    #  Observations
    # ──────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute per-agent observations.  Layout depends on variant."""
        cfg = self.cfg

        # Reuse cached sensors if available (discrete variants compute them
        # in _apply_action already), otherwise compute fresh
        cache = self._sensor_cache
        if cache is not None:
            prox_vals = cache["prox_vals"]
            light_vals = cache["light_vals"]
            ztilde = cache["ztilde"]
            rab_proj = cache["rab_proj"]
            self._sensor_cache = None  # consume cache
        else:
            prox_vals, _, _ = self.sensors.compute_proximity(
                self.agent_pos, self.agent_yaw,
                obstacle_segments=self.wall_segments,
                all_agent_pos=self.agent_pos,
                robot_radius=cfg.robot_radius,
            )
            light_vals, _, _ = self.sensors.compute_light(
                self.agent_pos, self.agent_yaw, self.light_pos,
            )
            ztilde, rab_proj = self.sensors.compute_rab(
                self.agent_pos, self.agent_yaw,
            )

        ground = self._ground_color(self.agent_pos)  # (E, N, 3)

        if cfg.variant in ("dandelion", "daisy"):
            obs_all = self.sensors.collect_obs_dandelion(
                prox_vals, light_vals, ground, ztilde, rab_proj,
            )  # (E, N, 24)
        else:
            # lily / tulip / cyclamen: 4-dim
            obs_all = EpuckSensors.collect_obs_lily(ground, ztilde)  # (E, N, 4)

        # Convert to per-agent dict
        obs_dict: dict[str, torch.Tensor] = {}
        for i, agent in enumerate(cfg.possible_agents):
            obs_dict[agent] = obs_all[:, i]  # (E, obs_dim)

        return obs_dict

    # ──────────────────────────────────────────────────────────────
    #  Rewards  —  r(t) = K⁺(t) − K⁻(t)
    # ──────────────────────────────────────────────────────────────

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Compute team reward: K⁺ − K⁻ gate crossings.

        The trigger line is at the middle of the white gate
        (y = gate_south + gate_length / 2).  A crossing is
        counted when a robot's y-position crosses this line
        while its x-position is within ±gate_width/2.

          K⁺: north → south  (prev_y > trigger, curr_y ≤ trigger)
          K⁻: south → north  (prev_y ≤ trigger, curr_y > trigger)
        """
        cfg = self.cfg
        curr_y = self.agent_pos[:, :, 1]          # (E, N)
        curr_x = self.agent_pos[:, :, 0]          # (E, N)
        prev_y = self.prev_y                      # (E, N)
        trig = self._gate_trigger_y
        hw = self._gate_x_halfwidth

        # Robot must be within the gate opening (x-axis)
        in_gate = curr_x.abs() < hw              # (E, N)

        # North-to-south crossing → K⁺
        n2s = (prev_y > trig) & (curr_y <= trig) & in_gate
        k_plus = n2s.float().sum(dim=1)           # (E,)

        # South-to-north crossing → K⁻
        s2n = (prev_y <= trig) & (curr_y > trig) & in_gate
        k_minus = s2n.float().sum(dim=1)          # (E,)

        # Update y tracking
        self.prev_y = curr_y.clone()

        # Team reward
        reward = k_plus - k_minus                 # (E,)
        self._episode_group_reward += reward

        # Return same reward for all agents (team reward)
        reward_dict = {
            agent: reward for agent in cfg.possible_agents
        }
        return reward_dict

    # ──────────────────────────────────────────────────────────────
    #  Done conditions
    # ──────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """No early termination; episode ends by time limit only."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: torch.zeros_like(time_out) for agent in self.cfg.possible_agents}
        truncated = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, truncated

    # ──────────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        idx = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        N = self.cfg.num_agents
        R = self.cfg.arena_circumradius

        # Snapshot completed reward before resetting
        self.completed_group_reward[idx] = self._episode_group_reward[idx]
        self._episode_group_reward[idx] = 0.0

        # Random positions inside arena (rejection-sample inside the dodecagon
        # approximated by a circle of radius R * cos(π/n) — the inradius)
        inradius = R * math.cos(math.pi / self.cfg.arena_num_sides)
        safe_r = inradius - self.cfg.robot_radius * 2

        # Vectorized: generate all positions at once for len(env_ids) envs × N agents
        n_reset = len(env_ids)
        r_rand = torch.sqrt(torch.rand(n_reset, N, device=self.device)) * safe_r
        theta = torch.rand(n_reset, N, device=self.device) * 2 * math.pi
        self.agent_pos[idx, :, 0] = r_rand * torch.cos(theta)
        self.agent_pos[idx, :, 1] = r_rand * torch.sin(theta)
        self.agent_yaw[idx] = torch.rand(n_reset, N, device=self.device) * 2 * math.pi - math.pi

        # Reset y-tracking for crossing detection
        self.prev_y[idx] = self.agent_pos[idx, :, 1]

        # Reset exploration state for these envs
        mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        mask[idx] = True
        self.behavior_modules.reset_exploration_state(mask)

    # ──────────────────────────────────────────────────────────────
    #  Critic state (called externally by trainer if needed)
    # ──────────────────────────────────────────────────────────────

    def get_critic_state(self) -> torch.Tensor:
        """Return 5-D polar critic state for all agents: (E, N, 5).

        s = (ρ, cos α, sin α, cos β, sin β)
        """
        return EpuckSensors.compute_critic_state_5d(
            self.agent_pos,
            self.agent_yaw,
            self.arena_center,
            self.cfg.arena_circumradius,
            self.light_dir,
        )
