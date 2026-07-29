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
        # Unity detects ground COLOR TRANSITIONS (not Y-position crossings):
        #   BLACK → WHITE  →  +1  (correct traversal: corridor → gate going south)
        #   WHITE → BLACK  →  −1  (reverse traversal: gate → corridor going north)
        # Track previous ground color per robot (0=black, 0.5=grey, 1=white)
        self.prev_ground_color = torch.full((E, N), 0.5, device=dev)  # grey default

        # ── Episode reward accumulator (for trainer compatibility) ─
        self.completed_group_reward = torch.zeros(E, device=dev)
        self._episode_group_reward = torch.zeros(E, device=dev)
        # IsaacLab auto-resets timed-out environments before returning from
        # ``step``. Keep the true final centralized state so interrupted Unity
        # trajectories can bootstrap from s_{t+1}, rather than from the reset.
        self.completed_terminal_critic_state = torch.zeros(E, N, 5, device=dev)

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
            rab_loss_probability=cfg.rab_loss_probability,
            light_threshold=cfg.light_threshold,
            light_intensity=cfg.light_intensity,
            alpha_rab=cfg.alpha_parameter,
            unity_unit_scale_m=cfg.unity_unit_scale_m,
            device=dev,
        )
        self.light_pos = torch.tensor(
            cfg.light_position[:2], dtype=torch.float32, device=dev,
        )

        # ── Behaviour modules (for ACB discrete variants) ─────────
        self.behavior_modules = BehaviorModules(
            max_speed=cfg.max_wheel_speed,
            alpha_parameter=cfg.alpha_parameter,
            device=dev,
        )
        self.behavior_modules.init_state(E, N)

        # ── Arena center / light direction for critic state ───────
        self.arena_center = torch.zeros(2, device=dev)
        light_vec = self.light_pos - self.arena_center
        self.light_dir = light_vec / (light_vec.norm() + 1e-8)
        # PerAgentState5DSensor.cs measures alpha from Unity world +Z. With the
        # Unity XZ plane mapped to Isaac XY, that reference is Isaac +Y.
        self.critic_reference_dir = torch.tensor((0.0, 1.0), device=dev)

        # ── Sensor cache (avoids double computation for discrete variants) ──
        self._sensor_cache = None
        # Reuse one low-level wheel command across physics substeps, so ACB modules
        # keep the same 10 Hz clock when GUI playback raises the physics rate.
        self._cached_left_vel = torch.zeros(E, N, device=dev)
        self._cached_right_vel = torch.zeros(E, N, device=dev)
        self._low_level_action_dirty = True

        # ── Precompute wall face normals/points for vectorized collision ──
        self._wall_normals, self._wall_points = self._precompute_wall_faces()

    # ──────────────────────────────────────────────────────────────
    #  Scene setup (visual only — physics are kinematic)
    # ──────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """Spawn the local arena, zones, and robot markers."""

        # Dome light so we can see
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # ── Visual arena geometry ─────────────────────────────────
        self._spawn_arena_visuals()

        # ── Robot instanced markers ───────────────────────────────
        self._robot_markers = self._create_robot_markers()
        self._heading_markers = self._create_heading_markers()
        if getattr(self.cfg, "debug_visual_sensors", False):
            self._sensor_line_markers = self._create_sensor_line_markers()
            self._sensor_point_markers = self._create_sensor_point_markers()
        else:
            self._sensor_line_markers = None
            self._sensor_point_markers = None

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
        wall_thick = cfg.arena_wall_thickness

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

    def _create_sensor_line_markers(self) -> VisualizationMarkers:
        """Create line-like cylinder markers for live sensor debugging."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/SensorLines",
            markers={
                "prox_clear": sim_utils.CylinderCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.25)),
                ),
                "prox_hit": sim_utils.CylinderCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.05)),
                ),
                "light": sim_utils.CylinderCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.05)),
                ),
                "rab_link": sim_utils.CylinderCfg(
                    radius=1.0,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 1.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _create_sensor_point_markers(self) -> VisualizationMarkers:
        """Create point markers for sensor ray endpoints and RAB range rings."""
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/SensorPoints",
            markers={
                "prox_clear": sim_utils.SphereCfg(
                    radius=0.006,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.25)),
                ),
                "prox_hit": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.05)),
                ),
                "light": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.05)),
                ),
                "rab_ring": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.25, 1.0)),
                ),
                "rab_neighbor": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 1.0)),
                ),
                "ground_black": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
                ),
                "ground_grey": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
                ),
                "ground_white": sim_utils.SphereCfg(
                    radius=0.010,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _compute_light_readings(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return light sensor readings, or zeros for missions without a light cue."""
        if getattr(self.cfg, "has_light", True):
            return self.sensors.compute_light(
                self.agent_pos, self.agent_yaw, self.light_pos,
            )

        E, N = self.num_envs, self.cfg.num_agents
        light_vals = torch.zeros(E, N, 8, device=self.device)
        light_value = torch.zeros(E, N, device=self.device)
        light_angle = torch.zeros(E, N, device=self.device)
        return light_vals, light_value, light_angle

    def _compute_sensor_bundle(self) -> dict[str, torch.Tensor]:
        """Sample all local sensors once for the current robot state."""
        cfg = self.cfg
        prox_vals, prox_value, prox_angle = self.sensors.compute_proximity(
            self.agent_pos,
            self.agent_yaw,
            obstacle_segments=self.wall_segments,
            all_agent_pos=self.agent_pos,
            robot_radius=cfg.robot_radius,
        )
        light_vals, light_value, light_angle = self._compute_light_readings()
        ztilde, rab_proj, rab_attr_x, rab_attr_y = self.sensors.compute_rab(
            self.agent_pos,
            self.agent_yaw,
            obstacle_segments=self.wall_segments,
        )
        return {
            "prox_vals": prox_vals,
            "prox_value": prox_value,
            "prox_angle": prox_angle,
            "light_vals": light_vals,
            "light_value": light_value,
            "light_angle": light_angle,
            "ztilde": ztilde,
            "rab_proj": rab_proj,
            "rab_attr_x": rab_attr_x,
            "rab_attr_y": rab_attr_y,
        }

    def _update_visual_markers(self):
        """Update robot and heading marker positions from kinematic state.

        Skipped entirely when running headless (no viewport) to avoid
        costly GPU→CPU transfers every step.
        """
        # Skip entirely in headless mode (no viewer)
        try:
            if not self.sim.has_gui():
                return
        except (AttributeError, RuntimeError):
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

        if getattr(self.cfg, "debug_visual_sensors", False):
            self._update_sensor_visual_markers(pos_2d, yaws)

    @staticmethod
    def _line_marker_arrays(
        starts: np.ndarray,
        ends: np.ndarray,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vec = ends - starts
        lengths = np.linalg.norm(vec, axis=1).clip(min=1e-6)
        dirs = vec / lengths[:, None]
        mid = 0.5 * (starts + ends)
        q = np.zeros((len(starts), 4), dtype=np.float32)
        q[:, 0] = math.sqrt(0.5)
        q[:, 1] = -dirs[:, 1] * math.sqrt(0.5)
        q[:, 2] = dirs[:, 0] * math.sqrt(0.5)
        scales = np.column_stack([
            np.full(len(starts), radius, dtype=np.float32),
            np.full(len(starts), radius, dtype=np.float32),
            lengths.astype(np.float32),
        ])
        return mid.astype(np.float32), q, scales

    def _selected_sensor_robot_indices(self) -> np.ndarray:
        N = self.cfg.num_agents
        idx = int(getattr(self.cfg, "sensor_visual_robot_index", -1))
        if idx >= 0:
            return np.array([max(0, min(idx, N - 1))], dtype=np.int64)
        return np.arange(N, dtype=np.int64)

    def _line_blocks_segment_np(self, start: np.ndarray, end: np.ndarray) -> bool:
        ray = end - start
        dist = float(np.linalg.norm(ray))
        if dist <= 1e-6:
            return False
        rdx, rdy = ray[0] / dist, ray[1] / dist
        ox, oy = float(start[0]), float(start[1])
        for ax, ay, bx, by in self.wall_segments:
            sx, sy = bx - ax, by - ay
            denom = rdx * sy - rdy * sx
            if abs(denom) <= 1e-8:
                continue
            t = ((ax - ox) * sy - (ay - oy) * sx) / denom
            u = ((ax - ox) * rdy - (ay - oy) * rdx) / denom
            if 1e-5 < t < dist - 1e-5 and 0.0 <= u <= 1.0:
                return True
        return False

    def _update_sensor_visual_markers(self, pos_2d: np.ndarray, yaws: np.ndarray):
        if self._sensor_line_markers is None or self._sensor_point_markers is None:
            return

        selected = self._selected_sensor_robot_indices()
        prox_vals, _, _ = self.sensors.compute_proximity(
            self.agent_pos[:1],
            self.agent_yaw[:1],
            obstacle_segments=self.wall_segments,
            all_agent_pos=self.agent_pos[:1],
            robot_radius=self.cfg.robot_radius,
        )
        light_vals, _, _ = self._compute_light_readings()
        ground_vals = self._ground_color(self.agent_pos[:1])
        prox_np = prox_vals[0, selected].detach().cpu().numpy()
        light_np = light_vals[0, selected].detach().cpu().numpy()
        ground_np = ground_vals[0, selected].detach().cpu().numpy()

        local_x = self.sensors._cos_a.detach().cpu().numpy()
        local_y = self.sensors._sin_a.detach().cpu().numpy()
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        line_idx: list[int] = []
        points: list[np.ndarray] = []
        point_idx: list[int] = []
        z_ray = self.cfg.robot_height + 0.012

        for row, robot_i in enumerate(selected):
            yaw = yaws[robot_i]
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            dirs_x = local_x * cos_y - local_y * sin_y
            dirs_y = local_x * sin_y + local_y * cos_y
            origin = np.array([pos_2d[robot_i, 0], pos_2d[robot_i, 1], z_ray], dtype=np.float32)

            ground_val = float(ground_np[row, 0])
            ground_marker = 5 if ground_val < 0.25 else (7 if ground_val > 0.75 else 6)
            # Epuck.cs uses one downward tag ray and copies it into all three
            # ground channels; draw three clustered channel dots with that value.
            for off_forward, off_left in [(-0.010, -0.012), (-0.010, 0.0), (-0.010, 0.012)]:
                gx = pos_2d[robot_i, 0] + off_forward * cos_y - off_left * sin_y
                gy = pos_2d[robot_i, 1] + off_forward * sin_y + off_left * cos_y
                points.append(np.array([gx, gy, z_ray + 0.018], dtype=np.float32))
                point_idx.append(ground_marker)

            for sensor_i in range(8):
                hit = prox_np[row, sensor_i] > 1e-4
                length = self.cfg.prox_range * (1.0 - prox_np[row, sensor_i] if hit else 1.0)
                end = origin + np.array(
                    [dirs_x[sensor_i] * length, dirs_y[sensor_i] * length, 0.0],
                    dtype=np.float32,
                )
                starts.append(origin)
                ends.append(end)
                line_idx.append(1 if hit else 0)
                points.append(end)
                point_idx.append(1 if hit else 0)

                light_len = float(np.clip(light_np[row, sensor_i], 0.0, 1.0)) * 0.5
                if light_len > 1e-4:
                    light_end = origin + np.array(
                        [dirs_x[sensor_i] * light_len, dirs_y[sensor_i] * light_len, 0.0],
                        dtype=np.float32,
                    )
                    starts.append(origin)
                    ends.append(light_end)
                    line_idx.append(2)
                    points.append(light_end)
                    point_idx.append(2)

            ring_segments = max(8, int(getattr(self.cfg, "sensor_visual_rab_ring_segments", 48)))
            for k in range(ring_segments):
                a = 2.0 * math.pi * k / ring_segments
                points.append(np.array([
                    pos_2d[robot_i, 0] + self.cfg.rab_range * math.cos(a),
                    pos_2d[robot_i, 1] + self.cfg.rab_range * math.sin(a),
                    z_ray,
                ], dtype=np.float32))
                point_idx.append(3)

            for other_i in range(self.cfg.num_agents):
                if other_i == robot_i:
                    continue
                delta = pos_2d[other_i] - pos_2d[robot_i]
                dist = float(np.linalg.norm(delta))
                if dist >= self.cfg.rab_range or self._line_blocks_segment_np(pos_2d[robot_i], pos_2d[other_i]):
                    continue
                neighbor = np.array([pos_2d[other_i, 0], pos_2d[other_i, 1], z_ray], dtype=np.float32)
                starts.append(origin)
                ends.append(neighbor)
                line_idx.append(3)
                points.append(neighbor)
                point_idx.append(4)

        if not starts:
            starts = [np.array([0.0, 0.0, -10.0], dtype=np.float32)]
            ends = [np.array([0.0, 0.0, -10.0], dtype=np.float32)]
            line_idx = [0]
        translations, orientations, scales = self._line_marker_arrays(
            np.stack(starts),
            np.stack(ends),
            0.003,
        )
        self._sensor_line_markers.visualize(
            translations=translations,
            orientations=orientations,
            scales=scales,
            marker_indices=np.array(line_idx, dtype=np.int32),
        )

        if not points:
            points = [np.array([0.0, 0.0, -10.0], dtype=np.float32)]
            point_idx = [0]
        self._sensor_point_markers.visualize(
            translations=np.stack(points).astype(np.float32),
            marker_indices=np.array(point_idx, dtype=np.int32),
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
        return self.cfg.arena_circumradius * math.cos(math.pi / self.cfg.arena_num_sides)

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
        self._low_level_action_dirty = True

    def _apply_action(self) -> None:
        """Convert actions → wheel velocities → kinematic integration.

        Dandelion: actions are directly (left_vel, right_vel).
        ACB variants: action is a module index; run the selected behaviour module.
        """
        cfg = self.cfg
        E, N = self.num_envs, cfg.num_agents
        dev = self.device

        if not self._low_level_action_dirty:
            left_vel = self._cached_left_vel
            right_vel = self._cached_right_vel
        elif cfg.discrete_actions:
            # ── ACB discrete dispatch ─────────────────────────────
            # Stack module indices: (E, N)
            module_ids = torch.stack(
                [self._raw_actions[a].squeeze(-1).long() for a in cfg.possible_agents],
                dim=1,
            )  # (E, N)

            # Use the exact sensor snapshot that produced the policy observation.
            # This also ensures one packet-loss sample is shared by the selector
            # and the behavior module for the same control cycle.
            if self._sensor_cache is None:
                self._sensor_cache = self._compute_sensor_bundle()
            sensors = self._sensor_cache

            left_vel, right_vel = self.behavior_modules.dispatch(
                module_ids,
                sensors["prox_value"], sensors["prox_angle"],
                sensors["light_value"], sensors["light_angle"],
                sensors["rab_attr_x"], sensors["rab_attr_y"],
                self._cached_left_vel, self._cached_right_vel,
            )
        else:
            # ── Dandelion continuous ──────────────────────────────
            # Stack (E, N, 2): [left_vel, right_vel]
            # Actions are in NORMALIZED [-1, 1] space (matching ML-Agents).
            # Unity's ContinuousActionOutputApplier clips to [-1, 1]
            # before the C# env scales by MaxVelocity.
            actions_stacked = torch.stack(
                [self._raw_actions[a] for a in cfg.possible_agents],
                dim=1,
            )  # (E, N, 2)
            # Clamp to [-1, 1] then scale to wheel velocity
            actions_clamped = actions_stacked.clamp(-1.0, 1.0)
            left_vel = actions_clamped[:, :, 0] * cfg.max_wheel_speed
            right_vel = actions_clamped[:, :, 1] * cfg.max_wheel_speed

        if self._low_level_action_dirty:
            self._cached_left_vel = left_vel
            self._cached_right_vel = right_vel
            self._low_level_action_dirty = False

        # ── Differential-drive kinematic integration ──────────────
        dt = cfg.sim.dt
        prev_pos = self.agent_pos.clone()
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
        self._resolve_collisions(prev_pos)

        # Positions and headings changed; the next observation must be sampled
        # from this new state, never from the pre-action state.
        self._sensor_cache = None

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
        normals_list = []
        points_list = []
        for ax, ay, bx, by in self.arena_wall_segments:
            mx = 0.5 * (ax + bx)
            my = 0.5 * (ay + by)
            norm = math.sqrt(mx * mx + my * my) + 1e-12
            # Inward normal points from the wall midpoint toward the arena center.
            nx = -mx / norm
            ny = -my / norm
            wx = mx
            wy = my
            normals_list.append([nx, ny])
            points_list.append([wx, wy])

        normals = torch.tensor(normals_list, dtype=torch.float32, device=self.device)  # (n, 2)
        points = torch.tensor(points_list, dtype=torch.float32, device=self.device)    # (n, 2)
        return normals, points

    def _resolve_collisions(self, prev_pos: torch.Tensor | None = None):
        """Resolve wall and robot contacts with a final wall-safe projection."""
        iterations = max(1, int(getattr(self.cfg, "collision_solver_iterations", 4)))

        self._resolve_wall_collisions()
        if prev_pos is not None:
            self._prevent_internal_wall_crossing(prev_pos)
        self._resolve_internal_wall_capsules(prev_pos)
        self._resolve_gate_wall_collisions()

        for _ in range(iterations):
            before_contacts = self.agent_pos.clone()
            self._resolve_robot_collisions()
            self._resolve_wall_collisions()
            self._prevent_internal_wall_crossing(before_contacts)
            self._resolve_internal_wall_capsules(before_contacts)
            self._resolve_gate_wall_collisions()

        self._resolve_wall_collisions()
        if prev_pos is not None:
            self._prevent_internal_wall_crossing(prev_pos)
        self._resolve_internal_wall_capsules(prev_pos)
        self._resolve_gate_wall_collisions()

    def _prevent_internal_wall_crossing(self, prev_pos: torch.Tensor):
        """Undo complete crossings through line-segment internal walls.

        Position-only push-out can miss a wall if crowd pressure moves a robot
        from one side of a thin wall to the other in one solver pass.  This
        swept side test puts the robot back on the side it occupied before
        the move/contact.
        """
        if not self.gate_wall_segments:
            return

        clearance = (
            self.cfg.robot_radius
            + 0.5 * float(getattr(self.cfg, "shelter_wall_thickness", 0.0))
            + float(getattr(self.cfg, "wall_contact_epsilon", 1e-4))
        )
        eps = 1e-8

        for ax, ay, bx, by in self.gate_wall_segments:
            abx = bx - ax
            aby = by - ay
            length_sq = abx * abx + aby * aby
            if length_sq <= eps:
                continue

            length = math.sqrt(length_sq)
            normal = torch.tensor(
                (-aby / length, abx / length),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )
            anchor = torch.tensor(
                (ax, ay),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )
            tangent = torch.tensor(
                (abx, aby),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )

            prev_rel = prev_pos - anchor.view(1, 1, 2)
            curr_rel = self.agent_pos - anchor.view(1, 1, 2)
            prev_signed = (prev_rel * normal.view(1, 1, 2)).sum(dim=-1)
            curr_signed = (curr_rel * normal.view(1, 1, 2)).sum(dim=-1)

            denom = prev_signed - curr_signed
            safe_denom = torch.where(denom.abs() > eps, denom, torch.ones_like(denom))
            sweep_t = torch.where(
                denom.abs() > eps,
                prev_signed / safe_denom,
                torch.zeros_like(denom),
            )
            intersection = prev_pos + (self.agent_pos - prev_pos) * sweep_t.unsqueeze(-1)
            wall_u = (
                ((intersection - anchor.view(1, 1, 2)) * tangent.view(1, 1, 2)).sum(dim=-1)
                / length_sq
            )

            crossed = (
                (prev_signed * curr_signed < 0.0)
                & (sweep_t >= 0.0)
                & (sweep_t <= 1.0)
                & (wall_u >= 0.0)
                & (wall_u <= 1.0)
            )
            if not crossed.any():
                continue

            prev_side = torch.sign(prev_signed)
            prev_side = torch.where(prev_side == 0.0, -torch.sign(curr_signed), prev_side)
            prev_side = torch.where(prev_side == 0.0, torch.ones_like(prev_side), prev_side)
            desired_signed = prev_side * clearance
            correction = (desired_signed - curr_signed).unsqueeze(-1) * normal.view(1, 1, 2)
            corrected_pos = self.agent_pos + correction
            self.agent_pos = torch.where(crossed.unsqueeze(-1), corrected_pos, self.agent_pos)

    def _resolve_internal_wall_capsules(self, prev_pos: torch.Tensor | None = None):
        """Resolve finite internal wall segments as capsules with wall thickness."""
        if not self.gate_wall_segments:
            return

        wall_thickness = float(getattr(
            self.cfg,
            "shelter_wall_thickness",
            getattr(self.cfg, "internal_wall_thickness", 0.01),
        ))
        clearance = (
            self.cfg.robot_radius
            + 0.5 * wall_thickness
            + float(getattr(self.cfg, "wall_contact_epsilon", 1e-4))
        )
        eps = 1e-8

        for ax, ay, bx, by in self.gate_wall_segments:
            abx = bx - ax
            aby = by - ay
            length_sq = abx * abx + aby * aby
            if length_sq <= eps:
                continue

            length = math.sqrt(length_sq)
            normal = torch.tensor(
                (-aby / length, abx / length),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )
            anchor = torch.tensor(
                (ax, ay),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )
            tangent = torch.tensor(
                (abx, aby),
                dtype=self.agent_pos.dtype,
                device=self.device,
            )

            rel = self.agent_pos - anchor.view(1, 1, 2)
            u = (rel * tangent.view(1, 1, 2)).sum(dim=-1) / length_sq
            u_clamped = u.clamp(0.0, 1.0)
            closest = anchor.view(1, 1, 2) + u_clamped.unsqueeze(-1) * tangent.view(1, 1, 2)
            delta = self.agent_pos - closest
            raw_dist = torch.linalg.norm(delta, dim=-1)
            dist = raw_dist.clamp_min(eps)

            curr_signed = (rel * normal.view(1, 1, 2)).sum(dim=-1)
            if prev_pos is not None:
                prev_rel = prev_pos - anchor.view(1, 1, 2)
                side = torch.sign((prev_rel * normal.view(1, 1, 2)).sum(dim=-1))
                side = torch.where(side == 0.0, torch.sign(curr_signed), side)
            else:
                side = torch.sign(curr_signed)
            side = torch.where(side == 0.0, torch.ones_like(side), side)

            side_dir = side.unsqueeze(-1) * normal.view(1, 1, 2)
            radial_dir = torch.where(
                (raw_dist > eps).unsqueeze(-1),
                delta / dist.unsqueeze(-1),
                side_dir,
            )
            on_segment_span = (u >= 0.0) & (u <= 1.0)
            push_dir = torch.where(on_segment_span.unsqueeze(-1), side_dir, radial_dir)

            penetration = clearance - dist
            near = penetration > 0.0
            corrected = self.agent_pos + penetration.clamp_min(0.0).unsqueeze(-1) * push_dir
            self.agent_pos = torch.where(near.unsqueeze(-1), corrected, self.agent_pos)

    def _resolve_wall_collisions(self):
        """Push robots inside the dodecagonal arena boundary (fully vectorized)."""
        r = (
            self.cfg.robot_radius
            + 0.5 * float(getattr(self.cfg, "arena_wall_thickness", 0.01))
            + float(getattr(self.cfg, "wall_contact_epsilon", 1e-4))
        )
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

        if self._sensor_cache is None:
            self._sensor_cache = self._compute_sensor_bundle()
        cache = self._sensor_cache
        prox_vals = cache["prox_vals"]
        light_vals = cache["light_vals"]
        ztilde = cache["ztilde"]
        rab_proj = cache["rab_proj"]

        ground = self._ground_color(self.agent_pos)  # (E, N, 3)

        if (
            cfg.variant in ("dandelion", "daisy")
            or cfg.full_policy_observations
        ):
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
        """Compute team reward via ground COLOR TRANSITIONS (matching Unity exactly).

        Unity DirGateEnvController.cs detects:
          BLACK → WHITE  →  +1  (correct traversal: robot exits corridor into gate)
          WHITE → BLACK  →  −1  (reverse traversal: robot enters corridor from gate)

        All other transitions (grey↔white, grey↔black, same→same) give 0 reward.
        """
        cfg = self.cfg

        # Current ground color scalar per agent: 0=black, 0.5=grey, 1=white
        curr_color = self._ground_color(self.agent_pos)[:, :, 0]  # (E, N)
        prev_color = self.prev_ground_color                       # (E, N)

        # Discretize: black < 0.25, grey ∈ [0.25, 0.75], white > 0.75
        prev_is_black = (prev_color < 0.25)
        prev_is_white = (prev_color > 0.75)
        curr_is_black = (curr_color < 0.25)
        curr_is_white = (curr_color > 0.75)

        # K⁺: BLACK → WHITE (correct traversal, going south through gate)
        black_to_white = prev_is_black & curr_is_white
        k_plus = black_to_white.float().sum(dim=1)                # (E,)

        # K⁻: WHITE → BLACK (reverse traversal, going north into corridor)
        white_to_black = prev_is_white & curr_is_black
        k_minus = white_to_black.float().sum(dim=1)               # (E,)

        # Update color tracking
        self.prev_ground_color = curr_color.clone()

        # Team reward
        reward = k_plus - k_minus                                 # (E,)
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
        time_out = self.episode_length_buf >= self.max_episode_length
        if time_out.any():
            terminal_state = self.get_critic_state()
            self.completed_terminal_critic_state[time_out] = terminal_state[time_out]

        terminated = {agent: torch.zeros_like(time_out) for agent in self.cfg.possible_agents}
        truncated = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, truncated

    # ──────────────────────────────────────────────────────────────
    #  Reset
    # ──────────────────────────────────────────────────────────────

    def _sample_spawn_positions(self, n_reset: int, n_agents: int) -> torch.Tensor:
        """Sample Unity-style spawn positions, scaled into Isaac meters."""
        cfg = self.cfg
        cx, cy = cfg.spawn_area_center
        sx, sy = cfg.spawn_area_size
        center = torch.tensor((cx, cy), dtype=torch.float32, device=self.device)

        def sample_rect(shape: tuple[int, int]) -> torch.Tensor:
            rand = torch.rand(*shape, 2, device=self.device) - 0.5
            scale = torch.tensor((sx, sy), dtype=torch.float32, device=self.device)
            return center.view(1, 1, 2) + rand * scale.view(1, 1, 2)

        pos = sample_rect((n_reset, n_agents))
        radius = float(getattr(cfg, "spawn_circle_radius", 0.0))
        if radius <= 0.0:
            return pos

        max_attempts = int(getattr(cfg, "spawn_max_attempts", 100))
        for _ in range(max_attempts):
            rel = pos - center.view(1, 1, 2)
            invalid = torch.linalg.norm(rel, dim=-1) > radius
            if not invalid.any():
                break
            replacement = sample_rect((n_reset, n_agents))
            pos = torch.where(invalid.unsqueeze(-1), replacement, pos)
        return pos

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        if isinstance(env_ids, torch.Tensor):
            idx = env_ids.to(device=self.device, dtype=torch.long)
        else:
            idx = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        N = self.cfg.num_agents

        # Snapshot completed reward before resetting
        self.completed_group_reward[idx] = self._episode_group_reward[idx]
        self._episode_group_reward[idx] = 0.0

        # Vectorized Unity-style spawn rectangle for len(env_ids) envs x N agents.
        n_reset = len(env_ids)
        self.agent_pos[idx] = self._sample_spawn_positions(n_reset, N)
        self.agent_yaw[idx] = torch.rand(n_reset, N, device=self.device) * 2 * math.pi - math.pi

        self._resolve_collisions()

        # Reset ground-color tracking for crossing detection
        reset_color = self._ground_color(self.agent_pos[idx])[:, :, 0]  # (len(idx), N)
        self.prev_ground_color[idx] = reset_color

        # Reset exploration state for these envs
        mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        mask[idx] = True
        self.behavior_modules.reset_exploration_state(mask)
        self._sensor_cache = None
        self._low_level_action_dirty = True

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
            self.cfg.critic_state_radius,
            self.critic_reference_dir,
        )
