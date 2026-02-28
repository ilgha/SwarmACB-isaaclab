#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Manual control in Isaac Sim — drive one e-puck in the Directional Gate arena.

3D visualisation (Isaac Sim viewport) with on-screen debug text.
You control robot #0 via keyboard; the remaining 19 robots run
a selected behaviour module (default: stop).

Controls:
  Z / UP    : Forward
  S / DOWN  : Backward
  Q / LEFT  : Turn left
  D / RIGHT : Turn right
  A         : Stop
  NUMPAD 0  : Others → Exploration
  NUMPAD 1  : Others → Stop
  NUMPAD 2  : Others → Phototaxis
  NUMPAD 3  : Others → Anti-phototaxis
  NUMPAD 4  : Others → Attraction
  NUMPAD 5  : Others → Repulsion
  R         : Reset episode
  ESC       : Quit

Usage:
  python scripts/manual_control_isaac.py
  python scripts/manual_control_isaac.py --num-agents 10
  python scripts/manual_control_isaac.py --speed 0.10
"""

from __future__ import annotations

import argparse

# ── Isaac Lab bootstrap (MUST happen before other Isaac Lab imports) ──
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SwarmACB — Manual control (Isaac Sim)")
parser.add_argument("--num-agents", type=int, default=20)
parser.add_argument("--speed", type=float, default=0.08, help="Keyboard control speed (m/s)")
parser.add_argument("--others-explore", action="store_true",
                    help="Start other robots in exploration mode instead of stop")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab & Omni packages ─────────────────────

import math
import weakref

import carb
import numpy as np
import omni
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaacsim.core.api.simulation_context import SimulationContext

# ── Import env components (bypass package chain) ────────────────────
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_EPUCK_DIR = os.path.join(
    _PROJECT_ROOT, "source", "SwarmACB_isaac", "SwarmACB_isaac",
    "tasks", "direct", "epuck",
)
sys.path.insert(0, _EPUCK_DIR)

from epuck_sensors import EpuckSensors
from behavior_modules import BehaviorModules


# =====================================================================
#  Standalone kinematic env (same as pygame version)
# =====================================================================

class StandaloneDGTEnv:
    """Lightweight DGT env (pure-PyTorch kinematic sim, no USD physics)."""

    def __init__(self, num_agents: int = 20, device: str = "cpu"):
        self.device = torch.device(device)
        self.E = 1
        self.N = num_agents

        # ── Arena geometry (dodecagon 4.91 m²) ─────────────────
        self.arena_n_sides = 12
        self.arena_circumradius = math.sqrt(
            2 * 4.91 / (12 * math.sin(2 * math.pi / 12))
        )

        # ── Robot params ────────────────────────────────────────
        self.robot_radius = 0.035
        self.max_speed = 0.12
        self.wheelbase = 0.053
        self.dt = 0.1

        # ── Ground zones ────────────────────────────────────────
        self.corridor_width = 0.50
        self.corridor_length = 1.06
        self.gate_width = 0.45
        self.gate_length = 0.33
        self.side_wall_length = 0.50

        inradius = self.arena_circumradius * math.cos(math.pi / self.arena_n_sides)
        self.north_inradius = inradius
        self.corr_south = inradius - self.corridor_length
        self.gate_south = self.corr_south - self.gate_length
        self.corr_hw = self.corridor_width / 2.0
        self.gate_hw = self.gate_width / 2.0

        # ── Light source ────────────────────────────────────────
        self.light_pos = torch.tensor([0.0, -1.4], device=self.device)

        # ── State ───────────────────────────────────────────────
        self.pos = torch.zeros(self.E, self.N, 2, device=self.device)
        self.yaw = torch.zeros(self.E, self.N, device=self.device)
        self.prev_y = torch.zeros(self.E, self.N, device=self.device)
        self.gate_trigger_y = self.gate_south + self.gate_length / 2.0

        # ── Sensors ─────────────────────────────────────────────
        self.sensors = EpuckSensors(
            prox_range=0.10, rab_range=0.20, light_threshold=0.2, device=device,
        )
        self.behavior_modules = BehaviorModules(max_speed=self.max_speed, device=device)
        self.behavior_modules.init_exploration_state(self.E, self.N)

        # ── Walls ───────────────────────────────────────────────
        self.arena_wall_segments = self._build_walls()
        self.gate_wall_segments = self._build_gate_walls()
        self.wall_segments = self.arena_wall_segments + self.gate_wall_segments

        # ── Reward ──────────────────────────────────────────────
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.k_plus_total = 0
        self.k_minus_total = 0

        self.reset()

    # ── Builder helpers ─────────────────────────────────────────

    def _build_walls(self):
        R = self.arena_circumradius
        n = self.arena_n_sides
        verts = []
        for i in range(n):
            a = 2 * math.pi * i / n + math.pi / n
            verts.append((R * math.cos(a), R * math.sin(a)))
        segs = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            segs.append((ax, ay, bx, by))
        return segs

    def _build_gate_walls(self):
        hw = self.corr_hw
        gs = self.gate_south
        wl = self.side_wall_length
        return [
            (-hw, gs, -hw, gs + wl),
            (hw, gs, hw, gs + wl),
        ]

    def reset(self):
        inradius = self.arena_circumradius * math.cos(math.pi / self.arena_n_sides)
        safe = inradius - self.robot_radius * 2
        r = torch.sqrt(torch.rand(self.N)) * safe
        th = torch.rand(self.N) * 2 * math.pi
        self.pos[0, :, 0] = r * torch.cos(th)
        self.pos[0, :, 1] = r * torch.sin(th)
        self.yaw[0] = torch.rand(self.N) * 2 * math.pi - math.pi
        self.prev_y[0] = self.pos[0, :, 1]
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.k_plus_total = 0
        self.k_minus_total = 0
        self.behavior_modules.reset_exploration_state(
            torch.ones(self.E, dtype=torch.bool, device=self.device))

    def step(self, left_vel, right_vel):
        lv = left_vel.clamp(-self.max_speed, self.max_speed)
        rv = right_vel.clamp(-self.max_speed, self.max_speed)
        dx, dy, dyaw = EpuckSensors.differential_drive(
            lv, rv, self.yaw, self.wheelbase, self.dt,
        )
        self.pos[:, :, 0] += dx
        self.pos[:, :, 1] += dy
        self.yaw += dyaw
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))
        self._resolve_walls()
        self._resolve_gate_walls()
        self._resolve_robots()

        # Gate reward
        curr_y = self.pos[0, :, 1]
        curr_x = self.pos[0, :, 0]
        prev_y = self.prev_y[0]
        trig = self.gate_trigger_y
        hw = self.gate_hw
        in_gate = curr_x.abs() < hw
        n2s = (prev_y > trig) & (curr_y <= trig) & in_gate
        s2n = (prev_y <= trig) & (curr_y > trig) & in_gate
        k_plus = n2s.float().sum().item()
        k_minus = s2n.float().sum().item()
        self.prev_y[0] = curr_y.clone()
        self.step_reward = k_plus - k_minus
        self.episode_reward += self.step_reward
        self.k_plus_total += int(k_plus)
        self.k_minus_total += int(k_minus)
        self.step_count += 1

    def compute_obs_robot0(self):
        prox_vals, prox_value, prox_angle = self.sensors.compute_proximity(
            self.pos, self.yaw, self.wall_segments, self.pos, self.robot_radius,
        )
        light_vals, light_value, light_angle = self.sensors.compute_light(
            self.pos, self.yaw, self.light_pos,
        )
        ground = self._ground_3ch(self.pos)
        ztilde, rab_proj = self.sensors.compute_rab(self.pos, self.yaw)
        dx = self.pos[0, :, 0] - self.pos[0, 0, 0]
        dy = self.pos[0, :, 1] - self.pos[0, 0, 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        not_self = torch.ones(self.N, dtype=torch.bool)
        not_self[0] = False
        in_range = (dist < self.sensors.rab_range) & not_self
        gv = ground[0, 0].cpu().tolist()
        return {
            "prox_val": prox_value[0, 0].item(),
            "prox_angle": math.degrees(prox_angle[0, 0].item()),
            "light_val": light_value[0, 0].item(),
            "light_angle": math.degrees(light_angle[0, 0].item()),
            "ground_3": gv,
            "ztilde": ztilde[0, 0].item(),
            "n_neighbors": int(in_range.sum().item()),
        }

    def _ground_scalar(self, pos_2d):
        x, y = pos_2d[:, 0], pos_2d[:, 1]
        c = torch.full_like(x, 0.5)
        in_gate = (x.abs() < self.gate_hw) & (y > self.gate_south) & (y < self.corr_south)
        c = torch.where(in_gate, torch.ones_like(c), c)
        in_corr = (x.abs() < self.corr_hw) & (y >= self.corr_south) & (y < self.north_inradius)
        c = torch.where(in_corr, torch.zeros_like(c), c)
        return c

    def _ground_3ch(self, pos):
        s = self._ground_scalar(pos[0])
        return s.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)

    def _resolve_gate_walls(self):
        r = self.robot_radius
        hw = self.corr_hw
        gs = self.gate_south
        wall_top = gs + self.side_wall_length
        px = self.pos[:, :, 0]
        py = self.pos[:, :, 1]
        in_wall_y = (py > gs) & (py < wall_top)
        dx_l = px - (-hw)
        pen_l = r - dx_l.abs()
        near_l = (pen_l > 0) & in_wall_y & (px < 0)
        sign_l = torch.sign(dx_l)
        sign_l = torch.where(sign_l == 0, -torch.ones_like(sign_l), sign_l)
        self.pos[:, :, 0] = torch.where(near_l, -hw + sign_l * r, self.pos[:, :, 0])
        px = self.pos[:, :, 0]
        dx_r = px - hw
        pen_r = r - dx_r.abs()
        near_r = (pen_r > 0) & in_wall_y & (px > 0)
        sign_r = torch.sign(dx_r)
        sign_r = torch.where(sign_r == 0, torch.ones_like(sign_r), sign_r)
        self.pos[:, :, 0] = torch.where(near_r, hw + sign_r * r, self.pos[:, :, 0])

    def _resolve_walls(self):
        R = self.arena_circumradius
        r = self.robot_radius
        n = self.arena_n_sides
        inradius = R * math.cos(math.pi / n)
        for i in range(n):
            a1 = 2 * math.pi * i / n + math.pi / n
            a2 = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
            mid = (a1 + a2) / 2.0
            nx, ny = -math.cos(mid), -math.sin(mid)
            wx = inradius * math.cos(mid)
            wy = inradius * math.sin(mid)
            dx = self.pos[:, :, 0] - wx
            dy = self.pos[:, :, 1] - wy
            sd = dx * nx + dy * ny
            pen = r - sd
            push = pen > 0
            self.pos[:, :, 0] += torch.where(push, pen * nx, torch.zeros_like(pen))
            self.pos[:, :, 1] += torch.where(push, pen * ny, torch.zeros_like(pen))

    def _resolve_robots(self):
        r = self.robot_radius
        md = 2 * r
        N = self.N
        dx = self.pos[:, :, 0].unsqueeze(2) - self.pos[:, :, 0].unsqueeze(1)
        dy = self.pos[:, :, 1].unsqueeze(2) - self.pos[:, :, 1].unsqueeze(1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1).unsqueeze(0)
        overlap = (md - dist).clamp(min=0) * mask.float()
        if overlap.sum() == 0:
            return
        nx = dx / (dist + 1e-8)
        ny = dy / (dist + 1e-8)
        self.pos[:, :, 0] += (overlap * nx * 0.5).sum(2)
        self.pos[:, :, 0] -= (overlap * nx * 0.5).sum(1)
        self.pos[:, :, 1] += (overlap * ny * 0.5).sum(2)
        self.pos[:, :, 1] -= (overlap * ny * 0.5).sum(1)


# =====================================================================
#  Isaac Sim visual scene builder
# =====================================================================

def _build_arena_visuals(env: StandaloneDGTEnv):
    """Spawn static visual geometry for the arena in USD."""
    R = env.arena_circumradius
    n = env.arena_n_sides
    wall_h = 0.08
    wall_thick = 0.01

    # ── Grey arena floor (large rectangle extending beyond arena) ──
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

    # ── White gate zone ─────────────────────────────────────────
    gate_w = env.gate_hw * 2
    gate_l = env.corr_south - env.gate_south
    gate_cy = (env.gate_south + env.corr_south) / 2.0
    gate_cfg = sim_utils.CuboidCfg(
        size=(gate_w, gate_l, 0.003),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
    )
    gate_cfg.func("/World/Arena/GateZone", gate_cfg, translation=(0.0, gate_cy, 0.002))

    # ── Black corridor zone ─────────────────────────────────────
    corr_w = env.corr_hw * 2
    corr_l = env.north_inradius - env.corr_south
    corr_cy = (env.corr_south + env.north_inradius) / 2.0
    corr_cfg = sim_utils.CuboidCfg(
        size=(corr_w, corr_l, 0.003),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
    )
    corr_cfg.func("/World/Arena/CorridorZone", corr_cfg, translation=(0.0, corr_cy, 0.003))

    # ── Arena dodecagonal walls (12 thin cuboid segments) ───────
    wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.70, 0.40))
    for i in range(n):
        a1 = 2 * math.pi * i / n + math.pi / n
        a2 = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
        ax, ay = R * math.cos(a1), R * math.sin(a1)
        bx, by = R * math.cos(a2), R * math.sin(a2)
        cx = (ax + bx) / 2.0
        cy = (ay + by) / 2.0
        seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        seg_angle = math.atan2(by - ay, bx - ax)

        wall_cfg = sim_utils.CuboidCfg(
            size=(seg_len, wall_thick, wall_h),
            visual_material=wall_mat,
        )
        # Orientation: rotate around Z axis
        qw = math.cos(seg_angle / 2)
        qz = math.sin(seg_angle / 2)
        wall_cfg.func(
            f"/World/Arena/Wall_{i}", wall_cfg,
            translation=(cx, cy, wall_h / 2),
            orientation=(qw, 0.0, 0.0, qz),
        )

    # ── Gate side walls (two vertical barriers) ─────────────────
    gate_wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.39, 0.20))
    hw = env.corr_hw
    gs = env.gate_south
    wl = env.side_wall_length
    wcy = gs + wl / 2.0
    for side_i, sx in enumerate([-hw, hw]):
        gw_cfg = sim_utils.CuboidCfg(
            size=(wall_thick, wl, wall_h),
            visual_material=gate_wall_mat,
        )
        gw_cfg.func(
            f"/World/Arena/GateWall_{side_i}", gw_cfg,
            translation=(sx, wcy, wall_h / 2),
        )

    # ── Light source indicator (red sphere at south) ────────────
    light_cfg = sim_utils.SphereCfg(
        radius=0.04,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15)),
    )
    light_cfg.func("/World/Arena/LightIndicator", light_cfg,
                    translation=(0.0, -1.4, 0.04))

    # ── N / S labels (small coloured cuboids as direction markers) ──
    n_marker_cfg = sim_utils.SphereCfg(
        radius=0.03,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
    )
    n_marker_cfg.func("/World/Arena/NorthMarker", n_marker_cfg,
                       translation=(0.0, R + 0.1, 0.03))
    s_marker_cfg = sim_utils.SphereCfg(
        radius=0.03,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.3)),
    )
    s_marker_cfg.func("/World/Arena/SouthMarker", s_marker_cfg,
                       translation=(0.0, -(R + 0.1), 0.03))


def _create_robot_markers(env: StandaloneDGTEnv) -> VisualizationMarkers:
    """Create instanced robot markers using VisualizationMarkers.

    Two prototypes:
      0 = controlled robot (green)
      1 = other robot (blue)
    """
    r = env.robot_radius
    h = 0.05  # robot height

    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Robots",
        markers={
            "controlled": sim_utils.CylinderCfg(
                radius=r,
                height=h,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.86, 0.33),
                ),
            ),
            "other": sim_utils.CylinderCfg(
                radius=r,
                height=h,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.31, 0.55, 0.86),
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _create_heading_markers(env: StandaloneDGTEnv) -> VisualizationMarkers:
    """Create small arrow-like cone markers for heading indication.

    Two prototypes:
      0 = controlled heading (yellow)
      1 = other heading (light grey)
    """
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/Headings",
        markers={
            "ctrl_heading": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 1.0, 0.3),
                ),
            ),
            "other_heading": sim_utils.SphereCfg(
                radius=0.008,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.7, 0.7, 0.78),
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg)


# =====================================================================
#  Keyboard handler
# =====================================================================

MODULE_NAMES = [
    "Exploration", "Stop", "Phototaxis",
    "Anti-photo", "Attraction", "Repulsion",
]

class KeyboardController:
    """Capture keyboard state via carb.input for manual robot control."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Pressed-keys set
        self._pressed: set[str] = set()
        # Events consumed once per frame
        self._events: list[str] = []

        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *a, obj=weakref.proxy(self): obj._on_key(event, *a),
        )

    def _on_key(self, event, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(name)
            self._events.append(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(name)
        return True

    def is_held(self, *keys: str) -> bool:
        return any(k in self._pressed for k in keys)

    def pop_events(self) -> list[str]:
        evts = self._events[:]
        self._events.clear()
        return evts

    def destroy(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)


# =====================================================================
#  Main loop
# =====================================================================

def main():
    N = args.num_agents
    speed = args.speed

    # ── Simulation context ────────────────────────────────────────
    sim = SimulationContext(physics_dt=0.1, rendering_dt=0.1)

    # ── Lighting ──────────────────────────────────────────────────
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.80, 0.80, 0.80))
    light_cfg.func("/World/DomeLight", light_cfg)

    # ── Ground plane ──────────────────────────────────────────────
    spawn_ground_plane("/World/GroundPlane", GroundPlaneCfg())

    # ── Kinematic env ─────────────────────────────────────────────
    env = StandaloneDGTEnv(num_agents=N, device="cpu")

    # ── Visual scene ──────────────────────────────────────────────
    _build_arena_visuals(env)
    robot_markers = _create_robot_markers(env)
    heading_markers = _create_heading_markers(env)

    # ── Keyboard ──────────────────────────────────────────────────
    kb = KeyboardController()
    others_module = 0 if args.others_explore else 1  # 0=explore, 1=stop

    # ── Let the sim initialise ────────────────────────────────────
    sim.reset()

    # Marker index arrays (robot 0 = prototype 0, rest = prototype 1)
    robot_proto = np.array([0] + [1] * (N - 1), dtype=np.int32)
    heading_proto = np.array([0] + [1] * (N - 1), dtype=np.int32)

    print("\n" + "=" * 60)
    print("  SwarmACB — Manual Control (Isaac Sim)")
    print("  Robot #0 = GREEN   |   Others = BLUE")
    print("  Z/↑=fwd  S/↓=bwd  Q/←=left  D/→=right  A=stop")
    print("  Numpad 0-5: set others' behaviour module")
    print("  R=reset  ESC=quit")
    print("=" * 60 + "\n")

    step_counter = 0

    while simulation_app.is_running():
        # ── Handle keyboard events ────────────────────────────────
        for evt in kb.pop_events():
            if evt == "ESCAPE":
                simulation_app.close()
                return
            elif evt == "R":
                env.reset()
                step_counter = 0
                print("[RESET] Episode reset")
            elif evt == "NUMPAD_0":
                others_module = 0
                print(f"[MODULE] Others → {MODULE_NAMES[0]}")
            elif evt == "NUMPAD_1":
                others_module = 1
                print(f"[MODULE] Others → {MODULE_NAMES[1]}")
            elif evt == "NUMPAD_2":
                others_module = 2
                print(f"[MODULE] Others → {MODULE_NAMES[2]}")
            elif evt == "NUMPAD_3":
                others_module = 3
                print(f"[MODULE] Others → {MODULE_NAMES[3]}")
            elif evt == "NUMPAD_4":
                others_module = 4
                print(f"[MODULE] Others → {MODULE_NAMES[4]}")
            elif evt == "NUMPAD_5":
                others_module = 5
                print(f"[MODULE] Others → {MODULE_NAMES[5]}")

        # ── Keyboard → wheel velocities for robot 0 ──────────────
        lv0, rv0 = 0.0, 0.0
        if kb.is_held("Z", "UP"):
            lv0, rv0 = speed, speed
        if kb.is_held("S", "DOWN"):
            lv0, rv0 = -speed, -speed
        if kb.is_held("Q", "LEFT"):
            lv0 -= speed * 0.5
            rv0 += speed * 0.5
        if kb.is_held("D", "RIGHT"):
            lv0 += speed * 0.5
            rv0 -= speed * 0.5
        if kb.is_held("A"):
            lv0, rv0 = 0.0, 0.0

        # Build velocity tensors
        left = torch.zeros(1, N)
        right = torch.zeros(1, N)
        left[0, 0] = lv0
        right[0, 0] = rv0

        # ── Others: run selected behaviour module ─────────────────
        if N > 1:
            prox_v, prox_val, prox_ang = env.sensors.compute_proximity(
                env.pos, env.yaw, env.wall_segments, env.pos, env.robot_radius,
            )
            module_ids = torch.full((1, N), others_module, dtype=torch.long)
            module_ids[0, 0] = 1  # robot 0 overridden by keyboard
            light_v, light_val, light_ang = env.sensors.compute_light(
                env.pos, env.yaw, env.light_pos,
            )
            zt, rp = env.sensors.compute_rab(env.pos, env.yaw)
            rab_ang = torch.atan2(rp[:, :, 1] + rp[:, :, 3], rp[:, :, 0] + rp[:, :, 2])
            el, er = env.behavior_modules.dispatch(
                module_ids, prox_val, prox_ang, light_val, light_ang, zt, rab_ang,
            )
            left[0, 1:] = el[0, 1:]
            right[0, 1:] = er[0, 1:]

        # ── Step kinematic env ────────────────────────────────────
        env.step(left, right)
        step_counter += 1

        # ── Update robot markers ──────────────────────────────────
        pos_2d = env.pos[0].cpu().numpy()  # (N, 2)
        yaws = env.yaw[0].cpu().numpy()    # (N,)

        # 3D positions: (N, 3) — robots sit on the ground at z=robot_height/2
        robot_z = 0.025
        robot_pos_3d = np.zeros((N, 3), dtype=np.float32)
        robot_pos_3d[:, 0] = pos_2d[:, 0]
        robot_pos_3d[:, 1] = pos_2d[:, 1]
        robot_pos_3d[:, 2] = robot_z

        # Orientation: yaw around Z → quaternion (w, x, y, z)
        robot_orient = np.zeros((N, 4), dtype=np.float32)
        robot_orient[:, 0] = np.cos(yaws / 2)   # w
        robot_orient[:, 3] = np.sin(yaws / 2)   # z

        robot_markers.visualize(
            translations=robot_pos_3d,
            orientations=robot_orient,
            marker_indices=robot_proto,
        )

        # ── Update heading markers (small sphere in front of each robot) ──
        arrow_len = env.robot_radius * 1.8
        heading_pos_3d = np.zeros((N, 3), dtype=np.float32)
        heading_pos_3d[:, 0] = pos_2d[:, 0] + arrow_len * np.cos(yaws)
        heading_pos_3d[:, 1] = pos_2d[:, 1] + arrow_len * np.sin(yaws)
        heading_pos_3d[:, 2] = robot_z + 0.01

        heading_markers.visualize(
            translations=heading_pos_3d,
            marker_indices=heading_proto,
        )

        # ── Periodic console readout ──────────────────────────────
        if step_counter % 10 == 0:
            info = env.compute_obs_robot0()
            gv = info["ground_3"]
            g_label = "BLACK" if gv[0] < 0.1 else ("WHITE" if gv[0] > 0.9 else "GREY")
            pos0 = pos_2d[0]
            print(
                f"[Step {env.step_count:4d}/1200] "
                f"pos=({pos0[0]:+.3f},{pos0[1]:+.3f}) "
                f"yaw={math.degrees(yaws[0]):+6.1f}° "
                f"ground={g_label} "
                f"prox={info['prox_val']:.2f} "
                f"light={info['light_val']:.2f} "
                f"ztilde={info['ztilde']:.2f} "
                f"neighbors={info['n_neighbors']} "
                f"reward={env.step_reward:+.0f} "
                f"K+={env.k_plus_total} K-={env.k_minus_total} "
                f"module={MODULE_NAMES[others_module]}"
            )

        # ── Sim step (renders the viewport) ───────────────────────
        sim.step()

    # ── Cleanup ───────────────────────────────────────────────────
    kb.destroy()
    simulation_app.close()


if __name__ == "__main__":
    main()
