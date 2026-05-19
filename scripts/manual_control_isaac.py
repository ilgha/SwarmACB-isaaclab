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
from _isaac_launch import apply_windows_kit_defaults

TASK_CHOICES = [
    "SwarmACB-DirectionalGate-v0",
    "SwarmACB-XOR-v0",
    "SwarmACB-Homing-v0",
    "SwarmACB-Foraging-v0",
    "SwarmACB-Sheltering-v0",
    "SwarmACB-SCA-v0",
    "SwarmACB-SHL-v0",
]


def _mission_from_task(task: str) -> str:
    if task == "SwarmACB-XOR-v0":
        return "xor"
    if task == "SwarmACB-Homing-v0":
        return "homing"
    if task == "SwarmACB-Foraging-v0":
        return "foraging"
    if task in ("SwarmACB-Sheltering-v0", "SwarmACB-SCA-v0", "SwarmACB-SHL-v0"):
        return "sheltering"
    return "dgt"


parser = argparse.ArgumentParser(description="SwarmACB — Manual control (Isaac Sim)")
parser.add_argument("--task", type=str, default="SwarmACB-DirectionalGate-v0",
                    choices=TASK_CHOICES,
                    help="Mission layout to load in the manual viewer")
parser.add_argument("--num-agents", type=int, default=20)
parser.add_argument("--speed", type=float, default=0.08, help="Keyboard control speed (m/s)")
parser.add_argument("--others-explore", action="store_true",
                    help="Start other robots in exploration mode instead of stop")
parser.add_argument("--no-keyboard", action="store_true",
                    help="Do not subscribe to keyboard events; useful for startup smoke tests")
parser.add_argument("--smoke-frames", type=int, default=0,
                    help="Run this many rendered frames, then exit; 0 means run until closed")
parser.add_argument("--sim-hz", type=float, default=60.0,
                    help="Kinematic integration and viewport update rate")
parser.add_argument("--control-hz", type=float, default=10.0,
                    help="Behaviour-module decision rate; 10 Hz matches the original 0.1 s step")
parser.add_argument("--policy-checkpoint", "--checkpoint", dest="policy_checkpoint",
                    type=str, default=None,
                    help="Optional POCA checkpoint to drive all robots in the fast viewer")
parser.add_argument("--config", type=str, default=None,
                    help="Training config used to infer the CASA variant for policy playback")
parser.add_argument("--variant", type=str, default=None,
                    choices=["dandelion", "daisy", "lily", "tulip", "cyclamen"],
                    help="CASA variant override for policy playback")
parser.add_argument("--deterministic", action="store_true",
                    help="Use deterministic policy actions during checkpoint playback")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_windows_kit_defaults(args, "ManualIsaac")

if getattr(args, "headless", False) and not args.no_keyboard:
    print("[ManualIsaac] Headless mode has no app window; enabling --no-keyboard.", flush=True)
    args.no_keyboard = True

print("[ManualIsaac] Launching Isaac Sim app...", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[ManualIsaac] Isaac Sim app launched.", flush=True)

# ── Now safe to import Isaac Lab & Omni packages ─────────────────────

import math
import weakref

import carb
import numpy as np
import omni
import omni.usd
import torch
from pxr import Gf, UsdGeom, Vt

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
_AGENTS_DIR = os.path.join(
    _PROJECT_ROOT, "source", "SwarmACB_isaac", "SwarmACB_isaac",
    "tasks", "direct", "agents",
)
for _path in (_EPUCK_DIR, _AGENTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from epuck_sensors import EpuckSensors
from behavior_modules import BehaviorModules
from poca_networks import Actor, DiscreteActor, RecurrentDiscreteActor


# =====================================================================
#  Standalone kinematic env (same as pygame version)
# =====================================================================

class StandaloneDGTEnv:
    """Lightweight DGT env (pure-PyTorch kinematic sim, no USD physics)."""

    def __init__(
        self,
        num_agents: int = 20,
        device: str = "cpu",
        dt: float = 0.1,
        task: str = "SwarmACB-DirectionalGate-v0",
    ):
        self.device = torch.device(device)
        self.E = 1
        self.N = num_agents
        self.task = task
        self.mission = _mission_from_task(task)

        # ── Arena geometry (dodecagon 4.91 m²) ─────────────────
        self.arena_n_sides = 12
        self.arena_circumradius = math.sqrt(
            2 * 4.91 / (12 * math.sin(2 * math.pi / 12))
        )

        # ── Robot params ────────────────────────────────────────
        self.robot_radius = 0.035
        self.max_speed = 0.12
        self.wheelbase = 0.053
        self.dt = dt

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
        self.has_light = self.mission not in ("xor", "homing")

        # XOR aggregation targets: two black disks, diameter 0.60 m,
        # centered 0.50 m left/right of the arena center.
        self.target_radius = 0.30
        self.target_centers = torch.tensor(
            [[-0.50, 0.0], [0.50, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        # Homing goal: a single black disk in the southern half.
        self.goal_radius = 0.30
        self.goal_center = torch.tensor([0.0, -0.70], dtype=torch.float32, device=self.device)

        # Foraging: two black food disks and one white nest near the light.
        self.food_radius = 0.15
        self.food_centers = torch.tensor(
            [[-0.75, 0.0], [0.75, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )
        self.nest_top_y = -0.63
        self.nest_center = torch.tensor([0.0, -0.78], dtype=torch.float32, device=self.device)
        self.nest_size = torch.tensor([1.10, 0.35], dtype=torch.float32, device=self.device)

        # Sheltering: white center shelter, three walls, two black side cues.
        self.shelter_center = torch.tensor([0.0, 0.0], dtype=torch.float32, device=self.device)
        self.shelter_size = torch.tensor([0.50, 0.30], dtype=torch.float32, device=self.device)
        self.shelter_wall_thickness = 0.03
        self.shelter_black_radius = 0.30
        self.shelter_black_centers = torch.tensor(
            [[-0.80, 0.0], [0.80, 0.0]],
            dtype=torch.float32,
            device=self.device,
        )

        # ── State ───────────────────────────────────────────────
        self.pos = torch.zeros(self.E, self.N, 2, device=self.device)
        self.yaw = torch.zeros(self.E, self.N, device=self.device)
        self.prev_y = torch.zeros(self.E, self.N, device=self.device)

        # Color-transition reward tracking (matches Unity DirGateEnvController.cs)
        # Unity detects ground COLOR TRANSITIONS, not Y-position crossings.
        self.prev_ground_color = torch.full((self.E, self.N), 0.5, device=self.device)  # grey default

        # ── Sensors ─────────────────────────────────────────────
        self.sensors = EpuckSensors(
            prox_range=0.10, rab_range=0.20, light_threshold=0.2, device=device,
        )
        self.behavior_modules = BehaviorModules(max_speed=self.max_speed, device=device)
        self.behavior_modules.init_state(self.E, self.N)

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
        self.has_food = torch.zeros(self.E, self.N, dtype=torch.bool, device=self.device)
        self.prev_in_nest = torch.zeros_like(self.has_food)

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
        if self.mission in ("xor", "homing", "foraging"):
            return []
        if self.mission == "sheltering":
            left, right, bottom, top = self._shelter_bounds()
            return [
                (left, bottom, left, top),
                (right, bottom, right, top),
                (left, top, right, top),
            ]
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
        if self.mission == "homing":
            th = torch.rand(self.N) * math.pi
        else:
            th = torch.rand(self.N) * 2 * math.pi
        self.pos[0, :, 0] = r * torch.cos(th)
        self.pos[0, :, 1] = r * torch.sin(th)
        if self.mission == "homing":
            self.pos[0, :, 1] = self.pos[0, :, 1].abs()
        self.yaw[0] = torch.rand(self.N) * 2 * math.pi - math.pi
        self.prev_y[0] = self.pos[0, :, 1]
        self.prev_ground_color = self._ground_scalar(self.pos[0]).unsqueeze(0)
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.k_plus_total = 0
        self.k_minus_total = 0
        self.has_food[:] = False
        self.prev_in_nest = self._nest_membership(self.pos[0]).unsqueeze(0)
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

        if self.mission == "xor":
            in_targets = self._target_membership(self.pos[0])
            counts = in_targets.float().sum(dim=0)
            self.step_reward = counts.max().item()
            self.episode_reward += self.step_reward
            self.step_count += 1
            return
        if self.mission == "homing":
            final_step = self.step_count + 1 >= 1200
            self.step_reward = self._goal_membership(self.pos[0]).float().sum().item() if final_step else 0.0
            self.episode_reward += self.step_reward
            self.step_count += 1
            return
        if self.mission == "foraging":
            in_food = self._food_membership(self.pos[0]).any(dim=-1).unsqueeze(0)
            in_nest = self._nest_membership(self.pos[0]).unsqueeze(0)
            self.has_food = self.has_food | in_food
            arrived = in_nest & (~self.prev_in_nest) & self.has_food
            self.step_reward = arrived.float().sum().item()
            self.has_food = torch.where(arrived, torch.zeros_like(self.has_food), self.has_food)
            self.prev_in_nest = in_nest.clone()
            self.episode_reward += self.step_reward
            self.step_count += 1
            return
        if self.mission == "sheltering":
            self.step_reward = self._shelter_membership(self.pos[0]).float().sum().item()
            self.episode_reward += self.step_reward
            self.step_count += 1
            return

        # Reward: colour transitions (matching Unity DirGateEnvController.cs)
        # BLACK → WHITE = +1 (K⁺), WHITE → BLACK = -1 (K⁻)
        curr_color = self._ground_scalar(self.pos[0])  # (N,)
        prev_color = self.prev_ground_color[0]          # (N,)

        prev_is_black = (prev_color < 0.25)
        prev_is_white = (prev_color > 0.75)
        curr_is_black = (curr_color < 0.25)
        curr_is_white = (curr_color > 0.75)

        black_to_white = prev_is_black & curr_is_white
        white_to_black = prev_is_white & curr_is_black

        k_plus = black_to_white.float().sum().item()
        k_minus = white_to_black.float().sum().item()
        self.prev_ground_color[0] = curr_color.clone()

        self.step_reward = k_plus - k_minus
        self.episode_reward += self.step_reward
        self.k_plus_total += int(k_plus)
        self.k_minus_total += int(k_minus)
        self.step_count += 1

    def _compute_light_readings(self):
        if self.has_light:
            return self.sensors.compute_light(self.pos, self.yaw, self.light_pos)
        light_vals = torch.zeros(self.E, self.N, 8, device=self.device)
        light_value = torch.zeros(self.E, self.N, device=self.device)
        light_angle = torch.zeros(self.E, self.N, device=self.device)
        return light_vals, light_value, light_angle

    def compute_obs_robot0(self):
        prox_vals, prox_value, prox_angle = self.sensors.compute_proximity(
            self.pos, self.yaw, self.wall_segments, self.pos, self.robot_radius,
        )
        light_vals, light_value, light_angle = self._compute_light_readings()
        ground = self._ground_3ch(self.pos)
        ztilde, rab_proj, _, _ = self.sensors.compute_rab(self.pos, self.yaw)
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
        if self.mission == "xor":
            in_target = self._target_membership(pos_2d).any(dim=-1)
            return torch.where(in_target, torch.zeros_like(c), c)
        if self.mission == "homing":
            return torch.where(self._goal_membership(pos_2d), torch.zeros_like(c), c)
        if self.mission == "foraging":
            in_food = self._food_membership(pos_2d).any(dim=-1)
            in_nest = self._nest_membership(pos_2d)
            c = torch.where(in_food, torch.zeros_like(c), c)
            return torch.where(in_nest, torch.ones_like(c), c)
        if self.mission == "sheltering":
            in_black = self._shelter_black_membership(pos_2d).any(dim=-1)
            in_shelter = self._shelter_membership(pos_2d)
            c = torch.where(in_black, torch.zeros_like(c), c)
            return torch.where(in_shelter, torch.ones_like(c), c)
        in_gate = (x.abs() < self.gate_hw) & (y > self.gate_south) & (y < self.corr_south)
        c = torch.where(in_gate, torch.ones_like(c), c)
        in_corr = (x.abs() < self.corr_hw) & (y >= self.corr_south) & (y < self.north_inradius)
        c = torch.where(in_corr, torch.zeros_like(c), c)
        return c

    def _target_membership(self, pos_2d):
        diff = pos_2d.unsqueeze(1) - self.target_centers.view(1, 2, 2)
        dist_sq = (diff * diff).sum(dim=-1)
        return dist_sq <= self.target_radius ** 2

    def _goal_membership(self, pos_2d):
        diff = pos_2d - self.goal_center.view(1, 2)
        return (diff * diff).sum(dim=-1) <= self.goal_radius ** 2

    def _food_membership(self, pos_2d):
        diff = pos_2d.unsqueeze(1) - self.food_centers.view(1, 2, 2)
        return (diff * diff).sum(dim=-1) <= self.food_radius ** 2

    def _nest_membership(self, pos_2d):
        return pos_2d[:, 1] <= self.nest_top_y

    def _shelter_bounds(self):
        half = self.shelter_size / 2.0
        left = (self.shelter_center[0] - half[0]).item()
        right = (self.shelter_center[0] + half[0]).item()
        bottom = (self.shelter_center[1] - half[1]).item()
        top = (self.shelter_center[1] + half[1]).item()
        return left, right, bottom, top

    def _shelter_membership(self, pos_2d):
        left, right, bottom, top = self._shelter_bounds()
        return (
            (pos_2d[:, 0] >= left) & (pos_2d[:, 0] <= right)
            & (pos_2d[:, 1] >= bottom) & (pos_2d[:, 1] <= top)
        )

    def _shelter_black_membership(self, pos_2d):
        diff = pos_2d.unsqueeze(1) - self.shelter_black_centers.view(1, 2, 2)
        return (diff * diff).sum(dim=-1) <= self.shelter_black_radius ** 2

    def _ground_3ch(self, pos):
        s = self._ground_scalar(pos[0])
        return s.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)

    def _resolve_gate_walls(self):
        if self.mission in ("xor", "homing", "foraging"):
            return
        if self.mission == "sheltering":
            r = self.robot_radius
            t = self.shelter_wall_thickness
            left, right, bottom, top = self._shelter_bounds()
            px = self.pos[:, :, 0]
            py = self.pos[:, :, 1]
            vertical_y = (py > bottom - r) & (py < top + r)
            for x0 in (left, right):
                dx = px - x0
                near = (dx.abs() < r + t / 2) & vertical_y
                sign = torch.sign(dx)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                self.pos[:, :, 0] = torch.where(near, x0 + sign * (r + t / 2), self.pos[:, :, 0])
            px = self.pos[:, :, 0]
            py = self.pos[:, :, 1]
            horizontal_x = (px > left - r) & (px < right + r)
            dy = py - top
            near_top = (dy.abs() < r + t / 2) & horizontal_x
            sign_y = torch.sign(dy)
            sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
            self.pos[:, :, 1] = torch.where(near_top, top + sign_y * (r + t / 2), self.pos[:, :, 1])
            return
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

def _spawn_flat_circle(
    prim_path: str,
    center_xy,
    radius: float,
    z: float = 0.004,
    segments: int = 96,
    color: tuple[float, float, float] = (0.02, 0.02, 0.02),
):
    """Spawn a true flat circular floor patch as a USD triangle-fan mesh."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    points = [Gf.Vec3f(cx, cy, z)]
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        points.append(Gf.Vec3f(
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
            z,
        ))

    indices: list[int] = []
    for i in range(segments):
        indices.extend([0, i + 1, (i + 1) % segments + 1])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * segments)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def _dodecagon_vertices(radius: float, n_sides: int = 12):
    return [
        (
            radius * math.cos(2 * math.pi * i / n_sides + math.pi / n_sides),
            radius * math.sin(2 * math.pi * i / n_sides + math.pi / n_sides),
        )
        for i in range(n_sides)
    ]


def _clip_polygon_below_y(points, y_max: float):
    if not points:
        return []
    clipped = []
    prev = points[-1]
    prev_inside = prev[1] <= y_max
    for curr in points:
        curr_inside = curr[1] <= y_max
        if curr_inside != prev_inside:
            denom = curr[1] - prev[1]
            alpha = 0.0 if abs(denom) < 1e-8 else (y_max - prev[1]) / denom
            clipped.append((prev[0] + alpha * (curr[0] - prev[0]), y_max))
        if curr_inside:
            clipped.append(curr)
        prev, prev_inside = curr, curr_inside
    return clipped


def _spawn_flat_polygon(
    prim_path: str,
    points_xy,
    z: float = 0.004,
    color: tuple[float, float, float] = (0.95, 0.95, 0.95),
):
    if len(points_xy) < 3:
        return
    cx = sum(p[0] for p in points_xy) / len(points_xy)
    cy = sum(p[1] for p in points_xy) / len(points_xy)
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = [Gf.Vec3f(cx, cy, z)]
    points.extend(Gf.Vec3f(float(x), float(y), z) for x, y in points_xy)
    indices: list[int] = []
    for i in range(len(points_xy)):
        indices.extend([0, i + 1, (i + 1) % len(points_xy) + 1])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(points_xy))
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))


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
    if env.mission != "dgt":
        if env.mission == "xor":
            for idx, center in enumerate(env.target_centers.cpu().numpy()):
                _spawn_flat_circle(
                    f"/World/Arena/BlackTarget_{idx}",
                    center,
                    env.target_radius,
                )
        elif env.mission == "homing":
            _spawn_flat_circle(
                "/World/Arena/HomingGoal",
                env.goal_center.cpu().numpy(),
                env.goal_radius,
            )
        elif env.mission == "foraging":
            nest_poly = _clip_polygon_below_y(
                _dodecagon_vertices(R, n),
                env.nest_top_y,
            )
            _spawn_flat_polygon(
                "/World/Arena/Nest",
                nest_poly,
                color=(0.95, 0.95, 0.95),
            )
            for idx, center in enumerate(env.food_centers.cpu().numpy()):
                _spawn_flat_circle(
                    f"/World/Arena/Food_{idx}",
                    center,
                    env.food_radius,
                )
        elif env.mission == "sheltering":
            for idx, center in enumerate(env.shelter_black_centers.cpu().numpy()):
                _spawn_flat_circle(
                    f"/World/Arena/BlackCue_{idx}",
                    center,
                    env.shelter_black_radius,
                )
            shelter_cfg = sim_utils.CuboidCfg(
                size=(float(env.shelter_size[0]), float(env.shelter_size[1]), 0.003),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
            )
            shelter_cfg.func(
                "/World/Arena/ShelterArea",
                shelter_cfg,
                translation=(float(env.shelter_center[0]), float(env.shelter_center[1]), 0.003),
            )
            left, right, bottom, top = env._shelter_bounds()
            sx, sy = float(env.shelter_size[0]), float(env.shelter_size[1])
            t = env.shelter_wall_thickness
            shelter_wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.39, 0.20))
            wall_specs = [
                ("ShelterWallLeft", (t, sy, wall_h), (left, float(env.shelter_center[1]), wall_h / 2)),
                ("ShelterWallRight", (t, sy, wall_h), (right, float(env.shelter_center[1]), wall_h / 2)),
                ("ShelterWallTop", (sx, t, wall_h), (float(env.shelter_center[0]), top, wall_h / 2)),
            ]
            for name, size, translation in wall_specs:
                wall_cfg = sim_utils.CuboidCfg(size=size, visual_material=shelter_wall_mat)
                wall_cfg.func(f"/World/Arena/{name}", wall_cfg, translation=translation)

        if env.has_light:
            light_cfg = sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15)),
            )
            light_cfg.func("/World/Arena/LightIndicator", light_cfg,
                           translation=(float(env.light_pos[0]), float(env.light_pos[1]), 0.04))

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
            wall_cfg.func(
                f"/World/Arena/Wall_{i}",
                wall_cfg,
                translation=(cx, cy, wall_h / 2),
                orientation=(math.cos(seg_angle / 2), 0.0, 0.0, math.sin(seg_angle / 2)),
            )

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
        return

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


def _variant_from_config(config_path: str) -> str:
    """Read the CASA variant from an ML-Agents-style YAML config."""
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    behaviors = raw.get("behaviors", raw)
    if not behaviors:
        raise ValueError("Config must have a top-level 'behaviors' key.")
    run_name = next(iter(behaviors))
    return behaviors[run_name].get("variant", "dandelion")


def _infer_variant_from_checkpoint(ckpt: dict) -> str:
    if not ckpt.get("discrete", False):
        return "dandelion"
    if ckpt.get("recurrent", False):
        return "cyclamen"
    if ckpt.get("obs_dim", 0) == 24:
        return "daisy"
    return "tulip"


def _expected_obs_dim(variant: str) -> int:
    return 24 if variant in ("dandelion", "daisy") else 4


def _policy_observations(env: StandaloneDGTEnv, variant: str):
    """Compute the same local observations used by the training environment."""
    prox_vals, prox_value, prox_angle = env.sensors.compute_proximity(
        env.pos, env.yaw, env.wall_segments, env.pos, env.robot_radius,
    )
    light_vals, light_value, light_angle = env._compute_light_readings()
    ground = env._ground_3ch(env.pos)
    ztilde, rab_proj, rab_attr_x, rab_attr_y = env.sensors.compute_rab(env.pos, env.yaw)

    if variant in ("dandelion", "daisy"):
        obs_all = env.sensors.collect_obs_dandelion(
            prox_vals, light_vals, ground, ztilde, rab_proj,
        )
    else:
        obs_all = EpuckSensors.collect_obs_lily(ground, ztilde)

    behavior_inputs = (prox_value, prox_angle, light_value, light_angle, rab_attr_x, rab_attr_y)
    return obs_all, behavior_inputs


def _load_policy_actor(
    checkpoint_path: str,
    config_path: str | None,
    cli_variant: str | None,
    device: torch.device,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    variant = cli_variant
    if variant is None and config_path:
        variant = _variant_from_config(config_path)
    if variant is None:
        variant = _infer_variant_from_checkpoint(ckpt)

    discrete = bool(ckpt.get("discrete", False))
    recurrent = bool(ckpt.get("recurrent", False))
    hidden_dim = int(ckpt.get("hidden_dim", 256))
    num_layers = int(ckpt.get("num_layers", 2))
    obs_dim = int(ckpt.get("obs_dim", _expected_obs_dim(variant)))
    num_actions = int(ckpt.get("num_actions", 6))
    act_dim = int(ckpt.get("act_dim", 2))
    memory_size = int(ckpt.get("memory_size", 128))

    if obs_dim != _expected_obs_dim(variant):
        print(
            f"[Policy] Warning: checkpoint obs_dim={obs_dim}, "
            f"but variant '{variant}' expects {_expected_obs_dim(variant)}.",
            flush=True,
        )

    if discrete:
        if recurrent:
            actor = RecurrentDiscreteActor(
                obs_dim, num_actions, hidden_dim, num_layers, memory_size,
            ).to(device)
        else:
            actor = DiscreteActor(obs_dim, num_actions, hidden_dim, num_layers).to(device)
    else:
        actor = Actor(obs_dim, act_dim, hidden_dim, num_layers).to(device)

    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    meta = {
        "variant": variant,
        "discrete": discrete,
        "recurrent": recurrent,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "act_dim": act_dim,
        "memory_size": memory_size,
    }
    return actor, meta


class KeyboardController:
    """Capture keyboard state via carb.input for manual robot control."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        if self._appwindow is None:
            raise RuntimeError("No Isaac Sim app window is available for keyboard input.")
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
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


# =====================================================================
#  Main loop
# =====================================================================

def main():
    N = args.num_agents
    speed = args.speed
    sim_hz = max(args.sim_hz, 1.0)
    control_hz = max(args.control_hz, 1.0)
    sim_dt = 1.0 / sim_hz
    control_dt = 1.0 / control_hz
    control_interval = max(1, round(control_dt / sim_dt))
    status_interval = max(1, round(1.0 / sim_dt))
    print(
        f"[ManualIsaac] sim_hz={sim_hz:.1f}, control_hz={control_hz:.1f}, "
        f"control interval={control_interval} frames",
        flush=True,
    )

    # ── Simulation context ────────────────────────────────────────
    print("[ManualIsaac] Creating SimulationContext...", flush=True)
    sim = SimulationContext(physics_dt=sim_dt, rendering_dt=sim_dt)
    print("[ManualIsaac] SimulationContext ready.", flush=True)

    # ── Lighting ──────────────────────────────────────────────────
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.80, 0.80, 0.80))
    light_cfg.func("/World/DomeLight", light_cfg)

    # ── Ground plane ──────────────────────────────────────────────
    spawn_ground_plane("/World/GroundPlane", GroundPlaneCfg())

    # ── Kinematic env ─────────────────────────────────────────────
    env = StandaloneDGTEnv(num_agents=N, device="cpu", dt=sim_dt, task=args.task)

    policy_mode = args.policy_checkpoint is not None
    policy_device = torch.device("cpu")
    policy_actor = None
    policy_meta: dict[str, object] = {}
    if policy_mode:
        policy_actor, policy_meta = _load_policy_actor(
            args.policy_checkpoint, args.config, args.variant, policy_device,
        )
        print(
            f"[Policy] Loaded {args.policy_checkpoint} "
            f"variant={policy_meta['variant']} "
            f"discrete={policy_meta['discrete']} "
            f"recurrent={policy_meta['recurrent']} "
            f"obs={policy_meta['obs_dim']}",
            flush=True,
        )

    # ── Visual scene ──────────────────────────────────────────────
    print("[ManualIsaac] Building arena visuals...", flush=True)
    _build_arena_visuals(env)
    robot_markers = _create_robot_markers(env)
    heading_markers = _create_heading_markers(env)
    print("[ManualIsaac] Visual scene ready.", flush=True)

    # ── Keyboard ──────────────────────────────────────────────────
    kb = None if args.no_keyboard else KeyboardController()
    others_module = 0 if args.others_explore else 1  # 0=explore, 1=stop

    # ── Let the sim initialise ────────────────────────────────────
    print("[ManualIsaac] Resetting simulation...", flush=True)
    sim.reset()
    print("[ManualIsaac] Entering main loop.", flush=True)

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
    if policy_mode:
        mode = "deterministic" if args.deterministic else "stochastic"
        print(
            f"[Policy] Fast viewer playback active: variant={policy_meta['variant']} "
            f"actions={mode}; policy drives all robots.",
            flush=True,
        )

    step_counter = 0
    other_left_cmd = torch.zeros(1, N)
    other_right_cmd = torch.zeros(1, N)
    policy_left_cmd = torch.zeros(1, N)
    policy_right_cmd = torch.zeros(1, N)
    if policy_mode and policy_meta["recurrent"]:
        policy_memory = policy_actor.initial_state(N, policy_device)
    else:
        policy_memory = None
    force_control_update = True

    while simulation_app.is_running():
        # ── Handle keyboard events ────────────────────────────────
        for evt in ([] if kb is None else kb.pop_events()):
            if evt == "ESCAPE":
                print("[ManualIsaac] Escape pressed; exiting.", flush=True)
                if kb is not None:
                    kb.destroy()
                simulation_app.close()
                return
            elif evt == "R":
                env.reset()
                step_counter = 0
                force_control_update = True
                if policy_mode and policy_meta["recurrent"]:
                    policy_memory = policy_actor.initial_state(N, policy_device)
                print("[RESET] Episode reset")
            elif evt == "NUMPAD_0":
                others_module = 0
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[0]}")
            elif evt == "NUMPAD_1":
                others_module = 1
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[1]}")
            elif evt == "NUMPAD_2":
                others_module = 2
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[2]}")
            elif evt == "NUMPAD_3":
                others_module = 3
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[3]}")
            elif evt == "NUMPAD_4":
                others_module = 4
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[4]}")
            elif evt == "NUMPAD_5":
                others_module = 5
                force_control_update = True
                print(f"[MODULE] Others → {MODULE_NAMES[5]}")

        # ── Keyboard → wheel velocities for robot 0 ──────────────
        lv0, rv0 = 0.0, 0.0
        if kb is not None and kb.is_held("Z", "UP"):
            lv0, rv0 = speed, speed
        if kb is not None and kb.is_held("S", "DOWN"):
            lv0, rv0 = -speed, -speed
        if kb is not None and kb.is_held("Q", "LEFT"):
            lv0 -= speed * 0.5
            rv0 += speed * 0.5
        if kb is not None and kb.is_held("D", "RIGHT"):
            lv0 += speed * 0.5
            rv0 -= speed * 0.5
        if kb is not None and kb.is_held("A"):
            lv0, rv0 = 0.0, 0.0

        # Build velocity tensors
        left = torch.zeros(1, N)
        right = torch.zeros(1, N)
        left[0, 0] = lv0
        right[0, 0] = rv0

        # ── Others: run selected behaviour module ─────────────────
        if (not policy_mode) and N > 1 and (force_control_update or step_counter % control_interval == 0):
            prox_v, prox_val, prox_ang = env.sensors.compute_proximity(
                env.pos, env.yaw, env.wall_segments, env.pos, env.robot_radius,
            )
            module_ids = torch.full((1, N), others_module, dtype=torch.long)
            module_ids[0, 0] = 1  # robot 0 overridden by keyboard
            light_v, light_val, light_ang = env._compute_light_readings()
            zt, rp, rab_ax, rab_ay = env.sensors.compute_rab(env.pos, env.yaw)
            el, er = env.behavior_modules.dispatch(
                module_ids, prox_val, prox_ang, light_val, light_ang, rab_ax, rab_ay,
            )
            other_left_cmd = el
            other_right_cmd = er
            force_control_update = False

        if (not policy_mode) and N > 1:
            left[0, 1:] = other_left_cmd[0, 1:]
            right[0, 1:] = other_right_cmd[0, 1:]

        if policy_mode:
            if force_control_update or step_counter % control_interval == 0:
                obs_all, behavior_inputs = _policy_observations(
                    env, str(policy_meta["variant"]),
                )
                flat_obs = obs_all.reshape(N, -1).to(policy_device)
                with torch.no_grad():
                    if policy_meta["discrete"]:
                        if policy_meta["recurrent"]:
                            logits, policy_memory = policy_actor.step(flat_obs, policy_memory)
                            policy_memory = (
                                policy_memory[0].detach(),
                                policy_memory[1].detach(),
                            )
                            dist = torch.distributions.Categorical(logits=logits)
                        else:
                            dist = policy_actor.get_dist(flat_obs)

                        if args.deterministic:
                            action_ids = dist.probs.argmax(dim=-1)
                        else:
                            action_ids = dist.sample()
                        module_ids = action_ids.view(1, N).cpu().long()
                        policy_left_cmd, policy_right_cmd = env.behavior_modules.dispatch(
                            module_ids, *behavior_inputs,
                        )
                    else:
                        dist = policy_actor.get_dist(flat_obs)
                        raw_actions = dist.mean if args.deterministic else dist.sample()
                        wheel_actions = raw_actions.clamp(-3.0, 3.0) / 3.0
                        wheel_actions = wheel_actions.view(1, N, 2).cpu()
                        policy_left_cmd = wheel_actions[:, :, 0] * env.max_speed
                        policy_right_cmd = wheel_actions[:, :, 1] * env.max_speed

                force_control_update = False

            left = policy_left_cmd.clone()
            right = policy_right_cmd.clone()

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
        if step_counter % status_interval == 0:
            info = env.compute_obs_robot0()
            gv = info["ground_3"]
            g_label = "BLACK" if gv[0] < 0.1 else ("WHITE" if gv[0] > 0.9 else "GREY")
            pos0 = pos_2d[0]
            mode_label = (
                f"policy:{policy_meta['variant']}"
                if policy_mode else MODULE_NAMES[others_module]
            )
            print(
                f"[t={env.step_count * env.dt:6.1f}s step={env.step_count:5d}] "
                f"pos=({pos0[0]:+.3f},{pos0[1]:+.3f}) "
                f"yaw={math.degrees(yaws[0]):+6.1f}° "
                f"ground={g_label} "
                f"prox={info['prox_val']:.2f} "
                f"light={info['light_val']:.2f} "
                f"ztilde={info['ztilde']:.2f} "
                f"neighbors={info['n_neighbors']} "
                f"reward={env.step_reward:+.0f} "
                f"K+={env.k_plus_total} K-={env.k_minus_total} "
                f"module={mode_label}"
            )

        # ── Sim step (renders the viewport) ───────────────────────
        sim.step()

        if args.smoke_frames > 0 and step_counter >= args.smoke_frames:
            print(f"[ManualIsaac] Smoke test completed after {step_counter} frames.", flush=True)
            break

    # ── Cleanup ───────────────────────────────────────────────────
    if kb is not None:
        kb.destroy()
    simulation_app.close()


if __name__ == "__main__":
    main()
