#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Manual control — drive one e-puck in the Directional Gate arena.

2D top-down visualisation (pygame) with live sensor readout.
You control robot #0 via keyboard; the remaining 19 robots stand still
(or run the exploration behaviour module if --others-explore is set).

Controls (AZERTY):
  Z / ↑    : Forward
  S / ↓    : Backward
  Q / ←    : Turn left
  D / →    : Turn right
  A        : Stop
  Numpad 0 : Others → Exploration
  Numpad 1 : Others → Stop
  Numpad 2 : Others → Phototaxis
  Numpad 3 : Others → Anti-phototaxis
  Numpad 4 : Others → Attraction
  Numpad 5 : Others → Repulsion
  R        : Reset episode
  ESC      : Quit

The window shows:
  - Dodecagonal arena with ground zones (grey / white gate / black corridor)
  - All 20 robots as coloured circles with heading arrows
  - Controlled robot highlighted in green
  - Live sensor panel: 8 prox, 8 light, 3 ground, ztilde, 4 RAB, reward

Usage:
  python scripts/manual_control.py
  python scripts/manual_control.py --others-explore
"""

from __future__ import annotations

import argparse
import math
import sys
import os

import torch

# ── Add source to path so we can import epuck modules directly ─────
# We bypass the package __init__.py chain (which pulls in Isaac Lab / pxr).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_EPUCK_DIR = os.path.join(
    _PROJECT_ROOT, "source", "SwarmACB_isaac", "SwarmACB_isaac",
    "tasks", "direct", "epuck",
)
sys.path.insert(0, _EPUCK_DIR)

# Import env components directly (no Isaac Sim needed — kinematic env)
from epuck_sensors import EpuckSensors
from behavior_modules import BehaviorModules


# =======================================================================
#  Standalone minimalist env (no Isaac Sim dependency)
# =======================================================================

class StandaloneDGTEnv:
    """Lightweight standalone version of DirectionalGateEnv for manual testing.

    No Isaac Sim / USD / Omniverse required.  Pure PyTorch kinematic sim
    matching the full env exactly.
    """

    def __init__(self, num_agents: int = 20, device: str = "cpu"):
        self.device = torch.device(device)
        self.E = 1  # single env
        self.N = num_agents

        # ── Arena geometry (dodecagon 4.91 m²) ────────────────────
        self.arena_n_sides = 12
        self.arena_circumradius = math.sqrt(
            2 * 4.91 / (12 * math.sin(2 * math.pi / 12))
        )  # ≈ 1.279 m

        # ── Robot params ──────────────────────────────────────────
        self.robot_radius = 0.035
        self.max_speed = 0.12
        self.wheelbase = 0.053
        self.dt = 0.1

        # ── Ground zones (from paper figure) ───────────────────
        self.corridor_width = 0.50   # m
        self.corridor_length = 1.06  # m
        self.gate_width = 0.45       # m
        self.gate_length = 0.33      # m
        self.side_wall_length = 0.50 # m

        # Derived Y positions
        inradius = self.arena_circumradius * math.cos(
            math.pi / self.arena_n_sides
        )
        self.north_inradius = inradius
        self.corr_south = inradius - self.corridor_length
        self.gate_south = self.corr_south - self.gate_length
        self.corr_hw = self.corridor_width / 2.0
        self.gate_hw = self.gate_width / 2.0

        # ── Light source (south edge) ─────────────────────────────
        self.light_pos = torch.tensor([0.0, -1.4], device=self.device)

        # ── State ─────────────────────────────────────────────────
        self.pos = torch.zeros(self.E, self.N, 2, device=self.device)
        self.yaw = torch.zeros(self.E, self.N, device=self.device)
        self.prev_y = torch.zeros(self.E, self.N, device=self.device)

        # Gate trigger line (middle of white gate)
        self.gate_trigger_y = self.gate_south + self.gate_length / 2.0

        # ── Sensors ───────────────────────────────────────────────
        self.sensors = EpuckSensors(
            prox_range=0.10, rab_range=0.20, light_threshold=0.2, device=device,
        )
        self.behavior_modules = BehaviorModules(
            max_speed=self.max_speed, device=device,
        )
        self.behavior_modules.init_exploration_state(self.E, self.N)

        # ── Wall segments ─────────────────────────────────────────
        self.arena_wall_segments = self._build_walls()
        self.gate_wall_segments = self._build_gate_walls()
        # Combined list for proximity sensor raycasts
        self.wall_segments = self.arena_wall_segments + self.gate_wall_segments

        # ── Reward tracking ───────────────────────────────────────
        self.step_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        self.k_plus_total = 0
        self.k_minus_total = 0

        self.reset()

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
        """Two vertical side walls at x = ±corr_hw."""
        hw = self.corr_hw
        gs = self.gate_south
        wl = self.side_wall_length
        return [
            (-hw, gs, -hw, gs + wl),   # left wall
            (hw, gs, hw, gs + wl),     # right wall
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

    def _ground_scalar(self, pos_2d):
        """(N, 2) → (N,)  scalar ground value."""
        x, y = pos_2d[:, 0], pos_2d[:, 1]
        c = torch.full_like(x, 0.5)
        # White gate
        in_gate = (
            (x.abs() < self.gate_hw)
            & (y > self.gate_south)
            & (y < self.corr_south)
        )
        c = torch.where(in_gate, torch.ones_like(c), c)
        # Black corridor
        in_corr = (
            (x.abs() < self.corr_hw)
            & (y >= self.corr_south)
            & (y < self.north_inradius)
        )
        c = torch.where(in_corr, torch.zeros_like(c), c)
        return c

    def _ground_3ch(self, pos):
        """(1, N, 2) → (1, N, 3)"""
        s = self._ground_scalar(pos[0])
        return s.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)

    def step(self, left_vel: torch.Tensor, right_vel: torch.Tensor):
        """Step all robots. left_vel, right_vel: (1, N)."""
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

        # Reward: gate crossing at trigger line
        curr_y = self.pos[0, :, 1]   # (N,)
        curr_x = self.pos[0, :, 0]   # (N,)
        prev_y = self.prev_y[0]      # (N,)
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
        """Return full sensor breakdown for robot 0."""
        prox_vals, prox_value, prox_angle = self.sensors.compute_proximity(
            self.pos, self.yaw,
            obstacle_segments=self.wall_segments,
            all_agent_pos=self.pos,
            robot_radius=self.robot_radius,
        )
        light_vals, light_value, light_angle = self.sensors.compute_light(
            self.pos, self.yaw, self.light_pos,
        )
        ground = self._ground_3ch(self.pos)  # (1, N, 3)
        ztilde, rab_proj = self.sensors.compute_rab(
            self.pos, self.yaw,
        )
        obs24 = self.sensors.collect_obs_dandelion(
            prox_vals, light_vals, ground, ztilde, rab_proj,
        )  # (1, N, 24)

        # Raw neighbor count for robot 0 (exclude self)
        dx = self.pos[0, :, 0] - self.pos[0, 0, 0]  # (N,)
        dy = self.pos[0, :, 1] - self.pos[0, 0, 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        not_self = torch.ones(self.N, dtype=torch.bool)
        not_self[0] = False
        in_range = (dist < self.sensors.rab_range) & not_self
        n_neighbors = int(in_range.sum().item())

        return {
            "prox_8": prox_vals[0, 0].cpu().tolist(),
            "prox_val": prox_value[0, 0].item(),
            "prox_angle": math.degrees(
                prox_angle[0, 0].item()),
            "light_8": light_vals[0, 0].cpu().tolist(),
            "light_val": light_value[0, 0].item(),
            "light_angle": math.degrees(
                light_angle[0, 0].item()),
            "ground_3": ground[0, 0].cpu().tolist(),
            "ztilde": ztilde[0, 0].item(),
            "n_neighbors": n_neighbors,
            "rab_4": rab_proj[0, 0].cpu().tolist(),
            "obs_24": obs24[0, 0].cpu().tolist(),
        }

    def _resolve_gate_walls(self):
        """Push robots out of the two vertical side walls."""
        r = self.robot_radius
        hw = self.corr_hw
        gs = self.gate_south
        wall_top = gs + self.side_wall_length

        px = self.pos[:, :, 0]
        py = self.pos[:, :, 1]
        in_wall_y = (py > gs) & (py < wall_top)

        # Left wall at x = -hw
        dx_l = px - (-hw)
        pen_l = r - dx_l.abs()
        near_l = (pen_l > 0) & in_wall_y & (px < 0)
        sign_l = torch.sign(dx_l)
        sign_l = torch.where(
            sign_l == 0, -torch.ones_like(sign_l), sign_l
        )
        self.pos[:, :, 0] = torch.where(
            near_l, -hw + sign_l * r, self.pos[:, :, 0]
        )

        # Right wall at x = +hw
        px = self.pos[:, :, 0]
        dx_r = px - hw
        pen_r = r - dx_r.abs()
        near_r = (pen_r > 0) & in_wall_y & (px > 0)
        sign_r = torch.sign(dx_r)
        sign_r = torch.where(
            sign_r == 0, torch.ones_like(sign_r), sign_r
        )
        self.pos[:, :, 0] = torch.where(
            near_r, hw + sign_r * r, self.pos[:, :, 0]
        )

    def _resolve_walls(self):
        R = self.arena_circumradius
        r = self.robot_radius
        n = self.arena_n_sides
        inradius = R * math.cos(math.pi / n)
        for i in range(n):
            a1 = 2 * math.pi * i / n + math.pi / n
            a2 = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
            mid = (a1 + a2) / 2.0
            # Inward normal (toward center)
            nx, ny = -math.cos(mid), -math.sin(mid)
            # Point on the wall face (at inradius, not circumradius)
            wx = inradius * math.cos(mid)
            wy = inradius * math.sin(mid)
            dx = self.pos[:, :, 0] - wx
            dy = self.pos[:, :, 1] - wy
            sd = dx * nx + dy * ny
            pen = r - sd
            push = pen > 0
            self.pos[:, :, 0] += torch.where(
                push, pen * nx, torch.zeros_like(pen))
            self.pos[:, :, 1] += torch.where(
                push, pen * ny, torch.zeros_like(pen))

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


# =======================================================================
#  Pygame visualisation + keyboard control
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Manual control for DGT env",
    )
    parser.add_argument(
        "--num-agents", type=int, default=20,
    )
    parser.add_argument(
        "--speed", type=float, default=0.08,
        help="Keyboard control speed (m/s)",
    )
    args = parser.parse_args()

    try:
        import pygame
    except ImportError:
        print("ERROR: pygame is required.  Install with:  pip install pygame")
        sys.exit(1)

    # ── Create env ────────────────────────────────────────────────
    env = StandaloneDGTEnv(num_agents=args.num_agents, device="cpu")

    # ── Pygame setup ──────────────────────────────────────────────
    WIDTH, HEIGHT = 900, 700
    PANEL_W = 300
    ARENA_W = WIDTH - PANEL_W
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("SwarmACB — Directional Gate — Manual Control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 12)
    font_big = pygame.font.SysFont("consolas", 14, bold=True)

    # Coordinate mapping: arena coords → pixel coords
    arena_R = env.arena_circumradius
    margin = 0.15
    scale = (ARENA_W / 2 - 20) / (arena_R + margin)
    cx, cy_screen = ARENA_W // 2, HEIGHT // 2

    def arena_to_px(ax, ay):
        return int(cx + ax * scale), int(cy_screen - ay * scale)

    # Colours
    COL_BG = (40, 40, 45)
    COL_GREY = (130, 130, 130)
    COL_WHITE = (240, 240, 240)
    COL_BLACK = (30, 30, 30)
    COL_WALL = (200, 180, 100)
    COL_ROBOT = (80, 140, 220)
    COL_CTRL = (50, 220, 80)
    COL_LIGHT = (220, 60, 60)
    COL_TEXT = (220, 220, 220)
    COL_PANEL_BG = (30, 30, 35)
    COL_HEADING = (255, 255, 100)

    # Behavior module names for HUD display
    MODULE_NAMES = [
        "Exploration", "Stop", "Phototaxis",
        "Anti-photo", "Attraction", "Repulsion",
    ]
    # Numpad key → module id mapping
    NUMPAD_MODULE = {
        pygame.K_KP0: 0,  # Exploration
        pygame.K_KP1: 1,  # Stop
        pygame.K_KP2: 2,  # Phototaxis
        pygame.K_KP3: 3,  # Anti-phototaxis
        pygame.K_KP4: 4,  # Attraction
        pygame.K_KP5: 5,  # Repulsion
    }

    others_module = 1  # default: Stop (stationary)
    running = True
    speed = args.speed

    while running:
        # ── Events ────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key in NUMPAD_MODULE:
                    others_module = NUMPAD_MODULE[event.key]

        # ── Keyboard → wheel velocities for robot 0 ──────────────
        keys = pygame.key.get_pressed()
        lv0, rv0 = 0.0, 0.0
        if keys[pygame.K_z] or keys[pygame.K_UP]:
            lv0, rv0 = speed, speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            lv0, rv0 = -speed, -speed
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            lv0 -= speed * 0.5
            rv0 += speed * 0.5
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            lv0 += speed * 0.5
            rv0 -= speed * 0.5
        if keys[pygame.K_a]:
            lv0, rv0 = 0.0, 0.0

        # Build velocity tensors
        N = env.N
        left = torch.zeros(1, N)
        right = torch.zeros(1, N)
        left[0, 0] = lv0
        right[0, 0] = rv0

        # Others: run selected behaviour module
        if N > 1:
            prox_v, prox_val, prox_ang = (
                env.sensors.compute_proximity(
                    env.pos, env.yaw,
                    env.wall_segments,
                    env.pos, env.robot_radius,
                ))
            module_ids = torch.full(
                (1, N), others_module, dtype=torch.long,
            )
            module_ids[0, 0] = 1  # robot 0 = stop (overridden)
            light_v, light_val, light_ang = (
                env.sensors.compute_light(
                    env.pos, env.yaw, env.light_pos,
                ))
            zt, rp = env.sensors.compute_rab(
                env.pos, env.yaw,
            )
            rab_ang = torch.atan2(
                rp[:, :, 1] + rp[:, :, 3],
                rp[:, :, 0] + rp[:, :, 2],
            )
            el, er = env.behavior_modules.dispatch(
                module_ids, prox_val, prox_ang,
                light_val, light_ang, zt, rab_ang,
            )
            left[0, 1:] = el[0, 1:]
            right[0, 1:] = er[0, 1:]

        # ── Step ──────────────────────────────────────────────────
        env.step(left, right)

        # ── Sensor readouts for robot 0 ───────────────────────────
        info = env.compute_obs_robot0()

        # ── Draw ──────────────────────────────────────────────────
        screen.fill(COL_BG)

        # -- Ground zones --
        # Grey arena floor (dodecagon fill)
        arena_segs = env.arena_wall_segments
        poly_pts = []
        for seg in arena_segs:
            poly_pts.append(arena_to_px(seg[0], seg[1]))
        pygame.draw.polygon(screen, COL_GREY, poly_pts)

        # White gate rectangle
        ghw = env.gate_hw
        gs = env.gate_south
        cs = env.corr_south
        gate_pts = [
            arena_to_px(-ghw, gs), arena_to_px(ghw, gs),
            arena_to_px(ghw, cs), arena_to_px(-ghw, cs),
        ]
        pygame.draw.polygon(screen, COL_WHITE, gate_pts)

        # Black corridor rectangle
        chw = env.corr_hw
        ni = env.north_inradius
        corr_pts = [
            arena_to_px(-chw, cs), arena_to_px(chw, cs),
            arena_to_px(chw, ni), arena_to_px(-chw, ni),
        ]
        pygame.draw.polygon(screen, COL_BLACK, corr_pts)

        # -- Arena walls --
        for seg in arena_segs:
            p1 = arena_to_px(seg[0], seg[1])
            p2 = arena_to_px(seg[2], seg[3])
            pygame.draw.line(screen, COL_WALL, p1, p2, 3)

        # -- Gate walls (red-orange barriers) --
        COL_GATE_WALL = (220, 100, 50)
        for seg in env.gate_wall_segments:
            p1 = arena_to_px(seg[0], seg[1])
            p2 = arena_to_px(seg[2], seg[3])
            pygame.draw.line(screen, COL_GATE_WALL, p1, p2, 3)

        # -- Light source (red dot at south) --
        lp = arena_to_px(env.light_pos[0].item(), env.light_pos[1].item())
        pygame.draw.circle(screen, COL_LIGHT, lp, 8)
        pygame.draw.circle(screen, (255, 120, 120), lp, 12, 2)

        # -- Robots --
        positions = env.pos[0].cpu()
        yaws = env.yaw[0].cpu()

        # Draw RAB neighbourhood circle for robot 0
        r0_px, r0_py = arena_to_px(
            positions[0, 0].item(),
            positions[0, 1].item(),
        )
        rab_r_px = int(env.sensors.rab_range * scale)
        # Semi-transparent filled circle via Surface
        rab_surf = pygame.Surface(
            (rab_r_px * 2, rab_r_px * 2), pygame.SRCALPHA,
        )
        pygame.draw.circle(
            rab_surf, (100, 200, 255, 40),
            (rab_r_px, rab_r_px), rab_r_px,
        )
        screen.blit(
            rab_surf,
            (r0_px - rab_r_px, r0_py - rab_r_px),
        )
        # Outline
        pygame.draw.circle(
            screen, (100, 200, 255),
            (r0_px, r0_py), rab_r_px, 1,
        )

        for i in range(N):
            px, py = arena_to_px(
                positions[i, 0].item(),
                positions[i, 1].item(),
            )
            rad_px = max(int(env.robot_radius * scale), 3)
            col = COL_CTRL if i == 0 else COL_ROBOT
            pygame.draw.circle(screen, col, (px, py), rad_px)
            # Heading arrow
            h_len = rad_px * 1.8
            hx = px + int(
                h_len * math.cos(yaws[i].item()))
            hy = py - int(
                h_len * math.sin(yaws[i].item()))
            line_col = (COL_HEADING if i == 0
                        else (180, 180, 200))
            pygame.draw.line(
                screen, line_col, (px, py), (hx, hy), 2)

        # -- N/S labels --
        n_lbl = font_big.render("N", True, COL_TEXT)
        screen.blit(n_lbl, (cx - 5, 10))
        s_lbl = font_big.render("S", True, COL_TEXT)
        screen.blit(s_lbl, (cx - 5, HEIGHT - 25))

        # ── Sensor Panel ──────────────────────────────────────────
        panel_x = ARENA_W + 5
        pygame.draw.rect(screen, COL_PANEL_BG, (ARENA_W, 0, PANEL_W, HEIGHT))
        y = 10

        def draw_text(txt, color=COL_TEXT, bold=False):
            nonlocal y
            f = font_big if bold else font
            surf = f.render(txt, True, color)
            screen.blit(surf, (panel_x, y))
            y += 16

        draw_text("═══ ROBOT 0 SENSORS ═══", COL_CTRL, True)
        y += 4
        draw_text(f"Position: ({positions[0,0]:.3f}, {positions[0,1]:.3f})")
        draw_text(f"Heading:  {math.degrees(yaws[0].item()):.1f}°")
        draw_text(f"Wheels:   L={lv0:.3f}  R={rv0:.3f}")
        y += 8

        draw_text("── Proximity (8 IR) ──", COL_HEADING, True)
        for j, v in enumerate(info["prox_8"]):
            bar = "█" * int(v * 20)
            draw_text(f"  [{j}] {v:.3f} {bar}")
        draw_text(f"  Aggr: val={info['prox_val']:.3f}  ang={info['prox_angle']:.1f}°")
        y += 4

        draw_text("── Light (8 sensors) ──", COL_HEADING, True)
        for j, v in enumerate(info["light_8"]):
            bar = "█" * int(v * 20)
            draw_text(f"  [{j}] {v:.3f} {bar}")
        draw_text(f"  Aggr: val={info['light_val']:.3f}  ang={info['light_angle']:.1f}°")
        y += 4

        draw_text("── Ground (3 ch) ──", COL_HEADING, True)
        gv = info["ground_3"]
        label = "BLACK" if gv[0] < 0.1 else ("WHITE" if gv[0] > 0.9 else "GREY")
        draw_text(f"  [{gv[0]:.2f}, {gv[1]:.2f}, {gv[2]:.2f}]  → {label}")
        y += 4

        draw_text("── RAB ──", COL_HEADING, True)
        nn = info['n_neighbors']
        draw_text(f"  Neighbors: {nn}  z̃={info['ztilde']:.3f}")
        rab_str = ', '.join(
            f'{v:.2f}' for v in info['rab_4'])
        draw_text(f"  proj[4] = [{rab_str}]")
        y += 8

        draw_text("── Reward ──", COL_HEADING, True)
        draw_text(f"  Step: {env.step_reward:+.0f}   K⁺={env.k_plus_total}  K⁻={env.k_minus_total}")
        draw_text(f"  Episode: {env.episode_reward:.0f}")
        draw_text(f"  Step #{env.step_count}  / 1200")
        y += 8

        draw_text("── Controls ──", (160, 160, 170), True)
        draw_text("Z/↑ forward  S/↓ back")
        draw_text("Q/← left     D/→ right")
        draw_text("A stop  R reset  ESC quit")
        y += 4
        draw_text("── Others (numpad) ──", (160, 160, 170), True)
        for mi, mname in enumerate(MODULE_NAMES):
            marker = "►" if mi == others_module else " "
            draw_text(
                f" {marker} [{mi}] {mname}",
                COL_HEADING if mi == others_module
                else COL_TEXT,
            )

        pygame.display.flip()
        clock.tick(10)  # 10 Hz matches control frequency

    pygame.quit()


if __name__ == "__main__":
    main()
