# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Sheltering with constrained access mission environment."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ..directional_gate.directional_gate_env import DirectionalGateEnv
from ..mission_visuals import spawn_flat_circle
from .sheltering_env_cfg import ShelteringEnvCfg


class ShelteringEnv(DirectionalGateEnv):
    """Sheltering mission for SwarmACB."""

    cfg: ShelteringEnvCfg

    def _shelter_bounds(self):
        cx, cy = self.cfg.shelter_center
        sx, sy = self.cfg.shelter_size
        return cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2

    def _build_gate_wall_segments(self):
        left, right, bottom, top = self._shelter_bounds()
        return [
            (left, bottom, left, top),
            (right, bottom, right, top),
            (left, top, right, top),
        ]

    def _spawn_arena_visuals(self):
        cfg = self.cfg
        R = cfg.arena_circumradius
        n = cfg.arena_num_sides
        wall_h = cfg.arena_wall_height
        wall_thick = 0.01

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        floor_side = R * 3.0
        floor_cfg = sim_utils.CuboidCfg(
            size=(floor_side, floor_side, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        )
        floor_cfg.func("/World/Arena/Floor", floor_cfg, translation=(0.0, 0.0, 0.001))

        for idx, center in enumerate(cfg.black_area_centers):
            spawn_flat_circle(
                f"/World/Arena/BlackCue_{idx}",
                center,
                cfg.black_area_radius,
                color=(0.02, 0.02, 0.02),
            )

        sx, sy = cfg.shelter_size
        cx, cy = cfg.shelter_center
        shelter_area_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
        )
        shelter_area_cfg.func("/World/Arena/ShelterArea", shelter_area_cfg, translation=(cx, cy, 0.003))

        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.39, 0.20))
        left, right, bottom, top = self._shelter_bounds()
        t = cfg.shelter_wall_thickness
        wall_specs = [
            ("ShelterWallLeft", (t, sy, wall_h), (left, cy, wall_h / 2)),
            ("ShelterWallRight", (t, sy, wall_h), (right, cy, wall_h / 2)),
            ("ShelterWallTop", (sx, t, wall_h), (cx, top, wall_h / 2)),
        ]
        for name, size, translation in wall_specs:
            wall_cfg = sim_utils.CuboidCfg(size=size, visual_material=wall_mat)
            wall_cfg.func(f"/World/Arena/{name}", wall_cfg, translation=translation)

        arena_wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.70, 0.40))
        for i in range(n):
            a1 = 2 * math.pi * i / n + math.pi / n
            a2 = 2 * math.pi * ((i + 1) % n) / n + math.pi / n
            ax, ay = R * math.cos(a1), R * math.sin(a1)
            bx, by = R * math.cos(a2), R * math.sin(a2)
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            seg_angle = math.atan2(by - ay, bx - ax)
            wall_cfg = sim_utils.CuboidCfg(
                size=(seg_len, wall_thick, wall_h),
                visual_material=arena_wall_mat,
            )
            wall_cfg.func(
                f"/World/Arena/Wall_{i}",
                wall_cfg,
                translation=(mx, my, wall_h / 2),
                orientation=(math.cos(seg_angle / 2), 0.0, 0.0, math.sin(seg_angle / 2)),
            )

        lx, ly, _ = cfg.light_position
        li_cfg = sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15)),
        )
        li_cfg.func("/World/Arena/LightIndicator", li_cfg, translation=(lx, ly, 0.04))

    def _black_membership(self, pos: torch.Tensor) -> torch.Tensor:
        centers = torch.tensor(self.cfg.black_area_centers, dtype=pos.dtype, device=pos.device)
        diff = pos.unsqueeze(2) - centers.view(1, 1, 2, 2)
        return (diff * diff).sum(dim=-1) <= self.cfg.black_area_radius ** 2

    def _shelter_membership(self, pos: torch.Tensor) -> torch.Tensor:
        left, right, bottom, top = self._shelter_bounds()
        x, y = pos[:, :, 0], pos[:, :, 1]
        return (x >= left) & (x <= right) & (y >= bottom) & (y <= top)

    def _ground_color(self, pos: torch.Tensor) -> torch.Tensor:
        in_black = self._black_membership(pos).any(dim=-1)
        in_shelter = self._shelter_membership(pos)
        grey = torch.full(pos.shape[:2], 0.5, dtype=pos.dtype, device=pos.device)
        color = torch.where(in_black, torch.zeros_like(grey), grey)
        color = torch.where(in_shelter, torch.ones_like(grey), color)
        return color.unsqueeze(-1).expand(-1, -1, 3)

    def _resolve_gate_wall_collisions(self):
        cfg = self.cfg
        r = cfg.robot_radius
        t = cfg.shelter_wall_thickness
        left, right, bottom, top = self._shelter_bounds()
        px = self.agent_pos[:, :, 0]
        py = self.agent_pos[:, :, 1]

        vertical_y = (py > bottom - r) & (py < top + r)
        for x0 in (left, right):
            dx = px - x0
            near = (dx.abs() < r + t / 2) & vertical_y
            sign = torch.sign(dx)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            self.agent_pos[:, :, 0] = torch.where(
                near,
                x0 + sign * (r + t / 2),
                self.agent_pos[:, :, 0],
            )

        px = self.agent_pos[:, :, 0]
        py = self.agent_pos[:, :, 1]
        horizontal_x = (px > left - r) & (px < right + r)
        dy = py - top
        near_top = (dy.abs() < r + t / 2) & horizontal_x
        sign_y = torch.sign(dy)
        sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
        self.agent_pos[:, :, 1] = torch.where(
            near_top,
            top + sign_y * (r + t / 2),
            self.agent_pos[:, :, 1],
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        reward = self._shelter_membership(self.agent_pos).float().sum(dim=1)
        self._episode_group_reward += reward
        return {agent: reward for agent in self.cfg.possible_agents}
