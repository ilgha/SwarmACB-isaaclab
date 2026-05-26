# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""XOR-Aggregation mission environment.

The swarm receives at each 10 Hz step the number of robots in whichever of the
two black circular aggregation areas currently hosts more robots.
"""

from __future__ import annotations

import math

import omni.usd
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils

from ..directional_gate.directional_gate_env import DirectionalGateEnv
from .xor_aggregation_env_cfg import XorAggregationEnvCfg


def _spawn_flat_circle(
    prim_path: str,
    center_xy: tuple[float, float],
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


class XorAggregationEnv(DirectionalGateEnv):
    """XOR-Aggregation mission for SwarmACB."""

    cfg: XorAggregationEnvCfg

    def _build_gate_wall_segments(self):
        """XOR has no mission-specific internal wall segments."""
        return []

    def _spawn_arena_visuals(self):
        """Spawn arena floor, two black target areas, and dodecagonal walls."""
        cfg = self.cfg
        R = cfg.arena_circumradius
        n = cfg.arena_num_sides
        wall_h = cfg.arena_wall_height
        wall_thick = 0.01

        floor_side = R * 3.0
        floor_cfg = sim_utils.CuboidCfg(
            size=(floor_side, floor_side, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.45, 0.45),
            ),
        )
        floor_cfg.func("/World/Arena/Floor", floor_cfg, translation=(0.0, 0.0, 0.001))

        for idx, (x, y) in enumerate(cfg.target_centers):
            _spawn_flat_circle(
                f"/World/Arena/BlackTarget_{idx}",
                (x, y),
                cfg.target_radius,
            )

        wall_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.70, 0.40))
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
                visual_material=wall_mat,
            )
            wall_cfg.func(
                f"/World/Arena/Wall_{i}",
                wall_cfg,
                translation=(mx, my, wall_h / 2),
                orientation=(math.cos(seg_angle / 2), 0.0, 0.0, math.sin(seg_angle / 2)),
            )

    def _target_membership(self, pos: torch.Tensor) -> torch.Tensor:
        """Return mask (E, N, 2) for membership in the two black targets."""
        centers = torch.tensor(
            self.cfg.target_centers, dtype=pos.dtype, device=pos.device,
        )
        diff = pos.unsqueeze(2) - centers.view(1, 1, 2, 2)
        dist_sq = (diff * diff).sum(dim=-1)
        return dist_sq <= self.cfg.target_radius ** 2

    def _ground_color(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute ground colour: 0=black target, 0.5=grey elsewhere."""
        in_target = self._target_membership(pos).any(dim=-1)
        grey = torch.full(pos.shape[:2], 0.5, dtype=pos.dtype, device=pos.device)
        color = torch.where(in_target, torch.zeros_like(grey), grey)
        return color.unsqueeze(-1).expand(-1, -1, 3)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        """Reward is the occupancy of the currently most populated target."""
        counts = self._target_membership(self.agent_pos).float().sum(dim=1)
        reward = counts.max(dim=1).values
        self._episode_group_reward += reward
        return {agent: reward for agent in self.cfg.possible_agents}
