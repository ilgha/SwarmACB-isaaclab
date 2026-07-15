# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Foraging mission environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils

from ..directional_gate.directional_gate_env import DirectionalGateEnv
from ..mission_visuals import (
    clip_polygon_below_y,
    dodecagon_vertices,
    spawn_flat_circle,
    spawn_flat_polygon,
)
from .foraging_env_cfg import ForagingEnvCfg


class ForagingEnv(DirectionalGateEnv):
    """Foraging mission for SwarmACB.

    A robot scores when it reaches the nest after having visited a food source
    since the beginning of the episode or since its previous nest visit.
    """

    cfg: ForagingEnvCfg

    def __init__(self, cfg: ForagingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._has_food = torch.zeros(self.num_envs, self.cfg.num_agents, dtype=torch.bool, device=self.device)
        self._prev_in_nest = torch.zeros_like(self._has_food)

    def _build_gate_wall_segments(self):
        return []

    def _resolve_gate_wall_collisions(self):
        return

    def _spawn_arena_visuals(self):
        cfg = self.cfg
        R = cfg.arena_circumradius
        n = cfg.arena_num_sides
        wall_h = cfg.arena_wall_height
        wall_thick = 0.01

        floor_side = R * 3.0
        floor_cfg = sim_utils.CuboidCfg(
            size=(floor_side, floor_side, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        )
        floor_cfg.func("/World/Arena/Floor", floor_cfg, translation=(0.0, 0.0, 0.001))

        nest_poly = clip_polygon_below_y(
            dodecagon_vertices(R, n),
            cfg.nest_top_y,
        )
        spawn_flat_polygon(
            "/World/Arena/Nest",
            nest_poly,
            color=(0.95, 0.95, 0.95),
        )

        for idx, center in enumerate(cfg.food_centers):
            spawn_flat_circle(
                f"/World/Arena/Food_{idx}",
                center,
                cfg.food_radius,
                color=(0.02, 0.02, 0.02),
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

        lx, ly, _ = cfg.light_position
        li_cfg = sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15)),
        )
        li_cfg.func("/World/Arena/LightIndicator", li_cfg, translation=(lx, ly, 0.04))

    def _food_ground_membership(self, pos: torch.Tensor) -> torch.Tensor:
        """Circular black floor patches seen by the Unity ground sensor."""
        centers = torch.tensor(self.cfg.food_centers, dtype=pos.dtype, device=pos.device)
        diff = pos.unsqueeze(2) - centers.view(1, 1, 2, 2)
        return (diff * diff).sum(dim=-1) <= self.cfg.food_radius ** 2

    def _food_reward_membership(self, pos: torch.Tensor) -> torch.Tensor:
        """Unity ForagingEnvController's axis-aligned pickup test."""
        centers = torch.tensor(self.cfg.food_centers, dtype=pos.dtype, device=pos.device)
        diff = (pos.unsqueeze(2) - centers.view(1, 1, 2, 2)).abs()
        return (diff <= self.cfg.food_radius).all(dim=-1)

    def _nest_membership(self, pos: torch.Tensor) -> torch.Tensor:
        return pos[:, :, 1] <= self.cfg.nest_top_y

    def _ground_color(self, pos: torch.Tensor) -> torch.Tensor:
        in_food = self._food_ground_membership(pos).any(dim=-1)
        in_nest = self._nest_membership(pos)
        grey = torch.full(pos.shape[:2], 0.5, dtype=pos.dtype, device=pos.device)
        color = torch.where(in_food, torch.zeros_like(grey), grey)
        color = torch.where(in_nest, torch.ones_like(grey), color)
        return color.unsqueeze(-1).expand(-1, -1, 3)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        in_food = self._food_reward_membership(self.agent_pos).any(dim=-1)
        in_nest = self._nest_membership(self.agent_pos)

        self._has_food = self._has_food | in_food
        arrived = in_nest & self._has_food
        reward = arrived.float().sum(dim=1)
        self._has_food = torch.where(arrived, torch.zeros_like(self._has_food), self._has_food)
        self._prev_in_nest = in_nest.clone()

        self._episode_group_reward += reward
        return {agent: reward for agent in self.cfg.possible_agents}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            idx = env_ids.to(device=self.device, dtype=torch.long)
        else:
            idx = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if not hasattr(self, "_has_food"):
            return
        self._has_food[idx] = False
        self._prev_in_nest[idx] = self._nest_membership(self.agent_pos[idx]).clone()
