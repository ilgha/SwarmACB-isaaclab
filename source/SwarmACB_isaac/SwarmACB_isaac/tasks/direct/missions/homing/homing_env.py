# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Homing mission environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils

from ..directional_gate.directional_gate_env import DirectionalGateEnv
from ..mission_visuals import spawn_flat_circle
from .homing_env_cfg import HomingEnvCfg


class HomingEnv(DirectionalGateEnv):
    """Homing mission for SwarmACB.

    Robots start in the northern half; reward is issued only on the final
    evaluation step and equals the number of robots inside the southern goal.
    """

    cfg: HomingEnvCfg

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

        spawn_flat_circle(
            "/World/Arena/HomingGoal",
            cfg.goal_center,
            cfg.goal_radius,
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

    def _goal_membership(self, pos: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.cfg.goal_center, dtype=pos.dtype, device=pos.device)
        diff = pos - center.view(1, 1, 2)
        return (diff * diff).sum(dim=-1) <= self.cfg.goal_radius ** 2

    def _ground_color(self, pos: torch.Tensor) -> torch.Tensor:
        in_goal = self._goal_membership(pos)
        grey = torch.full(pos.shape[:2], 0.5, dtype=pos.dtype, device=pos.device)
        color = torch.where(in_goal, torch.zeros_like(grey), grey)
        return color.unsqueeze(-1).expand(-1, -1, 3)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        is_final = self.episode_length_buf >= self.max_episode_length - 1
        counts = self._goal_membership(self.agent_pos).float().sum(dim=1)
        reward = torch.where(is_final, counts, torch.zeros_like(counts))
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

        n_reset = len(idx)
        n_agents = self.cfg.num_agents
        inradius = self.cfg.arena_circumradius * math.cos(math.pi / self.cfg.arena_num_sides)
        safe_r = inradius - self.cfg.robot_radius * 2
        r_rand = torch.sqrt(torch.rand(n_reset, n_agents, device=self.device)) * safe_r
        theta = torch.rand(n_reset, n_agents, device=self.device) * math.pi
        self.agent_pos[idx, :, 0] = r_rand * torch.cos(theta)
        self.agent_pos[idx, :, 1] = r_rand * torch.sin(theta).abs()
        self.agent_yaw[idx] = torch.rand(n_reset, n_agents, device=self.device) * 2 * math.pi - math.pi
        self.prev_ground_color[idx] = self._ground_color(self.agent_pos[idx])[:, :, 0]
