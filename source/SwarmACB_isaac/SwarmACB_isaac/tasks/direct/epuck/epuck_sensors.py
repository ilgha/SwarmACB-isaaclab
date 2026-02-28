# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized e-puck sensor suite for Isaac Lab.

Implements the e-puck reference model RM 1.1 sensors in a fully batched manner
(no Python loops over robots), operating on Isaac Lab's (E, N, ...) tensor layout
where E = num environments, N = num agents.

Sensors implemented:
  - 8 IR proximity sensors (aggregated to single value + angle)
  - 8 light sensors (aggregated to single value + angle)
  - 3 ground sensors (black=0, white=1, grey=0.5 — via floor-zone checking)
  - Range-and-bearing (neighbour detection with optional noise)

The per-step observation vector matches the paper's reference model:
  Dandelion/Daisy (24-dim): 8 prox + 8 light + 3 ground + 1 ztilde + 4 RAB projections
  Lily/Tulip/Cyclamen (4-dim): 3 ground + 1 ztilde
"""

from __future__ import annotations

import math
import torch


# E-puck IR sensor angles (from ARGoS / reference model RM 1.1)
_EPUCK_ANGLES_RAD = torch.tensor([
    math.pi / 10.5884,    # ~17°   — front-right
    math.pi / 3.5999,     # ~50°
    math.pi / 2.0,        # 90°    — right
    math.pi / 1.2,        # 150°
    math.pi / 0.8571,     # 210°
    math.pi / 0.6667,     # 270°   — left
    math.pi / 0.5806,     # 310°
    math.pi / 0.5247,     # 342°   — front-left
], dtype=torch.float32)

# RAB projection angles (four quadrant directions, in radians)
_RAB_PROJ_ANGLES_DEG = torch.tensor([45.0, 135.0, 225.0, 315.0], dtype=torch.float32)
_RAB_PROJ_ANGLES_RAD = _RAB_PROJ_ANGLES_DEG * (math.pi / 180.0)


class EpuckSensors:
    """Vectorized e-puck sensor computation.

    All methods operate on batched tensors: (E, N, ...) where
    E = number of parallel environments, N = number of agents.

    This class does NOT hold state — it's a stateless utility.
    Ground-color transition tracking is done by the environment.
    """

    def __init__(
        self,
        prox_range: float = 0.10,           # IR sensor range in meters
        rab_range: float = 0.20,             # range-and-bearing detection radius
        light_threshold: float = 0.2,        # minimum reading to register light
        alpha_rab: float = 5.0,              # attraction/repulsion gain
        device: str | torch.device = "cuda",
    ):
        self.prox_range = prox_range
        self.rab_range = rab_range
        self.light_threshold = light_threshold
        self.alpha_rab = alpha_rab
        self.device = torch.device(device)

        # Pre-compute sensor geometry on device
        self._angles = _EPUCK_ANGLES_RAD.to(self.device)      # (8,)
        self._cos_a = torch.cos(self._angles)                  # (8,)
        self._sin_a = torch.sin(self._angles)                  # (8,)
        self._rab_cos = torch.cos(_RAB_PROJ_ANGLES_RAD.to(self.device))  # (4,)
        self._rab_sin = torch.sin(_RAB_PROJ_ANGLES_RAD.to(self.device))  # (4,)

    # ──────────────────────────────────────────────────────────────
    #  Proximity sensor (8 IR raycasts → aggregated value + angle)
    # ──────────────────────────────────────────────────────────────

    def compute_proximity(
        self,
        agent_pos: torch.Tensor,     # (E, N, 2) XY world positions
        agent_yaw: torch.Tensor,     # (E, N) heading in radians
        obstacle_segments: list | None = None,  # arena wall segments for raycast
        all_agent_pos: torch.Tensor | None = None,  # (E, N, 2) for inter-robot detection
        robot_radius: float = 0.035,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute vectorized proximity sensor readings.

        Returns:
            prox_values: (E, N, 8) — per-sensor readings in [0, 1]
            prox_value:  (E, N)    — aggregated magnitude
            prox_angle:  (E, N)    — aggregated angle in (-pi, pi)
        """
        E, N = agent_pos.shape[:2]
        device = agent_pos.device

        # Sensor directions in world frame: rotate local sensor angles by agent yaw
        # Local directions: (cos(angle), sin(angle)) in robot frame
        # World directions: rotate by yaw
        cos_y = torch.cos(agent_yaw)  # (E, N)
        sin_y = torch.sin(agent_yaw)

        # Local sensor unit vectors: (8,) → broadcast to (E, N, 8)
        local_x = self._cos_a.view(1, 1, 8)   # forward component
        local_y = self._sin_a.view(1, 1, 8)    # lateral component

        # World-frame directions for each sensor: (E, N, 8)
        world_dx = local_x * cos_y.unsqueeze(-1) - local_y * sin_y.unsqueeze(-1)
        world_dy = local_x * sin_y.unsqueeze(-1) + local_y * cos_y.unsqueeze(-1)

        # Initialize readings at 0 (no obstacle detected)
        prox_values = torch.zeros(E, N, 8, device=device)

        # --- Detect walls (line segments) ---
        if obstacle_segments is not None:
            for (ax, ay, bx, by) in obstacle_segments:
                # Ray-segment intersection for all (E, N, 8) rays
                prox_values = self._raycast_segment(
                    agent_pos, world_dx, world_dy, prox_values,
                    ax, ay, bx, by,
                )

        # --- Detect other robots ---
        if all_agent_pos is not None and N > 1:
            prox_values = self._detect_robots_proximity(
                agent_pos, world_dx, world_dy, prox_values,
                all_agent_pos, robot_radius,
            )

        # Aggregate: weighted sum of unit vectors
        # local_dir for each sensor: (cos_a, sin_a) in body frame
        sum_x = (prox_values * self._cos_a.view(1, 1, 8)).sum(dim=-1)  # (E, N)
        sum_y = (prox_values * self._sin_a.view(1, 1, 8)).sum(dim=-1)

        prox_value = torch.clamp(torch.sqrt(sum_x ** 2 + sum_y ** 2), max=1.0)
        prox_angle = torch.atan2(sum_y, sum_x)

        return prox_values, prox_value, prox_angle

    def _raycast_segment(
        self,
        origins: torch.Tensor,    # (E, N, 2)
        dx: torch.Tensor,         # (E, N, 8)
        dy: torch.Tensor,         # (E, N, 8)
        readings: torch.Tensor,   # (E, N, 8)
        ax: float, ay: float,
        bx: float, by: float,
    ) -> torch.Tensor:
        """Ray-segment intersection for all rays simultaneously."""
        E, N = origins.shape[:2]
        device = origins.device

        # Ray origin (E, N, 1, 2) → broadcast with 8 sensors
        ox = origins[:, :, 0:1]  # (E, N, 1)
        oy = origins[:, :, 1:2]

        # Segment vector
        sx, sy = bx - ax, by - ay

        # Denominator: dx * sy - dy * sx
        denom = dx * sy - dy * sx  # (E, N, 8)
        # Avoid division by zero (parallel rays)
        valid = denom.abs() > 1e-8

        # t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        t = ((ax - ox) * sy - (ay - oy) * sx) / (denom + 1e-12)
        # u = ((ax - ox) * dy - (ay - oy) * dx) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / (denom + 1e-12)

        # Valid hit: t in [0, prox_range], u in [0, 1], and not parallel
        hit = valid & (t >= 0) & (t <= self.prox_range) & (u >= 0) & (u <= 1)

        # Reading = 1 - (distance / range)
        new_reading = torch.where(hit, 1.0 - t / self.prox_range, torch.zeros_like(t))
        # Take max (closest obstacle gives highest reading)
        readings = torch.max(readings, new_reading)

        return readings

    def _detect_robots_proximity(
        self,
        agent_pos: torch.Tensor,     # (E, N, 2) this agent's position
        dx: torch.Tensor,             # (E, N, 8) ray directions
        dy: torch.Tensor,
        readings: torch.Tensor,       # (E, N, 8)
        all_pos: torch.Tensor,        # (E, N, 2) all robots
        robot_radius: float,
    ) -> torch.Tensor:
        """Detect other robots in proximity sensor rays."""
        E, N = agent_pos.shape[:2]

        for j in range(N):
            # Vector from each agent to robot j
            diff_x = all_pos[:, j:j+1, 0] - agent_pos[:, :, 0]  # (E, N)
            diff_y = all_pos[:, j:j+1, 1] - agent_pos[:, :, 1]
            dist = torch.sqrt(diff_x ** 2 + diff_y ** 2 + 1e-12)

            # Skip self (distance ~ 0)
            is_self = dist < 1e-4  # (E, N)

            # Check if robot j is within prox_range
            in_range = dist < (self.prox_range + robot_radius)

            # For each of 8 sensors, check angular alignment
            # Dot product of ray direction with direction to robot j
            dot = dx * diff_x.unsqueeze(-1) + dy * diff_y.unsqueeze(-1)  # (E, N, 8)
            # Normalize by distance
            cos_angle = dot / (dist.unsqueeze(-1) + 1e-8)

            # Hit if cos_angle > ~cos(15°) — narrow cone
            angular_hit = cos_angle > 0.9659  # cos(15°)

            hit = in_range.unsqueeze(-1) & angular_hit & ~is_self.unsqueeze(-1)
            new_reading = torch.where(
                hit,
                (1.0 - dist.unsqueeze(-1) / (self.prox_range + robot_radius)).clamp(0, 1),
                torch.zeros_like(readings),
            )
            readings = torch.max(readings, new_reading)

        return readings

    # ──────────────────────────────────────────────────────────────
    #  Light sensor (8 readings → aggregated value + angle)
    # ──────────────────────────────────────────────────────────────

    def compute_light(
        self,
        agent_pos: torch.Tensor,     # (E, N, 2)
        agent_yaw: torch.Tensor,     # (E, N)
        light_pos: torch.Tensor,     # (2,) or (3,) world position of light source
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute vectorized light sensor readings.

        Matches Unity Epuck.cs: inverse-distance * directional dot product.
        If max reading > threshold → LightValue = max, LightAngle = net angle.

        Returns:
            light_values: (E, N, 8) — per-sensor readings
            light_value:  (E, N)    — aggregated value (0 or max reading)
            light_angle:  (E, N)    — aggregated angle in body frame
        """
        E, N = agent_pos.shape[:2]
        device = agent_pos.device

        # Direction to light in world frame
        lx = light_pos[0] - agent_pos[:, :, 0]  # (E, N)
        ly = light_pos[1] - agent_pos[:, :, 1]
        dist = torch.sqrt(lx ** 2 + ly ** 2 + 1e-6)
        intensity_base = 1.0 / dist  # inverse distance (light.intensity / dist with intensity=1)

        # Sensor world directions
        cos_y = torch.cos(agent_yaw)
        sin_y = torch.sin(agent_yaw)
        local_x = self._cos_a.view(1, 1, 8)
        local_y = self._sin_a.view(1, 1, 8)
        world_dx = local_x * cos_y.unsqueeze(-1) - local_y * sin_y.unsqueeze(-1)
        world_dy = local_x * sin_y.unsqueeze(-1) + local_y * cos_y.unsqueeze(-1)

        # Dot product: sensor direction · direction to light
        norm_lx = lx / (dist + 1e-8)
        norm_ly = ly / (dist + 1e-8)
        dot = world_dx * norm_lx.unsqueeze(-1) + world_dy * norm_ly.unsqueeze(-1)
        dot = dot.clamp(min=0.0)  # Only front-facing hemisphere

        light_values = (intensity_base.unsqueeze(-1) * dot).clamp(0, 1)  # (E, N, 8)

        # Aggregation: max sensor → threshold check
        max_val, _ = light_values.max(dim=-1)  # (E, N)

        # Weighted sum in body frame for angle
        # First convert light direction to body frame
        body_lx = lx * cos_y + ly * sin_y
        body_ly = -lx * sin_y + ly * cos_y
        sum_x = (light_values * self._cos_a.view(1, 1, 8)).sum(dim=-1)
        sum_y = (light_values * self._sin_a.view(1, 1, 8)).sum(dim=-1)

        net_angle = torch.atan2(sum_y, sum_x)

        # Apply threshold
        above = max_val > self.light_threshold
        light_value = torch.where(above, max_val, torch.zeros_like(max_val))
        light_angle = torch.where(above, net_angle, torch.zeros_like(net_angle))

        return light_values, light_value, light_angle

    # ──────────────────────────────────────────────────────────────
    #  Ground sensor (3 channels: black, white, grey)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_ground(
        agent_pos: torch.Tensor,     # (E, N, 2)
        zone_check_fn: callable,     # fn(pos) → str tensor or encoding
    ) -> torch.Tensor:
        """Compute ground sensor reading based on agent position.

        The zone_check_fn is mission-specific (provided by the environment).
        It returns (E, N, 3) sensor values: channel 0-2 all set to the same
        value: 0.0=black, 0.5=grey, 1.0=white.

        Returns:
            ground: (E, N, 3) — ground sensor readings
        """
        return zone_check_fn(agent_pos)

    # ──────────────────────────────────────────────────────────────
    #  Range-and-bearing (neighbor detection)
    # ──────────────────────────────────────────────────────────────

    def compute_rab(
        self,
        agent_pos: torch.Tensor,     # (E, N, 2)
        agent_yaw: torch.Tensor,     # (E, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute range-and-bearing observations.

        Returns:
            ztilde:     (E, N)    — normalized neighbor count: 1 - 2/(1+exp(n))
            rab_proj:   (E, N, 4) — 4 directional projections of RAB sum vector
        """
        E, N = agent_pos.shape[:2]
        device = agent_pos.device

        cos_y = torch.cos(agent_yaw)  # (E, N)
        sin_y = torch.sin(agent_yaw)

        # Pairwise distances (E, N, N)
        dx = agent_pos[:, :, 0].unsqueeze(2) - agent_pos[:, :, 0].unsqueeze(1)
        dy = agent_pos[:, :, 1].unsqueeze(2) - agent_pos[:, :, 1].unsqueeze(1)
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # (E, N, N)

        # Mask: in range and not self (explicit diagonal exclusion)
        not_self = ~torch.eye(
            N, dtype=torch.bool, device=device,
        ).unsqueeze(0)  # (1, N, N)
        in_range = (dist < self.rab_range) & not_self

        # Neighbor count
        n_neighbors = in_range.float().sum(dim=-1)  # (E, N)
        ztilde = 1.0 - 2.0 / (1.0 + torch.exp(n_neighbors))  # (E, N)

        # RAB sum vector in body frame
        # Direction to each neighbor in world frame
        inv_dist = 1.0 / (dist + 1e-8)  # (E, N, N)

        # Body-frame bearing to each neighbor
        # body_x = dx * cos_yaw + dy * sin_yaw (forward direction)
        # body_y = -dx * sin_yaw + dy * cos_yaw (left direction)
        body_x = dx * cos_y.unsqueeze(-1) + dy * sin_y.unsqueeze(-1)   # (E, N, N)
        body_y = -dx * sin_y.unsqueeze(-1) + dy * cos_y.unsqueeze(-1)

        bearing = torch.atan2(body_y, body_x)  # (E, N, N) — bearing in body frame

        # Weighted sum: w = 1/dist * (cos(bearing), sin(bearing))
        w_x = (inv_dist * torch.cos(bearing) * in_range.float()).sum(dim=-1)  # (E, N)
        w_y = (inv_dist * torch.sin(bearing) * in_range.float()).sum(dim=-1)

        # Project onto 4 directions: 45°, 135°, 225°, 315°
        rab_proj = (
            w_x.unsqueeze(-1) * self._rab_cos.view(1, 1, 4)
            + w_y.unsqueeze(-1) * self._rab_sin.view(1, 1, 4)
        )  # (E, N, 4)

        return ztilde, rab_proj

    # ──────────────────────────────────────────────────────────────
    #  Full observation assembly
    # ──────────────────────────────────────────────────────────────

    def collect_obs_dandelion(
        self,
        prox_values: torch.Tensor,   # (E, N, 8)
        light_values: torch.Tensor,  # (E, N, 8)
        ground: torch.Tensor,        # (E, N, 3)
        ztilde: torch.Tensor,        # (E, N)
        rab_proj: torch.Tensor,      # (E, N, 4)
    ) -> torch.Tensor:
        """Assemble 24-dim Dandelion/Daisy observation vector.

        Layout: [8 prox | 8 light | 3 ground | 1 ztilde | 4 RAB] = 24
        """
        return torch.cat([
            prox_values,                 # (E, N, 8)
            light_values,                # (E, N, 8)
            ground,                      # (E, N, 3)
            ztilde.unsqueeze(-1),        # (E, N, 1)
            rab_proj,                    # (E, N, 4)
        ], dim=-1)  # (E, N, 24)

    @staticmethod
    def collect_obs_lily(
        ground: torch.Tensor,        # (E, N, 3)
        ztilde: torch.Tensor,        # (E, N)
    ) -> torch.Tensor:
        """Assemble 4-dim Lily/Tulip/Cyclamen observation vector.

        Layout: [3 ground | 1 ztilde] = 4
        """
        return torch.cat([
            ground,                      # (E, N, 3)
            ztilde.unsqueeze(-1),        # (E, N, 1)
        ], dim=-1)  # (E, N, 4)

    # ──────────────────────────────────────────────────────────────
    #  Critic state (5D polar encoding per agent)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def compute_critic_state_5d(
        agent_pos: torch.Tensor,     # (E, N, 2) world XY
        agent_yaw: torch.Tensor,     # (E, N) heading
        arena_center: torch.Tensor,  # (2,) world XY of arena center
        arena_radius: float,         # normalization radius
        light_dir: torch.Tensor,     # (2,) unit vector from arena center toward light
    ) -> torch.Tensor:
        """Compute 5D critic state per robot: (rho, cos_alpha, sin_alpha, cos_beta, sin_beta).

        Matches the paper's state encoding (Fig. state):
          rho   = distance from arena center, normalized to [0, 1]
          alpha = angle from arena-center-to-light-source axis to arena-center-to-robot
          beta  = robot heading relative to arena-center-to-robot axis

        Returns:
            state: (E, N, 5)
        """
        # Vector from center to robot (E, N, 2)
        rel = agent_pos - arena_center.view(1, 1, 2)
        norm = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)  # (E, N, 1)
        rho = (norm / arena_radius).clamp(0, 1).squeeze(-1)  # (E, N)

        # Unit vector from center to robot
        rhat = rel / norm  # (E, N, 2)

        # alpha = angle from light_dir to rhat
        # cos_alpha = dot(light_dir, rhat), sin_alpha = cross(light_dir, rhat)
        cos_alpha = rhat[:, :, 0] * light_dir[0] + rhat[:, :, 1] * light_dir[1]
        sin_alpha = rhat[:, :, 0] * light_dir[1] - rhat[:, :, 1] * light_dir[0]

        # beta = robot heading relative to center-to-robot axis
        heading = torch.stack([torch.cos(agent_yaw), torch.sin(agent_yaw)], dim=-1)  # (E, N, 2)
        cos_beta = (heading * rhat).sum(dim=-1)
        sin_beta = rhat[:, :, 0] * heading[:, :, 1] - rhat[:, :, 1] * heading[:, :, 0]

        return torch.stack([rho, cos_alpha, sin_alpha, cos_beta, sin_beta], dim=-1)

    # ──────────────────────────────────────────────────────────────
    #  Differential drive kinematics
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def differential_drive(
        left_vel: torch.Tensor,      # (E, N) left wheel speed
        right_vel: torch.Tensor,     # (E, N) right wheel speed
        agent_yaw: torch.Tensor,     # (E, N) current heading
        wheelbase: float,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute differential-drive displacement and heading change.

        Returns:
            dx:       (E, N) world X displacement
            dy:       (E, N) world Y displacement
            d_yaw:    (E, N) heading change in radians
        """
        v = 0.5 * (left_vel + right_vel)
        omega = (right_vel - left_vel) / wheelbase

        cos_y = torch.cos(agent_yaw)
        sin_y = torch.sin(agent_yaw)

        dx = v * cos_y * dt
        dy = v * sin_y * dt
        d_yaw = omega * dt

        return dx, dy, d_yaw
