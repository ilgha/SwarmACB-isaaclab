# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Small USD visual helpers shared by mission environments."""

from __future__ import annotations

import math

import omni.usd
from pxr import Gf, UsdGeom, Vt


def spawn_flat_circle(
    prim_path: str,
    center_xy,
    radius: float,
    z: float = 0.004,
    segments: int = 96,
    color: tuple[float, float, float] = (0.02, 0.02, 0.02),
):
    """Spawn a flat circular floor patch as a USD triangle-fan mesh."""
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


def clip_polygon_below_y(points: list[tuple[float, float]], y_max: float) -> list[tuple[float, float]]:
    """Clip a 2-D polygon to the half-plane y <= y_max."""
    if not points:
        return []

    clipped: list[tuple[float, float]] = []
    prev = points[-1]
    prev_inside = prev[1] <= y_max
    for curr in points:
        curr_inside = curr[1] <= y_max
        if curr_inside != prev_inside:
            denom = curr[1] - prev[1]
            alpha = 0.0 if abs(denom) < 1e-8 else (y_max - prev[1]) / denom
            clipped.append((
                prev[0] + alpha * (curr[0] - prev[0]),
                y_max,
            ))
        if curr_inside:
            clipped.append(curr)
        prev, prev_inside = curr, curr_inside
    return clipped


def dodecagon_vertices(circumradius: float, n_sides: int = 12) -> list[tuple[float, float]]:
    """Return arena vertices in the same orientation as the environment walls."""
    return [
        (
            circumradius * math.cos(2 * math.pi * i / n_sides + math.pi / n_sides),
            circumradius * math.sin(2 * math.pi * i / n_sides + math.pi / n_sides),
        )
        for i in range(n_sides)
    ]


def spawn_flat_polygon(
    prim_path: str,
    points_xy: list[tuple[float, float]],
    z: float = 0.004,
    color: tuple[float, float, float] = (0.95, 0.95, 0.95),
):
    """Spawn a flat polygon floor patch as a triangle fan."""
    if len(points_xy) < 3:
        return
    cx = sum(p[0] for p in points_xy) / len(points_xy)
    cy = sum(p[1] for p in points_xy) / len(points_xy)

    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = [Gf.Vec3f(cx, cy, z)]
    points.extend(Gf.Vec3f(float(x), float(y), z) for x, y in points_xy)

    indices: list[int] = []
    n = len(points_xy)
    for i in range(n):
        indices.extend([0, i + 1, (i + 1) % n + 1])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * n)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
