#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Isaac Sim launch helpers for standalone scripts."""

from __future__ import annotations

import os


def _kit_setting_present(kit_args: str, setting_path: str) -> bool:
    """Return True if a Kit setting path is already present in the args string."""
    prefix = f"--{setting_path}"
    return any(token == prefix or token.startswith(f"{prefix}=") for token in kit_args.split())


def apply_windows_kit_defaults(args, label: str = "IsaacLaunch") -> None:
    """Apply Windows-specific Kit defaults that keep Isaac Sim stable locally.

    The RTX 5080 + recent Windows driver setup we use crashes in Isaac Sim 5.1's
    Vulkan RTX path. Prefer D3D12 on Windows unless the caller explicitly passes
    an /app/vulkan setting through --kit_args.
    """
    if os.name != "nt":
        return

    existing = (getattr(args, "kit_args", "") or "").strip()
    additions: list[str] = []

    if not _kit_setting_present(existing, "/app/vulkan"):
        additions.append("--/app/vulkan=false")

    if not _kit_setting_present(existing, "/crashreporter/preserveDump"):
        additions.append("--/crashreporter/preserveDump=true")

    if additions:
        args.kit_args = " ".join(part for part in [existing, *additions] if part)
        print(f"[{label}] Added Windows Kit args: {' '.join(additions)}", flush=True)
