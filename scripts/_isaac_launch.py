#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Isaac Sim launch helpers for standalone scripts."""

from __future__ import annotations

import os
import re
import sys


FORWARDED_KIT_ARGS_ENV = "SWARMACB_FORWARDED_KIT_ARGS"


def consume_forwarded_kit_args(args, label: str = "IsaacLaunch") -> None:
    """Merge custom Kit args forwarded by a parent script without CLI splitting."""
    forwarded = os.environ.pop(FORWARDED_KIT_ARGS_ENV, "").strip()
    if not forwarded:
        return

    direct = (getattr(args, "kit_args", "") or "").strip()
    # Put direct child arguments last so they take precedence if both sources
    # explicitly set the same Kit setting.
    args.kit_args = " ".join(part for part in (forwarded, direct) if part)
    print(f"[{label}] Received custom Kit args from the parent process.", flush=True)


def _kit_setting_present(kit_args: str, setting_path: str) -> bool:
    """Return True if a Kit setting path is already present in the args string."""
    prefix = f"--{setting_path}"
    return any(token == prefix or token.startswith(f"{prefix}=") for token in kit_args.split())


def _append_kit_args(args, additions: list[tuple[str, str]], label: str) -> None:
    """Append Kit settings unless the caller already supplied the same setting."""
    existing = (getattr(args, "kit_args", "") or "").strip()
    new_args = [
        argument for setting_path, argument in additions
        if not _kit_setting_present(existing, setting_path)
    ]
    if not new_args:
        return

    args.kit_args = " ".join(part for part in [existing, *new_args] if part)
    print(f"[{label}] Added Kit args: {' '.join(new_args)}", flush=True)


def _append_raw_kit_args(args, tokens: list[str], label: str) -> None:
    """Append raw Kit CLI tokens that are not /setting=value pairs."""
    if not tokens:
        return
    existing = (getattr(args, "kit_args", "") or "").strip()
    existing_tokens = existing.split()
    if all(token in existing_tokens for token in tokens):
        return

    args.kit_args = " ".join(part for part in [existing, *tokens] if part)
    print(f"[{label}] Added Kit args: {' '.join(tokens)}", flush=True)


def _parse_gui_resolution(value: str) -> tuple[int, int] | None:
    """Parse WIDTHxHEIGHT strings; return None for disabled resolution overrides."""
    normalized = (value or "").strip().lower()
    if normalized in ("", "native", "none", "off", "0", "0x0"):
        return None
    match = re.fullmatch(r"(\d+)\s*x\s*(\d+)", normalized)
    if match is None:
        raise ValueError(
            f"Invalid --gui-resolution value {value!r}. Use WIDTHxHEIGHT, native, or off."
        )
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _resolve_async_rendering(args, lightweight_viewer: bool) -> bool | None:
    """Resolve the async-rendering mode; ``None`` leaves Isaac's default intact."""
    mode = getattr(args, "gui_async_rendering", "auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    # Async rendering can leave the editor viewport black while a standalone
    # script manually drives SimulationContext.render(). Keep it opt-in.
    return None


def apply_gui_performance_defaults(
    args,
    label: str = "IsaacLaunch",
    *,
    lightweight_viewer: bool = False,
) -> None:
    """Apply GUI-preserving performance defaults for playback scripts.

    The default keeps the same rendering and visible scene features as the
    normal viewer, but uncaps Isaac's run loops and prevents RTX eco-mode from
    pausing an animated viewport. More aggressive quality and asynchronous
    rendering changes remain opt-in CLI flags.
    """
    if getattr(args, "headless", False):
        return

    preset = getattr(args, "gui_performance_preset", "same")
    if preset == "off":
        return

    existing_kit_args = (getattr(args, "kit_args", "") or "").strip()
    if "omni.kit.loop-isaac" not in existing_kit_args.split():
        _append_raw_kit_args(args, ["--enable", "omni.kit.loop-isaac"], label)

    use_rendering_tweaks = preset == "fast"
    if (
        use_rendering_tweaks
        and getattr(args, "rendering_mode", None) == "balanced"
        and not getattr(
            args,
            "rendering_mode_explicit",
            any(
                token == "--rendering_mode" or token.startswith("--rendering_mode=")
                for token in sys.argv
            ),
        )
    ):
        args.rendering_mode = "performance"
        print(f"[{label}] GUI rendering mode defaulted to performance.", flush=True)

    resolution = _parse_gui_resolution(getattr(args, "gui_resolution", "native"))
    if resolution is not None:
        width, height = resolution
        args.width = width
        args.height = height
        args.window_width = width
        args.window_height = height
        print(f"[{label}] GUI resolution defaulted to {width}x{height}.", flush=True)

    existing_kit_args = (getattr(args, "kit_args", "") or "").strip()
    material_setting_explicit = _kit_setting_present(existing_kit_args, "/rtx/debugMaterialType")
    texture_setting_explicit = _kit_setting_present(
        existing_kit_args,
        "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
    )

    disable_materials = (
        getattr(args, "gui_disable_materials", False)
        and not getattr(args, "gui_keep_materials", False)
    )
    texture_budget = float(getattr(args, "gui_texture_budget", 0.0))
    args._gui_runtime_material_override = (
        disable_materials
        and not material_setting_explicit
    )
    args._gui_runtime_texture_budget = texture_budget > 0.0 and not texture_setting_explicit

    bool_defaults: list[tuple[str, str]] = [
        ("/app/runLoopsGlobal/syncToPresent", "--/app/runLoopsGlobal/syncToPresent=false"),
        ("/app/runLoops/main/manualModeEnabled", "--/app/runLoops/main/manualModeEnabled=true"),
        ("/app/runLoops/main/rateLimitEnabled", "--/app/runLoops/main/rateLimitEnabled=false"),
        ("/app/runLoops/present/rateLimitEnabled", "--/app/runLoops/present/rateLimitEnabled=false"),
        ("/app/runLoops/rendering_0/rateLimitEnabled", "--/app/runLoops/rendering_0/rateLimitEnabled=false"),
        ("/rtx/ecoMode/enabled", "--/rtx/ecoMode/enabled=false"),
    ]
    # Convert the CLI strings above to runtime values while retaining the
    # original spelling for Kit startup.
    additions: list[tuple[str, str]] = list(bool_defaults)
    runtime_bools = {
        path: argument.rsplit("=", 1)[-1].lower() == "true"
        for path, argument in bool_defaults
        if not _kit_setting_present(existing_kit_args, path)
    }

    int_defaults: list[tuple[str, int, str]] = [
        ("/app/renderer/sleepMsOnFocus", 0, "--/app/renderer/sleepMsOnFocus=0"),
        ("/app/renderer/sleepMsOutOfFocus", 0, "--/app/renderer/sleepMsOutOfFocus=0"),
    ]
    additions.extend((path, argument) for path, _, argument in int_defaults)
    runtime_ints = {
        path: value
        for path, value, _ in int_defaults
        if not _kit_setting_present(existing_kit_args, path)
    }

    async_rendering = _resolve_async_rendering(args, lightweight_viewer)
    if async_rendering is not None:
        async_defaults = [
            ("/exts/isaacsim.core.throttling/enable_async", False),
            ("/app/asyncRendering", async_rendering),
            ("/app/omni.usd/asyncHandshake", async_rendering),
            ("/omni/replicator/asyncRendering", async_rendering),
        ]
        for path, value in async_defaults:
            argument = f"--{path}={'true' if value else 'false'}"
            additions.append((path, argument))
            if not _kit_setting_present(existing_kit_args, path):
                runtime_bools[path] = value

    args._gui_runtime_bool_settings = runtime_bools
    args._gui_runtime_int_settings = runtime_ints
    if use_rendering_tweaks:
        additions.append(("/rtx/post/dlss/execMode", "--/rtx/post/dlss/execMode=0"))
    if texture_budget > 0.0:
        additions.append((
            "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
            f"--/rtx-transient/resourcemanager/texturestreaming/memoryBudget={texture_budget:g}",
        ))
    if disable_materials:
        additions.append(("/rtx/debugMaterialType", "--/rtx/debugMaterialType=0"))

    cpu_threads = int(getattr(args, "gui_cpu_threads", 0) or 0)
    if cpu_threads > 0:
        additions += [
            (
                "/plugins/carb.tasking.plugin/threadCount",
                f"--/plugins/carb.tasking.plugin/threadCount={cpu_threads}",
            ),
            (
                "/persistent/physics/numThreads",
                f"--/persistent/physics/numThreads={cpu_threads}",
            ),
            (
                "/plugins/omni.tbb.globalcontrol/maxThreadCount",
                f"--/plugins/omni.tbb.globalcontrol/maxThreadCount={cpu_threads}",
            ),
        ]

    _append_kit_args(args, additions, label)


def apply_runtime_gui_performance_settings(args, label: str = "IsaacLaunch") -> None:
    """Re-apply selected settings after Kit starts, when carb is available."""
    if getattr(args, "headless", False):
        return
    if getattr(args, "gui_performance_preset", "same") == "off":
        return
    try:
        import carb

        settings = carb.settings.get_settings()
        for path, value in getattr(args, "_gui_runtime_bool_settings", {}).items():
            settings.set_bool(path, value)
        for path, value in getattr(args, "_gui_runtime_int_settings", {}).items():
            settings.set_int(path, value)
        if getattr(args, "_gui_runtime_material_override", False):
            settings.set_int("/rtx/debugMaterialType", 0)
        texture_budget = float(getattr(args, "gui_texture_budget", 0.0))
        if getattr(args, "_gui_runtime_texture_budget", False):
            settings.set_float(
                "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
                texture_budget,
            )
    except Exception as exc:
        print(f"[{label}] Warning: could not apply runtime GUI performance settings: {exc}", flush=True)


def add_gui_performance_args(parser) -> None:
    """Add shared GUI performance flags to a standalone script parser."""
    group = parser.add_argument_group("SwarmACB GUI performance")
    group.add_argument(
        "--gui-performance-preset",
        choices=["same", "fast", "off"],
        default="same",
        help=(
            "same uncaps Isaac's run loops without changing rendering; "
            "fast also enables rendering-side speed tweaks; off uses stock Isaac settings"
        ),
    )
    group.add_argument(
        "--gui-resolution",
        type=str,
        default="native",
        help="GUI render/window size as WIDTHxHEIGHT; use native/off to keep Isaac's default",
    )
    group.add_argument(
        "--gui-texture-budget",
        type=float,
        default=0.0,
        help="Texture streaming budget fraction of GPU memory; <=0 leaves it unchanged",
    )
    group.add_argument(
        "--gui-disable-materials",
        action="store_true",
        help="Use Isaac's low-cost debug material override for maximum viewport speed",
    )
    group.add_argument(
        "--gui-keep-materials",
        action="store_true",
        help="Deprecated compatibility flag; materials are kept by default",
    )
    group.add_argument(
        "--gui-cpu-threads",
        type=int,
        default=0,
        help="Optional cap for Isaac/PhysX worker threads; 0 leaves Isaac's default",
    )
    group.add_argument(
        "--gui-async-rendering",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Async render thread: auto keeps Isaac's stable default, on forces "
            "it, and off explicitly forces synchronous rendering"
        ),
    )


def apply_windows_kit_defaults(args, label: str = "IsaacLaunch") -> None:
    """Apply Windows-specific Kit defaults that keep Isaac Sim stable locally.

    The RTX 5080 + recent Windows driver setup we use crashes in Isaac Sim 5.1's
    Vulkan RTX path. Prefer D3D12 on Windows unless the caller explicitly passes
    an /app/vulkan setting through --kit_args.
    """
    if os.name != "nt":
        return

    additions: list[str] = []

    existing = (getattr(args, "kit_args", "") or "").strip()
    if not _kit_setting_present(existing, "/app/vulkan"):
        additions.append("--/app/vulkan=false")

    if not _kit_setting_present(existing, "/crashreporter/preserveDump"):
        additions.append("--/crashreporter/preserveDump=true")

    if additions:
        _append_kit_args(
            args,
            [(argument.split("=", 1)[0][2:], argument) for argument in additions],
            label,
        )
