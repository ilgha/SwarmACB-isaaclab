#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Train a POCA agent on any registered SwarmACB mission.

Usage:
    # Preferred: use a YAML config file (ML-Agents style)
    python scripts/train.py --config configs/DirGate_dandelion.yaml

    # Override specific settings via CLI
    python scripts/train.py --config configs/DirGate_dandelion.yaml --num_envs 10

    # Legacy (no config file — uses CLI args only)
    python scripts/train.py --task SwarmACB-DirectionalGate-v0 --variant daisy

    # Resume from checkpoint
    python scripts/train.py --config configs/DirGate_dandelion.yaml \
        --checkpoint checkpoints/DirGate_dandelion/poca_1000000.pt
"""

from __future__ import annotations

import argparse

# Isaac Lab bootstrap — must happen before other Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SwarmACB POCA Training")

# ── Config file (primary) ────────────────────────────────────────
parser.add_argument("--config", type=str, default=None,
                    help="Path to ML-Agents-style YAML config file")

# ── CLI overrides (used with or without --config) ────────────────
parser.add_argument("--task", type=str, default="SwarmACB-DirectionalGate-v0",
                    help="Registered Gymnasium task ID")
parser.add_argument("--variant", type=str, default=None,
                    choices=["dandelion", "daisy", "lily", "tulip", "cyclamen"],
                    help="CASA variant (overrides config file)")
parser.add_argument("--num_envs", type=int, default=None,
                    help="Override number of parallel envs")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from")
parser.add_argument("--total_timesteps", type=int, default=None,
                    help="Override total training timesteps")
parser.add_argument("--decision_period", type=int, default=None,
                    help="Decision period override")
parser.add_argument("--hidden_dim", type=int, default=None,
                    help="Hidden dim override")
parser.add_argument("--num_layers", type=int, default=None,
                    help="Number of hidden layers override")
parser.add_argument("--log_dir", type=str, default=None,
                    help="TensorBoard log directory override")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Checkpoint save directory override")

# AppLauncher adds its own args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import Isaac Lab packages ─────────────────────────

import importlib

import gymnasium as gym

# Trigger task registration
import SwarmACB_isaac.tasks  # noqa: F401

from SwarmACB_isaac.tasks.direct.agents.poca_trainer import (
    POCATrainer, POCAConfig,
)
from SwarmACB_isaac.tasks.direct.agents.config_loader import (
    load_config, print_config,
)


def _resolve_env_cfg(task_id: str):
    """Instantiate env config from the gym registry entry point."""
    spec = gym.spec(task_id)
    entry = spec.kwargs.get("env_cfg_entry_point")
    if entry is None:
        raise ValueError(
            f"Task {task_id} has no env_cfg_entry_point"
        )
    module_path, cls_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def main():
    # ── Load config ───────────────────────────────────────────────
    if args.config:
        run_name, variant, cfg, env_overrides = load_config(args.config)
    else:
        # Legacy: build config from CLI args alone
        variant = args.variant or "dandelion"
        run_name = f"poca_{variant}_{args.task}"

        if variant in ("tulip", "cyclamen"):
            hd, nl = 128, 1
        else:
            hd, nl = 512, 2

        cfg = POCAConfig(
            hidden_dim=args.hidden_dim or hd,
            num_layers=args.num_layers or nl,
            decision_period=args.decision_period or 1,
        )
        cfg.log_dir = f"runs/{run_name}"
        cfg.checkpoint_dir = f"checkpoints/poca_{variant}"
        env_overrides = {}

    # ── CLI overrides always win ──────────────────────────────────
    if args.variant is not None:
        variant = args.variant
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        cfg.num_layers = args.num_layers
    if args.decision_period is not None:
        cfg.decision_period = args.decision_period
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.num_envs is not None:
        env_overrides["num_envs"] = args.num_envs

    # ── Print resolved config ─────────────────────────────────────
    print_config(run_name, variant, cfg, env_overrides)

    # ── Build env config and apply overrides BEFORE gym.make ─────
    env_cfg = _resolve_env_cfg(args.task)

    if hasattr(env_cfg, "update_variant"):
        env_cfg.update_variant(variant)
    if "num_envs" in env_overrides:
        env_cfg.scene.num_envs = env_overrides["num_envs"]
    if "episode_length_s" in env_overrides:
        env_cfg.episode_length_s = env_overrides["episode_length_s"]

    # ── Create environment ────────────────────────────────────────
    env = gym.make(args.task, cfg=env_cfg)

    # ── Create trainer and run ────────────────────────────────────
    trainer = POCATrainer(env, cfg)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()

    # ── Cleanup ───────────────────────────────────────────────────
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
