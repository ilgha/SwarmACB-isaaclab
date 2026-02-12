# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Train multi-robot cooperative foraging with POCA.

Usage (from Isaac Lab root):
    python scripts/train_poca.py --num_envs 64
    python scripts/train_poca.py --num_envs 64 --headless
    python scripts/train_poca.py --num_envs 128 --headless --total_steps 20000000

All POCA hyper-parameters can be overridden via CLI flags.
"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="POCA training for SwarmACB Foraging")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Template-Swarmacb-Foraging-Direct-v0")

# POCA hyper-parameters (optional overrides)
parser.add_argument("--horizon", type=int, default=256)
parser.add_argument("--num_epochs", type=int, default=4)
parser.add_argument("--mini_batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--total_steps", type=int, default=10_000_000)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/poca")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports that require the Omniverse runtime ────────────────────────
import gymnasium as gym

import SwarmACB_isaac.tasks  # noqa: F401  — registers gym environments
from isaaclab_tasks.utils import parse_env_cfg

from SwarmACB_isaac.tasks.direct.swarmacb_foraging.agents.poca_trainer import (
    POCATrainer,
    POCAConfig,
)


def main():
    # Build env config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Build POCA config from CLI overrides
    poca_cfg = POCAConfig(
        horizon=args_cli.horizon,
        num_epochs=args_cli.num_epochs,
        mini_batch_size=args_cli.mini_batch_size,
        lr=args_cli.lr,
        gamma=args_cli.gamma,
        lam=args_cli.lam,
        clip_eps=args_cli.clip_eps,
        entropy_coef=args_cli.entropy_coef,
        total_timesteps=args_cli.total_steps,
        checkpoint_dir=args_cli.checkpoint_dir,
    )

    trainer = POCATrainer(env, poca_cfg)

    if args_cli.resume:
        trainer.load_checkpoint(args_cli.resume)

    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
