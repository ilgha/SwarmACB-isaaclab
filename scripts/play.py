#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a trained POCA agent.

Usage:
    # Preferred: config + checkpoint
    python scripts/play.py --config configs/DirGate_dandelion.yaml \
        --checkpoint checkpoints/DirGate_dandelion/poca_final.pt

    # Legacy (variant from CLI, architecture auto-detected from checkpoint)
    python scripts/play.py --task SwarmACB-DirectionalGate-v0 \
        --variant daisy --checkpoint checkpoints/poca_daisy/poca_final.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SwarmACB POCA Evaluation")

# ── Config file (primary) ────────────────────────────────────────
parser.add_argument("--config", type=str, default=None,
                    help="Path to ML-Agents-style YAML config file")

# ── CLI args ─────────────────────────────────────────────────────
parser.add_argument("--task", type=str, default="SwarmACB-DirectionalGate-v0")
parser.add_argument("--variant", type=str, default=None,
                    choices=["dandelion", "daisy", "lily", "tulip", "cyclamen"])
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to evaluate")
parser.add_argument("--deterministic", action="store_true",
                    help="Use deterministic (mean) actions instead of sampling")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────

import importlib

import gymnasium as gym
import torch

import SwarmACB_isaac.tasks  # noqa: F401

from SwarmACB_isaac.tasks.direct.agents.poca_networks import (
    Actor, DiscreteActor,
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
    # ── Resolve variant from config / CLI / checkpoint ────────────
    variant = args.variant  # may be None

    if args.config:
        from SwarmACB_isaac.tasks.direct.agents.config_loader import load_config, print_config
        run_name, cfg_variant, cfg, env_overrides = load_config(args.config)
        if variant is None:
            variant = cfg_variant
        print_config(run_name, variant, cfg, env_overrides)

    if variant is None:
        variant = "dandelion"  # fallback

    # ── Build env config and apply variant BEFORE gym.make ────────
    env_cfg = _resolve_env_cfg(args.task)
    if hasattr(env_cfg, "update_variant"):
        env_cfg.update_variant(variant)

    # ── Create environment ────────────────────────────────────────
    env = gym.make(args.task, cfg=env_cfg)

    unwrapped = env.unwrapped
    device = unwrapped.device
    agents = unwrapped.cfg.possible_agents

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    discrete = ckpt.get("discrete", False)
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 2)
    num_actions = ckpt.get("num_actions", 6)

    obs_dict, _ = env.reset()
    obs_dim = obs_dict[agents[0]].shape[-1]
    act_dim = ckpt.get("act_dim", 2)

    print(f"[Play] variant={variant}  discrete={discrete}  "
          f"hidden={hidden_dim}  layers={num_layers}  "
          f"obs={obs_dim}  act={'discrete(' + str(num_actions) + ')' if discrete else str(act_dim)}")

    if discrete:
        actor = DiscreteActor(obs_dim, num_actions, hidden_dim, num_layers).to(device)
    else:
        actor = Actor(obs_dim, act_dim, hidden_dim, num_layers).to(device)

    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # ── Evaluation loop ───────────────────────────────────────────
    episode_rewards = []
    episode_count = 0

    obs_dict, _ = env.reset()
    ep_reward = torch.zeros(args.num_envs, device=device)

    print(f"[Play] Evaluating {args.num_episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'})...")

    while episode_count < args.num_episodes:
        with torch.no_grad():
            action_dict = {}
            for i, agent in enumerate(agents):
                obs = obs_dict[agent]  # (E, obs_dim)
                dist = actor.get_dist(obs)
                if args.deterministic:
                    if discrete:
                        act = dist.probs.argmax(dim=-1)  # (E,)
                    else:
                        act = dist.mean  # (E, act_dim)
                else:
                    act = dist.sample()

                if discrete:
                    action_dict[agent] = act.unsqueeze(-1)  # (E, 1)
                else:
                    action_dict[agent] = act

        obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)

        ep_reward += reward_dict[agents[0]]

        # Check for done envs
        for ei in range(args.num_envs):
            done = (terminated_dict[agents[0]][ei] | truncated_dict[agents[0]][ei]).item()
            if done:
                episode_rewards.append(ep_reward[ei].item())
                ep_reward[ei] = 0.0
                episode_count += 1
                if episode_count >= args.num_episodes:
                    break

    # ── Print results ─────────────────────────────────────────────
    import statistics
    print(f"\n{'='*50}")
    print(f"Results over {len(episode_rewards)} episodes:")
    print(f"  Mean reward : {statistics.mean(episode_rewards):.2f}")
    print(f"  Std reward  : {statistics.stdev(episode_rewards):.2f}" if len(episode_rewards) > 1 else "")
    print(f"  Min reward  : {min(episode_rewards):.2f}")
    print(f"  Max reward  : {max(episode_rewards):.2f}")
    print(f"  Median      : {statistics.median(episode_rewards):.2f}")
    print(f"{'='*50}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
