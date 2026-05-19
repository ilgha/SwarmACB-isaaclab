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
import math
import os
import sys
import time

from isaaclab.app import AppLauncher
from _isaac_launch import apply_windows_kit_defaults

parser = argparse.ArgumentParser(description="SwarmACB POCA Evaluation")

# ── Config file (primary) ────────────────────────────────────────
parser.add_argument("--config", type=str, default=None,
                    help="Path to ML-Agents-style YAML config file")

# ── CLI args ─────────────────────────────────────────────────────
parser.add_argument("--task", type=str, default=None,
                    help="Registered Gymnasium task ID; overrides config task")
parser.add_argument("--variant", type=str, default=None,
                    choices=["dandelion", "daisy", "lily", "tulip", "cyclamen"])
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to evaluate")
parser.add_argument("--deterministic", action="store_true",
                    help="Use deterministic (mean) actions instead of sampling")
parser.add_argument("--fast-viewer", action="store_true",
                    help="Use the lightweight visual playback loop instead of the exact IsaacLab env")
parser.add_argument("--exact-env", action="store_true",
                    help=argparse.SUPPRESS)
parser.add_argument("--sim-hz", type=float, default=60.0,
                    help="Fast viewer render / kinematic update rate")
parser.add_argument("--control-hz", type=float, default=10.0,
                    help="Fast viewer policy decision rate")
parser.add_argument("--visual-hz", type=float, default=10.0,
                    help="GUI simulation/render substep rate; 10 Hz stays closest to real time")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_windows_kit_defaults(args, "Play")
if (
    not getattr(args, "headless", False)
    and "--rendering_mode" not in sys.argv
    and getattr(args, "rendering_mode", None) == "balanced"
):
    args.rendering_mode = "performance"
    print("[Play] GUI rendering mode defaulted to performance.", flush=True)


def _launch_fast_viewer_from_play():
    """Replace this process with the fast visual playback script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_script = os.path.join(script_dir, "manual_control_isaac.py")
    task_id = args.task or _task_from_config(args.config) or "SwarmACB-DirectionalGate-v0"
    cmd = [
        sys.executable,
        viewer_script,
        "--task", task_id,
        "--checkpoint", args.checkpoint,
        "--sim-hz", str(args.sim_hz),
        "--control-hz", str(args.control_hz),
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.variant:
        cmd += ["--variant", args.variant]
    if args.deterministic:
        cmd.append("--deterministic")
    print(
        "[Play] GUI mode uses fast 60 Hz visual playback. "
        "Use --exact-env for the slower exact IsaacLab viewer.",
        flush=True,
    )
    os.execv(sys.executable, cmd)


def _task_from_config(config_path: str | None) -> str | None:
    if not config_path:
        return None
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        behaviors = raw.get("behaviors", raw)
        if not behaviors:
            return None
        block = behaviors[next(iter(behaviors))]
        environment = block.get("environment", {})
        return block.get("task", environment.get("task", None))
    except Exception:
        return None


if not getattr(args, "headless", False) and args.fast_viewer and not args.exact_env:
    _launch_fast_viewer_from_play()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────

import importlib

import gymnasium as gym
import torch

import SwarmACB_isaac.tasks  # noqa: F401

from SwarmACB_isaac.tasks.direct.agents.poca_networks import (
    Actor, DiscreteActor, RecurrentDiscreteActor,
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
    env_overrides = {}

    if args.config:
        from SwarmACB_isaac.tasks.direct.agents.config_loader import load_config, print_config
        run_name, cfg_variant, cfg, env_overrides = load_config(args.config)
        if variant is None:
            variant = cfg_variant

    # Evaluation defaults to one environment unless explicitly overridden.
    if args.num_envs is not None:
        env_overrides["num_envs"] = args.num_envs

    if variant is None:
        variant = "dandelion"  # fallback
    task_id = args.task or env_overrides.pop("task", None) or "SwarmACB-DirectionalGate-v0"

    if args.config:
        print_config(run_name, variant, cfg, env_overrides)

    # ── Build env config and apply variant BEFORE gym.make ────────
    env_cfg = _resolve_env_cfg(task_id)
    if hasattr(env_cfg, "update_variant"):
        env_cfg.update_variant(variant)
    if "num_envs" in env_overrides:
        env_cfg.scene.num_envs = env_overrides["num_envs"]
    if "episode_length_s" in env_overrides:
        env_cfg.episode_length_s = env_overrides["episode_length_s"]
    decision_dt = env_cfg.sim.dt * env_cfg.decimation
    if not getattr(args, "headless", False):
        visual_hz = max(args.visual_hz, 1.0)
        env_cfg.decimation = max(1, round(decision_dt * visual_hz))
        env_cfg.sim.dt = decision_dt / env_cfg.decimation
        env_cfg.sim.render_interval = 1
        mode = "smooth" if env_cfg.decimation > 1 else "real-time"
        print(
            f"[Play] Exact {mode} GUI playback: policy_hz={1.0 / decision_dt:.1f}, "
            f"sim_hz={1.0 / env_cfg.sim.dt:.1f}, "
            f"render_hz={1.0 / env_cfg.sim.dt:.1f}, "
            f"decimation={env_cfg.decimation}.",
            flush=True,
        )

    # ── Create environment ────────────────────────────────────────
    env = gym.make(task_id, cfg=env_cfg)

    unwrapped = env.unwrapped
    device = unwrapped.device
    agents = unwrapped.cfg.possible_agents

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    discrete = ckpt.get("discrete", False)
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 2)
    num_actions = ckpt.get("num_actions", 6)
    recurrent = bool(ckpt.get("recurrent", False))
    memory_size = ckpt.get("memory_size", 128)
    if recurrent and not discrete:
        raise ValueError("Recurrent playback is only implemented for discrete actors")

    obs_dict, _ = env.reset()
    obs_dim = obs_dict[agents[0]].shape[-1]
    act_dim = ckpt.get("act_dim", 2)

    print(f"[Play] variant={variant}  discrete={discrete}  "
          f"recurrent={recurrent}  "
          f"hidden={hidden_dim}  layers={num_layers}  "
          f"obs={obs_dim}  act={'discrete(' + str(num_actions) + ')' if discrete else str(act_dim)}")

    if discrete:
        if recurrent:
            actor = RecurrentDiscreteActor(
                obs_dim, num_actions, hidden_dim, num_layers, memory_size,
            ).to(device)
        else:
            actor = DiscreteActor(obs_dim, num_actions, hidden_dim, num_layers).to(device)
    else:
        actor = Actor(obs_dim, act_dim, hidden_dim, num_layers).to(device)

    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # ── Evaluation loop ───────────────────────────────────────────
    episode_rewards = []
    episode_count = 0

    obs_dict, _ = env.reset()
    num_envs = unwrapped.num_envs
    ep_reward = torch.zeros(num_envs, device=device)
    if recurrent:
        memory_h, memory_c = actor.initial_state(num_envs * len(agents), device)
    else:
        memory_h = None
        memory_c = None

    print(f"[Play] Evaluating {args.num_episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'})...")

    eval_start = time.perf_counter()

    while episode_count < args.num_episodes:
        with torch.no_grad():
            action_dict = {}
            if recurrent:
                obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
                flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])
                logits, next_memory = actor.step(flat_obs, (memory_h, memory_c))
                memory_h, memory_c = next_memory[0].detach(), next_memory[1].detach()
                dist = torch.distributions.Categorical(logits=logits)
                if args.deterministic:
                    flat_act = dist.probs.argmax(dim=-1)
                else:
                    flat_act = dist.sample()
                all_actions = flat_act.view(num_envs, len(agents), 1)
                action_dict = {a: all_actions[:, i] for i, a in enumerate(agents)}
            else:
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
                        # ML-Agents preprocessing: clamp(-3,3)/3 before env
                        action_dict[agent] = act.clamp(-3, 3) / 3

        obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)

        ep_reward += reward_dict[agents[0]]

        # Check for done envs
        for ei in range(num_envs):
            done = (terminated_dict[agents[0]][ei] | truncated_dict[agents[0]][ei]).item()
            if done:
                episode_rewards.append(ep_reward[ei].item())
                ep_reward[ei] = 0.0
                if recurrent:
                    start = ei * len(agents)
                    end = start + len(agents)
                    memory_h[:, start:end, :] = 0.0
                    memory_c[:, start:end, :] = 0.0
                episode_count += 1
                if episode_count >= args.num_episodes:
                    break

    # ── Print results ─────────────────────────────────────────────
    import statistics
    eval_elapsed = time.perf_counter() - eval_start
    realtime_target = unwrapped.cfg.episode_length_s * math.ceil(
        args.num_episodes / max(1, num_envs)
    )
    print(f"\n{'='*50}")
    print(f"Results over {len(episode_rewards)} episodes:")
    print(f"  Mean reward : {statistics.mean(episode_rewards):.2f}")
    print(f"  Std reward  : {statistics.stdev(episode_rewards):.2f}" if len(episode_rewards) > 1 else "")
    print(f"  Min reward  : {min(episode_rewards):.2f}")
    print(f"  Max reward  : {max(episode_rewards):.2f}")
    print(f"  Median      : {statistics.median(episode_rewards):.2f}")
    print(f"  Eval time   : {eval_elapsed:.1f}s wall / {realtime_target:.1f}s real-time target")
    print(f"{'='*50}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
