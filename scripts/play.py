#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Play / evaluate a trained POCA or fixed-option Option-Critic agent.

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
from _isaac_launch import (
    add_gui_performance_args,
    apply_gui_performance_defaults,
    apply_runtime_gui_performance_settings,
    apply_windows_kit_defaults,
)

parser = argparse.ArgumentParser(description="SwarmACB Evaluation")

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
parser.add_argument("--seed", type=int, default=0,
                    help="Environment and policy sampling seed")
parser.add_argument("--fast-viewer", action="store_true",
                    help="Use the lightweight visual playback loop. This is the GUI default.")
parser.add_argument("--exact-env", action="store_true",
                    help="Use the full IsaacLab environment in GUI instead of the smooth fast viewer")
parser.add_argument("--sim-hz", type=float, default=60.0,
                    help="Fast viewer render / kinematic update rate")
parser.add_argument("--control-hz", type=float, default=10.0,
                    help="Fast viewer policy decision rate")
parser.add_argument("--visual-hz", type=float, default=60.0,
                    help="GUI simulation/render substep rate; 10 Hz stays closest to real time")
parser.add_argument("--status-interval", type=float, default=1.0,
                    help="Seconds between live score/time updates; <=0 disables live status")
parser.add_argument("--no-editor-hud", action="store_true",
                    help="Disable the small Isaac editor playback status window")

add_gui_performance_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_windows_kit_defaults(args, "Play")
apply_gui_performance_defaults(args, "Play")


def _launch_fast_viewer_from_play():
    """Replace this process with the fast visual playback script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viewer_script = os.path.join(script_dir, "manual_control_isaac.py")
    task_id = args.task or _task_from_config(args.config) or "SwarmACB-DirectionalGate-v0"
    control_hz = args.control_hz
    if "--control-hz" not in sys.argv:
        control_hz = 10.0 / max(1, _decision_period_from_config(args.config) or 1)
    cmd = [
        sys.executable,
        viewer_script,
        "--task", task_id,
        "--checkpoint", args.checkpoint,
        "--sim-hz", str(args.sim_hz),
        "--control-hz", str(control_hz),
        "--status-interval", str(args.status_interval),
        "--seed", str(args.seed),
        "--gui-performance-preset", args.gui_performance_preset,
        "--gui-resolution", args.gui_resolution,
        "--gui-texture-budget", str(args.gui_texture_budget),
        "--gui-cpu-threads", str(args.gui_cpu_threads),
    ]
    if args.config:
        cmd += ["--config", args.config]
    if args.variant:
        cmd += ["--variant", args.variant]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.no_editor_hud:
        cmd.append("--no-editor-hud")
    if args.gui_keep_materials:
        cmd.append("--gui-keep-materials")
    if getattr(args, "gui_disable_materials", False):
        cmd.append("--gui-disable-materials")
    if getattr(args, "device", None):
        cmd += ["--device", str(args.device)]
    if getattr(args, "rendering_mode", None):
        cmd += ["--rendering_mode", str(args.rendering_mode)]
    print(
        f"[Play] GUI mode uses fast {args.sim_hz:g} Hz visual playback "
        f"with policy_hz={control_hz:g}. "
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


def _decision_period_from_config(config_path: str | None) -> int | None:
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
        value = environment.get("decision_period", None)
        return int(value) if value is not None else None
    except Exception:
        return None


if not getattr(args, "headless", False) and not args.exact_env:
    _launch_fast_viewer_from_play()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
apply_runtime_gui_performance_settings(args, "Play")

# ── Post-launch imports ───────────────────────────────────────────

import importlib
import random

import gymnasium as gym
import numpy as np
import torch

import SwarmACB_isaac.tasks  # noqa: F401

from SwarmACB_isaac.tasks.direct.agents.poca_networks import (
    Actor, DiscreteActor, RecurrentDiscreteActor, checkpoint_memory_size,
)
from SwarmACB_isaac.tasks.direct.agents.option_critic_networks import (
    FixedOptionManager,
)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(math.ceil(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class _PlaybackStatusHud:
    """Small Isaac editor window for live playback status."""

    def __init__(self, enabled: bool):
        self._label = None
        self._window = None
        if not enabled:
            return
        try:
            import omni.ui as ui

            self._window = ui.Window("SwarmACB Playback", width=360, height=132)
            with self._window.frame:
                with ui.VStack(spacing=4):
                    ui.Label("SwarmACB Playback", height=22)
                    self._label = ui.Label("", word_wrap=True)
        except Exception as exc:
            print(f"[Play] Warning: could not create editor HUD: {exc}", flush=True)

    def update(self, text: str):
        if self._label is not None:
            self._label.text = text


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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Resolve variant from config / CLI / checkpoint ────────────
    variant = args.variant  # may be None
    env_overrides = {}
    decision_period = 1

    if args.config:
        from SwarmACB_isaac.tasks.direct.agents.config_loader import load_config, print_config
        run_name, cfg_variant, cfg, env_overrides = load_config(args.config)
        cfg.seed = args.seed
        decision_period = max(1, int(getattr(cfg, "decision_period", decision_period)))
        if variant is None:
            variant = cfg_variant

    ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    if variant is None and ckpt_meta.get("variant") is not None:
        variant = ckpt_meta["variant"]

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
    env_cfg.seed = args.seed
    if hasattr(env_cfg, "update_variant"):
        env_cfg.update_variant(variant)
    for key, value in env_overrides.items():
        if key == "num_envs":
            env_cfg.scene.num_envs = value
        elif hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
        else:
            print(f"[Play] Warning: ignored unknown environment override {key!r}")
    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    policy_dt = env_step_dt * decision_period
    if not getattr(args, "headless", False):
        visual_hz = max(args.visual_hz, 1.0)
        env_cfg.decimation = max(1, round(env_step_dt * visual_hz))
        env_cfg.sim.dt = env_step_dt / env_cfg.decimation
        env_cfg.sim.render_interval = 1
        mode = "smooth" if env_cfg.decimation > 1 else "real-time"
        print(
            f"[Play] Exact {mode} GUI playback: policy_hz={1.0 / policy_dt:.1f}, "
            f"decision_period={decision_period}, "
            f"env_step_hz={1.0 / env_step_dt:.1f}, "
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
    trainer_type = ckpt.get("trainer_type", "poca")
    discrete = ckpt.get("discrete", False)
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 2)
    num_actions = ckpt.get("num_actions", 6)
    num_options = ckpt.get("num_options", num_actions)
    recurrent = bool(ckpt.get("recurrent", False))
    memory_size = checkpoint_memory_size(ckpt)
    if trainer_type != "option_critic" and recurrent and not discrete:
        raise ValueError("Recurrent playback is only implemented for discrete actors")

    obs_dict, _ = env.reset()
    obs_dim = obs_dict[agents[0]].shape[-1]
    act_dim = ckpt.get("act_dim", 2)

    print(f"[Play] trainer={trainer_type}  variant={variant}  discrete={discrete}  "
          f"recurrent={recurrent}  "
          f"hidden={hidden_dim}  layers={num_layers}  "
          f"obs={obs_dim}  act={'discrete(' + str(num_actions) + ')' if discrete else str(act_dim)}")

    if trainer_type == "option_critic":
        manager = FixedOptionManager(
            obs_dim, num_options, hidden_dim, num_layers, memory_size,
        ).to(device)
        manager.load_state_dict(ckpt["manager"])
        manager.eval()
        actor = None
    elif discrete:
        if recurrent:
            actor = RecurrentDiscreteActor(
                obs_dim, num_actions, hidden_dim, num_layers, memory_size,
            ).to(device)
        else:
            actor = DiscreteActor(obs_dim, num_actions, hidden_dim, num_layers).to(device)
    else:
        actor = Actor(obs_dim, act_dim, hidden_dim, num_layers).to(device)

    if trainer_type != "option_critic":
        actor.load_state_dict(ckpt["actor"])
        actor.eval()

    # ── Evaluation loop ───────────────────────────────────────────
    episode_rewards = []
    episode_count = 0

    obs_dict, _ = env.reset()
    num_envs = unwrapped.num_envs
    ep_reward = torch.zeros(num_envs, device=device)
    if trainer_type == "option_critic":
        memory_h, memory_c = manager.initial_state(num_envs * len(agents), device)
        current_options = torch.full(
            (num_envs, len(agents)),
            -1,
            dtype=torch.long,
            device=device,
        )
    elif recurrent:
        memory_h, memory_c = actor.initial_state(num_envs * len(agents), device)
    else:
        memory_h = None
        memory_c = None

    print(f"[Play] Evaluating {args.num_episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'})...")
    if trainer_type == "option_critic" and args.deterministic:
        print(
            "[Play] Warning: deterministic Option-Critic playback thresholds "
            "termination probabilities at 0.5. Use stochastic playback to "
            "evaluate the learned call-and-return policy."
        )

    eval_start = time.perf_counter()
    episode_steps = int(getattr(
        unwrapped,
        "max_episode_length",
        round(unwrapped.cfg.episode_length_s / env_step_dt),
    ))
    editor_hud = _PlaybackStatusHud(
        not getattr(args, "headless", False)
        and not args.no_editor_hud
        and args.status_interval > 0.0
    )
    last_status_wall = 0.0

    def _episode_step_zero() -> int:
        step_buf = getattr(unwrapped, "episode_length_buf", None)
        if step_buf is None:
            return 0
        return int(step_buf[0].item())

    def _emit_status(now: float):
        step0 = min(max(_episode_step_zero(), 0), episode_steps)
        elapsed_s = step0 * env_step_dt
        total_s = episode_steps * env_step_dt
        remaining_s = max(0.0, total_s - elapsed_s)
        score0 = float(ep_reward[0].item())
        mean_score = float(ep_reward.mean().item())
        last_score = episode_rewards[-1] if episode_rewards else None
        last_text = f" | last={last_score:.2f}" if last_score is not None else ""
        terminal_text = (
            f"[Play] ep={episode_count}/{args.num_episodes} "
            f"| env0={_format_duration(elapsed_s)}/{_format_duration(total_s)} "
            f"remaining={_format_duration(remaining_s)} "
            f"| score={score0:.2f} mean={mean_score:.2f}"
            f"{last_text} | wall={now - eval_start:.1f}s"
        )
        print(terminal_text, flush=True)

        hud_lines = [
            f"Episodes: {episode_count}/{args.num_episodes}",
            f"Env 0 time: {_format_duration(elapsed_s)} / {_format_duration(total_s)}",
            f"Remaining: {_format_duration(remaining_s)}",
            f"Score: {score0:.2f}   Mean live: {mean_score:.2f}",
        ]
        if last_score is not None:
            hud_lines.append(f"Last completed: {last_score:.2f}")
        editor_hud.update("\n".join(hud_lines))

    if args.status_interval > 0.0:
        last_status_wall = time.perf_counter()
        _emit_status(last_status_wall)

    while episode_count < args.num_episodes:
        with torch.no_grad():
            action_dict = {}
            if trainer_type == "option_critic":
                obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
                flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])
                option_logits, termination_logits, next_memory = manager.step(
                    flat_obs,
                    (memory_h, memory_c),
                )
                memory_h, memory_c = next_memory[0].detach(), next_memory[1].detach()
                option_dist = torch.distributions.Categorical(logits=option_logits)
                if args.deterministic:
                    proposed = option_dist.probs.argmax(dim=-1)
                else:
                    proposed = option_dist.sample()
                proposed = proposed.view(num_envs, len(agents))

                force_new = current_options < 0
                safe_current = current_options.clamp(min=0).reshape(-1)
                beta_logits = termination_logits.gather(
                    -1,
                    safe_current.unsqueeze(-1),
                ).squeeze(-1)
                if args.deterministic:
                    terminate = (torch.sigmoid(beta_logits) > 0.5).view(num_envs, len(agents))
                else:
                    terminate = torch.distributions.Bernoulli(
                        logits=beta_logits,
                    ).sample().bool().view(num_envs, len(agents))
                switch = terminate | force_new
                current_options = torch.where(switch, proposed, current_options)
                all_actions = current_options.unsqueeze(-1)
                action_dict = {a: all_actions[:, i] for i, a in enumerate(agents)}
            elif recurrent:
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

        substep_active = torch.ones(num_envs, dtype=torch.bool, device=device)
        for _ in range(decision_period):
            obs_dict, reward_dict, terminated_dict, truncated_dict, info = env.step(action_dict)

            ep_reward += reward_dict[agents[0]] * substep_active.float()

            done_tensor = terminated_dict[agents[0]] | truncated_dict[agents[0]]
            newly_done = substep_active & done_tensor
            if newly_done.any():
                for ei in newly_done.nonzero(as_tuple=False).flatten().tolist():
                    episode_rewards.append(ep_reward[ei].item())
                    ep_reward[ei] = 0.0
                    if trainer_type == "option_critic":
                        current_options[ei] = -1
                        start = ei * len(agents)
                        end = start + len(agents)
                        memory_h[:, start:end, :] = 0.0
                        memory_c[:, start:end, :] = 0.0
                    elif recurrent:
                        start = ei * len(agents)
                        end = start + len(agents)
                        memory_h[:, start:end, :] = 0.0
                        memory_c[:, start:end, :] = 0.0
                    episode_count += 1
                    if episode_count >= args.num_episodes:
                        break
                for agent in agents:
                    action_dict[agent][newly_done] = 0
                substep_active = substep_active & ~done_tensor
                if episode_count >= args.num_episodes or not substep_active.any():
                    break
        now = time.perf_counter()
        if args.status_interval > 0.0 and now - last_status_wall >= args.status_interval:
            last_status_wall = now
            _emit_status(now)

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
