#!/usr/bin/env python3
# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Headless behavior, performance, and persistence evaluation for Cyclamen controllers.

The script evaluates one episode per checkpoint and measures how much robot-time
is spent in each of the six fixed behavior modules. It also compares episode
reward and temporal persistence, measured from uninterrupted behavior dwell
segments. It supports both classical Cyclamen POCA checkpoints and fixed-option
Option-Critic checkpoints, for any implemented benchmark mission.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path

from isaaclab.app import AppLauncher
from _isaac_launch import apply_windows_kit_defaults


BEHAVIOR_NAMES = [
    "Stop",
    "Exploration",
    "Attraction",
    "Repulsion",
    "Phototaxis",
    "Anti-phototaxis",
]

MISSION_PRESETS = {
    "dirgate": {
        "prefix": "DirGate",
        "display": "Directional Gate",
        "task": "SwarmACB-DirectionalGate-v0",
    },
    "directional_gate": {
        "prefix": "DirGate",
        "display": "Directional Gate",
        "task": "SwarmACB-DirectionalGate-v0",
    },
    "xor": {
        "prefix": "XOR",
        "display": "XOR Aggregation",
        "task": "SwarmACB-XOR-v0",
    },
    "homing": {
        "prefix": "Homing",
        "display": "Homing",
        "task": "SwarmACB-Homing-v0",
    },
    "foraging": {
        "prefix": "Foraging",
        "display": "Foraging",
        "task": "SwarmACB-Foraging-v0",
    },
    "sheltering": {
        "prefix": "Sheltering",
        "display": "Sheltering/SCA",
        "task": "SwarmACB-Sheltering-v0",
    },
    "sca": {
        "prefix": "Sheltering",
        "display": "Sheltering/SCA",
        "task": "SwarmACB-Sheltering-v0",
    },
}


parser = argparse.ArgumentParser(
    description="Evaluate behavior time for Cyclamen and fixed-option OC-Cyclamen checkpoints."
)
parser.add_argument(
    "--mission",
    type=str,
    default="dirgate",
    choices=sorted(MISSION_PRESETS.keys()),
    help="Mission to evaluate.",
)
parser.add_argument(
    "--classical-config",
    type=str,
    default=None,
    help="Classical Cyclamen config. Defaults to configs/<Mission>_cyclamen.yaml.",
)
parser.add_argument(
    "--oc-config",
    type=str,
    default=None,
    help="OC-Cyclamen config. Defaults to configs/OC_<Mission>_cyclamen.yaml.",
)
parser.add_argument("--checkpoint-root", type=str, default="checkpoints")
parser.add_argument(
    "--classical-pattern",
    type=str,
    default=None,
    help=(
        "Checkpoint path pattern relative to --checkpoint-root. Supports "
        "{mission}, {prefix}, {index}, {i}, and {run}. Defaults to "
        "{prefix}_cyclamen_hpc_{index}/poca_final.pt."
    ),
)
parser.add_argument(
    "--oc-pattern",
    type=str,
    default=None,
    help=(
        "Checkpoint path pattern relative to --checkpoint-root. Supports "
        "{mission}, {prefix}, {index}, {i}, and {run}. Defaults to "
        "OC_{prefix}_cyclamen_hpc_{index}/option_critic_final.pt."
    ),
)
parser.add_argument("--num-runs", type=int, default=10)
parser.add_argument(
    "--batch-size",
    type=int,
    default=0,
    help="Number of controllers to evaluate concurrently per method; 0 evaluates all available runs together.",
)
parser.add_argument("--num-envs", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--sequential",
    action="store_true",
    help="Debug mode: evaluate one checkpoint at a time instead of batching controllers.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory. Defaults to analysis/<mission>_behavior_time.",
)
parser.add_argument("--seed", type=int, default=0, help="Base reset seed; use -1 to leave env stochastic.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use argmax actions and threshold OC termination at 0.5. Default is stochastic playback.",
)
parser.add_argument(
    "--allow-missing",
    action="store_true",
    help="Skip missing checkpoints instead of failing.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task override. Defaults to the task in the config, then the selected mission task.",
)
parser.add_argument("--gui", action="store_true", help=argparse.SUPPRESS)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
apply_windows_kit_defaults(args, "BehaviorTime")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import SwarmACB_isaac.tasks  # noqa: F401
from SwarmACB_isaac.tasks.direct.agents.config_loader import load_config
from SwarmACB_isaac.tasks.direct.agents.option_critic_networks import FixedOptionManager
from SwarmACB_isaac.tasks.direct.agents.poca_networks import (
    DiscreteActor,
    RecurrentDiscreteActor,
    checkpoint_memory_size,
)


def _mission_key() -> str:
    return args.mission.lower()


def _mission_preset() -> dict:
    return MISSION_PRESETS[_mission_key()]


def _mission_prefix() -> str:
    return _mission_preset()["prefix"]


def _mission_display() -> str:
    return _mission_preset()["display"]


def _default_classical_config() -> str:
    return f"configs/{_mission_prefix()}_cyclamen.yaml"


def _default_oc_config() -> str:
    return f"configs/OC_{_mission_prefix()}_cyclamen.yaml"


def _classical_config() -> str:
    return args.classical_config or _default_classical_config()


def _oc_config() -> str:
    return args.oc_config or _default_oc_config()


def _classical_pattern() -> str:
    return args.classical_pattern or "{prefix}_cyclamen_hpc_{index}/poca_final.pt"


def _oc_pattern() -> str:
    return args.oc_pattern or "OC_{prefix}_cyclamen_hpc_{index}/option_critic_final.pt"


def _output_dir() -> Path:
    return Path(args.output_dir or f"analysis/{_mission_prefix().lower()}_behavior_time")


def _task_from_overrides(env_overrides: dict) -> str | None:
    return env_overrides.pop("task", None) if "task" in env_overrides else None


def _resolve_env_cfg(task_id: str):
    spec = gym.spec(task_id)
    entry = spec.kwargs.get("env_cfg_entry_point")
    if entry is None:
        raise ValueError(f"Task {task_id} has no env_cfg_entry_point")
    module_path, cls_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()


def _load_env(config_path: str, num_envs: int):
    _run_name, variant, cfg, env_overrides = load_config(config_path)
    decision_period = max(1, int(getattr(cfg, "decision_period", 1)))
    task_id = args.task or _task_from_overrides(env_overrides) or _mission_preset()["task"]
    env_cfg = _resolve_env_cfg(task_id)
    if hasattr(env_cfg, "update_variant"):
        env_cfg.update_variant(variant)
    env_cfg.scene.num_envs = num_envs
    for key, value in env_overrides.items():
        if key == "num_envs":
            continue
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
        else:
            print(f"[BehaviorTime] Warning: ignored unknown environment override {key!r}", flush=True)
    env = gym.make(task_id, cfg=env_cfg)
    env_step_dt = env_cfg.sim.dt * env_cfg.decimation
    decision_dt = env_step_dt * decision_period
    return env, variant, decision_dt, decision_period


def _reset_env(env, seed: int | None):
    if seed is None:
        return env.reset()
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _make_checkpoint_paths(pattern: str) -> list[Path]:
    root = Path(args.checkpoint_root)
    return [
        root / pattern.format(
            mission=_mission_key(),
            prefix=_mission_prefix(),
            index=i,
            i=i,
            run=i,
        )
        for i in range(args.num_runs)
    ]


def _checkpoint_step(path: Path) -> int:
    tail = path.stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _latest_checkpoint_fallback(path: Path) -> Path | None:
    if not path.parent.exists():
        return None
    if path.name == "poca_final.pt":
        candidates = sorted(path.parent.glob("poca_*.pt"), key=_checkpoint_step)
    elif path.name == "option_critic_final.pt":
        candidates = sorted(path.parent.glob("option_critic_*.pt"), key=_checkpoint_step)
    else:
        candidates = []
    candidates = [candidate for candidate in candidates if candidate.name != path.name]
    return candidates[-1] if candidates else None


def _validate_checkpoints(paths: list[Path], method: str) -> list[Path]:
    existing: list[Path] = []
    missing: list[Path] = []
    for path in paths:
        if path.exists():
            existing.append(path)
            continue
        fallback = _latest_checkpoint_fallback(path)
        if fallback is not None:
            print(
                f"[BehaviorTime] {method}: using latest checkpoint {fallback} "
                f"instead of missing {path.name}.",
                flush=True,
            )
            existing.append(fallback)
        else:
            missing.append(path)
    if missing and not args.allow_missing:
        sample = "\n".join(f"  - {path}" for path in missing[:5])
        extra = "" if len(missing) <= 5 else f"\n  ... and {len(missing) - 5} more"
        raise FileNotFoundError(
            f"Missing {len(missing)} {method} checkpoint(s):\n{sample}{extra}\n"
            "Use --allow-missing to skip them, or adjust --checkpoint-root/--*-pattern."
        )
    if missing:
        print(f"[BehaviorTime] {method}: skipping {len(missing)} missing checkpoint(s).", flush=True)
    return existing


def _build_policy(ckpt: dict, obs_dim: int, device: torch.device):
    trainer_type = ckpt.get("trainer_type", "poca")
    discrete = bool(ckpt.get("discrete", False))
    hidden_dim = ckpt.get("hidden_dim", 256)
    num_layers = ckpt.get("num_layers", 2)
    num_actions = ckpt.get("num_actions", 6)
    num_options = ckpt.get("num_options", num_actions)
    recurrent = bool(ckpt.get("recurrent", False))
    memory_size = checkpoint_memory_size(ckpt)

    if trainer_type == "option_critic":
        manager = FixedOptionManager(
            obs_dim, num_options, hidden_dim, num_layers, memory_size,
        ).to(device)
        manager.load_state_dict(ckpt["manager"])
        manager.eval()
        return {
            "trainer_type": trainer_type,
            "model": manager,
            "recurrent": True,
            "memory_size": memory_size,
            "num_actions": num_options,
        }

    if not discrete:
        raise ValueError("This behavior-time evaluator expects discrete fixed-module Cyclamen checkpoints.")

    if recurrent:
        actor = RecurrentDiscreteActor(
            obs_dim, num_actions, hidden_dim, num_layers, memory_size,
        ).to(device)
    else:
        actor = DiscreteActor(obs_dim, num_actions, hidden_dim, num_layers).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return {
        "trainer_type": trainer_type,
        "model": actor,
        "recurrent": recurrent,
        "memory_size": memory_size,
        "num_actions": num_actions,
    }


def _count_actions(action_ids: torch.Tensor, active_mask: torch.Tensor, counts: torch.Tensor):
    active_actions = action_ids[active_mask]
    if active_actions.numel() == 0:
        return
    counts += torch.bincount(
        active_actions.reshape(-1).clamp(min=0, max=len(BEHAVIOR_NAMES) - 1),
        minlength=len(BEHAVIOR_NAMES),
    ).to(counts)


def _count_actions_for_env(action_ids: torch.Tensor, counts: torch.Tensor):
    counts += torch.bincount(
        action_ids.reshape(-1).clamp(min=0, max=len(BEHAVIOR_NAMES) - 1),
        minlength=len(BEHAVIOR_NAMES),
    ).to(counts)


def _obs_for_env(obs_dict: dict, agents: list[str], env_index: int) -> torch.Tensor:
    obs = torch.stack([obs_dict[agent][env_index] for agent in agents], dim=0)
    if obs.ndim > 2:
        obs = obs.reshape(obs.shape[0], -1)
    return obs


def _behavior_key(behavior: str) -> str:
    return behavior.lower().replace("-", "_").replace(" ", "_")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _usage_entropy(fractions: torch.Tensor) -> tuple[float, float]:
    valid = fractions[fractions > 0]
    if valid.numel() == 0:
        return 0.0, 0.0
    entropy = -(valid * valid.log()).sum().item()
    return entropy, entropy / math.log(len(BEHAVIOR_NAMES))


def _record_persistence_step(
    env_index: int,
    action_ids: torch.Tensor,
    prev_actions: torch.Tensor,
    dwell_steps: torch.Tensor,
    dwell_segments: list[list[tuple[int, int]]],
    switch_counts: torch.Tensor,
):
    previous = prev_actions[env_index]
    lengths = dwell_steps[env_index]
    valid_previous = previous >= 0
    changed = valid_previous & (action_ids != previous)

    if bool(changed.any().item()):
        switch_counts[env_index] += changed.sum().to(
            dtype=switch_counts.dtype,
            device=switch_counts.device,
        )
        for robot_index in changed.nonzero(as_tuple=False).flatten().tolist():
            length = int(lengths[robot_index].item())
            if length > 0:
                dwell_segments[env_index].append((int(previous[robot_index].item()), length))

    new_segment = (~valid_previous) | changed
    lengths[new_segment] = 0
    prev_actions[env_index] = action_ids
    lengths += 1


def _finalize_persistence_env(
    env_index: int,
    prev_actions: torch.Tensor,
    dwell_steps: torch.Tensor,
    dwell_segments: list[list[tuple[int, int]]],
):
    previous = prev_actions[env_index]
    lengths = dwell_steps[env_index]
    valid_segments = (previous >= 0) & (lengths > 0)
    for robot_index in valid_segments.nonzero(as_tuple=False).flatten().tolist():
        dwell_segments[env_index].append((
            int(previous[robot_index].item()),
            int(lengths[robot_index].item()),
        ))
    previous.fill_(-1)
    lengths.zero_()


def _evaluate_checkpoints_batch(
    env,
    controller_specs: list[dict],
    decision_dt: float,
    decision_period: int,
    seed: int | None,
) -> list[dict]:
    obs_dict, _ = _reset_env(env, seed)
    unwrapped = env.unwrapped
    device = unwrapped.device
    agents = unwrapped.cfg.possible_agents
    num_envs = unwrapped.num_envs
    num_agents = len(agents)
    active_count = len(controller_specs)

    if num_envs < active_count:
        raise ValueError(
            f"Batch env count ({num_envs}) must be >= controller count ({active_count})."
        )

    policies = []
    for env_index, spec in enumerate(controller_specs):
        checkpoint_path = spec["checkpoint"]
        obs_dim = _obs_for_env(obs_dict, agents, env_index).shape[-1]
        ckpt = torch.load(checkpoint_path, map_location=device)
        policy = _build_policy(ckpt, obs_dim, device)
        policy["method"] = spec["method"]
        policy["checkpoint"] = checkpoint_path
        policy["run_index"] = _infer_run_index(checkpoint_path)
        if policy["trainer_type"] == "option_critic":
            memory_h, memory_c = policy["model"].initial_state(num_agents, device)
            policy["memory_h"] = memory_h
            policy["memory_c"] = memory_c
            policy["current_options"] = torch.full(
                (num_agents,),
                -1,
                dtype=torch.long,
                device=device,
            )
        elif policy["recurrent"]:
            memory_h, memory_c = policy["model"].initial_state(num_agents, device)
            policy["memory_h"] = memory_h
            policy["memory_c"] = memory_c
            policy["current_options"] = None
        else:
            policy["memory_h"] = None
            policy["memory_c"] = None
            policy["current_options"] = None
        policies.append(policy)

    counts = torch.zeros(
        (active_count, len(BEHAVIOR_NAMES)),
        dtype=torch.float64,
        device=device,
    )
    ep_reward = torch.zeros(active_count, device=device)
    active_envs = torch.ones(active_count, dtype=torch.bool, device=device)
    episode_lengths = torch.zeros(active_count, dtype=torch.long, device=device)
    prev_actions = torch.full(
        (active_count, num_agents),
        -1,
        dtype=torch.long,
        device=device,
    )
    dwell_steps = torch.zeros(
        (active_count, num_agents),
        dtype=torch.long,
        device=device,
    )
    switch_counts = torch.zeros(active_count, dtype=torch.float64, device=device)
    dwell_segments: list[list[tuple[int, int]]] = [[] for _ in range(active_count)]

    with torch.no_grad():
        while active_envs.any():
            action_ids = torch.zeros(
                (num_envs, num_agents),
                dtype=torch.long,
                device=device,
            )

            for env_index, policy in enumerate(policies):
                if not bool(active_envs[env_index].item()):
                    continue

                model = policy["model"]
                obs = _obs_for_env(obs_dict, agents, env_index)
                if policy["trainer_type"] == "option_critic":
                    option_logits, termination_logits, next_memory = model.step(
                        obs,
                        (policy["memory_h"], policy["memory_c"]),
                    )
                    policy["memory_h"] = next_memory[0].detach()
                    policy["memory_c"] = next_memory[1].detach()
                    option_dist = torch.distributions.Categorical(logits=option_logits)
                    if args.deterministic:
                        proposed = option_dist.probs.argmax(dim=-1)
                    else:
                        proposed = option_dist.sample()

                    current_options = policy["current_options"]
                    force_new = current_options < 0
                    safe_current = current_options.clamp(min=0)
                    beta_logits = termination_logits.gather(
                        -1,
                        safe_current.unsqueeze(-1),
                    ).squeeze(-1)
                    if args.deterministic:
                        terminate = torch.sigmoid(beta_logits) > 0.5
                    else:
                        terminate = torch.distributions.Bernoulli(
                            logits=beta_logits,
                        ).sample().bool()
                    switch = terminate | force_new
                    current_options = torch.where(switch, proposed, current_options)
                    policy["current_options"] = current_options
                    action_ids[env_index] = current_options

                elif policy["recurrent"]:
                    logits, next_memory = model.step(
                        obs,
                        (policy["memory_h"], policy["memory_c"]),
                    )
                    policy["memory_h"] = next_memory[0].detach()
                    policy["memory_c"] = next_memory[1].detach()
                    dist = torch.distributions.Categorical(logits=logits)
                    if args.deterministic:
                        action_ids[env_index] = dist.probs.argmax(dim=-1)
                    else:
                        action_ids[env_index] = dist.sample()
                else:
                    dist = model.get_dist(obs)
                    if args.deterministic:
                        action_ids[env_index] = dist.probs.argmax(dim=-1)
                    else:
                        action_ids[env_index] = dist.sample()

                _count_actions_for_env(action_ids[env_index], counts[env_index])
                _record_persistence_step(
                    env_index,
                    action_ids[env_index],
                    prev_actions,
                    dwell_steps,
                    dwell_segments,
                    switch_counts,
                )

            action_dict = {
                agent: action_ids[:, i].unsqueeze(-1)
                for i, agent in enumerate(agents)
            }
            decision_active = active_envs.clone()
            done = torch.zeros_like(active_envs)
            for _ in range(decision_period):
                obs_dict, reward_dict, terminated_dict, truncated_dict, _info = env.step(action_dict)

                active_rewards = reward_dict[agents[0]][:active_count]
                ep_reward += active_rewards * decision_active.float()
                episode_lengths[decision_active] += 1
                step_done = (terminated_dict[agents[0]] | truncated_dict[agents[0]])[:active_count]
                newly_done_step = decision_active & step_done
                done = done | newly_done_step
                if newly_done_step.any():
                    for agent in agents:
                        action_dict[agent][newly_done_step] = 0
                decision_active = decision_active & ~step_done
                if not decision_active.any():
                    break
            newly_done = active_envs & done
            if newly_done.any():
                for done_index in newly_done.nonzero(as_tuple=False).flatten().tolist():
                    _finalize_persistence_env(
                        done_index,
                        prev_actions,
                        dwell_steps,
                        dwell_segments,
                    )
            active_envs = active_envs & ~done

    results = []
    for env_index, policy in enumerate(policies):
        total_counts = counts[env_index].sum().clamp(min=1.0)
        seconds = counts[env_index] * decision_dt
        fractions = counts[env_index] / total_counts
        entropy, normalized_entropy = _usage_entropy(fractions)
        segment_steps = [steps for _behavior_id, steps in dwell_segments[env_index]]
        segment_seconds = [steps * decision_dt for steps in segment_steps]
        per_behavior_steps = [
            [steps for behavior_id, steps in dwell_segments[env_index] if behavior_id == target_id]
            for target_id in range(len(BEHAVIOR_NAMES))
        ]
        results.append({
            "mission": _mission_display(),
            "method": policy["method"],
            "run_index": policy["run_index"],
            "checkpoint": str(policy["checkpoint"]),
            "reward_mean": ep_reward[env_index].item(),
            "episode_steps_mean": episode_lengths[env_index].float().item(),
            "total_robot_seconds": seconds.sum().item(),
            "switch_count": switch_counts[env_index].item(),
            "switch_rate": switch_counts[env_index].item() / total_counts.item(),
            "segment_count": len(segment_steps),
            "mean_dwell_steps": _mean(segment_steps),
            "median_dwell_steps": _median(segment_steps),
            "mean_dwell_seconds": _mean(segment_seconds),
            "median_dwell_seconds": _median(segment_seconds),
            "behavior_usage_entropy": entropy,
            "behavior_usage_entropy_norm": normalized_entropy,
            "per_behavior_mean_dwell_seconds": [
                _mean([steps * decision_dt for steps in behavior_steps])
                for behavior_steps in per_behavior_steps
            ],
            "per_behavior_median_dwell_seconds": [
                _median([steps * decision_dt for steps in behavior_steps])
                for behavior_steps in per_behavior_steps
            ],
            "dwell_segments": [
                {
                    "behavior_id": behavior_id,
                    "behavior": BEHAVIOR_NAMES[behavior_id],
                    "dwell_steps": steps,
                    "dwell_seconds": steps * decision_dt,
                }
                for behavior_id, steps in dwell_segments[env_index]
            ],
            "counts": counts[env_index].cpu().tolist(),
            "seconds": seconds.cpu().tolist(),
            "fractions": fractions.cpu().tolist(),
        })
    return results


def _evaluate_checkpoint(
    env,
    checkpoint_path: Path,
    method: str,
    run_index: int,
    decision_dt: float,
    decision_period: int,
    seed: int | None,
) -> dict:
    obs_dict, _ = _reset_env(env, seed)
    unwrapped = env.unwrapped
    device = unwrapped.device
    agents = unwrapped.cfg.possible_agents
    num_envs = unwrapped.num_envs
    num_agents = len(agents)
    obs_dim = obs_dict[agents[0]].shape[-1]

    ckpt = torch.load(checkpoint_path, map_location=device)
    policy = _build_policy(ckpt, obs_dim, device)
    model = policy["model"]
    trainer_type = policy["trainer_type"]
    recurrent = policy["recurrent"]

    counts = torch.zeros(len(BEHAVIOR_NAMES), dtype=torch.float64, device=device)
    ep_reward = torch.zeros(num_envs, device=device)
    active_envs = torch.ones(num_envs, dtype=torch.bool, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)

    if trainer_type == "option_critic":
        memory_h, memory_c = model.initial_state(num_envs * num_agents, device)
        current_options = torch.full(
            (num_envs, num_agents),
            -1,
            dtype=torch.long,
            device=device,
        )
    elif recurrent:
        memory_h, memory_c = model.initial_state(num_envs * num_agents, device)
        current_options = None
    else:
        memory_h = memory_c = current_options = None

    with torch.no_grad():
        while active_envs.any():
            obs_stacked = torch.stack([obs_dict[a] for a in agents], dim=1)
            flat_obs = obs_stacked.reshape(-1, obs_stacked.shape[-1])

            if trainer_type == "option_critic":
                option_logits, termination_logits, next_memory = model.step(
                    flat_obs,
                    (memory_h, memory_c),
                )
                memory_h, memory_c = next_memory[0].detach(), next_memory[1].detach()
                option_dist = torch.distributions.Categorical(logits=option_logits)
                if args.deterministic:
                    proposed = option_dist.probs.argmax(dim=-1)
                else:
                    proposed = option_dist.sample()
                proposed = proposed.view(num_envs, num_agents)

                force_new = current_options < 0
                safe_current = current_options.clamp(min=0).reshape(-1)
                beta_logits = termination_logits.gather(
                    -1,
                    safe_current.unsqueeze(-1),
                ).squeeze(-1)
                if args.deterministic:
                    terminate = (torch.sigmoid(beta_logits) > 0.5).view(num_envs, num_agents)
                else:
                    terminate = torch.distributions.Bernoulli(
                        logits=beta_logits,
                    ).sample().bool().view(num_envs, num_agents)
                switch = terminate | force_new
                current_options = torch.where(switch, proposed, current_options)
                action_ids = current_options

            elif recurrent:
                logits, next_memory = model.step(flat_obs, (memory_h, memory_c))
                memory_h, memory_c = next_memory[0].detach(), next_memory[1].detach()
                dist = torch.distributions.Categorical(logits=logits)
                if args.deterministic:
                    flat_actions = dist.probs.argmax(dim=-1)
                else:
                    flat_actions = dist.sample()
                action_ids = flat_actions.view(num_envs, num_agents)
            else:
                per_agent_actions = []
                for agent in agents:
                    dist = model.get_dist(obs_dict[agent])
                    if args.deterministic:
                        per_agent_actions.append(dist.probs.argmax(dim=-1))
                    else:
                        per_agent_actions.append(dist.sample())
                action_ids = torch.stack(per_agent_actions, dim=1)

            _count_actions(action_ids, active_envs, counts)
            action_dict = {
                agent: action_ids[:, i].unsqueeze(-1)
                for i, agent in enumerate(agents)
            }
            decision_active = active_envs.clone()
            done = torch.zeros_like(active_envs)
            for _ in range(decision_period):
                obs_dict, reward_dict, terminated_dict, truncated_dict, _info = env.step(action_dict)

                ep_reward += reward_dict[agents[0]] * decision_active.float()
                episode_lengths[decision_active] += 1
                step_done = terminated_dict[agents[0]] | truncated_dict[agents[0]]
                newly_done_step = decision_active & step_done
                done = done | newly_done_step
                if newly_done_step.any():
                    for agent in agents:
                        action_dict[agent][newly_done_step] = 0
                decision_active = decision_active & ~step_done
                if not decision_active.any():
                    break
            newly_done = active_envs & done
            if newly_done.any():
                if trainer_type == "option_critic":
                    current_options[newly_done] = -1
                    for env_index in newly_done.nonzero(as_tuple=False).flatten():
                        start = int(env_index.item()) * num_agents
                        end = start + num_agents
                        memory_h[:, start:end, :] = 0.0
                        memory_c[:, start:end, :] = 0.0
                elif recurrent:
                    for env_index in newly_done.nonzero(as_tuple=False).flatten():
                        start = int(env_index.item()) * num_agents
                        end = start + num_agents
                        memory_h[:, start:end, :] = 0.0
                        memory_c[:, start:end, :] = 0.0
                active_envs = active_envs & ~done

    total_counts = counts.sum().clamp(min=1.0)
    seconds = counts * decision_dt
    fractions = counts / total_counts
    return {
        "mission": _mission_display(),
        "method": method,
        "run_index": run_index,
        "checkpoint": str(checkpoint_path),
        "reward_mean": ep_reward.mean().item(),
        "episode_steps_mean": episode_lengths.float().mean().item(),
        "total_robot_seconds": seconds.sum().item(),
        "counts": counts.cpu().tolist(),
        "seconds": seconds.cpu().tolist(),
        "fractions": fractions.cpu().tolist(),
    }


def _write_csv(results: list[dict], output_dir: Path):
    long_path = output_dir / "behavior_time_long.csv"
    summary_path = output_dir / "behavior_time_summary.csv"
    dwell_path = output_dir / "behavior_dwell_segments.csv"
    method_path = output_dir / "method_comparison_summary.csv"

    with long_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "mission",
                "run_index",
                "checkpoint",
                "behavior_id",
                "behavior",
                "count",
                "seconds",
                "fraction",
                "reward_mean",
                "episode_steps_mean",
                "switch_rate",
                "mean_dwell_seconds",
            ],
        )
        writer.writeheader()
        for row in results:
            for behavior_id, behavior in enumerate(BEHAVIOR_NAMES):
                writer.writerow({
                    "method": row["method"],
                    "mission": row["mission"],
                    "run_index": row["run_index"],
                    "checkpoint": row["checkpoint"],
                    "behavior_id": behavior_id,
                    "behavior": behavior,
                    "count": row["counts"][behavior_id],
                    "seconds": row["seconds"][behavior_id],
                    "fraction": row["fractions"][behavior_id],
                    "reward_mean": row["reward_mean"],
                    "episode_steps_mean": row["episode_steps_mean"],
                    "switch_rate": row["switch_rate"],
                    "mean_dwell_seconds": row["mean_dwell_seconds"],
                })

    fieldnames = [
        "method",
        "mission",
        "run_index",
        "checkpoint",
        "reward_mean",
        "episode_steps_mean",
        "total_robot_seconds",
        "switch_count",
        "switch_rate",
        "segment_count",
        "mean_dwell_steps",
        "median_dwell_steps",
        "mean_dwell_seconds",
        "median_dwell_seconds",
        "behavior_usage_entropy",
        "behavior_usage_entropy_norm",
    ]
    for behavior in BEHAVIOR_NAMES:
        key = _behavior_key(behavior)
        fieldnames += [
            f"{key}_seconds",
            f"{key}_fraction",
            f"{key}_mean_dwell_seconds",
            f"{key}_median_dwell_seconds",
        ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out = {
                "method": row["method"],
                "mission": row["mission"],
                "run_index": row["run_index"],
                "checkpoint": row["checkpoint"],
                "reward_mean": row["reward_mean"],
                "episode_steps_mean": row["episode_steps_mean"],
                "total_robot_seconds": row["total_robot_seconds"],
                "switch_count": row["switch_count"],
                "switch_rate": row["switch_rate"],
                "segment_count": row["segment_count"],
                "mean_dwell_steps": row["mean_dwell_steps"],
                "median_dwell_steps": row["median_dwell_steps"],
                "mean_dwell_seconds": row["mean_dwell_seconds"],
                "median_dwell_seconds": row["median_dwell_seconds"],
                "behavior_usage_entropy": row["behavior_usage_entropy"],
                "behavior_usage_entropy_norm": row["behavior_usage_entropy_norm"],
            }
            for behavior_id, behavior in enumerate(BEHAVIOR_NAMES):
                key = _behavior_key(behavior)
                out[f"{key}_seconds"] = row["seconds"][behavior_id]
                out[f"{key}_fraction"] = row["fractions"][behavior_id]
                out[f"{key}_mean_dwell_seconds"] = row["per_behavior_mean_dwell_seconds"][behavior_id]
                out[f"{key}_median_dwell_seconds"] = row["per_behavior_median_dwell_seconds"][behavior_id]
            writer.writerow(out)

    with dwell_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "mission",
                "run_index",
                "checkpoint",
                "behavior_id",
                "behavior",
                "dwell_steps",
                "dwell_seconds",
            ],
        )
        writer.writeheader()
        for row in results:
            for segment in row["dwell_segments"]:
                writer.writerow({
                    "method": row["method"],
                    "mission": row["mission"],
                    "run_index": row["run_index"],
                    "checkpoint": row["checkpoint"],
                    **segment,
                })

    methods = list(dict.fromkeys(row["method"] for row in results))
    metric_names = [
        "reward_mean",
        "episode_steps_mean",
        "total_robot_seconds",
        "switch_rate",
        "mean_dwell_seconds",
        "median_dwell_seconds",
        "behavior_usage_entropy_norm",
    ]
    with method_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "mission", "num_runs"]
        for metric in metric_names:
            fieldnames += [f"{metric}_mean", f"{metric}_std"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            method_rows = [row for row in results if row["method"] == method]
            out = {
                "method": method,
                "mission": _mission_display(),
                "num_runs": len(method_rows),
            }
            for metric in metric_names:
                values = [float(row[metric]) for row in method_rows]
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_std"] = _std(values)
            writer.writerow(out)

    return long_path, summary_path, dwell_path, method_path


def _plot_results(results: list[dict], output_dir: Path):
    colors = ["#6b7280", "#2563eb", "#16a34a", "#ef4444", "#f59e0b", "#7c3aed"]
    method_colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b"]
    methods = list(dict.fromkeys(row["method"] for row in results))
    x = torch.arange(len(BEHAVIOR_NAMES), dtype=torch.float64)
    width = 0.34 if len(methods) <= 2 else 0.8 / max(1, len(methods))

    def _plot_metric_by_method(ax, metric: str, ylabel: str, title: str):
        positions = list(range(len(methods)))
        means = []
        stds = []
        for method in methods:
            values = [float(row[metric]) for row in results if row["method"] == method]
            means.append(_mean(values))
            stds.append(_std(values))
        ax.bar(
            positions,
            means,
            yerr=stds,
            capsize=4,
            color=[method_colors[i % len(method_colors)] for i in range(len(methods))],
            alpha=0.82,
        )
        for method_index, method in enumerate(methods):
            values = [float(row[metric]) for row in results if row["method"] == method]
            if not values:
                continue
            if len(values) == 1:
                jitter = [0.0]
            else:
                jitter = [
                    (i - (len(values) - 1) / 2) * min(0.055, 0.32 / len(values))
                    for i in range(len(values))
                ]
            ax.scatter(
                [method_index + offset for offset in jitter],
                values,
                color="#111827",
                s=22,
                alpha=0.72,
                zorder=3,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for method_index, method in enumerate(methods):
        method_rows = [row for row in results if row["method"] == method]
        values = torch.tensor([row["fractions"] for row in method_rows], dtype=torch.float64)
        mean = values.mean(dim=0)
        std = values.std(dim=0) if values.shape[0] > 1 else torch.zeros_like(mean)
        offset = (method_index - (len(methods) - 1) / 2) * width
        ax.bar(
            (x + offset).numpy(),
            mean.numpy(),
            width=width,
            yerr=std.numpy(),
            capsize=3,
            label=method,
            alpha=0.86,
        )
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(BEHAVIOR_NAMES, rotation=20, ha="right")
    ax.set_ylabel("Fraction of robot-time")
    ax.set_title(f"{_mission_display()} Cyclamen Behavior Usage")
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1]))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    mean_path = output_dir / "behavior_fraction_mean.png"
    fig.savefig(mean_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    labels = [f"{row['method']}\n{row['run_index']}" for row in results]
    bottoms = torch.zeros(len(results), dtype=torch.float64)
    for behavior_id, behavior in enumerate(BEHAVIOR_NAMES):
        values = torch.tensor([row["seconds"][behavior_id] for row in results], dtype=torch.float64)
        ax.bar(labels, values.numpy(), bottom=bottoms.numpy(), label=behavior, color=colors[behavior_id])
        bottoms += values
    ax.set_ylabel("Robot-seconds")
    ax.set_title(f"{_mission_display()} Behavior Time per Controller")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    stacked_path = output_dir / "behavior_seconds_by_controller.png"
    fig.savefig(stacked_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _plot_metric_by_method(
        ax,
        "reward_mean",
        "Episode reward",
        f"{_mission_display()} Performance Comparison",
    )
    fig.tight_layout()
    performance_path = output_dir / "performance_by_method.png"
    fig.savefig(performance_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    _plot_metric_by_method(
        axes[0],
        "switch_rate",
        "Behavior switches / robot-step",
        f"{_mission_display()} Switch Rate",
    )
    _plot_metric_by_method(
        axes[1],
        "mean_dwell_seconds",
        "Mean dwell time (s)",
        f"{_mission_display()} Temporal Persistence",
    )
    fig.tight_layout()
    persistence_path = output_dir / "temporal_persistence_by_method.png"
    fig.savefig(persistence_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for method_index, method in enumerate(methods):
        method_rows = [row for row in results if row["method"] == method]
        ax.scatter(
            [row["mean_dwell_seconds"] for row in method_rows],
            [row["reward_mean"] for row in method_rows],
            label=method,
            s=48,
            color=method_colors[method_index % len(method_colors)],
            alpha=0.82,
        )
        for row in method_rows:
            ax.annotate(
                str(row["run_index"]),
                (row["mean_dwell_seconds"], row["reward_mean"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                alpha=0.75,
            )
    ax.set_xlabel("Mean dwell time (s)")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"{_mission_display()} Reward vs Temporal Persistence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    reward_persistence_path = output_dir / "reward_vs_persistence.png"
    fig.savefig(reward_persistence_path, dpi=180)
    plt.close(fig)

    return (
        mean_path,
        stacked_path,
        performance_path,
        persistence_path,
        reward_persistence_path,
    )


def main():
    output_dir = _output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    classical_config = _classical_config()
    oc_config = _oc_config()
    classical_pattern = _classical_pattern()
    oc_pattern = _oc_pattern()

    print(
        f"[BehaviorTime] Mission={_mission_display()} "
        f"classical_config={classical_config} oc_config={oc_config}",
        flush=True,
    )

    method_specs = [
        (
            "Cyclamen",
            classical_config,
            _validate_checkpoints(_make_checkpoint_paths(classical_pattern), "Cyclamen"),
        ),
        (
            "OC-Cyclamen",
            oc_config,
            _validate_checkpoints(_make_checkpoint_paths(oc_pattern), "OC-Cyclamen"),
        ),
    ]

    controller_specs: list[dict] = []
    for method, config_path, checkpoints in method_specs:
        if not checkpoints:
            print(f"[BehaviorTime] {method}: no checkpoints to evaluate.", flush=True)
            continue
        for checkpoint in checkpoints:
            controller_specs.append({
                "method": method,
                "config": config_path,
                "checkpoint": checkpoint,
            })

    if not controller_specs:
        raise RuntimeError("No evaluations completed.")

    # Avoid creating a second IsaacLab environment in the same process: evaluate
    # both methods in one vectorized env by default. This is also the fastest path.
    base_config = classical_config if any(
        spec["method"] == "Cyclamen" for spec in controller_specs
    ) else controller_specs[0]["config"]
    requested_batch_size = args.num_envs if args.num_envs is not None else args.batch_size
    if args.sequential:
        requested_batch_size = 1
    if requested_batch_size <= 0:
        requested_batch_size = len(controller_specs)
    batch_size = max(1, min(requested_batch_size, len(controller_specs)))

    results: list[dict] = []
    env, variant, decision_dt, decision_period = _load_env(base_config, batch_size)
    print(
        f"[BehaviorTime] Reusing one IsaacLab env: envs={batch_size} "
        f"config={base_config} variant={variant} "
        f"decision_period={decision_period} decision_dt={decision_dt:.3f}s",
        flush=True,
    )
    try:
        for start in range(0, len(controller_specs), batch_size):
            batch = controller_specs[start:start + batch_size]
            batch_seed = None if args.seed < 0 else args.seed + start
            method_counts = {}
            for spec in batch:
                method_counts[spec["method"]] = method_counts.get(spec["method"], 0) + 1
            method_summary = ", ".join(f"{method}={count}" for method, count in method_counts.items())
            print(
                f"[BehaviorTime] Batch {start // batch_size + 1}: controllers={len(batch)} "
                f"{method_summary}",
                flush=True,
            )
            batch_results = _evaluate_checkpoints_batch(
                env,
                batch,
                decision_dt,
                decision_period,
                batch_seed,
            )
            for result in batch_results:
                results.append(result)
                print(
                    f"[BehaviorTime] {result['method']} run={result['run_index']}: "
                    f"reward={result['reward_mean']:.2f} "
                    f"switch={result['switch_rate']:.3f} "
                    f"dwell={result['mean_dwell_seconds']:.2f}s "
                    f"fractions={[round(v, 3) for v in result['fractions']]}",
                    flush=True,
                )
    finally:
        env.close()

    if not results:
        raise RuntimeError("No evaluations completed.")

    csv_paths = _write_csv(results, output_dir)
    plot_paths = _plot_results(results, output_dir)
    print("\n[BehaviorTime] Wrote:")
    for path in (*csv_paths, *plot_paths):
        print(f"  {path}")


def _infer_run_index(path: Path) -> int:
    for part in reversed(path.parts):
        tail = part.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    stem_tail = path.stem.rsplit("_", 1)[-1]
    return int(stem_tail) if stem_tail.isdigit() else 0


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
