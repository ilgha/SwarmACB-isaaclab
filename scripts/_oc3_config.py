"""Resolve strict OC3 campaign settings from existing mini mission configs."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import yaml

from _oc3_runtime import agents_module


ALLOWED = {
    "name", "missions", "steps_per_mission", "num_envs_per_mission", "rollout_steps",
    "checkpoint_rounds", "keep_checkpoints", "worker_threads", "worker_timeout_s",
    "reward_scales",
}
TASKS = {
    "DirGate": "SwarmACB-DirectionalGate-v0", "XOR": "SwarmACB-XOR-v0",
    "Homing": "SwarmACB-Homing-v0", "Foraging": "SwarmACB-Foraging-v0",
    "Sheltering": "SwarmACB-Sheltering-v0",
}


def resolve(path, args):
    if not 0 <= args.seed < 2**32 - 5045:
        raise ValueError("Seed must remain a valid NumPy seed after mission offsets")
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"oc3"}:
        raise ValueError("Expected one top-level oc3 configuration")
    config = raw["oc3"]
    if set(config) - ALLOWED:
        raise ValueError(f"Unknown OC3 settings: {set(config) - ALLOWED}")
    required = ALLOWED - {"reward_scales"}
    if required - set(config):
        raise ValueError(f"Missing OC3 settings: {required - set(config)}")
    missions = config["missions"]
    if not isinstance(missions, dict) or not missions:
        raise ValueError("missions must be a nonempty name-to-config mapping")
    if set(missions) - set(TASKS):
        raise ValueError("Use canonical benchmark mission names")
    requested = args.missions or list(missions)
    excluded = args.exclude or []
    if len(set(requested)) != len(requested) or (set(requested) | set(excluded)) - set(missions):
        raise ValueError("Duplicate or unknown mission name")
    selected = [name for name in requested if name not in excluded]
    if not selected:
        raise ValueError("At least one training mission is required")
    if args.transfer and (len(selected) != 1 or args.checkpoint):
        raise ValueError("--transfer needs exactly one --missions target and no --checkpoint")
    steps = args.steps_per_mission if args.steps_per_mission is not None else config["steps_per_mission"]
    envs = args.num_envs if args.num_envs is not None else config["num_envs_per_mission"]
    horizon = args.rollout_steps if args.rollout_steps is not None else config["rollout_steps"]
    for name, value in {**{k: config[k] for k in ("checkpoint_rounds", "keep_checkpoints", "worker_threads", "worker_timeout_s")},
                        "steps_per_mission": steps, "num_envs": envs, "rollout_steps": horizon}.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if steps % (20 * envs):
        raise ValueError("steps_per_mission must be divisible by num_envs * 20 robots")
    scales = config.get("reward_scales", {})
    if set(scales) - set(missions):
        raise ValueError("Unknown mission in reward_scales")
    for value in scales.values():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Reward scales must be finite and positive")
    label = config["name"]
    if args.transfer:
        label += "_transfer_" + selected[0]
    elif selected != list(missions):
        label += "_on_" + "_".join(selected)
    log_dir = Path(args.log_dir or f"runs/{label}").resolve()
    checkpoint_dir = Path(args.checkpoint_dir or f"checkpoints/{label}").resolve()
    load = agents_module("config_loader").load_config
    specs = []
    for name in selected:
        source = (path.parent / missions[name]).resolve()
        _, variant, cfg, environment = load(source)
        if (variant != "cyclamen" or cfg.trainer_type != "learned_option_critic"
                or not cfg.reactive_intra_options or cfg.linear_intra_options or cfg.adaptive_actor_lr):
            raise ValueError("OC3 requires mini configs with non-adaptive actor LR")
        cfg.seed = args.seed + 1009 * list(missions).index(name)
        cfg.total_timesteps = steps
        cfg.horizon = horizon
        cfg.buffer_size_hint = 0
        cfg.reward_strength *= scales.get(name, 1.0)
        cfg.log_dir = str(log_dir / name)
        cfg.checkpoint_dir = str(checkpoint_dir / name)
        task = environment.pop("task")
        if task != TASKS[name]:
            raise ValueError(f"Mission {name} must refer to {TASKS[name]}, not {task}")
        environment["num_envs"] = envs
        specs.append({"name": name, "task": task, "environment": environment, "trainer": asdict(cfg)})
    # Only task reward scaling, RNG seeds and output paths may differ. This
    # prevents mismatched libraries, optimizers or schedules across workers.
    ignored = {"reward_strength", "seed", "checkpoint_interval", "summary_freq", "log_dir", "checkpoint_dir"}
    comparable = [{k: v for k, v in s["trainer"].items() if k not in ignored} for s in specs]
    if any(c != comparable[0] for c in comparable[1:]):
        raise ValueError("Selected base configs have incompatible network/training settings")
    manifest = {
        "specs": [{**s, "trainer": {k: v for k, v in s["trainer"].items() if k not in ("log_dir", "checkpoint_dir")}} for s in specs],
        "seed": args.seed, "steps_per_mission": steps, "num_envs": envs,
        "rollout_steps": horizon, "device": args.device,
    }
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return config, specs, manifest, fingerprint, log_dir, checkpoint_dir
