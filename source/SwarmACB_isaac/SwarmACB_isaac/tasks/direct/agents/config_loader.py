# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""YAML configuration loader for SwarmACB training configs.

Usage
-----
    from SwarmACB_isaac.tasks.direct.agents.config_loader import load_config

    run_name, variant, cfg, env_overrides = load_config("configs/DirGate_cyclamen.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .poca_trainer import POCAConfig
from .option_critic_trainer import FixedOptionCriticConfig


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> tuple[str, str, Any, dict[str, Any]]:
    """Load an ML-Agents-style YAML and return everything the trainer needs.

    Returns
    -------
    run_name : str
        The behaviour key (e.g. ``"DirGate_dandelion"``).
    variant : str
        CASA variant extracted from the config.
    cfg : POCAConfig | FixedOptionCriticConfig
        Fully populated trainer config.
    env_overrides : dict
        Keys that should be applied to the env cfg.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    behaviors = raw.get("behaviors", raw)
    if not behaviors:
        raise ValueError("Config must have a top-level 'behaviors' key.")

    # Take the first (and usually only) behaviour block
    run_name = next(iter(behaviors))
    block = behaviors[run_name]

    variant = block.get("variant", "dandelion")
    trainer_type = block.get("trainer_type", "poca").lower()
    hypers = block.get("hyperparameters", {})
    network = block.get("network_settings", {})
    reward = block.get("reward_signals", {})
    environment = block.get("environment", {})
    task_id = block.get("task", environment.get("task", None))

    # Build the selected trainer config.
    if trainer_type in ("option_critic", "fixed_option_critic", "fixed_oc", "oc"):
        cfg = FixedOptionCriticConfig()
        cfg.trainer_type = "option_critic"
    elif trainer_type == "poca":
        cfg = POCAConfig()
        cfg.trainer_type = "poca"
    else:
        raise ValueError(f"Unsupported trainer_type: {trainer_type}")

    # Hyperparameters
    cfg.mini_batch_size = hypers.get("batch_size", cfg.mini_batch_size)
    cfg.lr = hypers.get("learning_rate", cfg.lr)
    cfg.beta = hypers.get("beta", cfg.beta)
    cfg.clip_eps = hypers.get("epsilon", cfg.clip_eps)
    cfg.lam = hypers.get("lambd", cfg.lam)
    cfg.num_epochs = hypers.get("num_epoch", cfg.num_epochs)
    if hasattr(cfg, "termination_penalty"):
        cfg.termination_penalty = hypers.get("termination_penalty", cfg.termination_penalty)
        cfg.termination_coef = hypers.get("termination_coef", cfg.termination_coef)
        cfg.termination_entropy_coef = hypers.get(
            "termination_entropy_coef",
            cfg.termination_entropy_coef,
        )
        cfg.value_coef = hypers.get("value_coef", cfg.value_coef)
        cfg.option_value_coef = hypers.get("option_value_coef", cfg.option_value_coef)
        cfg.baseline_coef = hypers.get("baseline_coef", cfg.baseline_coef)

    # Schedules: "linear" or "constant"  (default: constant)
    cfg.lr_schedule = hypers.get("learning_rate_schedule", "constant")
    cfg.eps_schedule = hypers.get("epsilon_schedule", "constant")
    cfg.beta_schedule = hypers.get("beta_schedule", "constant")

    # Network
    cfg.hidden_dim = network.get("hidden_units", cfg.hidden_dim)
    cfg.num_layers = network.get("num_layers", cfg.num_layers)
    if hasattr(cfg, "num_options"):
        cfg.num_options = network.get("num_options", block.get("num_options", cfg.num_options))
    memory = network.get("memory", {})
    cfg.recurrent = bool(memory) or variant == "cyclamen"
    if cfg.recurrent:
        cfg.memory_size = memory.get("memory_size", cfg.memory_size)
        cfg.sequence_length = memory.get("sequence_length", cfg.sequence_length)

    # Reward signals
    extrinsic = reward.get("extrinsic", {})
    cfg.gamma = extrinsic.get("gamma", cfg.gamma)
    cfg.reward_strength = extrinsic.get("strength", 1.0)

    # Training control
    cfg.total_timesteps = block.get("max_steps", cfg.total_timesteps)
    cfg.horizon = block.get("time_horizon", cfg.horizon)
    cfg.summary_freq = block.get("summary_freq", 120000)
    cfg.checkpoint_interval = block.get("checkpoint_interval", 120000)
    cfg.keep_checkpoints = block.get("keep_checkpoints", 5)

    # buffer_size hint  (for reference / validation logging only)
    cfg.buffer_size_hint = hypers.get("buffer_size", 0)

    # Environment
    cfg.decision_period = environment.get("decision_period", cfg.decision_period)

    # Logging / checkpoints named after the run
    cfg.log_dir = f"runs/{run_name}"
    cfg.checkpoint_dir = f"checkpoints/{run_name}"

    # ── Environment overrides ─────────────────────────────────────
    env_overrides: dict[str, Any] = {}
    if task_id is not None:
        env_overrides["task"] = task_id
    if "num_envs" in environment:
        env_overrides["num_envs"] = environment["num_envs"]
    if "episode_length_s" in environment:
        env_overrides["episode_length_s"] = environment["episode_length_s"]

    return run_name, variant, cfg, env_overrides


# ──────────────────────────────────────────────────────────────────────
#  Pretty-print
# ──────────────────────────────────────────────────────────────────────

def print_config(run_name: str, variant: str, cfg: Any, env_ov: dict):
    """Print a human-readable summary matching ML-Agents console output."""
    trainer_type = getattr(cfg, "trainer_type", "poca")
    trainer_label = "Fixed Option-Critic" if trainer_type == "option_critic" else "POCA"
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  SwarmACB Training Config")
    print(f"{sep}")
    print(f"  Run name        : {run_name}")
    print(f"  CASA variant    : {variant}")
    print(f"  Trainer         : {trainer_label}")
    print(f"{sep}")
    print(f"  Hyperparameters")
    print(f"    batch_size          : {cfg.mini_batch_size}")
    print(f"    learning_rate       : {cfg.lr}  (schedule: {cfg.lr_schedule})")
    print(f"    beta                : {cfg.beta}  (schedule: {cfg.beta_schedule})")
    print(f"    epsilon             : {cfg.clip_eps}  (schedule: {cfg.eps_schedule})")
    print(f"    lambd               : {cfg.lam}")
    print(f"    num_epoch           : {cfg.num_epochs}")
    print(f"    gamma               : {cfg.gamma}")
    if hasattr(cfg, "termination_penalty"):
        print(f"    termination_penalty : {cfg.termination_penalty}")
        print(f"    termination_coef    : {cfg.termination_coef}")
        print(f"    termination_entropy : {cfg.termination_entropy_coef}")
        print(f"    value_coef          : {cfg.value_coef}")
        print(f"    option_value_coef   : {cfg.option_value_coef}")
        print(f"    baseline_coef       : {cfg.baseline_coef}")
    print(f"  Network")
    print(f"    hidden_units        : {cfg.hidden_dim}")
    print(f"    num_layers          : {cfg.num_layers}")
    if hasattr(cfg, "num_options"):
        print(f"    fixed_options       : {cfg.num_options}")
    if cfg.recurrent:
        print(f"    memory_size         : {cfg.memory_size}")
        print(f"    sequence_length     : {cfg.sequence_length}")
    print(f"  Training")
    print(f"    max_steps           : {cfg.total_timesteps:,}")
    print(f"    time_horizon        : {cfg.horizon}")
    print(f"    decision_period     : {cfg.decision_period}")
    print(f"    checkpoint_interval : {cfg.checkpoint_interval:,}")
    print(f"    summary_freq        : {cfg.summary_freq:,}")
    if cfg.reward_strength != 1.0:
        print(f"    reward_strength     : {cfg.reward_strength}")
    if env_ov:
        print(f"  Environment overrides")
        for k, v in env_ov.items():
            print(f"    {k:22s}: {v}")
    print(f"{sep}\n")
