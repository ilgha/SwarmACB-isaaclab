#!/usr/bin/env python3
"""Fail-fast audit of the SwarmACB paper-parity training configuration.

This checker deliberately avoids importing Isaac Lab, so it can run on a login
node before an expensive job is submitted.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml


MISSIONS = {
    "DirGate": ("SwarmACB-DirectionalGate-v0", 120.0, 120_000_000, 120_000),
    "Homing": ("SwarmACB-Homing-v0", 120.0, 120_000_000, 120_000),
    "XOR": ("SwarmACB-XOR-v0", 180.0, 180_000_000, 180_000),
    "Foraging": ("SwarmACB-Foraging-v0", 180.0, 180_000_000, 180_000),
    "Sheltering": ("SwarmACB-Sheltering-v0", 180.0, 180_000_000, 180_000),
}

VARIANTS = {
    "dandelion": (512, 2, None, 3e-4),
    "daisy": (512, 2, None, 3e-4),
    "lily": (512, 2, None, 3e-4),
    "tulip": (128, 1, None, 5e-4),
    "cyclamen": (128, 1, (128, 128), 3e-4),
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.checks = 0

    def equal(self, label: str, actual, expected) -> None:
        self.checks += 1
        if actual != expected:
            self.errors.append(f"{label}: expected {expected!r}, got {actual!r}")

    def close(self, label: str, actual, expected: float, tolerance: float = 1e-9) -> None:
        self.checks += 1
        try:
            matches = abs(float(actual) - expected) <= tolerance
        except (TypeError, ValueError):
            matches = False
        if not matches:
            self.errors.append(f"{label}: expected {expected!r}, got {actual!r}")


def load_network_config_module(root: Path):
    """Load the runtime resolver without importing Isaac Lab."""
    path = (
        root
        / "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/agents/network_config.py"
    )
    spec = importlib.util.spec_from_file_location("swarmacb_network_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load network configuration helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_behavior(path: Path, audit: Audit) -> tuple[str, dict]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    behaviors = document.get("behaviors", {}) if isinstance(document, dict) else {}
    audit.equal(f"{path.name}.behavior_count", len(behaviors), 1)
    if len(behaviors) != 1:
        return "", {}
    return next(iter(behaviors.items()))


def audit_config(
    path: Path,
    mission: str,
    variant: str,
    is_oc: bool,
    audit: Audit,
    network_config,
    learned_oc: bool = False,
) -> None:
    run_name, block = one_behavior(path, audit)
    prefix = path.name
    task, duration, max_steps, interval = MISSIONS[mission]
    hidden, layers, memory, learning_rate = VARIANTS[variant]
    environment = block.get("environment", {})

    audit.equal(f"{prefix}.run_name", run_name, path.stem)
    audit.equal(f"{prefix}.task", block.get("task", environment.get("task")), task)
    audit.equal(f"{prefix}.variant", block.get("variant"), variant)
    expected_trainer = (
        "learned_option_critic"
        if learned_oc
        else ("option_critic" if is_oc else "poca")
    )
    audit.equal(
        f"{prefix}.trainer_type",
        block.get("trainer_type"),
        expected_trainer,
    )

    hyper = block.get("hyperparameters", {})
    audit.equal(f"{prefix}.batch_size", hyper.get("batch_size"), 2048)
    audit.equal(f"{prefix}.buffer_size", hyper.get("buffer_size"), 20480)
    audit.close(f"{prefix}.learning_rate", hyper.get("learning_rate"), learning_rate)
    audit.close(f"{prefix}.epsilon", hyper.get("epsilon"), 0.2)
    audit.close(f"{prefix}.lambda", hyper.get("lambd"), 0.95)
    audit.equal(f"{prefix}.num_epoch", hyper.get("num_epoch"), 3)
    audit.equal(f"{prefix}.lr_schedule", hyper.get("learning_rate_schedule"), "linear")
    audit.equal(f"{prefix}.epsilon_schedule", hyper.get("epsilon_schedule"), "linear")
    audit.equal(f"{prefix}.beta_schedule", hyper.get("beta_schedule"), "linear")

    network = block.get("network_settings", {})
    audit.equal(f"{prefix}.hidden_units", network.get("hidden_units"), hidden)
    audit.equal(f"{prefix}.num_layers", network.get("num_layers"), layers)
    if is_oc:
        audit.equal(f"{prefix}.num_options", network.get("num_options"), 6)
    if learned_oc:
        learned_defaults = {
            "beta": 0.001,
            "intra_option_coef": 1.0,
            "selector_coef": 1.0,
            "local_option_value_coef": 0.5,
            "option_entropy_coef": 0.005,
            "termination_penalty": 0.01,
            "termination_coef": 1.0,
            "termination_entropy_coef": 0.0,
            "initial_termination_probability": 0.05,
            "action_baseline_coef": 0.25,
            "option_baseline_coef": 0.25,
            "attention_diversity_coef": 0.01,
            "attention_temporal_coef": 0.01,
            "initial_log_std": -0.7,
            "min_log_std": -2.5,
            "max_log_std": 0.0,
            "max_grad_norm": 10.0,
            "actor_learning_rate": 0.0001,
            "actor_max_grad_norm": 1.0,
            "target_kl": 0.03,
            "option_value_temperature": 1.0,
        }
        for key, expected in learned_defaults.items():
            audit.close(f"{prefix}.{key}", hyper.get(key), expected)
        audit.equal(
            f"{prefix}.option_hidden_units",
            network.get("option_hidden_units"),
            512,
        )
        audit.equal(
            f"{prefix}.option_num_layers",
            network.get("option_num_layers"),
            2,
        )
    if memory is None:
        audit.equal(f"{prefix}.memory", network.get("memory"), None)
    else:
        memory_block = network.get("memory", {})
        audit.equal(f"{prefix}.memory_size", memory_block.get("memory_size"), memory[0])
        audit.equal(f"{prefix}.sequence_length", memory_block.get("sequence_length"), memory[1])
        if learned_oc:
            audit.equal(
                f"{prefix}.option_memory_size",
                memory_block.get("option_memory_size"),
                64,
            )

    # Exercise the exact dependency-free resolver used by config_loader.py.
    # This catches a valid YAML value silently diverging at trainer runtime.
    resolved = SimpleNamespace(
        hidden_dim=128 if is_oc else 512,
        num_layers=1 if is_oc else 2,
        critic_hidden_dim=128,
        critic_num_layers=2,
        critic_num_heads=4,
        recurrent=is_oc,
        memory_size=128,
        sequence_length=128,
    )
    if is_oc:
        resolved.num_options = 6
    if learned_oc:
        resolved.option_hidden_dim = 512
        resolved.option_num_layers = 2
        resolved.option_memory_size = 64
    network_config.apply_network_settings(
        resolved,
        network,
        block.get("critic_settings", {}),
        variant,
        block,
    )
    audit.equal(f"{prefix}.resolved_actor_hidden", resolved.hidden_dim, hidden)
    audit.equal(f"{prefix}.resolved_actor_layers", resolved.num_layers, layers)
    expected_critic_hidden = 512 if learned_oc else hidden
    expected_critic_layers = 2 if learned_oc else layers
    audit.equal(
        f"{prefix}.resolved_critic_hidden",
        resolved.critic_hidden_dim,
        expected_critic_hidden,
    )
    audit.equal(
        f"{prefix}.resolved_critic_layers",
        resolved.critic_num_layers,
        expected_critic_layers,
    )
    audit.equal(f"{prefix}.resolved_critic_heads", resolved.critic_num_heads, 4)
    audit.equal(f"{prefix}.resolved_recurrent", resolved.recurrent, memory is not None)
    if memory is not None:
        audit.equal(f"{prefix}.resolved_memory_size", resolved.memory_size, memory[0])
        audit.equal(
            f"{prefix}.resolved_sequence_length",
            resolved.sequence_length,
            memory[1],
        )
    if is_oc:
        audit.equal(f"{prefix}.resolved_num_options", resolved.num_options, 6)
    if learned_oc:
        audit.equal(
            f"{prefix}.resolved_option_hidden",
            resolved.option_hidden_dim,
            512,
        )
        audit.equal(
            f"{prefix}.resolved_option_layers",
            resolved.option_num_layers,
            2,
        )
        audit.equal(
            f"{prefix}.resolved_option_memory",
            resolved.option_memory_size,
            64,
        )

    reward = block.get("reward_signals", {}).get("extrinsic", {})
    audit.close(f"{prefix}.gamma", reward.get("gamma"), 0.99)
    audit.close(f"{prefix}.reward_strength", reward.get("strength"), 1.0)
    audit.equal(f"{prefix}.max_steps", block.get("max_steps"), max_steps)
    audit.equal(f"{prefix}.time_horizon", block.get("time_horizon"), 1000)
    audit.equal(f"{prefix}.summary_freq", block.get("summary_freq"), interval)
    audit.equal(f"{prefix}.checkpoint_interval", block.get("checkpoint_interval"), interval)

    audit.equal(f"{prefix}.num_envs", environment.get("num_envs"), 5)
    audit.equal(f"{prefix}.decision_period", environment.get("decision_period"), 1)
    audit.close(f"{prefix}.episode_length_s", environment.get("episode_length_s"), duration)


def class_constants(path: Path, class_name: str) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            values: dict[str, object] = {}
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    try:
                        values[child.target.id] = ast.literal_eval(child.value)
                    except (ValueError, TypeError):
                        pass
            return values
    return {}


def function_call_count(path: Path, function_name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    )


def audit_network_resolution(root: Path, network_config, audit: Audit) -> None:
    audit.equal("paper parity version", network_config.PAPER_PARITY_VERSION, 4)

    loader = (
        root
        / "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/agents/config_loader.py"
    )
    audit.equal(
        "config_loader shared network resolver calls",
        function_call_count(loader, "apply_network_settings"),
        1,
    )

    # An explicit Isaac-only critic override remains supported.
    override = SimpleNamespace(
        hidden_dim=512,
        num_layers=2,
        critic_hidden_dim=128,
        critic_num_layers=2,
        critic_num_heads=4,
        recurrent=False,
        memory_size=128,
        sequence_length=128,
    )
    network_config.apply_network_settings(
        override,
        {"hidden_units": 512, "num_layers": 2},
        {"hidden_units": 64, "num_layers": 3, "num_heads": 2},
        "dandelion",
        {},
    )
    audit.equal("explicit critic hidden override", override.critic_hidden_dim, 64)
    audit.equal("explicit critic layers override", override.critic_num_layers, 3)
    audit.equal("explicit critic heads override", override.critic_num_heads, 2)


def audit_environment_sources(root: Path, audit: Audit) -> None:
    missions = root / "source/SwarmACB_isaac/SwarmACB_isaac/tasks/direct/missions"
    directional = class_constants(
        missions / "directional_gate/directional_gate_env_cfg.py",
        "DirectionalGateEnvCfg",
    )
    source_expectations = {
        "decimation": 1,
        "episode_length_s": 120.0,
        "critic_state_radius": 1.2,
        "max_wheel_speed": 0.16,
        "wheelbase": 0.055,
        "prox_range": 0.10,
        "rab_range": 0.60,
        "rab_loss_probability": 0.85,
        "light_position": (0.0, -1.5, 0.0),
    }
    for name, expected in source_expectations.items():
        if isinstance(expected, tuple):
            audit.equal(f"DirectionalGateEnvCfg.{name}", directional.get(name), expected)
        else:
            audit.close(f"DirectionalGateEnvCfg.{name}", directional.get(name), expected)

    inherited = {
        "homing/homing_env_cfg.py": (
            "HomingEnvCfg",
            {
                "episode_length_s": 120.0,
                "spawn_area_center": (0.0, 0.7),
                "spawn_area_size": (2.0, 0.6),
                "spawn_circle_radius": 0.8,
                "goal_center": (0.0, -0.7),
                "goal_radius": 0.3,
            },
        ),
        "foraging/foraging_env_cfg.py": (
            "ForagingEnvCfg",
            {
                "episode_length_s": 180.0,
                "spawn_area_size": (1.8, 1.8),
                "food_radius": 0.15,
                "food_centers": ((-0.75, 0.0), (0.75, 0.0)),
                "nest_top_y": -0.58,
                "light_position": (0.0, -1.5, 0.0),
            },
        ),
        "sheltering/sheltering_env_cfg.py": (
            "ShelteringEnvCfg",
            {
                "episode_length_s": 180.0,
                "spawn_area_size": (1.8, 1.8),
                "shelter_center": (0.0, 0.0),
                "shelter_size": (0.5, 0.3),
                "light_position": (0.0, -1.5, 0.0),
            },
        ),
    }
    for relative, (class_name, expected_values) in inherited.items():
        values = class_constants(missions / relative, class_name)
        for name, expected in expected_values.items():
            audit.equal(f"{class_name}.{name}", values.get(name), expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of scripts/)",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config_dir = root / "configs"
    audit = Audit()
    network_config = load_network_config_module(root)
    audit_network_resolution(root, network_config, audit)

    expected_files: set[str] = set()
    for mission in MISSIONS:
        for variant in VARIANTS:
            filename = f"{mission}_{variant}.yaml"
            expected_files.add(filename)
            audit_config(
                config_dir / filename,
                mission,
                variant,
                False,
                audit,
                network_config,
            )
        filename = f"OC_{mission}_cyclamen.yaml"
        expected_files.add(filename)
        audit_config(
            config_dir / filename,
            mission,
            "cyclamen",
            True,
            audit,
            network_config,
        )
        learned_filename = f"OC2_{mission}_cyclamen.yaml"
        expected_files.add(learned_filename)
        audit_config(
            config_dir / learned_filename,
            mission,
            "cyclamen",
            True,
            audit,
            network_config,
            learned_oc=True,
        )

    actual_files = {path.name for path in config_dir.glob("*.yaml")}
    audit.equal("config file set", actual_files, expected_files)
    audit_environment_sources(root, audit)

    if audit.errors:
        print(f"Paper-parity audit FAILED ({len(audit.errors)} errors):", file=sys.stderr)
        for error in audit.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"Paper-parity audit passed: {len(expected_files)} configs, "
        f"{audit.checks} checks."
    )
    print("Clock: 10 Hz | robots: 20 | envs: 5 | episodes/design: 5000")
    print(
        f"Fresh paper-parity-v{network_config.PAPER_PARITY_VERSION} checkpoints "
        "are required for comparison."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
