# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Run the matched Option-Critic validation battery as isolated Isaac processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _condition_commands(fixed_duration_s: float) -> dict[str, list[str]]:
    duration_tag = f"{fixed_duration_s:g}".replace(".", "p")
    conditions = {
        "dandelion": ["--methods", "dandelion"],
        "cyclamen": ["--methods", "cyclamen"],
        "oc1_learned": ["--methods", "oc1"],
        "oc1_never": ["--methods", "oc1", "--termination-mode", "never"],
        f"oc1_fixed_{duration_tag}s": [
            "--methods", "oc1",
            "--termination-mode", "fixed",
            "--fixed-option-duration-s", str(fixed_duration_s),
        ],
        "oc2_learned": ["--methods", "oc2"],
        "oc2_never": ["--methods", "oc2", "--termination-mode", "never"],
        f"oc2_fixed_{duration_tag}s": [
            "--methods", "oc2",
            "--termination-mode", "fixed",
            "--fixed-option-duration-s", str(fixed_duration_s),
        ],
    }
    for option in range(6):
        conditions[f"oc2_force_{option}"] = [
            "--methods", "oc2",
            "--termination-mode", "never",
            "--force-option", str(option),
        ]
    return conditions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dandelion, Cyclamen, OC1 termination controls, OC2 "
            "termination controls, and every forced OC2 option."
        )
    )
    parser.add_argument("--mission", default="dirgate")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument(
        "--output-root",
        default="analysis/dirgate_option_critic_tests",
    )
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--episodes-per-checkpoint", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=904000)
    parser.add_argument("--fixed-option-duration-s", type=float, default=35.0)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only the named conditions; omit to run the complete battery.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a condition when its controller_summary.csv already exists.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not build the consolidated report after evaluation.",
    )
    return parser.parse_args()


def _run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = _parse_args()
    conditions = _condition_commands(args.fixed_option_duration_s)
    selected = list(conditions) if args.only is None else args.only
    unknown = sorted(set(selected) - set(conditions))
    if unknown:
        raise ValueError(
            f"Unknown condition(s): {', '.join(unknown)}. "
            f"Available: {', '.join(conditions)}"
        )

    scripts_dir = Path(__file__).resolve().parent
    evaluator = scripts_dir / "evaluate_behavior_time.py"
    reporter = scripts_dir / "report_option_critic_validation.py"
    output_root = Path(args.output_root).resolve()
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(evaluator),
        "--mission", args.mission,
        "--checkpoint-root", str(Path(args.checkpoint_root).resolve()),
        "--num-runs", str(args.num_runs),
        "--episodes-per-checkpoint", str(args.episodes_per_checkpoint),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
    ]
    if args.deterministic:
        common.append("--deterministic")
    if args.allow_missing:
        common.append("--allow-missing")

    for index, condition in enumerate(selected, start=1):
        condition_dir = raw_root / condition
        summary_path = condition_dir / "controller_summary.csv"
        if args.skip_existing and summary_path.exists():
            print(
                f"[OCValidation] [{index}/{len(selected)}] Skipping {condition}: "
                f"{summary_path} exists.",
                flush=True,
            )
            continue
        print(
            f"\n[OCValidation] [{index}/{len(selected)}] Running {condition}",
            flush=True,
        )
        command = common + conditions[condition] + [
            "--output-dir", str(condition_dir),
        ]
        _run_logged(command, condition_dir / "evaluation.log")

    if not args.no_report:
        print("\n[OCValidation] Building consolidated report", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(reporter),
                "--input-root", str(output_root),
                "--fixed-option-duration-s", str(args.fixed_option_duration_s),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
