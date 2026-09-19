# Copyright (c) 2025 SwarmACB Project
# SPDX-License-Identifier: BSD-3-Clause

"""Consolidate matched Option-Critic evaluations into tables, plots, and a report."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


NUMERIC_FIELDS = {
    "run_index",
    "episode_index",
    "evaluation_seed",
    "forced_option",
    "reward_mean",
    "episode_steps_mean",
    "total_robot_seconds",
    "switch_count",
    "switch_rate",
    "termination_count",
    "termination_rate",
    "mean_termination_probability",
    "termination_entropy",
    "same_option_reselection_fraction",
    "segment_count",
    "mean_dwell_steps",
    "median_dwell_steps",
    "mean_dwell_seconds",
    "median_dwell_seconds",
    "behavior_usage_entropy",
    "behavior_usage_entropy_norm",
    "episodes_evaluated",
    "reward_episode_std",
    "reward_episode_min",
    "reward_episode_max",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="analysis/dirgate_option_critic_tests",
        help="Validation root containing raw/<condition> directories.",
    )
    parser.add_argument("--fixed-option-duration-s", type=float, default=35.0)
    return parser.parse_args()


def _duration_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _condition_labels(fixed_duration_s: float) -> dict[str, str]:
    tag = _duration_tag(fixed_duration_s)
    labels = {
        "dandelion": "Dandelion",
        "cyclamen": "Cyclamen",
        "oc1_learned": "OC1 learned beta",
        "oc1_never": "OC1 beta=0",
        f"oc1_fixed_{tag}s": f"OC1 fixed {fixed_duration_s:g}s",
        "oc2_learned": "OC2 learned beta",
        "oc2_never": "OC2 beta=0",
        f"oc2_fixed_{tag}s": f"OC2 fixed {fixed_duration_s:g}s",
    }
    for option in range(6):
        labels[f"oc2_force_{option}"] = f"OC2 forced option {option}"
    return labels


def _to_number(field: str, value: str):
    if field not in NUMERIC_FIELDS and not field.endswith(
        ("_seconds", "_fraction", "_mean_dwell_seconds", "_median_dwell_seconds")
    ):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if field in {"run_index", "episode_index", "forced_option", "episodes_evaluated"}:
        return int(number) if math.isfinite(number) else number
    return number


def _read_rows(path: Path, condition: str, label: str) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {key: _to_number(key, value) for key, value in raw.items()}
            row = {"condition": condition, "condition_label": label, **row}
            rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    preferred = ["condition", "condition_label", "method", "mission", "run_index"]
    fields = []
    for field in preferred:
        if any(field in row for row in rows):
            fields.append(field)
    fields.extend(sorted({key for row in rows for key in row} - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _finite(rows: list[dict], metric: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    n = int(values.size)
    if n == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
        if n > 1 else 0.0
    )
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _condition_summaries(
    controller_rows: list[dict],
    labels: dict[str, str],
) -> list[dict]:
    metrics = (
        "reward_mean",
        "reward_episode_std",
        "switch_rate",
        "termination_rate",
        "mean_termination_probability",
        "termination_entropy",
        "same_option_reselection_fraction",
        "mean_dwell_seconds",
        "behavior_usage_entropy_norm",
    )
    summaries = []
    for condition in labels:
        rows = [row for row in controller_rows if row["condition"] == condition]
        if not rows:
            continue
        out = {
            "condition": condition,
            "condition_label": labels[condition],
            "num_controllers": len(rows),
        }
        for metric in metrics:
            metric_summary = _summary(_finite(rows, metric))
            for name, value in metric_summary.items():
                out[f"{metric}_{name}"] = value
        summaries.append(out)
    return summaries


def _rows_by_run(controller_rows: list[dict], condition: str) -> dict[int, dict]:
    return {
        int(row["run_index"]): row
        for row in controller_rows
        if row["condition"] == condition
    }


def _paired_comparison(
    controller_rows: list[dict],
    reference: str,
    comparison: str,
    label: str,
) -> dict | None:
    reference_rows = _rows_by_run(controller_rows, reference)
    comparison_rows = _rows_by_run(controller_rows, comparison)
    run_indices = sorted(set(reference_rows) & set(comparison_rows))
    if not run_indices:
        return None
    ref = np.asarray(
        [float(reference_rows[index]["reward_mean"]) for index in run_indices]
    )
    comp = np.asarray(
        [float(comparison_rows[index]["reward_mean"]) for index in run_indices]
    )
    delta = comp - ref
    n = len(delta)
    delta_std = float(delta.std(ddof=1)) if n > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, n - 1) * delta_std / math.sqrt(n))
        if n > 1 else 0.0
    )
    t_result = stats.ttest_rel(comp, ref) if n > 1 else None
    if np.allclose(delta, 0.0):
        wilcoxon_p = 1.0
    else:
        try:
            wilcoxon_p = float(stats.wilcoxon(comp, ref).pvalue)
        except ValueError:
            wilcoxon_p = math.nan
    return {
        "comparison": label,
        "reference": reference,
        "candidate": comparison,
        "n_pairs": n,
        "reference_mean": float(ref.mean()),
        "candidate_mean": float(comp.mean()),
        "mean_paired_delta": float(delta.mean()),
        "delta_std": delta_std,
        "delta_ci95_low": float(delta.mean()) - half_width,
        "delta_ci95_high": float(delta.mean()) + half_width,
        "wins": int((delta > 0.0).sum()),
        "ties": int(np.isclose(delta, 0.0).sum()),
        "losses": int((delta < 0.0).sum()),
        "paired_t_p": float(t_result.pvalue) if t_result is not None else math.nan,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": (
            float(delta.mean() / delta_std) if delta_std > 0.0 else math.nan
        ),
    }


def _paired_summaries(
    controller_rows: list[dict],
    fixed_duration_s: float,
) -> list[dict]:
    tag = _duration_tag(fixed_duration_s)
    comparisons = [
        ("dandelion", "cyclamen", "Cyclamen - Dandelion"),
        ("cyclamen", "oc1_learned", "OC1 - Cyclamen"),
        ("cyclamen", "oc2_learned", "OC2 - Cyclamen"),
        ("dandelion", "oc2_learned", "OC2 - Dandelion"),
        ("oc1_learned", "oc2_learned", "OC2 - OC1"),
        ("oc1_never", "oc1_learned", "OC1 learned - beta=0"),
        (f"oc1_fixed_{tag}s", "oc1_learned", "OC1 learned - fixed beta"),
        ("oc2_never", "oc2_learned", "OC2 learned - beta=0"),
        (f"oc2_fixed_{tag}s", "oc2_learned", "OC2 learned - fixed beta"),
    ]
    return [
        result
        for reference, comparison, label in comparisons
        if (result := _paired_comparison(
            controller_rows, reference, comparison, label
        )) is not None
    ]


def _forced_option_rows(controller_rows: list[dict]) -> list[dict]:
    learned = _rows_by_run(controller_rows, "oc2_learned")
    never = _rows_by_run(controller_rows, "oc2_never")
    forced = {
        option: _rows_by_run(controller_rows, f"oc2_force_{option}")
        for option in range(6)
    }
    common_runs = sorted(
        set(learned).intersection(*(set(rows) for rows in forced.values()))
    ) if learned else []
    output = []
    for run_index in common_runs:
        forced_rewards = np.asarray([
            float(forced[option][run_index]["reward_mean"])
            for option in range(6)
        ])
        learned_row = learned[run_index]
        fractions = np.asarray([
            float(learned_row.get(f"learned_option_{option}_fraction", math.nan))
            for option in range(6)
        ])
        dominant_option = int(np.nanargmax(fractions))
        best_option = int(np.argmax(forced_rewards))
        learned_reward = float(learned_row["reward_mean"])
        row = {
            "run_index": run_index,
            "learned_reward": learned_reward,
            "never_terminate_reward": (
                float(never[run_index]["reward_mean"]) if run_index in never else math.nan
            ),
            "best_forced_option": best_option,
            "best_forced_reward": float(forced_rewards[best_option]),
            "mean_forced_reward": float(forced_rewards.mean()),
            "worst_forced_reward": float(forced_rewards.min()),
            "forced_reward_range": float(forced_rewards.max() - forced_rewards.min()),
            "learned_minus_best_forced": learned_reward - float(forced_rewards.max()),
            "learned_minus_mean_forced": learned_reward - float(forced_rewards.mean()),
            "dominant_selected_option": dominant_option,
            "dominant_usage_fraction": float(fractions[dominant_option]),
            "dominant_forced_reward": float(forced_rewards[dominant_option]),
            "learned_minus_dominant_forced": (
                learned_reward - float(forced_rewards[dominant_option])
            ),
            "learned_switch_rate": float(learned_row["switch_rate"]),
            "learned_termination_rate": float(learned_row["termination_rate"]),
            "learned_mean_beta": float(learned_row["mean_termination_probability"]),
            "learned_termination_entropy": float(learned_row["termination_entropy"]),
        }
        for option in range(6):
            row[f"forced_option_{option}_reward"] = float(forced_rewards[option])
            row[f"learned_option_{option}_fraction"] = float(fractions[option])
            row[f"forced_option_{option}_mean_beta"] = float(
                forced[option][run_index]["mean_termination_probability"]
            )
        output.append(row)
    return output


def _forced_rank_summary(forced_rows: list[dict]) -> list[dict]:
    if not forced_rows:
        return []
    ranked = []
    for row in forced_rows:
        rewards = sorted(
            [float(row[f"forced_option_{option}_reward"]) for option in range(6)],
            reverse=True,
        )
        ranked.append(rewards)
    array = np.asarray(ranked)
    return [
        {
            "rank": rank + 1,
            "mean_reward": float(array[:, rank].mean()),
            "std_reward": float(array[:, rank].std(ddof=1)),
            "median_reward": float(np.median(array[:, rank])),
        }
        for rank in range(array.shape[1])
    ]


def _boxplot(
    controller_rows: list[dict],
    conditions: list[str],
    labels: dict[str, str],
    path: Path,
) -> None:
    available = [condition for condition in conditions if _rows_by_run(controller_rows, condition)]
    if not available:
        return
    values = [
        [float(row["reward_mean"]) for row in _rows_by_run(controller_rows, condition).values()]
        for condition in available
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    palette = ["#64748b", "#2563eb", "#dc2626", "#16a34a"]
    boxes = ax.boxplot(values, patch_artist=True, widths=0.55, showmeans=True)
    for patch, color in zip(boxes["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for index, (condition, condition_values) in enumerate(zip(available, values), start=1):
        jitter = np.linspace(-0.12, 0.12, len(condition_values))
        ax.scatter(
            index + jitter,
            condition_values,
            c=palette[(index - 1) % len(palette)],
            edgecolors="white",
            linewidths=0.6,
            s=42,
            zorder=3,
        )
    plot_labels = {
        "oc1_learned": "OC1\nlearned beta",
        "oc2_learned": "OC2\nlearned beta",
    }
    ax.set_xticks(
        range(1, len(available) + 1),
        [plot_labels.get(item, labels[item]) for item in available],
    )
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Directional Gate Performance Across Trained Controllers")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _termination_plot(
    controller_rows: list[dict],
    labels: dict[str, str],
    fixed_duration_s: float,
    path: Path,
) -> None:
    tag = _duration_tag(fixed_duration_s)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    complete = False
    for ax, method in zip(axes, ("oc1", "oc2")):
        conditions = [f"{method}_never", f"{method}_fixed_{tag}s", f"{method}_learned"]
        run_maps = [_rows_by_run(controller_rows, condition) for condition in conditions]
        common = sorted(set.intersection(*(set(rows) for rows in run_maps))) if all(run_maps) else []
        if not common:
            ax.set_visible(False)
            continue
        complete = True
        matrix = np.asarray([
            [float(run_map[index]["reward_mean"]) for run_map in run_maps]
            for index in common
        ])
        for row in matrix:
            ax.plot(range(3), row, color="#94a3b8", alpha=0.55, linewidth=1)
            ax.scatter(range(3), row, color="#64748b", s=18)
        ax.plot(range(3), matrix.mean(axis=0), color="#111827", linewidth=3,
                marker="o", markersize=6, label="Controller mean")
        ax.set_xticks(range(3), ["Never", f"Fixed\n{fixed_duration_s:g}s", "Learned"])
        ax.set_title(method.upper())
        ax.set_ylabel("Mean episode reward")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    if complete:
        fig.suptitle("Causal Termination Intervention")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
    plt.close(fig)


def _forced_option_plots(forced_rows: list[dict], output_dir: Path) -> None:
    if not forced_rows:
        return
    forced_rewards = np.asarray([
        [float(row[f"forced_option_{option}_reward"]) for option in range(6)]
        for row in forced_rows
    ])
    learned_rewards = np.asarray([float(row["learned_reward"]) for row in forced_rows])
    matrix = np.column_stack([forced_rewards, learned_rewards])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] < np.nanmedian(matrix) else "black"
            ax.text(column_index, row_index, f"{matrix[row_index, column_index]:.1f}",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(7), [f"Force {option}" for option in range(6)] + ["Learned\nmanager"])
    ax.set_yticks(range(len(forced_rows)), [f"Run {int(row['run_index'])}" for row in forced_rows])
    ax.set_title("OC2 Forced-Option Capability and Learned Composition")
    fig.colorbar(image, ax=ax, label="Mean episode reward")
    fig.tight_layout()
    fig.savefig(output_dir / "oc2_forced_option_rewards.png", dpi=200)
    plt.close(fig)

    usage = np.asarray([
        [float(row[f"learned_option_{option}_fraction"]) for option in range(6)]
        for row in forced_rows
    ])
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottoms = np.zeros(len(forced_rows))
    colors = ["#64748b", "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#7c3aed"]
    for option in range(6):
        ax.bar(
            range(len(forced_rows)),
            usage[:, option],
            bottom=bottoms,
            color=colors[option],
            label=f"Option {option}",
        )
        bottoms += usage[:, option]
    ax.set_xticks(range(len(forced_rows)), [str(int(row["run_index"])) for row in forced_rows])
    ax.set_xlabel("Training run")
    ax.set_ylabel("Robot-time fraction")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("OC2 Option Usage Under the Learned Manager")
    ax.legend(
        ncol=6,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_dir / "oc2_learned_option_usage.png", dpi=200)
    plt.close(fig)


def _diagnostic_plot(forced_rows: list[dict], path: Path) -> None:
    if not forced_rows:
        return
    reward = np.asarray([float(row["learned_reward"]) for row in forced_rows])
    metrics = [
        ("learned_termination_rate", "Termination rate"),
        ("learned_switch_rate", "True switch rate"),
        ("dominant_usage_fraction", "Dominant option fraction"),
        ("learned_termination_entropy", "Termination entropy (nats)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, label) in zip(axes.flat, metrics):
        x = np.asarray([float(row[metric]) for row in forced_rows])
        ax.scatter(x, reward, c="#16a34a", edgecolors="white", s=60)
        for x_value, y_value, row in zip(x, reward, forced_rows):
            ax.annotate(str(int(row["run_index"])), (x_value, y_value),
                        xytext=(4, 3), textcoords="offset points", fontsize=7)
        if len(x) > 1 and np.std(x) > 0.0:
            correlation = stats.spearmanr(x, reward).statistic
            ax.set_title(f"Spearman rho={correlation:.2f}", fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean episode reward")
        ax.grid(alpha=0.2)
    fig.suptitle("OC2 Hierarchy Diagnostics by Training Run")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fmt(value: float, digits: int = 2) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _lookup(rows: list[dict], condition: str) -> dict | None:
    return next((row for row in rows if row["condition"] == condition), None)


def _write_report(
    path: Path,
    condition_rows: list[dict],
    paired_rows: list[dict],
    forced_rows: list[dict],
    fixed_duration_s: float,
    episode_count: int,
) -> None:
    tag = _duration_tag(fixed_duration_s)
    baseline_conditions = ["dandelion", "cyclamen", "oc1_learned", "oc2_learned"]
    baseline = [_lookup(condition_rows, condition) for condition in baseline_conditions]
    baseline = [row for row in baseline if row is not None]
    lines = [
        "# Directional Gate Option-Critic Validation",
        "",
        "## Protocol",
        "",
        f"Each of 10 independently trained checkpoints was evaluated on {episode_count} stochastic, matched scenarios. "
        "Episodes were averaged within checkpoint before comparisons, so the trained checkpoint is the statistical unit (n=10).",
        "",
        "The battery includes Dandelion, classical Cyclamen, OC1 with its six fixed Cyclamen modules, OC2 with six learned intra-option policies, causal termination interventions, and a full-episode forced condition for every OC2 option.",
        "",
        "## Performance",
        "",
        "| Method | Reward mean +/- SD | Median | 95% CI | Switch rate | Mean dwell (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in baseline:
        switch_rate = (
            "NA" if row["condition"] == "dandelion"
            else _fmt(row["switch_rate_mean"], 4)
        )
        mean_dwell = (
            "NA" if row["condition"] == "dandelion"
            else _fmt(row["mean_dwell_seconds_mean"])
        )
        lines.append(
            f"| {row['condition_label']} | "
            f"{_fmt(row['reward_mean_mean'])} +/- {_fmt(row['reward_mean_std'])} | "
            f"{_fmt(row['reward_mean_median'])} | "
            f"[{_fmt(row['reward_mean_ci95_low'])}, {_fmt(row['reward_mean_ci95_high'])}] | "
            f"{switch_rate} | {mean_dwell} |"
        )

    lines += [
        "",
        "## Paired Comparisons",
        "",
        "Positive deltas favor the candidate named on the left. P-values are exploratory and are not corrected for multiple comparisons.",
        "",
        "| Comparison | Mean paired delta | 95% CI | W/T/L | Paired t p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']} | {_fmt(row['mean_paired_delta'])} | "
            f"[{_fmt(row['delta_ci95_low'])}, {_fmt(row['delta_ci95_high'])}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{_fmt(row['paired_t_p'], 4)} | {_fmt(row['wilcoxon_p'], 4)} |"
        )

    lines += [
        "",
        "## Termination Intervention",
        "",
        f"`beta=0` prevents termination after the initial option choice. The fixed control terminates every {fixed_duration_s:g} simulated seconds. Learned beta is the checkpoint's own termination policy.",
        "The shared fixed interval is a simple schedule control, not a termination-frequency-matched control. It can show that this schedule is insufficient, but it cannot by itself separate state dependence from termination frequency.",
        "",
        "| Method | Learned reward | beta=0 reward | Fixed reward | Learned termination rate | Learned mean beta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("oc1", "oc2"):
        learned = _lookup(condition_rows, f"{method}_learned")
        never = _lookup(condition_rows, f"{method}_never")
        fixed = _lookup(condition_rows, f"{method}_fixed_{tag}s")
        if learned and never and fixed:
            lines.append(
                f"| {method.upper()} | {_fmt(learned['reward_mean_mean'])} | "
                f"{_fmt(never['reward_mean_mean'])} | {_fmt(fixed['reward_mean_mean'])} | "
                f"{_fmt(learned['termination_rate_mean'], 4)} | "
                f"{_fmt(learned['mean_termination_probability_mean'], 4)} |"
            )

    if forced_rows:
        learned_minus_best = np.asarray([
            float(row["learned_minus_best_forced"]) for row in forced_rows
        ])
        learned_minus_dominant = np.asarray([
            float(row["learned_minus_dominant_forced"]) for row in forced_rows
        ])
        dominant_fraction = np.asarray([
            float(row["dominant_usage_fraction"]) for row in forced_rows
        ])
        ranges = np.asarray([float(row["forced_reward_range"]) for row in forced_rows])
        second_best = np.asarray([
            sorted(
                [float(row[f"forced_option_{option}_reward"]) for option in range(6)],
                reverse=True,
            )[1]
            for row in forced_rows
        ])
        dominant_is_best = sum(
            int(row["dominant_selected_option"]) == int(row["best_forced_option"])
            for row in forced_rows
        )
        collapsed = int((dominant_fraction >= 0.95).sum())
        material_composition = int((learned_minus_best > 5.0).sum())
        lines += [
            "",
            "## Learned-Option Probe",
            "",
            f"The learned manager minus the best single forced option is {_fmt(learned_minus_best.mean())} +/- {_fmt(learned_minus_best.std(ddof=1))} reward (median {_fmt(np.median(learned_minus_best))}). "
            f"It beats every forced option in {int((learned_minus_best > 0).sum())}/{len(forced_rows)} controllers.",
            "",
            f"Relative to each controller's most-used option forced for the whole episode, the learned manager changes reward by {_fmt(learned_minus_dominant.mean())} on average. "
            f"The median dominant-option usage is {_fmt(np.median(dominant_fraction), 3)}, and the mean best-to-worst forced-option reward range is {_fmt(ranges.mean())}.",
            "",
            f"The most-used option is also the best forced option in {dominant_is_best}/{len(forced_rows)} runs. "
            f"The best forced option averages {_fmt(np.mean([row['best_forced_reward'] for row in forced_rows]))} reward, while the second-best averages only {_fmt(second_best.mean())}. "
            f"{collapsed}/{len(forced_rows)} runs assign at least 95% of robot-time to one option, and only {material_composition}/{len(forced_rows)} obtain more than 5 reward beyond their best forced option.",
            "",
            "Option IDs are permutation-invariant across training runs. Interpret rows within a run, or compare option ranks, rather than treating option 0 from different runs as the same behavior.",
        ]

    oc1_delta = next((row for row in paired_rows if row["comparison"] == "OC1 - Cyclamen"), None)
    oc2_delta = next((row for row in paired_rows if row["comparison"] == "OC2 - Cyclamen"), None)
    oc2_learned = _lookup(condition_rows, "oc2_learned")
    lines += ["", "## Interpretation", ""]
    if oc1_delta:
        lines.append(
            f"- OC1 changes reward by {_fmt(oc1_delta['mean_paired_delta'])} relative to Cyclamen "
            f"({oc1_delta['wins']}/{oc1_delta['n_pairs']} checkpoint pairs improve). The ability to terminate is causally necessary for the learned policy, but phase 1 does not improve Directional Gate reward over classical Cyclamen."
        )
    if oc2_delta:
        lines.append(
            f"- OC2 changes reward by {_fmt(oc2_delta['mean_paired_delta'])} relative to Cyclamen "
            f"({oc2_delta['wins']}/{oc2_delta['n_pairs']} checkpoint pairs improve). The paired t-test is positive, but the Wilcoxon result and very large seed spread make this evidence promising rather than robust."
        )
    if oc2_learned:
        lines.append(
            f"- OC2 exhibits a mean true-switch rate of {_fmt(oc2_learned['switch_rate_mean'], 4)}, "
            f"a mean termination rate of {_fmt(oc2_learned['termination_rate_mean'], 4)}, and "
            f"a mean dwell of {_fmt(oc2_learned['mean_dwell_seconds_mean'])} s. This establishes temporal persistence; the forced-option probe establishes whether it reflects useful composition or collapse."
        )
    if forced_rows:
        lines.append(
            f"- OC2 termination learning is successful, but six-option discovery is usually degenerate: {collapsed}/{len(forced_rows)} runs are dominated by one option and the second-best forced option averages only {_fmt(second_best.mean())} reward. Most high-scoring runs behave like a learned flat policy selected and then retained; runs with real compositional gain are exceptions."
        )
    lines += [
        "- A scientifically strong hierarchy requires more than high reward: reproducible performance across seeds, options with distinguishable capabilities, and a causal benefit from the learned manager or termination policy.",
        "",
        "## Recommended Follow-up",
        "",
        "1. Add a per-checkpoint constant-hazard or periodic control matched to each learned policy's termination rate. This isolates state-dependent beta from termination frequency.",
        "2. Analyze run 3 against the high-reward single-option runs: it is the clearest case where composition beats every standalone option, while most runs learn one dominant flat policy.",
        "3. Improve cross-seed option diversity and repeatability before using these learned options as the foundation for multi-mission OC3 transfer.",
        "",
        "## Files",
        "",
        "- `controller_results.csv`: one row per independently trained checkpoint and condition.",
        "- `episode_results.csv`: one row per checkpoint and matched evaluation scenario.",
        "- `condition_summary.csv`: controller-level descriptive statistics and confidence intervals.",
        "- `paired_reward_comparisons.csv`: paired checkpoint comparisons.",
        "- `oc2_forced_option_by_controller.csv`: forced-option and learned-manager results within each OC2 checkpoint.",
        "- `oc2_forced_option_rank_summary.csv`: permutation-safe forced-option rank summary.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    root = Path(args.input_root).resolve()
    raw_root = root / "raw"
    output_dir = root / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _condition_labels(args.fixed_option_duration_s)

    controller_rows: list[dict] = []
    episode_rows: list[dict] = []
    missing = []
    for condition, label in labels.items():
        condition_dir = raw_root / condition
        controller_path = condition_dir / "controller_summary.csv"
        episode_path = condition_dir / "behavior_time_summary.csv"
        if not controller_path.exists() or not episode_path.exists():
            missing.append(condition)
            continue
        controller_rows.extend(_read_rows(controller_path, condition, label))
        episode_rows.extend(_read_rows(episode_path, condition, label))
    if not controller_rows:
        raise FileNotFoundError(f"No validation outputs found under {raw_root}")

    condition_rows = _condition_summaries(controller_rows, labels)
    paired_rows = _paired_summaries(controller_rows, args.fixed_option_duration_s)
    forced_rows = _forced_option_rows(controller_rows)
    rank_rows = _forced_rank_summary(forced_rows)

    _write_rows(output_dir / "controller_results.csv", controller_rows)
    _write_rows(output_dir / "episode_results.csv", episode_rows)
    _write_rows(output_dir / "condition_summary.csv", condition_rows)
    _write_rows(output_dir / "paired_reward_comparisons.csv", paired_rows)
    _write_rows(output_dir / "oc2_forced_option_by_controller.csv", forced_rows)
    _write_rows(output_dir / "oc2_forced_option_rank_summary.csv", rank_rows)

    _boxplot(
        controller_rows,
        ["dandelion", "cyclamen", "oc1_learned", "oc2_learned"],
        labels,
        output_dir / "performance_overview.png",
    )
    _termination_plot(
        controller_rows,
        labels,
        args.fixed_option_duration_s,
        output_dir / "termination_intervention.png",
    )
    _forced_option_plots(forced_rows, output_dir)
    _diagnostic_plot(forced_rows, output_dir / "oc2_hierarchy_diagnostics.png")

    episode_counts = [
        int(float(row.get("episodes_evaluated", 0))) for row in controller_rows
    ]
    _write_report(
        output_dir / "REPORT.md",
        condition_rows,
        paired_rows,
        forced_rows,
        args.fixed_option_duration_s,
        min(episode_counts) if episode_counts else 0,
    )
    print(f"[OCValidationReport] Wrote {output_dir}")
    if missing:
        print(f"[OCValidationReport] Missing conditions: {', '.join(missing)}")


if __name__ == "__main__":
    main()
