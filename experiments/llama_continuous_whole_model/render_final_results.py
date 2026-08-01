#!/usr/bin/env python3
"""Render the compact detector/throughput tradeoff from validated result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CANDIDATES = {
    "bnb-micro8": {
        "label": "Micro-8 (no cover)",
        "benchmark": "micro8-acc16.json",
        "color": "#D92D20",
        "annotation_offset": (5, -18),
    },
    "fused-micro16-cover1": {
        "label": "Micro-16 + decode each",
        "benchmark": "fused-micro16-acc8-cover1.json",
        "color": "#F79009",
        "annotation_offset": (5, 8),
    },
    "fused-cover8": {
        "label": "8 decode/update",
        "benchmark": "fused128-cover8-long.json",
        "color": "#2E90FA",
        "annotation_offset": (5, -18),
    },
    "fused-cover12": {
        "label": "12 decode/update",
        "benchmark": "fused128-cover12-long.json",
        "color": "#039855",
        "annotation_offset": (-8, -20),
        "annotation_ha": "right",
    },
    "fused-cover12-jitter4": {
        "label": "12 decode ±4",
        "benchmark": "fused128-cover12-jitter4.json",
        "metrics": "jitter_final/continuous_power_metrics.json",
        "color": "#C11574",
        "annotation_offset": (-8, 12),
        "annotation_ha": "right",
    },
    "fused-cover16": {
        "label": "16 decode/update",
        "benchmark": "fused128-cover16-long.json",
        "color": "#7F56D9",
        "annotation_offset": (5, 8),
    },
    "layercover3": {
        "label": "11 decode, layer-interleaved",
        "benchmark": "fused128-layercover3.json",
        "metrics": "layercover_final/continuous_power_metrics.json",
        "color": "#0E7090",
        "annotation_offset": (8, 14),
        "annotation_ha": "left",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result: {path}")
    return json.loads(path.read_text())


def build_summary(results_root: Path) -> dict[str, Any]:
    baseline = load_json(results_root / "baseline_v1" / "continuous_power_metrics.json")
    baseline_benchmark = load_json(results_root / "benchmarks" / "baseline-seq128.json")
    candidates = {}
    for key, specification in CANDIDATES.items():
        metrics_path = specification.get(
            "metrics", f"final_eval/{key}/continuous_power_metrics.json"
        )
        metrics = load_json(results_root / metrics_path)
        benchmark = load_json(results_root / "benchmarks" / specification["benchmark"])
        candidates[key] = {
            "label": specification["label"],
            "stationary_similarities": metrics["similarities"],
            "ridge_detector_by_horizon_ms": metrics["detector_by_horizon_ms"],
            "useful_loss_tokens_per_second": benchmark["useful_loss_tokens_per_second"],
            "throughput_relative_to_original": (
                benchmark["useful_loss_tokens_per_second"]
                / baseline_benchmark["useful_loss_tokens_per_second"]
            ),
            "trace_counts": metrics["trace_counts"],
            "session_counts": metrics["session_counts"],
            "all_health_checks_passed": metrics["all_health_checks_passed"],
        }
        cnn_paths = sorted((results_root / "final_eval" / key).glob("cnn_*.json"))
        if cnn_paths:
            evaluations = []
            for cnn_path in cnn_paths:
                cnn = load_json(cnn_path)
                evaluations.append(
                    {
                        "source_file": cnn_path.name,
                        **{
                            field: cnn[field]
                            for field in (
                                "horizon_ms",
                                "balanced_accuracy",
                                "fold_standard_deviation",
                                "minimum_fold_accuracy",
                                "maximum_fold_accuracy",
                                "aggregate_confusion_matrix_true_rows_predicted_columns",
                                "split",
                                "attacker_input",
                                "epochs_per_fold",
                                "batch_size",
                                "learning_rate",
                            )
                        },
                    }
                )
            candidates[key]["cnn_evaluations"] = evaluations
            five_ms = [row for row in evaluations if row["horizon_ms"] == 5.0]
            if five_ms:
                candidates[key]["strongest_cnn_5ms"] = max(
                    five_ms, key=lambda row: row["balanced_accuracy"]
                )
    return {
        "experiment": "continuous whole-model Llama training versus quantized inference",
        "attacker_observable": "ADC current-probe samples only",
        "sample_rate_hz": baseline["sample_rate_hz"],
        "capture_duration_ms": baseline["trace_samples"] / baseline["sample_rate_hz"] * 1e3,
        "original_baseline": {
            "stationary_similarities": baseline["similarities"],
            "ridge_detector_by_horizon_ms": baseline["detector_by_horizon_ms"],
            "useful_loss_tokens_per_second": baseline_benchmark[
                "useful_loss_tokens_per_second"
            ],
            "trace_counts": baseline["trace_counts"],
            "session_counts": baseline["session_counts"],
            "all_health_checks_passed": baseline["all_health_checks_passed"],
        },
        "candidates": candidates,
    }


def render(summary: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    baseline = summary["original_baseline"]
    horizons = np.asarray(
        sorted(float(value) for value in baseline["ridge_detector_by_horizon_ms"])
    )

    axes[0].plot(
        horizons,
        [
            baseline["ridge_detector_by_horizon_ms"][str(value)]["balanced_accuracy"]
            for value in horizons
        ],
        color="#475467",
        marker="o",
        label="Ordinary baseline",
        lw=2,
    )
    for key, specification in CANDIDATES.items():
        candidate = summary["candidates"][key]
        detector = candidate["ridge_detector_by_horizon_ms"]
        axes[0].plot(
            horizons,
            [detector[str(value)]["balanced_accuracy"] for value in horizons],
            color=specification["color"],
            marker="o",
            label=specification["label"],
            lw=1.7,
        )
    cover12_cnn = summary["candidates"]["fused-cover12"].get("strongest_cnn_5ms")
    if cover12_cnn:
        axes[0].scatter(
            [5],
            [cover12_cnn["balanced_accuracy"]],
            marker="X",
            s=90,
            color="#027A48",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
            label="12-decode CNN",
        )
    axes[0].axhline(0.5, color="#98A2B3", ls="--", lw=1)
    axes[0].set(
        xscale="log",
        ylim=(0.45, 1.02),
        xlabel="Observation horizon (ms)",
        ylabel="Session-held-out balanced accuracy",
        title="Power-only detector",
    )

    baseline_throughput = baseline["useful_loss_tokens_per_second"]
    axes[1].scatter(
        [baseline_throughput],
        [baseline["ridge_detector_by_horizon_ms"]["5.0"]["balanced_accuracy"]],
        color="#475467",
        marker="s",
        s=70,
        label="Ordinary baseline",
    )
    axes[1].annotate(
        "Ordinary baseline",
        (
            baseline_throughput,
            baseline["ridge_detector_by_horizon_ms"]["5.0"]["balanced_accuracy"],
        ),
        xytext=(-5, -18),
        textcoords="offset points",
        ha="right",
        fontsize=8,
    )
    for key, specification in CANDIDATES.items():
        candidate = summary["candidates"][key]
        axes[1].scatter(
            [candidate["useful_loss_tokens_per_second"]],
            [candidate["ridge_detector_by_horizon_ms"]["5.0"]["balanced_accuracy"]],
            color=specification["color"],
            s=75,
        )
        axes[1].annotate(
            specification["label"],
            (
                candidate["useful_loss_tokens_per_second"],
                candidate["ridge_detector_by_horizon_ms"]["5.0"]["balanced_accuracy"],
            ),
            xytext=specification["annotation_offset"],
            textcoords="offset points",
            ha=specification.get("annotation_ha", "left"),
            fontsize=8,
        )
    axes[1].axhline(0.5, color="#98A2B3", ls="--", lw=1)
    axes[1].axvline(baseline_throughput, color="#D0D5DD", ls=":", lw=1)
    axes[1].set(
        xlim=(0, 430),
        ylim=(0.45, 1.02),
        xlabel="Useful causal-loss tokens/s",
        ylabel="5 ms balanced accuracy",
        title="Camouflage–throughput frontier",
    )

    labels = ["Ordinary", *[CANDIDATES[key]["label"] for key in CANDIDATES]]
    similarities = [
        baseline["stationary_similarities"]["mean_similarity"],
        *[
            summary["candidates"][key]["stationary_similarities"]["mean_similarity"]
            for key in CANDIDATES
        ],
    ]
    colors = ["#475467", *[CANDIDATES[key]["color"] for key in CANDIDATES]]
    positions = np.arange(len(labels))
    axes[2].bar(positions, similarities, color=colors)
    axes[2].set_xticks(positions, labels, rotation=35, ha="right")
    axes[2].set(
        ylim=(0.5, 1.0),
        ylabel="Mean stationary similarity",
        title="Physical distribution match",
    )
    for position, value in zip(positions, similarities):
        axes[2].text(position, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    for axis in axes:
        axis.grid(True, color="#EAECF0", lw=0.7, alpha=0.9, axis="y")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Making full-parameter Llama training resemble quantized inference",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot", type=Path, required=True)
    args = parser.parse_args()
    summary = build_summary(args.results_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    render(summary, args.output_plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
