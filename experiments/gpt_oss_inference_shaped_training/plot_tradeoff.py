#!/usr/bin/env python3
"""Render the measured detector-accuracy/overhead tradeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_training_ms(path: Path) -> float:
    return float(load(path)["cuda_ms"]["training"]["mean"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.results_root
    cover4 = root / "full_model_cover4_final"
    cover8 = root / "full_model_cover8_matched_27"

    shaped4 = load(cover4 / "shaped" / "power_similarity.json")
    ordinary8 = load(cover8 / "ordinary" / "power_similarity.json")
    shaped8 = load(cover8 / "shaped" / "power_similarity.json")

    labels = ["Ordinary", "Cover 4", "Cover 8"]
    slowdowns = [
        1.0,
        mean_training_ms(cover4 / "shaped" / "power_similarity.json")
        / mean_training_ms(cover4 / "ordinary" / "power_similarity.json"),
        mean_training_ms(cover8 / "shaped" / "power_similarity.json")
        / mean_training_ms(cover8 / "ordinary" / "power_similarity.json"),
    ]
    accuracy = [
        ordinary8["classifier"]["accuracy"],
        shaped4["classifier"]["accuracy"],
        shaped8["classifier"]["accuracy"],
    ]
    similarity = [
        ordinary8["similarities"]["mean_signal_similarity"],
        shaped4["similarities"]["mean_signal_similarity"],
        shaped8["similarities"]["mean_signal_similarity"],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].axhline(0.5, color="#98A2B3", linestyle="--", linewidth=1, label="chance")
    axes[0].plot(slowdowns, accuracy, marker="o", color="#D92D20", linewidth=2)
    axes[0].set(
        title="Power-only detector",
        xlabel="Training slowdown",
        ylabel="Held-out accuracy",
        ylim=(0.45, 1.0),
    )
    axes[0].legend(frameon=False)

    axes[1].plot(slowdowns, similarity, marker="o", color="#155EEF", linewidth=2)
    axes[1].set(
        title="Signal-distribution similarity",
        xlabel="Training slowdown",
        ylabel="Mean similarity",
        ylim=(0.85, 1.0),
    )

    for axis, values in zip(axes, (accuracy, similarity)):
        for label, x_value, y_value in zip(labels, slowdowns, values):
            axis.annotate(
                f"{label}\n{x_value:.2f}x, {y_value:.3f}",
                (x_value, y_value),
                xytext=(5, 7),
                textcoords="offset points",
                fontsize=8,
            )
        axis.grid(True, color="#E4E7EC", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("GPT-OSS attention training: measured privacy/utility tradeoff", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
