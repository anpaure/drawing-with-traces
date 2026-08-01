#!/usr/bin/env python3
"""Render the concise, evidence-backed figure used by the branch README."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "llama_continuous_whole_model"
SUMMARY = RESULTS / "final_summary.json"
KERNELS = RESULTS / "kernel_audit_summary.json"
OUTPUT = RESULTS / "readme_explainer.png"

BLUE = "#2563eb"
ORANGE = "#f59e0b"
RED = "#dc2626"
PURPLE = "#7c3aed"
GREEN = "#059669"
INK = "#172033"
MUTED = "#5f6b7a"
GRID = "#dce2ea"
PALE_BLUE = "#eff6ff"
PALE_ORANGE = "#fff7ed"
PALE_RED = "#fef2f2"


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _block(ax, x: float, y: float, width: float, label: str, color: str, *, height: float = 0.52) -> None:
    patch = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(patch)
    if label:
        ax.text(x + width / 2, y, label, ha="center", va="center", color="white", fontsize=9)


def _card(ax, x: float, width: float, title: str, body: str, *, facecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, 0.05),
        width,
        0.86,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1,
        edgecolor=GRID,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03, 0.77, title, color=INK, fontsize=13, fontweight="bold", va="top")
    ax.text(
        x + 0.03,
        0.63,
        textwrap.fill(body, width=43),
        color=MUTED,
        fontsize=10,
        va="top",
        linespacing=1.35,
    )


def render() -> Path:
    summary = _load(SUMMARY)
    kernels = _load(KERNELS)
    ordinary = summary["original_baseline"]
    cover = summary["candidates"]["fused-cover12"]
    horizons = np.array([5, 10, 20, 50, 100], dtype=float)
    ordinary_accuracy = np.array(
        [ordinary["ridge_detector_by_horizon_ms"][f"{value:.1f}"]["balanced_accuracy"] for value in horizons]
    )
    cover_accuracy = np.array(
        [cover["ridge_detector_by_horizon_ms"][f"{value:.1f}"]["balanced_accuracy"] for value in horizons]
    )
    cnn_5ms = cover["strongest_cnn_5ms"]["balanced_accuracy"]
    retained = cover["useful_loss_tokens_per_second"] / ordinary["useful_loss_tokens_per_second"]
    kernel_similarity = kernels["candidates"]["fused-cover12"]["similarity_to_quantized_cached_decode"][
        "continuous_kernel_similarity"
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(3, 12, height_ratios=[1.15, 1.35, 0.9], hspace=0.38, wspace=0.7)
    fig.suptitle(
        "Full-parameter Llama training made harder to distinguish from inference",
        fontsize=24,
        fontweight="bold",
        x=0.05,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.05,
        0.925,
        "Measured H100 current-probe experiment  •  schematic schedules are not to scale",
        fontsize=12,
        color=MUTED,
    )

    timeline = fig.add_subplot(grid[0, :])
    timeline.set_xlim(-0.19, 1.0)
    timeline.set_ylim(-0.65, 2.65)
    timeline.axis("off")
    timeline.text(-0.19, 2.5, "WHAT ACTUALLY RUNS", fontsize=11, color=MUTED, fontweight="bold")
    rows = [(1.85, "Inference reference"), (0.85, "Ordinary training"), (-0.15, "Best schedule")]
    for y, label in rows:
        timeline.text(-0.18, y, label, va="center", fontsize=12, fontweight="bold")
    for i in range(8):
        _block(timeline, 0.015 + i * 0.12, 1.85, 0.095, "decode", BLUE)
    for cycle in range(3):
        base = 0.015 + cycle * 0.325
        _block(timeline, base, 0.85, 0.085, "F", ORANGE)
        _block(timeline, base + 0.095, 0.85, 0.115, "B", RED)
        _block(timeline, base + 0.22, 0.85, 0.075, "opt", PURPLE)
    _block(timeline, 0.015, -0.15, 0.085, "F", ORANGE)
    _block(timeline, 0.11, -0.15, 0.115, "B", RED)
    _block(timeline, 0.235, -0.15, 0.075, "opt", PURPLE)
    for i in range(12):
        _block(timeline, 0.335 + i * 0.052, -0.15, 0.043, "", BLUE)
    timeline.text(
        0.665, -0.51, "12 real NF4 cached-decode tokens after each real update", ha="center", color=MUTED
    )
    timeline.text(
        0.99,
        2.45,
        "32 layers  •  all 8.03B training parameters  •  causal loss + backward + AdamW",
        ha="right",
        fontsize=10.5,
        color=MUTED,
    )

    detector = fig.add_subplot(grid[1, :7])
    detector.plot(horizons, ordinary_accuracy * 100, "o-", color=INK, lw=2.4, label="ordinary training")
    detector.plot(horizons, cover_accuracy * 100, "o-", color=GREEN, lw=2.8, label="training + 12 decodes")
    detector.scatter([5], [cnn_5ms * 100], marker="D", s=80, color=PURPLE, zorder=5, label="raw 1D CNN")
    detector.axhline(50, color="#94a3b8", lw=1.5, ls="--")
    detector.set_xscale("log")
    detector.set_xticks(horizons, [f"{int(value)}" for value in horizons])
    detector.set_ylim(45, 104)
    detector.set_xlabel("observation window (ms)")
    detector.set_ylabel("session-held-out balanced accuracy (%)")
    detector.set_title(
        "The best schedule reduces—rather than eliminates—the signal", loc="left", fontweight="bold"
    )
    detector.grid(axis="both", color=GRID, alpha=0.75)
    detector.legend(frameon=False, ncols=3, loc="lower right", fontsize=9.5)
    detector.annotate(
        f"{cover_accuracy[0] * 100:.1f}% ridge",
        (5, cover_accuracy[0] * 100),
        xytext=(8, 56),
        textcoords="offset points",
        color=GREEN,
        fontweight="bold",
    )
    detector.annotate(
        f"{cnn_5ms * 100:.1f}% CNN",
        (5, cnn_5ms * 100),
        xytext=(8, 7),
        textcoords="offset points",
        color=PURPLE,
        fontweight="bold",
    )

    metrics = fig.add_subplot(grid[1, 7:])
    metrics.axis("off")
    metrics.set_xlim(0, 1)
    metrics.set_ylim(0, 1)
    metrics.text(0, 1.0, "BEST VALIDATED TRADE-OFF", fontsize=11, color=MUTED, fontweight="bold", va="top")
    values = [
        (f"{cover['stationary_similarities']['mean_similarity']:.3f}", "stationary signal similarity"),
        (f"{kernel_similarity:.3f}", "continuous kernel-process similarity"),
        (f"{cover['useful_loss_tokens_per_second']:.0f}/s", "useful causal-loss targets"),
        (f"{retained * 100:.0f}%", "of original 8-bit-Adam throughput"),
    ]
    for i, (value, label) in enumerate(values):
        x = 0.02 + (i % 2) * 0.5
        y = 0.74 - (i // 2) * 0.42
        metrics.text(x, y, value, fontsize=26, fontweight="bold", color=GREEN)
        metrics.text(x, y - 0.11, label, fontsize=10.5, color=MUTED, linespacing=1.2)
    metrics.text(
        0.02,
        0.02,
        "Cost context: 307 targets/s is 86% of the original 8-bit baseline,\n"
        "but only 33% of the faster no-cover fused-Adam implementation.",
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.5", "fc": PALE_ORANGE, "ec": "#fed7aa"},
    )

    implications = fig.add_subplot(grid[2, :])
    implications.axis("off")
    implications.set_xlim(0, 1)
    implications.set_ylim(0, 1)
    _card(
        implications,
        0.0,
        0.31,
        "What happened",
        "Real inference cover changed the measured process enough to cut a strong 5 ms detector from 99.98% to 62.77%.",
        facecolor=PALE_BLUE,
    )
    _card(
        implications,
        0.345,
        0.31,
        "What did not happen",
        "Backward and AdamW did not become inference. A raw CNN still reaches 71.70%, and 100 ms ridge accuracy is 77.73%.",
        facecolor=PALE_RED,
    )
    _card(
        implications,
        0.69,
        0.31,
        "What it implies",
        "Power-only labels are manipulable under an adaptive scheduler. Defenses need long context, cadence tests, or independent telemetry.",
        facecolor=PALE_ORANGE,
    )

    fig.subplots_adjust(left=0.05, right=0.97, top=0.89, bottom=0.045)
    fig.savefig(OUTPUT, dpi=160, facecolor="white")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(render())
