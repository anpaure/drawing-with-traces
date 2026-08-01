#!/usr/bin/env python3
"""Render the concise, evidence-backed figure used by the GPT-OSS branch README."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "gpt_oss_inference_shaped_training"
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


def _run_metrics(run: str, variant: str) -> dict:
    return _load(RESULTS / run / variant / "power_similarity.json")


def render() -> Path:
    ordinary = _run_metrics("full_model_cover8_matched_27", "ordinary")
    cover4_ordinary = _run_metrics("full_model_cover4_final", "ordinary")
    cover4 = _run_metrics("full_model_cover4_final", "shaped")
    cover8 = _run_metrics("full_model_cover8_matched_27", "shaped")

    names = ["ordinary", "cover 4", "cover 8"]
    accuracy = np.array(
        [
            ordinary["classifier"]["accuracy"],
            cover4["classifier"]["accuracy"],
            cover8["classifier"]["accuracy"],
        ]
    )
    similarity = np.array(
        [
            ordinary["similarities"]["mean_signal_similarity"],
            cover4["similarities"]["mean_signal_similarity"],
            cover8["similarities"]["mean_signal_similarity"],
        ]
    )
    slowdown = np.array(
        [
            1.0,
            cover4["cuda_ms"]["training"]["mean"] / cover4_ordinary["cuda_ms"]["training"]["mean"],
            cover8["cuda_ms"]["training"]["mean"] / ordinary["cuda_ms"]["training"]["mean"],
        ]
    )
    phase_accuracy = cover8["classifier"]["accuracy_by_training_phase_offset"]
    phases = np.arange(9)
    phase_values = np.array([phase_accuracy[str(value)] for value in phases])

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
    grid = fig.add_gridspec(3, 12, height_ratios=[1.15, 1.35, 0.9], hspace=0.42, wspace=0.95)
    fig.suptitle(
        "GPT-OSS attention training hidden inside real inference cover",
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
    rows = [(1.85, "Inference reference"), (0.85, "Ordinary training"), (-0.15, "Cover-8 process")]
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
    for i in range(8):
        _block(timeline, 0.34 + i * 0.08, -0.15, 0.064, "", BLUE)
    timeline.text(0.655, -0.51, "one real update + eight real cached-decode covers", ha="center", color=MUTED)
    timeline.text(
        0.99,
        2.45,
        "full GPT-OSS forward/backward  •  updates 96 attention projections (637M parameters)",
        ha="right",
        fontsize=10.5,
        color=MUTED,
    )

    detector = fig.add_subplot(grid[1, :4])
    colors = [INK, ORANGE, GREEN]
    detector.bar(names, accuracy * 100, color=colors, width=0.65)
    detector.axhline(50, color="#94a3b8", lw=1.5, ls="--")
    detector.set_ylim(45, 102)
    detector.set_ylabel("held-out 5 ms accuracy (%)")
    detector.set_title("Aggregate detector", loc="left", fontweight="bold")
    detector.grid(axis="y", color=GRID, alpha=0.75)
    for i, value in enumerate(accuracy):
        detector.text(i, value * 100 + 1.3, f"{value * 100:.1f}%", ha="center", fontweight="bold")

    match = fig.add_subplot(grid[1, 4:8])
    match.bar(names, similarity * 100, color=colors, width=0.65)
    match.set_ylim(86, 101)
    match.set_ylabel("mean signal similarity (%)")
    match.set_title("Distribution match and cost", loc="left", fontweight="bold")
    match.grid(axis="y", color=GRID, alpha=0.75)
    for i, (value, cost) in enumerate(zip(similarity, slowdown, strict=True)):
        match.text(i, value * 100 + 0.35, f"{value * 100:.1f}%\n{cost:.2f}×", ha="center", fontweight="bold")

    offsets = fig.add_subplot(grid[1, 8:])
    phase_colors = [RED if value == 0 else ORANGE if value == 8 else BLUE for value in phases]
    offsets.bar(phases, phase_values * 100, color=phase_colors, width=0.72)
    offsets.axhline(50, color="#94a3b8", lw=1.5, ls="--")
    offsets.set_ylim(40, 88)
    offsets.set_xticks(phases)
    offsets.set_xlabel("first cycle offset")
    offsets.set_ylabel("held-out accuracy (%)")
    offsets.set_title("Cover-8 by phase", loc="left", fontweight="bold")
    offsets.grid(axis="y", color=GRID, alpha=0.75)
    offsets.text(0, phase_values[0] * 100 + 1.5, "training first", ha="center", fontsize=8, fontweight="bold")
    offsets.text(8, phase_values[8] * 100 + 1.5, "decode → train", ha="center", fontsize=8, fontweight="bold")

    implications = fig.add_subplot(grid[2, :])
    implications.axis("off")
    implications.set_xlim(0, 1)
    implications.set_ylim(0, 1)
    _card(
        implications,
        0.0,
        0.31,
        "What happened",
        "Across all rotated phases, cover-8 cut the 5 ms detector from 95.83% to 57.59% while the same 27 updates reduced loss.",
        facecolor=PALE_BLUE,
    )
    _card(
        implications,
        0.345,
        0.31,
        "What did not happen",
        "Training itself did not become invisible. Windows starting on training remain 80.83% detectable, and the process is 4.48× slower.",
        facecolor=PALE_RED,
    )
    _card(
        implications,
        0.69,
        0.31,
        "What it implies",
        "The 57.59% aggregate is mostly a cover-window result. Phase-aware or long-horizon monitoring can still expose recurring backward work.",
        facecolor=PALE_ORANGE,
    )

    fig.subplots_adjust(left=0.05, right=0.97, top=0.89, bottom=0.045)
    fig.savefig(OUTPUT, dpi=160, facecolor="white")
    plt.close(fig)
    return OUTPUT


if __name__ == "__main__":
    print(render())
