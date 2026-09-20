"""Render measured signals and metrics without changing their values or alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..llama_continuous_whole_model.analyze_continuous import load_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    saved = args.results
    performance = json.loads((saved / "certification.json").read_text())
    power = json.loads((saved / "grouped_power_metrics.json").read_text())
    nested = json.loads((saved / "nested_detector.json").read_text())
    traces = {role: load_traces(args.root, role) for role in ("inference", "training")}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), constrained_layout=True)
    for column, (role, color) in enumerate((("inference", "#2166ac"), ("training", "#d6604d"))):
        trace = traces[role][0]
        x = np.arange(len(trace.values)) / trace.sample_rate_hz * 1e3
        axes[0, column].plot(x, trace.values, color=color, lw=0.35, rasterized=True)
        axes[0, column].set(title=f'Mixed computation · host label “{role}”',
                            xlabel="Time (ms)", ylabel="Raw ADC units", ylim=(-0.13, 0.13))
    horizons = [5, 10, 20, 50, 100]
    ridge = [power["detector_by_horizon_ms"][str(float(h))]["balanced_accuracy"] for h in horizons]
    cnn = [json.loads((saved / f"cnn_{h}ms.json").read_text())["balanced_accuracy"] for h in horizons]
    audited = [nested["horizons"][str(float(h))]["balanced_accuracy"] for h in horizons]
    ax = axes[1, 0]
    for values, label, color in ((ridge, "Ridge · crossed folds", "#777777"),
                                 (cnn, "CNN · crossed folds", "#7b3294"),
                                 (audited, "Nested ridge · paired folds", "#008837")):
        ax.plot(horizons, np.asarray(values) * 100, marker="o", label=label, color=color)
    ax.axhline(50, color="black", ls=":", lw=1)
    ax.axhline(60, color="#b2182b", ls="--", lw=1)
    ax.set(xscale="log", xticks=horizons, xticklabels=horizons, ylim=(25, 75),
           title="Five session pairs: detector evidence remains limited",
           xlabel="Observation window (ms)", ylabel="Balanced accuracy (%)")
    ax.legend(fontsize=9, loc="lower right")
    ax = axes[1, 1]
    timing = performance["wall_timing"]
    values = [timing["slowdown"][role] for role in ("inference", "training")]
    spreads = [timing["stdev_per_round_slowdown"][role] for role in ("inference", "training")]
    ax.bar([0, 1], values, yerr=spreads, width=0.55, capsize=5, color=["#2166ac", "#d6604d"])
    ax.axhline(2, color="black", ls="--", lw=1)
    ax.set(xticks=[0, 1], xticklabels=["Inference", "Training"], ylim=(0, 2.4),
           title=f'Native service ratio {timing["native_service_ratio"]:.4f}:1',
           ylabel="Slowdown versus dedicated service")
    for index, value in enumerate(values):
        ax.text(index, 0.8, f"{value:.3f}×", color="white", weight="bold", ha="center", fontsize=15)
    ax.text(0.5, 0.96, "Error bars: per-round SD; not confidence intervals", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)
    fig.suptitle("Llama 1B · real inference + SGD · H100 / ChipWhisperer", fontsize=16)
    fig.savefig(saved / "overview.png", dpi=170, facecolor="white")
    plt.close(fig)
    manifest = {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.root.rglob("*.npy"))}
    (saved / "trace_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
