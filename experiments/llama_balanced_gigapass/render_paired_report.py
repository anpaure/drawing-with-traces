"""Render an audited paired evaluation without selecting traces by their appearance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..llama_continuous_whole_model.analyze_continuous import load_traces

HORIZONS = (5, 10, 20, 50, 100)


def detector_series(payload: dict, pairs: int) -> list[float]:
    """Refuse to present a partial run as a complete detector evaluation."""
    result = []
    for horizon in HORIZONS:
        entry = payload["horizons"].get(str(float(horizon)))
        if entry is None or entry["pair_count"] != pairs or len(entry["folds"]) != pairs:
            raise ValueError(f"incomplete paired detector evaluation at {horizon} ms")
        value = entry["balanced_accuracy"]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid detector accuracy at {horizon} ms")
        result.append(value)
    return result


def select_pair(traces: dict, seed: int):
    """Selection uses identifiers and a fixed seed, never trace values or scores."""
    by_role = {
        role: {(int(t.session_id[1:]), t.index): t for t in values}
        for role, values in traces.items()
    }
    keys = sorted(set(by_role["inference"]) & set(by_role["training"]))
    if not keys:
        raise ValueError("no matching physical capture identifiers")
    key = keys[int(np.random.default_rng(seed).integers(len(keys)))]
    return key, {role: mapping[key] for role, mapping in by_role.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()
    audit = json.loads((args.root / "progress_audit.json").read_text())
    if not audit["passed"]:
        raise ValueError("physical records must pass the progress audit before rendering")
    traces = {role: load_traces(args.root, role) for role in ("inference", "training")}
    key, selected = select_pair(traces, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "path.simplify": False})
    colors = {"inference": "#2166ac", "training": "#d6604d"}
    bounds = (min(t.values.min() for t in selected.values()),
              max(t.values.max() for t in selected.values()))
    margin = (bounds[1] - bounds[0]) * 0.05
    fig, axes = plt.subplots(3, 2, figsize=(13, 8), layout="constrained", sharey=True)
    for column, (role, trace) in enumerate(selected.items()):
        for row, duration in enumerate((1, 5, 100)):
            count = round(duration * 1e-3 * trace.sample_rate_hz)
            start = (len(trace.values) - count) // 2
            values = trace.values[start:start + count]
            time = (start + np.arange(count)) / trace.sample_rate_hz * 1e3
            ax = axes[row, column]
            ax.plot(time, values, lw=0.45 if row < 2 else 0.3, color=colors[role])
            ax.set(title=f'{duration} ms · host label “{role}”', xlabel="Time within capture (ms)",
                   ylabel="Raw ADC units", ylim=(bounds[0] - margin, bounds[1] + margin))
    fig.suptitle(f"Identical mixed computation · pair {key[0]}, capture {key[1]}\n"
                 "Raw samples; shared amplitude scale; no alignment or normalization", fontsize=14)
    fig.savefig(args.output_dir / "eager_raw_windows.png", dpi=170, facecolor="white")
    plt.close(fig)
    manifest = {"pair": key[0], "capture_index": key[1], "selection_seed": args.seed,
                "selection": "uniform over sorted matching capture identifiers; no signal-dependent selection",
                "raw_window_durations_ms": [1, 5, 100], "window_location": "capture center",
                "sample_rate_hz": selected["inference"].sample_rate_hz,
                "processing": "none; no time alignment, normalization, filtering or resampling"}
    (args.output_dir / "eager_figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.raw_only:
        return
    ridge = json.loads((args.root / "nested_detector.json").read_text())
    cnn = json.loads((args.root / "paired_cnn.json").read_text())
    timing = json.loads(args.benchmark.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    series = [(detector_series(payload, audit["pairs"]), label, color) for payload, label, color in
              ((ridge, "Nested ridge", "#008837"), (cnn, "Raw-waveform CNN", "#7b3294"))]
    for index, (values, label, color) in enumerate(series):
        axes[0].plot(HORIZONS, 100 * np.asarray(values), marker="o", color=color, label=label)
        for h, value, other in zip(HORIZONS, values, series[1 - index][0]):
            axes[0].annotate(f"{value:.1%}", (h, 100 * value), xytext=(0, 10 if value >= other else -16),
                             textcoords="offset points", ha="center", fontsize=9, color=color)
    axes[0].axhline(50, color="black", lw=1, ls=":", label="Chance")
    axes[0].axhline(60, color="#b2182b", lw=1, ls="--", label="60% accuracy threshold")
    axes[0].set(title=f'{audit["pairs"]} held-out session pairs · {audit["record_count"]} captures',
                xscale="log", xticks=HORIZONS, xticklabels=HORIZONS, ylim=(0, 100),
                xlabel="Observation window (ms)", ylabel="Balanced accuracy (%)")
    axes[0].legend(fontsize=9, loc="lower right")
    wall = timing["wall_timing"]
    for index, role in enumerate(("inference", "training")):
        value = wall["slowdown"][role]
        axes[1].bar(index, value, width=0.55, color=colors[role])
        rounds = wall["per_round_slowdown"][role]
        axes[1].scatter(index + np.linspace(-0.12, 0.12, len(rounds)), rounds, s=16,
                        color="#111111", alpha=0.7)
        axes[1].text(index, 0.9, f"{value:.3f}×", ha="center", color="white", fontsize=16, weight="bold")
    axes[1].axhline(2, color="black", ls="--", lw=1)
    axes[1].set(title=f'Separate timing run · native service {wall["native_service_ratio"]:.3f}:1',
                xticks=[0, 1], xticklabels=["Inference", "Training"], ylim=(0, 2.45),
                ylabel="Slowdown versus dedicated workload")
    axes[1].text(0.5, 0.97, "Bars: aggregate timing; dots: individual rounds", ha="center", va="top",
                 transform=axes[1].transAxes, fontsize=9)
    fig.suptitle("Llama 1B · full-model inference + SGD under both labels\n"
                 "Not a test against unmodified inference-only serving", fontsize=14)
    fig.savefig(args.output_dir / "eager_summary.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
