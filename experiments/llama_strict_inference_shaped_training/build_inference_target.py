#!/usr/bin/env python3
"""Convert one physical inference trace into a calibrated actuator target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from drawing_with_traces.fast import FastCalibration


def bin_feature(trace: np.ndarray, samples_per_bin: int, feature_name: str) -> np.ndarray:
    if trace.ndim != 1 or trace.size < samples_per_bin:
        raise ValueError("trace must be a one-dimensional array containing at least one bin")
    bins = trace.size // samples_per_bin
    rows = trace[: bins * samples_per_bin].reshape(bins, samples_per_bin)
    center = float(np.median(trace))
    centered = rows - center
    if feature_name == "rms":
        return np.sqrt(np.mean(np.square(centered), axis=1))
    if feature_name == "std":
        return np.std(rows, axis=1)
    if feature_name == "mean_abs":
        return np.mean(np.abs(centered), axis=1)
    if feature_name == "diff_rms":
        return np.sqrt(np.mean(np.square(np.diff(rows, axis=1)), axis=1))
    if feature_name == "q98_q02_span":
        return np.quantile(rows, 0.98, axis=1) - np.quantile(rows, 0.02, axis=1)
    raise ValueError(f"unsupported calibration feature: {feature_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate-hz", type=float, default=1_500_000)
    parser.add_argument("--bin-ms", type=float, default=1.0)
    args = parser.parse_args()
    if args.sample_rate_hz <= 0 or args.bin_ms <= 0:
        raise ValueError("sample rate and bin duration must be positive")
    samples_per_bin = int(round(args.sample_rate_hz * args.bin_ms / 1e3))
    calibration = FastCalibration.from_dict(json.loads(args.calibration.read_text()))
    trace = np.load(args.trace, allow_pickle=False).astype(np.float64)
    feature = bin_feature(trace, samples_per_bin, calibration.feature_name)
    activity = calibration.feature_sign * feature
    unclipped = (activity - calibration.minimum_activity) / (
        calibration.maximum_activity - calibration.minimum_activity
    )
    target = np.clip(unclipped, 0, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "inference_feature.npy", feature)
    np.save(args.output / "inference_target.npy", target)
    metadata = {
        "source_trace": str(args.trace),
        "calibration": str(args.calibration),
        "feature": calibration.feature_name,
        "feature_sign": calibration.feature_sign,
        "sample_rate_hz": args.sample_rate_hz,
        "samples_per_bin": samples_per_bin,
        "bin_ms": args.bin_ms,
        "points": int(target.size),
        "calibration_activity_range": [
            calibration.minimum_activity,
            calibration.maximum_activity,
        ],
        "feature_quantiles": np.quantile(feature, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "target_quantiles": np.quantile(target, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "unclipped_target_range": [float(unclipped.min()), float(unclipped.max())],
        "fraction_clipped_low": float(np.mean(unclipped < 0)),
        "fraction_clipped_high": float(np.mean(unclipped > 1)),
    }
    (args.output / "inference_target_metadata.json").write_text(json.dumps(metadata, indent=2))
    x = (np.arange(target.size) + 0.5) * args.bin_ms
    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True, facecolor="white")
    axis.plot(x, target, color="#1D4ED8", lw=1.6)
    axis.fill_between(x, 0, target, color="#DBEAFE", alpha=0.7)
    axis.set(
        xlim=(0, target.size * args.bin_ms),
        ylim=(-0.03, 1.03),
        xlabel="time (ms)",
        ylabel="normalized physical activity",
        title="Held-design-session inference target",
    )
    axis.grid(color="#E5E7EB", lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(args.output / "inference_target.png", dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
