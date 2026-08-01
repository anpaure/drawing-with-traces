"""Calibration, telemetry analysis, and reference-style rendering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from sidecapture.errors import ConfigurationError
from sidecapture.store import DirectoryStore, atomic_json, atomic_numpy

from .envelope import PowerEnvelope, load_envelope


@dataclass(frozen=True)
class Telemetry:
    time_s: np.ndarray
    power_w: np.ndarray
    timestamps_ns: np.ndarray
    record: dict[str, Any]
    annotations: list[dict[str, Any]]


@dataclass(frozen=True)
class CalibrationCurve:
    duty: np.ndarray
    measured_power_w: np.ndarray
    monotonic_power_w: np.ndarray

    @property
    def minimum_power_w(self) -> float:
        return float(self.monotonic_power_w[0])

    @property
    def maximum_power_w(self) -> float:
        return float(self.monotonic_power_w[-1])

    def duties_for(self, target_normalized: np.ndarray) -> np.ndarray:
        desired = self.minimum_power_w + np.asarray(target_normalized) * (
            self.maximum_power_w - self.minimum_power_w
        )
        return np.interp(desired, self.monotonic_power_w, self.duty).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duty": self.duty.tolist(),
            "measured_power_w": self.measured_power_w.tolist(),
            "monotonic_power_w": self.monotonic_power_w.tolist(),
            "minimum_power_w": self.minimum_power_w,
            "maximum_power_w": self.maximum_power_w,
            "dynamic_range_w": self.maximum_power_w - self.minimum_power_w,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CalibrationCurve":
        return cls(
            np.asarray(value["duty"], dtype=np.float64),
            np.asarray(value["measured_power_w"], dtype=np.float64),
            np.asarray(value["monotonic_power_w"], dtype=np.float64),
        )


def load_telemetry(root: str | Path) -> Telemetry:
    root = Path(root)
    records = DirectoryStore(root, resume=True).records()
    if len(records) != 1:
        raise ConfigurationError(f"expected one long telemetry capture in {root}, found {len(records)}")
    record = records[0]
    channel_name = next((name for name in record["channels"] if name.endswith(".power_w")), None)
    if channel_name is None:
        raise ConfigurationError(f"capture has no NVML power channel: {sorted(record['channels'])}")
    descriptor = record["channels"][channel_name]
    if "timestamps_path" not in descriptor:
        raise ConfigurationError("NVML power channel has no host timestamps")
    power = np.load(root / descriptor["path"], allow_pickle=False).astype(np.float64)
    timestamps = np.load(root / descriptor["timestamps_path"], allow_pickle=False).astype(np.int64)
    trigger_ns = int(record["trigger"]["host_monotonic_ns"])
    annotations = json.loads((root / record["annotations"]["path"]).read_text())
    return Telemetry((timestamps - trigger_ns) / 1e9, power, timestamps, record, annotations)


def interval(telemetry: Telemetry, name: str) -> dict[str, Any]:
    matches = [event for event in telemetry.annotations if event["name"] == name]
    if len(matches) != 1:
        raise ConfigurationError(f"expected one {name!r} interval, found {len(matches)}")
    return matches[0]


def segments(telemetry: Telemetry) -> list[dict[str, Any]]:
    values = [event for event in telemetry.annotations if event["name"] == "power_target.segment"]
    values.sort(key=lambda event: int(event["metadata"]["index"]))
    if not values:
        raise ConfigurationError("capture has no power_target.segment annotations")
    return values


def _isotonic_increasing(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=np.float64)
    blocks = [[float(value), float(weight), index, index + 1] for index, (value, weight) in enumerate(zip(values, weights))]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][0] <= blocks[index + 1][0]:
            index += 1
            continue
        left, right = blocks[index], blocks[index + 1]
        weight = left[1] + right[1]
        mean = (left[0] * left[1] + right[0] * right[1]) / weight
        blocks[index : index + 2] = [[mean, weight, left[2], right[3]]]
        index = max(0, index - 1)
    result = np.empty_like(values)
    for mean, _, start, end in blocks:
        result[int(start) : int(end)] = mean
    return result


def calibration_from_run(root: str | Path, *, settle_fraction: float = 0.45) -> CalibrationCurve:
    if not 0 <= settle_fraction < 1:
        raise ValueError("settle_fraction must be between 0 and 1")
    telemetry = load_telemetry(root)
    rows = []
    for event in segments(telemetry):
        start = int(event["start"])
        end = int(event["end"])
        settled_start = start + int((end - start) * settle_fraction)
        selected = telemetry.power_w[
            (telemetry.timestamps_ns >= settled_start) & (telemetry.timestamps_ns <= end)
        ]
        if selected.size < 3:
            raise ConfigurationError(
                f"calibration segment {event['metadata']['index']} has only {selected.size} settled samples"
            )
        rows.append((float(event["metadata"]["duty_cycle"]), float(np.median(selected))))
    levels = sorted({duty for duty, _ in rows})
    measured = np.asarray(
        [np.median([power for duty, power in rows if duty == level]) for level in levels],
        dtype=np.float64,
    )
    duty = np.asarray(levels, dtype=np.float64)
    monotonic = _isotonic_increasing(measured)
    if monotonic[-1] - monotonic[0] < 10:
        raise ConfigurationError(
            f"calibration changed measured power by only {monotonic[-1] - monotonic[0]:.1f} W; "
            "the training workload is not controlling GPU power strongly enough"
        )
    return CalibrationCurve(duty, measured, monotonic)


def save_calibration(root: str | Path, curve: CalibrationCurve) -> Path:
    path = Path(root) / "calibration.json"
    atomic_json(curve.to_dict(), path)
    return path


def load_calibration(path: str | Path) -> CalibrationCurve:
    path = Path(path)
    if path.is_dir():
        path = path / "calibration.json"
    if not path.exists():
        raise ConfigurationError(f"calibration file does not exist: {path}")
    return CalibrationCurve.from_dict(json.loads(path.read_text()))


def render_target_preview(envelope: PowerEnvelope, output: str | Path) -> Path:
    output = Path(output)
    x = np.linspace(0, 1, envelope.values.size)
    figure, axis = plt.subplots(figsize=(14, 3.2), constrained_layout=True, facecolor="white")
    axis.fill_between(x, 0, envelope.values, color="black", linewidth=0)
    axis.plot(x, envelope.values, color="black", linewidth=1.2)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.05)
    axis.axis("off")
    descriptions = {
        "height": "Silhouette lowered to a common baseline · column height becomes target power",
        "upper-boundary": "Upper image boundary lowered to a baseline · edge position becomes target power",
        "lower-boundary": "Lower image boundary lowered to a baseline · edge position becomes target power",
    }
    axis.set_title(descriptions[envelope.extraction_mode], fontsize=13, color="#101828")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def render_calibration(root: str | Path, curve: CalibrationCurve, output: str | Path) -> Path:
    output = Path(output)
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True, facecolor="white")
    axis.plot(curve.duty * 100, curve.measured_power_w, "o", color="#D92D20", label="Measured median")
    axis.plot(
        curve.duty * 100,
        curve.monotonic_power_w,
        color="#101828",
        linewidth=1.5,
        label="Monotonic controller map",
    )
    axis.set_xlabel("Training duty cycle (%)")
    axis.set_ylabel("GPU power (W, NVML)")
    axis.set_title("H100 training-duty calibration")
    axis.grid(True, color="#E4E7EC", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def _moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    samples = max(1, int(samples))
    if samples == 1:
        return values.copy()
    left = samples // 2
    right = samples - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(samples) / samples, mode="valid")


def _best_lag(target: np.ndarray, measured: np.ndarray, max_samples: int) -> tuple[int, float]:
    best_lag, best_correlation = 0, float("-inf")
    for lag in range(-max_samples, max_samples + 1):
        if lag < 0:
            left, right = target[-lag:], measured[:lag]
        elif lag > 0:
            left, right = target[:-lag], measured[lag:]
        else:
            left, right = target, measured
        if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_lag, best_correlation = lag, correlation
    return best_lag, best_correlation


def analyze_drawing(
    root: str | Path,
    calibration: CalibrationCurve,
    *,
    smooth_s: float = 0.25,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(root)
    telemetry = load_telemetry(root)
    envelope = load_envelope(root)
    profile = interval(telemetry, "power_target.profile")
    start_ns, end_ns = int(profile["start"]), int(profile["end"])
    selected = (telemetry.timestamps_ns >= start_ns) & (telemetry.timestamps_ns <= end_ns)
    timestamps = telemetry.timestamps_ns[selected]
    power = telemetry.power_w[selected]
    if power.size < 10:
        raise ConfigurationError(f"profile contains only {power.size} NVML power samples")
    time_s = (timestamps - start_ns) / 1e9
    median_dt = float(np.median(np.diff(time_s)))
    smoothed = _moving_average(power, round(smooth_s / median_dt))
    x_envelope = np.linspace(0, time_s[-1], envelope.values.size)
    target_normalized = np.interp(time_s, x_envelope, envelope.values)
    target_power = calibration.minimum_power_w + target_normalized * (
        calibration.maximum_power_w - calibration.minimum_power_w
    )
    measured_normalized = (smoothed - calibration.minimum_power_w) / (
        calibration.maximum_power_w - calibration.minimum_power_w
    )
    pearson = float(np.corrcoef(target_normalized, measured_normalized)[0, 1])
    error = measured_normalized - target_normalized
    normalized_rmse = float(np.sqrt(np.mean(np.square(error))))
    target_variance = float(np.sum(np.square(target_normalized - target_normalized.mean())))
    lag_samples, lagged_correlation = _best_lag(
        target_normalized,
        measured_normalized,
        max(1, round(1.0 / median_dt)),
    )
    metrics = {
        "profile_duration_s": float(time_s[-1]),
        "nvml_samples": int(power.size),
        "nvml_median_interval_s": median_dt,
        "display_smoothing_s": float(smooth_s),
        "measured_power_min_w": float(smoothed.min()),
        "measured_power_max_w": float(smoothed.max()),
        "measured_power_dynamic_range_w": float(smoothed.max() - smoothed.min()),
        "target_power_min_w": calibration.minimum_power_w,
        "target_power_max_w": calibration.maximum_power_w,
        "pearson_r_unshifted": pearson,
        "normalized_mae": float(np.mean(np.abs(error))),
        "normalized_rmse": normalized_rmse,
        "r_squared": float(1 - np.sum(np.square(error)) / max(target_variance, 1e-15)),
        "shape_accuracy_percent": float(max(0.0, 1.0 - normalized_rmse) * 100),
        "best_lag_s": float(lag_samples * median_dt),
        "pearson_r_at_best_lag": lagged_correlation,
        "plot_uses_lag_shift": False,
        "power_source": "NVML power usage in watts",
    }
    arrays = {
        "time_s": time_s,
        "raw_power_w": power,
        "smoothed_power_w": smoothed,
        "target_power_w": target_power,
        "target_normalized": target_normalized,
        "measured_normalized": measured_normalized,
    }
    return metrics, arrays


def render_drawing(
    root: str | Path,
    calibration: CalibrationCurve,
    output: str | Path | None = None,
    *,
    compressed_duration_s: float = 2.0,
    smooth_s: float = 0.25,
) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    envelope = load_envelope(root)
    metrics, arrays = analyze_drawing(root, calibration, smooth_s=smooth_s)
    output = Path(output) if output is not None else root / "measured_silhouette.png"
    figure = plt.figure(figsize=(14, 8.2), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(3, 1, height_ratios=[1.35, 1, 1])
    axes = [figure.add_subplot(grid[index, 0]) for index in range(3)]

    shape_x = np.linspace(0, 1, envelope.values.size)
    axes[0].fill_between(shape_x, 0, envelope.values, color="black", linewidth=0)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.05)
    axes[0].axis("off")
    axes[0].set_title("Lowered target silhouette", fontsize=13, pad=6)

    full_time = arrays["time_s"]
    measured = arrays["smoothed_power_w"]
    target = arrays["target_power_w"]
    baseline = min(float(measured.min()), calibration.minimum_power_w) - 5
    for axis, x, title in [
        (axes[1], full_time, f"{metrics['profile_duration_s']:.0f} seconds (measured)"),
        (
            axes[2],
            full_time / max(full_time[-1], 1e-12) * compressed_duration_s,
            f"{compressed_duration_s:g} seconds (time-compressed view)",
        ),
    ]:
        axis.fill_between(x, baseline, measured, color="#D0D5DD", alpha=0.8, linewidth=0)
        axis.plot(x, measured, color="#D92D20", linewidth=1.6, label="Measured H100 power")
        axis.plot(x, target, color="#475467", linewidth=1.0, linestyle="--", label="Target power")
        axis.set_xlim(float(x[0]), float(x[-1]))
        axis.set_ylim(baseline, max(float(measured.max()), calibration.maximum_power_w) + 10)
        axis.set_ylabel("GPU power (W)")
        axis.set_title(title, fontsize=15, pad=7)
        axis.grid(True, color="#E4E7EC", linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time (s)")
    axes[2].set_xlabel("time (s)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    figure.suptitle(
        "Silhouette drawn with real model-training power",
        fontsize=17,
        weight="bold",
        color="#101828",
    )
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    atomic_json(metrics, root / "artwork" / "drawing_metrics.json")
    for name, values in arrays.items():
        atomic_numpy(np.asarray(values, dtype=np.float32), root / "artwork" / f"{name}.npy")
    return output, metrics
