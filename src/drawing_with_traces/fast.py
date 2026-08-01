"""Millisecond-scale tiled-gradient drawing through ChipWhisperer."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import sidecapture as sc
from sidecapture.errors import ConfigurationError
from sidecapture.store import DirectoryStore, atomic_json, atomic_numpy
from sidecapture.workloads import Workload

from .analysis import _isotonic_increasing
from .envelope import PowerEnvelope
from .workload import sleep_until


def array_hash(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


class TiledLinearTrainingWorkload(Workload):
    """Hand-written exact gradient for ``0.5/B * ||XW-Y||²``.

    Every controlled tile computes both a real forward block ``X @ W[:, j:k]``
    and its real weight-gradient block ``X.T @ residual[:, j:k]``. Repeated
    visits are averaged by output column before the accepted SGD update, so the
    delivered gradient is independent of the tile-width pattern.
    """

    replay_safe = True

    def __init__(
        self,
        commands: np.ndarray,
        *,
        profile_duration_s: float,
        hidden_size: int = 4096,
        batch_size: int = 512,
        learning_rate: float = 0.01,
        lead_s: float = 0.002,
        tail_s: float = 0.002,
        seed: int = 1729,
    ):
        commands = np.asarray(commands, dtype=np.int64)
        if commands.ndim != 1 or commands.size < 2:
            raise ValueError("commands must be a one-dimensional array with at least two bins")
        if np.any(commands < 0) or np.any(commands > hidden_size):
            raise ValueError("tile commands must be between 0 and hidden_size")
        nonzero = commands[commands > 0]
        if nonzero.size and np.any(nonzero % 32):
            raise ValueError("non-zero tile widths must be multiples of 32")
        if profile_duration_s <= 0 or lead_s < 0 or tail_s < 0:
            raise ValueError("profile duration must be positive and lead/tail cannot be negative")
        if hidden_size < 32 or hidden_size % 32:
            raise ValueError("hidden_size must be a positive multiple of 32")
        if batch_size < 1 or learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        self.commands = np.ascontiguousarray(commands)
        self.profile_duration_s = float(profile_duration_s)
        self.hidden_size = int(hidden_size)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.lead_s = float(lead_s)
        self.tail_s = float(tail_s)
        self.seed = int(seed)
        self.torch = None
        self.inputs = None
        self.targets = None
        self.weight = None
        self.master_weight = None
        self.gradient_sum = None
        self.column_visits = None
        self.cursor = 0
        self.loss_before = None
        self.loss_after = None

    @property
    def parameter_count(self) -> int:
        return self.hidden_size * self.hidden_size

    @property
    def bin_duration_s(self) -> float:
        return self.profile_duration_s / self.commands.size

    @property
    def total_duration_s(self) -> float:
        return self.lead_s + self.profile_duration_s + self.tail_s

    def setup(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise ConfigurationError("fast tiled training requires PyTorch") from error
        if not torch.cuda.is_available():
            raise ConfigurationError("fast tiled training requires a CUDA GPU")
        self.torch = torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.targets = torch.randn_like(self.inputs)
        self.master_weight = torch.randn(
            self.hidden_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.float32,
        ) / math.sqrt(self.hidden_size)
        self.weight = self.master_weight.to(torch.bfloat16)
        self.gradient_sum = torch.zeros_like(self.master_weight)
        self.column_visits = np.zeros(self.hidden_size, dtype=np.int64)
        self.loss_before = self.full_loss()
        torch.cuda.synchronize()

    def require_setup(self):
        if self.torch is None or self.weight is None or self.gradient_sum is None:
            raise RuntimeError("TiledLinearTrainingWorkload.setup() has not completed")
        return self.torch

    def full_loss(self) -> float:
        torch = self.torch
        with torch.no_grad():
            residual = self.inputs @ self.weight - self.targets
            loss = 0.5 * residual.float().square().mean()
        torch.cuda.synchronize()
        return float(loss)

    def gradient_tile(self, start: int, width: int, *, accumulate: bool = True) -> None:
        torch = self.require_setup()
        end = start + width
        prediction = self.inputs @ self.weight[:, start:end]
        residual = prediction - self.targets[:, start:end]
        gradient = self.inputs.T @ residual
        if accumulate:
            self.gradient_sum[:, start:end].add_(gradient.float() / self.batch_size)
            self.column_visits[start:end] += 1
        torch.cuda.synchronize()

    def next_tile(self, requested_width: int) -> None:
        if requested_width == 0:
            return
        width = min(int(requested_width), self.hidden_size)
        if self.cursor + width > self.hidden_size:
            self.cursor = 0
        self.gradient_tile(self.cursor, width)
        self.cursor = (self.cursor + width) % self.hidden_size

    def clear_gradient(self) -> None:
        if self.gradient_sum is not None:
            self.gradient_sum.zero_()
        if self.column_visits is not None:
            self.column_visits.fill(0)
        self.cursor = 0

    def warmup(self, iteration: int) -> None:
        del iteration
        self.gradient_tile(0, self.hidden_size, accumulate=False)

    def run(self, context):
        self.clear_gradient()
        context.labels.update(
            task="chipwhisperer-tiled-linear-training",
            profile_duration_s=self.profile_duration_s,
            profile_bins=int(self.commands.size),
            model_parameters=self.parameter_count,
        )
        context.add_artifact("tile_width_commands", self.commands.astype(np.int32))
        with context.region("tile.lead_idle", duration_s=self.lead_s):
            sleep_until(time.monotonic_ns() + int(round(self.lead_s * 1e9)))
        bin_ns = int(round(self.bin_duration_s * 1e9))
        operation_counts = np.zeros(self.commands.size, dtype=np.int64)
        overruns = np.zeros(self.commands.size, dtype=np.int64)
        profile_start_ns = time.monotonic_ns()
        with context.region(
            "tile.profile",
            duration_s=self.profile_duration_s,
            bins=int(self.commands.size),
        ):
            for index, width in enumerate(self.commands):
                deadline_ns = profile_start_ns + (index + 1) * bin_ns
                with context.region(
                    "tile.bin",
                    index=index,
                    requested_width=int(width),
                ):
                    if width == 0:
                        sleep_until(deadline_ns)
                    else:
                        while time.monotonic_ns() < deadline_ns:
                            self.next_tile(int(width))
                            operation_counts[index] += 1
                overruns[index] = max(0, time.monotonic_ns() - deadline_ns)
        with context.region("tile.tail_idle", duration_s=self.tail_s):
            sleep_until(time.monotonic_ns() + int(round(self.tail_s * 1e9)))
        max_overrun_ns = int(overruns.max(initial=0))
        if max_overrun_ns > max(250_000, bin_ns // 2):
            raise RuntimeError(
                f"a {self.bin_duration_s * 1e3:.3f} ms tile bin overran by "
                f"{max_overrun_ns / 1e6:.3f} ms; rejecting the timing-distorted trace"
            )
        context.add_artifact("tile_operations_per_bin", operation_counts)
        context.labels.update(
            tiled_gradient_operations=int(operation_counts.sum()),
            max_bin_overrun_us=max_overrun_ns / 1e3,
            loss_before_update=self.loss_before,
        )
        return {
            "tiled_gradient_operations": int(operation_counts.sum()),
            "max_bin_overrun_us": max_overrun_ns / 1e3,
            "loss_before_update": self.loss_before,
        }

    def complete_exact_gradient(self) -> None:
        for start in range(0, self.hidden_size, 256):
            end = min(self.hidden_size, start + 256)
            if np.any(self.column_visits[start:end] == 0):
                self.gradient_tile(start, end - start)

    def on_accept(self, result: Any) -> None:
        del result
        torch = self.require_setup()
        self.complete_exact_gradient()
        visits = torch.from_numpy(self.column_visits).to(device="cuda", dtype=torch.float32)
        average_gradient = self.gradient_sum / visits.clamp_min(1).unsqueeze(0)
        self.master_weight.add_(average_gradient, alpha=-self.learning_rate)
        self.weight.copy_(self.master_weight.to(torch.bfloat16))
        self.loss_after = self.full_loss()
        if not math.isfinite(self.loss_after) or self.loss_after > self.loss_before:
            raise RuntimeError(
                f"exact tiled SGD update did not reduce loss: {self.loss_before} -> {self.loss_after}"
            )
        self.clear_gradient()

    def on_reject(self, error: BaseException) -> None:
        del error
        self.clear_gradient()

    def teardown(self) -> None:
        self.inputs = None
        self.targets = None
        self.weight = None
        self.master_weight = None
        self.gradient_sum = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "drawing_with_traces.exact_tiled_linear_gradient",
            "objective": "0.5 / batch * ||XW - Y||^2",
            "gradient": "X.T @ (XW - Y) / batch, tiled over output columns",
            "commands_sha256": array_hash(self.commands, "<i8"),
            "profile_bins": int(self.commands.size),
            "profile_duration_s": self.profile_duration_s,
            "bin_duration_s": self.bin_duration_s,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "parameter_count": self.parameter_count,
            "dtype": "bfloat16 matmuls, float32 gradient accumulation/master weights",
            "optimizer": "exact deferred SGD",
            "learning_rate": self.learning_rate,
            "lead_s": self.lead_s,
            "tail_s": self.tail_s,
            "seed": self.seed,
            "transactional_update": True,
        }


@dataclass(frozen=True)
class FastCalibration:
    widths: np.ndarray
    feature_name: str
    feature_sign: float
    measured_feature: np.ndarray
    monotonic_activity: np.ndarray
    within_width_std: np.ndarray

    @property
    def minimum_activity(self) -> float:
        return float(self.monotonic_activity[0])

    @property
    def maximum_activity(self) -> float:
        return float(self.monotonic_activity[-1])

    def commands_for(self, target: np.ndarray) -> np.ndarray:
        desired = self.minimum_activity + np.asarray(target) * (
            self.maximum_activity - self.minimum_activity
        )
        continuous = np.interp(desired, self.monotonic_activity, self.widths)
        distances = np.abs(continuous[:, None] - self.widths[None, :])
        return self.widths[np.argmin(distances, axis=1)].astype(np.int64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "widths": self.widths.tolist(),
            "feature_name": self.feature_name,
            "feature_sign": self.feature_sign,
            "measured_feature": self.measured_feature.tolist(),
            "monotonic_activity": self.monotonic_activity.tolist(),
            "within_width_std": self.within_width_std.tolist(),
            "minimum_activity": self.minimum_activity,
            "maximum_activity": self.maximum_activity,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FastCalibration":
        return cls(
            np.asarray(value["widths"], dtype=np.float64),
            str(value["feature_name"]),
            float(value["feature_sign"]),
            np.asarray(value["measured_feature"], dtype=np.float64),
            np.asarray(value["monotonic_activity"], dtype=np.float64),
            np.asarray(value["within_width_std"], dtype=np.float64),
        )


def _trace_and_bins(root: str | Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    records = DirectoryStore(root, resume=True).records()
    if len(records) != 1:
        raise ConfigurationError(f"expected one ChipWhisperer trace in {root}, found {len(records)}")
    record = records[0]
    descriptor = record["channels"][record["primary_channel"]]
    trace = np.load(root / descriptor["path"], allow_pickle=False).astype(np.float64)
    annotations = json.loads((root / record["annotations"]["path"]).read_text())
    bins = [event for event in annotations if event["name"] == "tile.bin"]
    bins.sort(key=lambda event: int(event["metadata"]["index"]))
    if not bins:
        raise ConfigurationError("trace has no tile.bin annotations")
    return trace, bins, record


def _features(segment: np.ndarray, trace_center: float) -> dict[str, float]:
    centered = segment - trace_center
    differences = np.diff(segment)
    return {
        "rms": float(np.sqrt(np.mean(np.square(centered)))),
        "std": float(np.std(segment)),
        "mean_abs": float(np.mean(np.abs(centered))),
        "diff_rms": float(np.sqrt(np.mean(np.square(differences)))) if differences.size else 0.0,
        "q98_q02_span": float(np.quantile(segment, 0.98) - np.quantile(segment, 0.02)),
    }


def measured_bin_features(root: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    trace, bins, record = _trace_and_bins(root)
    center = float(np.median(trace))
    by_name: dict[str, list[float]] = {}
    widths = []
    for event in bins:
        start, end = int(event["start_sample"]), int(event["end_sample"])
        segment = trace[max(0, start) : min(trace.size, end)]
        if segment.size < 8:
            raise ConfigurationError(
                f"tile bin {event['metadata']['index']} has only {segment.size} ADC samples"
            )
        for name, value in _features(segment, center).items():
            by_name.setdefault(name, []).append(value)
        widths.append(int(event["metadata"]["requested_width"]))
    return {name: np.asarray(value) for name, value in by_name.items()}, np.asarray(widths), record


def calibrate_fast(root: str | Path) -> FastCalibration:
    features, commands, _ = measured_bin_features(root)
    widths = np.asarray(sorted(set(commands)), dtype=np.float64)
    x = np.log2(widths + 32)
    best = None
    for name, values in features.items():
        medians = np.asarray([np.median(values[commands == width]) for width in widths])
        correlation = float(np.corrcoef(x, medians)[0, 1])
        score = abs(correlation)
        if best is None or score > best[0]:
            best = (score, name, medians, correlation)
    _, name, measured, correlation = best
    sign = 1.0 if correlation >= 0 else -1.0
    activity = sign * measured
    # The AC-coupled feature can turn over once very wide GEMMs become more
    # efficient. Widths beyond the measured activity peak are not invertible
    # and must not be used by the controller.
    peak = int(np.argmax(activity))
    widths = widths[: peak + 1]
    measured = measured[: peak + 1]
    activity = activity[: peak + 1]
    monotonic = _isotonic_increasing(activity)
    within = np.asarray(
        [np.std(features[name][commands == width]) for width in widths],
        dtype=np.float64,
    )
    span = float(monotonic[-1] - monotonic[0])
    noise = float(np.median(within))
    if span <= max(1e-9, 2 * noise):
        raise ConfigurationError(
            f"tile-width calibration is not separable: activity span {span:.4g}, median noise {noise:.4g}"
        )
    return FastCalibration(widths, name, sign, measured, monotonic, within)


def save_fast_calibration(root: str | Path, calibration: FastCalibration) -> Path:
    path = Path(root) / "fast_calibration.json"
    atomic_json(calibration.to_dict(), path)
    return path


def render_fast_calibration(
    calibration: FastCalibration,
    output: str | Path,
) -> Path:
    output = Path(output)
    activity = calibration.feature_sign * calibration.measured_feature
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True, facecolor="white")
    axis.errorbar(
        calibration.widths,
        activity,
        yerr=calibration.within_width_std,
        fmt="o",
        color="#D92D20",
        capsize=3,
        label=f"Measured {calibration.feature_name}",
    )
    axis.plot(
        calibration.widths,
        calibration.monotonic_activity,
        color="#101828",
        linewidth=1.5,
        label="Monotonic controller map",
    )
    axis.set_xscale("symlog", linthresh=32, base=2)
    axis.set_xlabel("Output-column tile width")
    axis.set_ylabel("Oriented ChipWhisperer feature")
    axis.set_title("Tiled-gradient width calibration")
    axis.grid(True, color="#E4E7EC", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return output


def load_fast_calibration(path: str | Path) -> FastCalibration:
    path = Path(path)
    if path.is_dir():
        path = path / "fast_calibration.json"
    if not path.exists():
        raise ConfigurationError(f"fast calibration does not exist: {path}")
    return FastCalibration.from_dict(json.loads(path.read_text()))


def _smooth_points(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.copy()
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma))
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def analyze_fast_drawing(
    root: str | Path,
    target: np.ndarray,
    calibration: FastCalibration,
    *,
    smoothing_sigma_points: float = 0.8,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features, commands, record = measured_bin_features(root)
    measured_feature = features[calibration.feature_name]
    activity = calibration.feature_sign * measured_feature
    measured = (activity - calibration.minimum_activity) / max(
        calibration.maximum_activity - calibration.minimum_activity,
        1e-15,
    )
    measured_smoothed = _smooth_points(measured, smoothing_sigma_points)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != measured.shape:
        raise ConfigurationError(
            f"target has {target.size} points but trace contains {measured.size} tile bins"
        )
    error = measured_smoothed - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    target_variance = float(np.sum(np.square(target - target.mean())))
    metrics = {
        "requested_duration_ms": float(record["labels"]["profile_duration_s"] * 1e3),
        "bins": int(target.size),
        "feature": calibration.feature_name,
        "feature_sign": calibration.feature_sign,
        "smoothing_sigma_points": smoothing_sigma_points,
        "pearson_r": float(np.corrcoef(target, measured_smoothed)[0, 1]),
        "normalized_mae": float(np.mean(np.abs(error))),
        "normalized_rmse": rmse,
        "r_squared": float(1 - np.sum(np.square(error)) / max(target_variance, 1e-15)),
        "shape_accuracy_percent": float(max(0, 1 - rmse) * 100),
        "measurement": "ChipWhisperer normalized ADC activity; not calibrated watts",
        "loss_before_update": record["labels"].get("loss_before_update"),
    }
    arrays = {
        "target": target,
        "measured_raw": measured,
        "measured_smoothed": measured_smoothed,
        "commands": commands,
        "feature_raw": measured_feature,
    }
    return metrics, arrays


def render_fast_drawing(
    root: str | Path,
    target: np.ndarray,
    envelope: PowerEnvelope,
    calibration: FastCalibration,
    *,
    output: str | Path | None = None,
    smoothing_sigma_points: float = 0.8,
) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    metrics, arrays = analyze_fast_drawing(
        root,
        target,
        calibration,
        smoothing_sigma_points=smoothing_sigma_points,
    )
    output = Path(output) if output is not None else root / "fast_measured_silhouette.png"
    duration_ms = metrics["requested_duration_ms"]
    x = np.linspace(0, duration_ms, target.size)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 6.5),
        gridspec_kw={"height_ratios": [1.2, 1]},
        constrained_layout=True,
        facecolor="white",
    )
    shape_x = np.linspace(0, duration_ms, envelope.values.size)
    axes[0].fill_between(shape_x, 0, envelope.values, color="black", linewidth=0)
    axes[0].set_xlim(0, duration_ms)
    axes[0].set_ylim(0, 1.05)
    axes[0].axis("off")
    axes[0].set_title("Lowered target silhouette", fontsize=13)
    axes[1].plot(x, target, color="#475467", linestyle="--", linewidth=1.4, label="Target")
    axes[1].plot(
        x,
        arrays["measured_smoothed"],
        color="#D92D20",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Measured ChipWhisperer activity",
    )
    axes[1].fill_between(x, 0, arrays["measured_smoothed"], color="#D0D5DD", alpha=0.65)
    axes[1].set_xlim(0, duration_ms)
    low = min(0.0, float(arrays["measured_smoothed"].min()) - 0.05)
    high = max(1.0, float(arrays["measured_smoothed"].max()) + 0.05)
    axes[1].set_ylim(low, high)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("normalized activity")
    axes[1].set_title(
        f"{duration_ms:g} ms real tiled-gradient capture · accuracy {metrics['shape_accuracy_percent']:.1f}%"
    )
    axes[1].grid(True, color="#E4E7EC", linewidth=0.65)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    figure.suptitle(
        "Silhouette drawn with hand-tiled training gradients",
        fontsize=17,
        weight="bold",
        color="#101828",
    )
    figure.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    atomic_json(metrics, root / "fast_metrics.json")
    for name, values in arrays.items():
        atomic_numpy(np.asarray(values, dtype=np.float32), root / f"fast_{name}.npy")
    return output, metrics


def capture_fast(
    root: str | Path,
    workload: TiledLinearTrainingWorkload,
    *,
    sample_rate: str,
    margin_s: float,
    gain_db: float = 10.0,
    max_attempts: int = 4,
) -> dict[str, Any]:
    root = Path(root)
    request_duration = workload.total_duration_s + margin_s
    sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=request_duration,
            sample_rate=sample_rate,
            mode="burst",
            gain_db=gain_db,
        ),
        usb_read_mode="auto",
    )
    started = time.monotonic()
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(root, trace_dtype="float16"),
        retry=sc.RetryPolicy(max_attempts=max_attempts, backoff_s=0.2),
        warmup=2,
        workload_sync="cuda",
    ) as experiment:
        plan = experiment.resolved.to_dict()
        records = experiment.run(1)
    record = records[0]
    return {
        "run": str(root.resolve()),
        "elapsed_s": time.monotonic() - started,
        "capture_plan": plan,
        "record": {
            "index": record["index"],
            "attempt": record["attempt"],
            "labels": record["labels"],
            "health": record["health"],
        },
        "loss_before_update": workload.loss_before,
        "loss_after_update": workload.loss_after,
    }
