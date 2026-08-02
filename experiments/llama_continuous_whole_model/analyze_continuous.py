#!/usr/bin/env python3
"""Evaluate continuous training versus inference using power samples only."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from .power_features import (
        block_rms,
        histogram_similarity,
        js_similarity,
        standardized,
        welch_psd,
        window_features,
    )
except ImportError:  # pragma: no cover - direct hardware invocation.
    from power_features import (
        block_rms,
        histogram_similarity,
        js_similarity,
        standardized,
        welch_psd,
        window_features,
    )


@dataclass(frozen=True)
class ContinuousTrace:
    process: str
    session_id: str
    index: int
    sample_rate_hz: float
    values: np.ndarray
    health_ok: bool


def load_traces(root: Path, process: str) -> list[ContinuousTrace]:
    process_root = root / process
    if not process_root.exists():
        raise FileNotFoundError(f"Missing process directory: {process_root}")
    traces = []
    for session_root in sorted(path for path in process_root.iterdir() if path.is_dir()):
        for record_path in sorted((session_root / "records").rglob("*.json")):
            record = json.loads(record_path.read_text())
            labels = record["labels"]
            if labels.get("process") != process:
                continue
            descriptor = record["channels"][record["primary_channel"]]
            values = np.load(session_root / descriptor["path"], allow_pickle=False).astype(np.float64)
            traces.append(
                ContinuousTrace(
                    process=process,
                    session_id=str(labels.get("session_id", session_root.name)),
                    index=int(record["index"]),
                    sample_rate_hz=float(descriptor["sample_rate_hz"]),
                    values=values,
                    health_ok=bool(record["health"]["ok"]),
                )
            )
    if not traces:
        raise ValueError(f"No {process} records found below {process_root}")
    return traces


def enhanced_power_features(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Handcrafted attacker using only one contiguous ADC window."""

    base = window_features(values, sample_rate_hz)
    centered = values - np.median(values)
    envelope_features = []
    for block_us in (25, 50, 100, 250, 500, 1000):
        block_samples = max(2, int(round(block_us * 1e-6 * sample_rate_hz)))
        envelope = block_rms(centered, block_samples)
        envelope_features.extend(
            [
                envelope.mean(),
                envelope.std(),
                *np.quantile(envelope, (0.1, 0.5, 0.9, 0.99)),
                envelope.max(),
            ]
        )

    normalized = standardized(values)
    spectrum = np.abs(np.fft.rfft(normalized * np.hanning(len(normalized)))) ** 2
    spectrum /= max(float(spectrum.sum()), np.finfo(np.float64).tiny)
    frequencies = np.fft.rfftfreq(len(normalized), d=1 / sample_rate_hz)
    spectral_entropy = -float(np.sum(spectrum * np.log(spectrum + 1e-30))) / math.log(len(spectrum))
    spectral_centroid = float(np.sum(frequencies * spectrum) / sample_rate_hz)

    autocorrelation_features = []
    for lag_us in (10, 25, 50, 100, 250, 500, 1000, 2000):
        lag = int(round(lag_us * 1e-6 * sample_rate_hz))
        if lag >= len(normalized):
            autocorrelation_features.append(0.0)
        else:
            autocorrelation_features.append(
                float(np.dot(normalized[:-lag], normalized[lag:]) / (len(normalized) - lag))
            )
    return np.concatenate(
        (
            base,
            np.asarray(envelope_features, dtype=np.float64),
            np.asarray([spectral_entropy, spectral_centroid], dtype=np.float64),
            np.asarray(autocorrelation_features, dtype=np.float64),
        )
    )


def trace_windows(trace: ContinuousTrace, horizon_ms: float) -> np.ndarray:
    samples = int(round(horizon_ms * 1e-3 * trace.sample_rate_hz))
    if samples < 2:
        raise ValueError(f"{horizon_ms} ms corresponds to fewer than two samples")
    if samples > len(trace.values):
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(
        [
            enhanced_power_features(
                trace.values[start : start + samples],
                trace.sample_rate_hz,
            )
            for start in range(0, len(trace.values) - samples + 1, samples)
        ]
    )


def balanced_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([np.mean(prediction[truth == label] == label) for label in (-1.0, 1.0)]))


def classification_rates(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return oriented rates for an inference=-1, training=+1 classifier."""

    inference_mask = truth == -1.0
    training_mask = truth == 1.0
    if not inference_mask.any() or not training_mask.any():
        raise ValueError("classification rates require both inference and training labels")
    inference_recall = float(np.mean(prediction[inference_mask] == -1.0))
    training_recall = float(np.mean(prediction[training_mask] == 1.0))
    accuracy = (inference_recall + training_recall) / 2
    return {
        "balanced_accuracy": accuracy,
        "classifier_error_rate": 1.0 - accuracy,
        "inference_recall": inference_recall,
        "training_recall": training_recall,
        "training_as_inference_rate": 1.0 - training_recall,
        "orientation_normalized_balanced_accuracy": max(accuracy, 1.0 - accuracy),
    }


def grouped_ridge_accuracy(
    inference: list[ContinuousTrace],
    training: list[ContinuousTrace],
    *,
    horizon_ms: float,
) -> dict[str, Any] | None:
    inference_sessions = sorted({trace.session_id for trace in inference})
    training_sessions = sorted({trace.session_id for trace in training})
    if len(inference_sessions) < 2 or len(training_sessions) < 2:
        return None
    feature_rows = {
        (trace.process, trace.session_id, trace.index): trace_windows(trace, horizon_ms)
        for trace in [*inference, *training]
    }
    if any(not len(features) for features in feature_rows.values()):
        return None

    folds = []
    for held_inference in inference_sessions:
        for held_training in training_sessions:
            train_x = []
            train_y = []
            test_x = []
            test_y = []
            for trace in [*inference, *training]:
                features = feature_rows[(trace.process, trace.session_id, trace.index)]
                label = -1.0 if trace.process == "inference" else 1.0
                held_out = (trace.process == "inference" and trace.session_id == held_inference) or (
                    trace.process == "training" and trace.session_id == held_training
                )
                target_x = test_x if held_out else train_x
                target_y = test_y if held_out else train_y
                target_x.append(features)
                target_y.append(np.full(len(features), label, dtype=np.float64))
            train_x_array = np.concatenate(train_x)
            train_y_array = np.concatenate(train_y)
            test_x_array = np.concatenate(test_x)
            test_y_array = np.concatenate(test_y)
            mean = train_x_array.mean(axis=0)
            scale = train_x_array.std(axis=0)
            scale[scale < 1e-12] = 1.0
            train_standardized = (train_x_array - mean) / scale
            test_standardized = (test_x_array - mean) / scale
            design = np.column_stack((np.ones(len(train_standardized)), train_standardized))
            test_design = np.column_stack((np.ones(len(test_standardized)), test_standardized))
            regularizer = np.eye(design.shape[1])
            regularizer[0, 0] = 0
            weights = np.linalg.solve(
                design.T @ design + regularizer,
                design.T @ train_y_array,
            )
            prediction = np.where(test_design @ weights >= 0, 1.0, -1.0)
            folds.append(
                {
                    "held_out_inference_session": held_inference,
                    "held_out_training_session": held_training,
                    **classification_rates(prediction, test_y_array),
                    "test_windows": len(test_y_array),
                }
            )
    accuracies = np.asarray([fold["balanced_accuracy"] for fold in folds])
    aggregate_rates = {
        key: float(np.mean([fold[key] for fold in folds]))
        for key in (
            "classifier_error_rate",
            "inference_recall",
            "training_recall",
            "training_as_inference_rate",
            "orientation_normalized_balanced_accuracy",
        )
    }
    return {
        "balanced_accuracy": float(accuracies.mean()),
        **aggregate_rates,
        "fold_standard_deviation": float(accuracies.std(ddof=1)),
        "minimum_fold_accuracy": float(accuracies.min()),
        "maximum_fold_accuracy": float(accuracies.max()),
        "folds": folds,
        "split": "leave one complete inference session and one complete training session out",
        "attacker_input": "power samples only",
    }


def stationary_similarities(
    inference: list[ContinuousTrace], training: list[ContinuousTrace]
) -> dict[str, float]:
    inference_values = np.concatenate([trace.values for trace in inference])
    training_values = np.concatenate([trace.values for trace in training])
    inference_normalized = np.concatenate([standardized(trace.values) for trace in inference])
    training_normalized = np.concatenate([standardized(trace.values) for trace in training])
    inference_psd = np.mean([welch_psd(trace.values) for trace in inference], axis=0)
    training_psd = np.mean([welch_psd(trace.values) for trace in training], axis=0)
    metrics = {
        "raw_amplitude_js_similarity": histogram_similarity(inference_values, training_values),
        "normalized_amplitude_js_similarity": histogram_similarity(inference_normalized, training_normalized),
        "welch_psd_js_similarity": js_similarity(inference_psd, training_psd),
    }
    metrics["mean_similarity"] = float(np.mean(list(metrics.values())))
    return metrics


def render_overview(
    output: Path,
    inference: list[ContinuousTrace],
    training: list[ContinuousTrace],
    similarities: dict[str, float],
    horizon_results: dict[str, Any],
) -> None:
    sample_rate_hz = inference[0].sample_rate_hz
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    snippet_samples = int(round(0.005 * sample_rate_hz))
    time_ms = np.arange(snippet_samples) / sample_rate_hz * 1e3
    axes[0, 0].plot(
        time_ms,
        inference[0].values[:snippet_samples],
        color="#155EEF",
        lw=0.7,
        label="BF16 inference",
    )
    axes[0, 0].plot(
        time_ms,
        training[0].values[:snippet_samples],
        color="#D92D20",
        lw=0.7,
        label="Full-parameter training",
        alpha=0.85,
    )
    axes[0, 0].set(title="Unaligned raw 5 ms windows", xlabel="Time (ms)", ylabel="ADC")
    axes[0, 0].legend(frameon=False)

    block = max(2, int(round(100e-6 * sample_rate_hz)))
    for traces, label, color in (
        (inference, "Inference", "#155EEF"),
        (training, "Training", "#D92D20"),
    ):
        envelopes = np.stack([block_rms(trace.values, block) for trace in traces])
        x = (np.arange(envelopes.shape[1]) + 0.5) * block / sample_rate_hz * 1e3
        axes[0, 1].plot(x, envelopes.mean(axis=0), color=color, label=label)
        axes[0, 1].fill_between(
            x,
            envelopes.mean(axis=0) - envelopes.std(axis=0),
            envelopes.mean(axis=0) + envelopes.std(axis=0),
            color=color,
            alpha=0.15,
        )
    axes[0, 1].set(title="100 µs RMS envelope", xlabel="Time (ms)", ylabel="RMS ADC")
    axes[0, 1].legend(frameon=False)

    for traces, label, color in (
        (inference, "Inference", "#155EEF"),
        (training, "Training", "#D92D20"),
    ):
        psd = np.mean([welch_psd(trace.values) for trace in traces], axis=0)
        frequencies = np.fft.rfftfreq(min(8192, len(traces[0].values)), d=1 / sample_rate_hz)
        axes[1, 0].semilogx(
            frequencies[1:] / 1e3,
            10 * np.log10(psd[1:] + 1e-30),
            color=color,
            label=label,
        )
    axes[1, 0].set(title="Stationary Welch spectrum", xlabel="Frequency (kHz)", ylabel="Power (dB)")
    axes[1, 0].legend(frameon=False)

    available = [
        (float(horizon), result["balanced_accuracy"])
        for horizon, result in horizon_results.items()
        if result is not None
    ]
    if available:
        axes[1, 1].axhline(0.5, color="#98A2B3", ls="--", lw=1)
        axes[1, 1].plot(
            [item[0] for item in available],
            [item[1] for item in available],
            marker="o",
            color="#7F56D9",
        )
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_ylim(0.45, 1.02)
        axes[1, 1].set(title="Session-held-out detector", xlabel="Observation (ms)", ylabel="Accuracy")
    else:
        axes[1, 1].text(0.5, 0.55, "Need ≥2 sessions per process", ha="center")
        axes[1, 1].text(
            0.5,
            0.42,
            f"Current stationary similarity: {similarities['mean_similarity']:.3f}",
            ha="center",
        )
        axes[1, 1].set_axis_off()

    for axis in axes.flat:
        axis.grid(True, color="#E4E7EC", lw=0.6, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Continuous whole-model power-only comparison", fontsize=14)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def analyze(root: Path, horizons_ms: list[float]) -> dict[str, Any]:
    inference = load_traces(root, "inference")
    training = load_traces(root, "training")
    rates = {trace.sample_rate_hz for trace in [*inference, *training]}
    lengths = {len(trace.values) for trace in [*inference, *training]}
    if len(rates) != 1 or len(lengths) != 1:
        raise ValueError(f"All traces must share rate and length; got rates={rates}, lengths={lengths}")
    similarities = stationary_similarities(inference, training)
    horizon_results = {
        str(horizon): grouped_ridge_accuracy(
            inference,
            training,
            horizon_ms=horizon,
        )
        for horizon in horizons_ms
    }
    result = {
        "root": str(root),
        "sample_rate_hz": rates.pop(),
        "trace_samples": lengths.pop(),
        "trace_counts": {"inference": len(inference), "training": len(training)},
        "session_counts": {
            "inference": len({trace.session_id for trace in inference}),
            "training": len({trace.session_id for trace in training}),
        },
        "all_health_checks_passed": all(trace.health_ok for trace in [*inference, *training]),
        "attacker_observable": "ADC power samples only",
        "similarities": similarities,
        "detector_by_horizon_ms": horizon_results,
        "per_trace": [
            {
                "process": trace.process,
                "session_id": trace.session_id,
                "index": trace.index,
                "mean": float(trace.values.mean()),
                "std": float(trace.values.std()),
                "minimum": float(trace.values.min()),
                "maximum": float(trace.values.max()),
            }
            for trace in [*inference, *training]
        ],
    }
    (root / "continuous_power_metrics.json").write_text(json.dumps(result, indent=2))
    render_overview(
        root / "continuous_power_overview.png",
        inference,
        training,
        similarities,
        horizon_results,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizons-ms", default="5,10,20,50,100")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [float(value) for value in args.horizons_ms.split(",")]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("Every observation horizon must be positive")
    print(json.dumps(analyze(args.root, horizons), indent=2))


if __name__ == "__main__":
    main()
