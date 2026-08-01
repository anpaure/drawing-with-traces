#!/usr/bin/env python3
"""Analyze continuous inference-vs-training SideCapture windows."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TraceRecord:
    process: str
    index: int
    phase: str
    phase_offset: int | None
    sample_rate_hz: float
    values: np.ndarray
    metadata: dict[str, Any]


def load_records(root: Path, process: str) -> list[TraceRecord]:
    process_root = root / process
    if not (process_root / "records").exists() and (root / "mixed" / "records").exists():
        process_root = root / "mixed"
    records = []
    for record_path in sorted((process_root / "records").rglob("*.json")):
        metadata = json.loads(record_path.read_text())
        labels = metadata["labels"]
        if labels.get("process") != process:
            continue
        descriptor = metadata["channels"][metadata["primary_channel"]]
        values = np.load(process_root / descriptor["path"], allow_pickle=False).astype(np.float64)
        phase_offset = labels.get("cycle_phase_offset")
        records.append(
            TraceRecord(
                process=process,
                index=int(metadata["index"]),
                phase=str(
                    labels.get("cycle_first", labels.get("phase_first", process))
                ),
                phase_offset=None if phase_offset is None else int(phase_offset),
                sample_rate_hz=float(descriptor["sample_rate_hz"]),
                values=values,
                metadata=metadata,
            )
        )
    if not records:
        raise ValueError(f"No SideCapture records found under {process_root}")
    return records


def js_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / left.sum()
    right = right / right.sum()
    midpoint = (left + right) / 2
    divergence = 0.0
    for distribution in (left, right):
        mask = distribution > 0
        divergence += 0.5 * float(
            np.sum(distribution[mask] * np.log2(distribution[mask] / midpoint[mask]))
        )
    return float(np.clip(1.0 - divergence, 0.0, 1.0))


def histogram_similarity(left: np.ndarray, right: np.ndarray, bins: int = 256) -> float:
    low = float(min(left.min(), right.min()))
    high = float(max(left.max(), right.max()))
    if high == low:
        return 1.0
    left_histogram, _ = np.histogram(left, bins=bins, range=(low, high))
    right_histogram, _ = np.histogram(right, bins=bins, range=(low, high))
    return js_similarity(left_histogram + 1e-12, right_histogram + 1e-12)


def standardized(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = np.std(centered)
    return centered / max(float(scale), np.finfo(np.float64).tiny)


def welch_psd(values: np.ndarray, segment_samples: int = 8192) -> np.ndarray:
    segment_samples = min(segment_samples, len(values))
    step = max(1, segment_samples // 2)
    window = np.hanning(segment_samples)
    spectra = []
    for start in range(0, len(values) - segment_samples + 1, step):
        segment = values[start : start + segment_samples]
        segment = segment - segment.mean()
        spectra.append(np.abs(np.fft.rfft(segment * window)) ** 2)
    if not spectra:
        raise ValueError("Trace is shorter than one Welch segment")
    result = np.mean(spectra, axis=0)
    result[0] = 0
    return result / max(float(result.sum()), np.finfo(np.float64).tiny)


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    normalized = standardized(values)
    fft_length = 1 << (2 * len(normalized) - 1).bit_length()
    spectrum = np.fft.rfft(normalized, n=fft_length)
    correlation = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_length)[:max_lag]
    overlap = np.arange(len(normalized), len(normalized) - max_lag, -1)
    correlation = correlation / overlap
    return correlation / max(float(correlation[0]), np.finfo(np.float64).tiny)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def block_rms(values: np.ndarray, block_samples: int = 150) -> np.ndarray:
    usable = len(values) // block_samples * block_samples
    blocks = values[:usable].reshape(-1, block_samples)
    blocks = blocks - np.median(blocks, axis=1, keepdims=True)
    return np.sqrt(np.mean(blocks**2, axis=1))


def window_features(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    centered = values - np.median(values)
    standard_deviation = max(float(centered.std()), np.finfo(np.float64).tiny)
    normalized = centered / standard_deviation
    quantiles = np.quantile(centered, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    difference = np.diff(centered)
    basic = np.asarray(
        [
            values.mean(),
            values.std(),
            np.sqrt(np.mean(values**2)),
            np.median(np.abs(centered)),
            *quantiles,
            centered.min(),
            centered.max(),
            difference.std(),
            np.mean(np.signbit(normalized[1:]) != np.signbit(normalized[:-1])),
            np.mean(normalized**3),
            np.mean(normalized**4),
        ],
        dtype=np.float64,
    )

    power = np.abs(np.fft.rfft(normalized * np.hanning(len(normalized)))) ** 2
    frequencies = np.fft.rfftfreq(len(normalized), d=1 / sample_rate_hz)
    edges = np.asarray([0, 2e3, 5e3, 10e3, 20e3, 40e3, 80e3, 120e3, 200e3, 350e3, 500e3, 750e3])
    bands = []
    total_power = max(float(power.sum()), np.finfo(np.float64).tiny)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(math.log10(float(power[mask].sum()) / total_power + 1e-15))
    return np.concatenate((basic, np.asarray(bands)))


def trace_windows(record: TraceRecord, window_ms: float = 5.0) -> np.ndarray:
    samples = int(round(window_ms * 1e-3 * record.sample_rate_hz))
    return np.stack(
        [
            window_features(record.values[start : start + samples], record.sample_rate_hz)
            for start in range(0, len(record.values) - samples + 1, samples)
        ]
    )


def held_out_linear_accuracy(
    inference: list[TraceRecord], training: list[TraceRecord]
) -> dict[str, Any]:
    if len(inference) != len(training):
        raise ValueError("Held-out paired evaluation requires equal trace counts")
    if len(inference) < 2:
        raise ValueError("Held-out paired evaluation requires at least two traces per process")
    inference_features = [trace_windows(record) for record in inference]
    training_features = [trace_windows(record) for record in training]
    fold_rows = []
    for held_out in range(len(inference)):
        train_x = np.concatenate(
            [
                features
                for index, pair in enumerate(zip(inference_features, training_features))
                if index != held_out
                for features in pair
            ]
        )
        train_y = np.concatenate(
            [
                np.full(len(features), label, dtype=np.float64)
                for index, pair in enumerate(zip(inference_features, training_features))
                if index != held_out
                for features, label in zip(pair, (-1.0, 1.0))
            ]
        )
        test_x = np.concatenate((inference_features[held_out], training_features[held_out]))
        test_y = np.concatenate(
            (
                np.full(len(inference_features[held_out]), -1.0),
                np.full(len(training_features[held_out]), 1.0),
            )
        )

        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        standardized_train = (train_x - mean) / scale
        standardized_test = (test_x - mean) / scale
        design = np.column_stack((np.ones(len(standardized_train)), standardized_train))
        test_design = np.column_stack((np.ones(len(standardized_test)), standardized_test))
        regularizer = np.eye(design.shape[1]) * 1.0
        regularizer[0, 0] = 0
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
        prediction = np.where(test_design @ weights >= 0, 1.0, -1.0)
        accuracy = float(np.mean(prediction == test_y))
        fold_rows.append(
            {
                "held_out_pair": held_out,
                "training_cycle_first": training[held_out].phase,
                "training_phase_offset": training[held_out].phase_offset,
                "accuracy": accuracy,
            }
        )
    accuracy = float(np.mean([row["accuracy"] for row in fold_rows]))
    fold_standard_deviation = float(np.std([row["accuracy"] for row in fold_rows], ddof=1))
    standard_error = fold_standard_deviation / math.sqrt(len(fold_rows))
    offset_accuracy = {
        str(offset): float(
            np.mean(
                [
                    row["accuracy"]
                    for row in fold_rows
                    if row["training_phase_offset"] == offset
                ]
            )
        )
        for offset in sorted(
            {
                row["training_phase_offset"]
                for row in fold_rows
                if row["training_phase_offset"] is not None
            }
        )
    }
    return {
        "accuracy": accuracy,
        "chance_accuracy": 0.5,
        "indistinguishability": float(np.clip(1 - 2 * abs(accuracy - 0.5), 0.0, 1.0)),
        "fold_accuracy_standard_deviation": fold_standard_deviation,
        "fold_accuracy_standard_error": standard_error,
        "approximate_95_percent_interval": [
            float(np.clip(accuracy - 1.96 * standard_error, 0.0, 1.0)),
            float(np.clip(accuracy + 1.96 * standard_error, 0.0, 1.0)),
        ],
        "accuracy_by_training_phase_offset": offset_accuracy,
        "folds": fold_rows,
        "window_ms": 5.0,
        "split": "leave one inference trace and one training trace out per fold",
    }


def describe_parameter_scope(record: TraceRecord) -> str:
    scope = record.metadata["labels"].get("parameter_scope", [])
    if not isinstance(scope, list) or not scope:
        return "training"
    projection_names = {str(name).split(".")[-2] for name in scope}
    if len(scope) == 96 and projection_names == {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "all 96 attention-projection weights"
    if projection_names == {"o_proj"}:
        return f"{len(scope)} attention-output projection weight{'s' if len(scope) != 1 else ''}"
    return f"{len(scope)} projection weight{'s' if len(scope) != 1 else ''}"


def describe_experiment(inference: list[TraceRecord], training: list[TraceRecord]) -> str:
    labels = training[0].metadata["labels"]
    model = str(labels.get("model", "model")).split("/")[-1]
    scope = describe_parameter_scope(training[0])
    covers = int(labels.get("cover_decodes_per_training_step", 0))
    variant = str(labels.get("variant", "ordinary"))
    training_kind = "inference-shaped training" if variant != "ordinary" else "ordinary training"
    if covers:
        training_description = f"{training_kind} of {scope} + {covers} cached-decode covers/step"
    else:
        training_description = f"{training_kind} of {scope}"
    inference_task = inference[0].metadata["labels"].get("task", "inference")
    inference_description = "cached decode" if inference_task == "cached_decode" else str(inference_task)
    return f"{model}: {inference_description} vs {training_description}"


def summarize_cuda_ms(records: list[TraceRecord]) -> dict[str, float] | None:
    values = [record.metadata.get("result_summary", {}).get("cuda_ms") for record in records]
    if any(value is None for value in values):
        return None
    timings = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(timings.mean()),
        "median": float(np.median(timings)),
        "minimum": float(timings.min()),
        "maximum": float(timings.max()),
    }


def analyze(root: Path, *, title: str | None = None) -> dict[str, Any]:
    inference = load_records(root, "inference")
    training = load_records(root, "training")
    rates = {record.sample_rate_hz for record in [*inference, *training]}
    lengths = {len(record.values) for record in [*inference, *training]}
    if len(rates) != 1 or len(lengths) != 1:
        raise ValueError(f"All traces must share rate and length; got rates={rates}, lengths={lengths}")
    sample_rate_hz = rates.pop()
    trace_samples = lengths.pop()

    inference_values = np.concatenate([record.values for record in inference])
    training_values = np.concatenate([record.values for record in training])
    normalized_inference = np.concatenate([standardized(record.values) for record in inference])
    normalized_training = np.concatenate([standardized(record.values) for record in training])

    inference_psd = np.mean([welch_psd(record.values) for record in inference], axis=0)
    training_psd = np.mean([welch_psd(record.values) for record in training], axis=0)
    inference_psd /= inference_psd.sum()
    training_psd /= training_psd.sum()
    max_lag = int(round(0.005 * sample_rate_hz))
    inference_acf = np.mean([autocorrelation(record.values, max_lag) for record in inference], axis=0)
    training_acf = np.mean([autocorrelation(record.values, max_lag) for record in training], axis=0)
    inference_rms = np.concatenate([block_rms(record.values) for record in inference])
    training_rms = np.concatenate([block_rms(record.values) for record in training])

    similarities = {
        "raw_amplitude_js_similarity": histogram_similarity(inference_values, training_values),
        "normalized_amplitude_js_similarity": histogram_similarity(
            normalized_inference, normalized_training
        ),
        "welch_psd_js_similarity": js_similarity(inference_psd, training_psd),
        "autocorrelation_cosine_similarity": (cosine_similarity(inference_acf, training_acf) + 1) / 2,
        "block_rms_js_similarity": histogram_similarity(inference_rms, training_rms),
    }
    similarities["mean_signal_similarity"] = float(np.mean(list(similarities.values())))
    classifier = held_out_linear_accuracy(inference, training)

    result = {
        "root": str(root),
        "sample_rate_hz": sample_rate_hz,
        "trace_samples": trace_samples,
        "duration_ms": trace_samples / sample_rate_hz * 1e3,
        "trace_counts": {"inference": len(inference), "training": len(training)},
        "cuda_ms": {
            "inference": summarize_cuda_ms(inference),
            "training": summarize_cuda_ms(training),
        },
        "training_phase_counts": dict(
            sorted(
                {
                    phase: sum(record.phase == phase for record in training)
                    for phase in {record.phase for record in training}
                }.items()
            )
        ),
        "training_phase_offset_counts": {
            str(offset): sum(record.phase_offset == offset for record in training)
            for offset in sorted(
                {record.phase_offset for record in training if record.phase_offset is not None}
            )
        },
        "all_health_checks_passed": all(
            record.metadata["health"]["ok"] for record in [*inference, *training]
        ),
        "similarities": similarities,
        "classifier": classifier,
        "per_trace": [
            {
                "process": record.process,
                "index": record.index,
                "phase": record.phase,
                "phase_offset": record.phase_offset,
                "mean": float(record.values.mean()),
                "std": float(record.values.std()),
                "minimum": float(record.values.min()),
                "maximum": float(record.values.max()),
            }
            for record in [*inference, *training]
        ],
    }

    frequency_hz = np.fft.rfftfreq(min(8192, trace_samples), d=1 / sample_rate_hz)
    time_ms = np.arange(trace_samples) / sample_rate_hz * 1e3
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    snippet_samples = int(round(0.002 * sample_rate_hz))
    axes[0, 0].plot(
        time_ms[:snippet_samples], inference[0].values[:snippet_samples], label="Inference", lw=0.7
    )
    representative_training = training[0]
    phase_suffix = (
        ""
        if representative_training.phase_offset is None
        else f", offset {representative_training.phase_offset}"
    )
    axes[0, 0].plot(
        time_ms[:snippet_samples],
        representative_training.values[:snippet_samples],
        label=f"Training ({representative_training.phase}{phase_suffix})",
        lw=0.7,
        alpha=0.82,
        color="#D92D20",
    )
    axes[0, 0].set(title="Raw 2 ms window", xlabel="Time (ms)", ylabel="Normalized ADC")
    axes[0, 0].legend(frameon=False, ncol=2)

    block = 150
    rms_time = (np.arange(trace_samples // block) + 0.5) * block / sample_rate_hz * 1e3
    inference_envelopes = np.stack([block_rms(record.values, block) for record in inference])
    training_envelopes = np.stack([block_rms(record.values, block) for record in training])
    for values, label, color in (
        (inference_envelopes, "Inference", "#155EEF"),
        (training_envelopes, "Training (all phases)", "#D92D20"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axes[0, 1].plot(rms_time, mean, label=label, color=color, lw=1.1)
        axes[0, 1].fill_between(rms_time, mean - std, mean + std, color=color, alpha=0.16)
    axes[0, 1].set(title="100 µs RMS envelope", xlabel="Time (ms)", ylabel="RMS ADC")
    axes[0, 1].legend(frameon=False)

    floor = 1e-20
    axes[1, 0].semilogx(
        frequency_hz[1:] / 1e3,
        10 * np.log10(inference_psd[1:] + floor),
        label="Inference",
        color="#155EEF",
    )
    axes[1, 0].semilogx(
        frequency_hz[1:] / 1e3,
        10 * np.log10(training_psd[1:] + floor),
        label="Training",
        color="#D92D20",
    )
    axes[1, 0].set(title="Normalized Welch spectrum", xlabel="Frequency (kHz)", ylabel="Power (dB)")
    axes[1, 0].legend(frameon=False)

    metric_names = [
        "Amplitude",
        "Norm. amplitude",
        "Spectrum",
        "Autocorrelation",
        "RMS envelope",
        "Linear monitor",
    ]
    metric_values = [
        similarities["raw_amplitude_js_similarity"],
        similarities["normalized_amplitude_js_similarity"],
        similarities["welch_psd_js_similarity"],
        similarities["autocorrelation_cosine_similarity"],
        similarities["block_rms_js_similarity"],
        classifier["indistinguishability"],
    ]
    axes[1, 1].barh(metric_names, metric_values, color=["#155EEF"] * 5 + ["#7F56D9"])
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set(title="Continuous-window similarity (higher is closer)", xlabel="Similarity")
    for index, value in enumerate(metric_values):
        axes[1, 1].text(min(value + 0.015, 0.94), index, f"{value:.3f}", va="center", fontsize=9)

    for axis in axes.flat:
        axis.grid(True, color="#E4E7EC", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title or describe_experiment(inference, training), fontsize=14)
    figure.savefig(root / "continuous_power_comparison.png", dpi=180, facecolor="white")
    plt.close(figure)

    (root / "power_similarity.json").write_text(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--title")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(analyze(arguments.root, title=arguments.title), indent=2))
