"""Dependency-free signal features shared by the Llama power analyses."""

from __future__ import annotations

import math

import numpy as np


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


def block_rms(values: np.ndarray, block_samples: int = 150) -> np.ndarray:
    usable = len(values) // block_samples * block_samples
    if usable == 0:
        raise ValueError(
            f"block_rms requires at least {block_samples} samples, got {len(values)}"
        )
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
    edges = np.asarray(
        [0, 2e3, 5e3, 10e3, 20e3, 40e3, 80e3, 120e3, 200e3, 350e3, 500e3, 750e3]
    )
    bands = []
    total_power = max(float(power.sum()), np.finfo(np.float64).tiny)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(math.log10(float(power[mask].sum()) / total_power + 1e-15))
    return np.concatenate((basic, np.asarray(bands)))
