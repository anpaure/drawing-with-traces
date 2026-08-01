"""Boundary-free similarity metrics for continuous GPU kernel processes."""

from __future__ import annotations

import collections
import math
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np


Event = dict[str, Any]


def _categorical_distribution(
    events: Sequence[Event], key: Callable[[Event], Hashable], weight: Callable[[Event], float]
) -> dict[Hashable, float]:
    counts: dict[Hashable, float] = collections.defaultdict(float)
    for event in events:
        counts[key(event)] += max(0.0, float(weight(event)))
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: value / total for name, value in counts.items()}


def _js_similarity(left: dict[Hashable, float], right: dict[Hashable, float]) -> float:
    """Return one minus Jensen-Shannon divergence, in the closed range [0, 1]."""

    keys = left.keys() | right.keys()
    if not keys:
        return 1.0
    divergence = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        midpoint = (p + q) / 2
        if p:
            divergence += 0.5 * p * math.log2(p / midpoint)
        if q:
            divergence += 0.5 * q * math.log2(q / midpoint)
    return max(0.0, min(1.0, 1.0 - divergence))


def _cyclic_ngrams(values: Sequence[Hashable], size: int) -> list[tuple[Hashable, ...]]:
    if size < 1:
        raise ValueError(f"ngram size must be >= 1, got {size}")
    if not values:
        return []
    return [tuple(values[(start + offset) % len(values)] for offset in range(size)) for start in range(len(values))]


def _numeric_histogram(values: Sequence[float], upper_log: float, bins: int = 64) -> dict[int, float]:
    if not values:
        return {}
    transformed = np.log1p(np.maximum(0.0, np.asarray(values, dtype=np.float64)))
    histogram, _ = np.histogram(transformed, bins=bins, range=(0.0, max(upper_log, 1e-12)))
    total = int(histogram.sum())
    if total == 0:
        return {}
    return {index: float(count) / total for index, count in enumerate(histogram) if count}


def _shared_histogram_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    maximum = max([0.0, *left, *right])
    upper_log = math.log1p(maximum)
    return _js_similarity(
        _numeric_histogram(left, upper_log),
        _numeric_histogram(right, upper_log),
    )


def _launch_gaps(events: Sequence[Event]) -> list[float]:
    timed = [event for event in events if event.get("start_us") is not None]
    timed.sort(key=lambda event: float(event["start_us"]))
    if not timed:
        return []
    gaps = []
    for current, following in zip(timed, timed[1:]):
        current_end = float(current["start_us"]) + float(current.get("duration_us", 0.0))
        gaps.append(max(0.0, float(following["start_us"]) - current_end))

    # A profile contains one iteration, but the monitored process repeats it.
    # Include one cyclic loop-boundary gap using the observed launch span.  The
    # zero lower bound is conservative because host-side loop delay is omitted.
    if len(timed) > 1:
        gaps.append(0.0)
    return gaps


def continuous_kernel_similarity(left: Sequence[Event], right: Sequence[Event]) -> dict[str, float]:
    """Compare indefinitely repeated kernel streams without aligning iterations.

    All component distributions are normalized, so duplicating either process
    any number of times leaves the score unchanged.  Cyclic n-grams explicitly
    model the transition from the end of one iteration to the next.
    """

    family_duration = _js_similarity(
        _categorical_distribution(left, lambda event: event["family"], lambda event: event["duration_us"]),
        _categorical_distribution(right, lambda event: event["family"], lambda event: event["duration_us"]),
    )
    exact_duration = _js_similarity(
        _categorical_distribution(left, lambda event: event["name"], lambda event: event["duration_us"]),
        _categorical_distribution(right, lambda event: event["name"], lambda event: event["duration_us"]),
    )

    left_families = [event["family"] for event in left]
    right_families = [event["family"] for event in right]
    bigrams = _js_similarity(
        _categorical_distribution(
            [{"value": value} for value in _cyclic_ngrams(left_families, 2)],
            lambda event: event["value"],
            lambda _event: 1.0,
        ),
        _categorical_distribution(
            [{"value": value} for value in _cyclic_ngrams(right_families, 2)],
            lambda event: event["value"],
            lambda _event: 1.0,
        ),
    )
    trigrams = _js_similarity(
        _categorical_distribution(
            [{"value": value} for value in _cyclic_ngrams(left_families, 3)],
            lambda event: event["value"],
            lambda _event: 1.0,
        ),
        _categorical_distribution(
            [{"value": value} for value in _cyclic_ngrams(right_families, 3)],
            lambda event: event["value"],
            lambda _event: 1.0,
        ),
    )
    launch_gaps = _shared_histogram_similarity(_launch_gaps(left), _launch_gaps(right))
    durations = _shared_histogram_similarity(
        [float(event["duration_us"]) for event in left],
        [float(event["duration_us"]) for event in right],
    )

    components = {
        "family_duration_js_similarity": family_duration,
        "exact_kernel_duration_js_similarity": exact_duration,
        "family_bigram_js_similarity": bigrams,
        "family_trigram_js_similarity": trigrams,
        "launch_gap_js_similarity": launch_gaps,
        "kernel_duration_js_similarity": durations,
    }
    weights = {
        "family_duration_js_similarity": 0.30,
        "exact_kernel_duration_js_similarity": 0.15,
        "family_bigram_js_similarity": 0.15,
        "family_trigram_js_similarity": 0.15,
        "launch_gap_js_similarity": 0.15,
        "kernel_duration_js_similarity": 0.10,
    }
    components["continuous_kernel_similarity"] = sum(
        weights[name] * components[name] for name in weights
    )
    return components
