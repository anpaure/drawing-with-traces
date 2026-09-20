"""Deterministic error-diffusion scheduling for equal native GPU service.

One training batch and one inference batch are completed per mixed cycle.  A
training batch is materially more expensive than an equal-sized inference
batch, so equal row counts are not equal service.  Calibration expresses each
inference candidate in units of one native training batch.  The controller
then alternates the two candidates bracketing 1.0 and keeps cumulative service
debt bounded, exactly like error diffusion in a rasterizer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidateTiming:
    """Native timing for one inference batch, normalized to one training batch."""

    batch_size: int
    native_seconds: float
    ratio_to_training: float
    paired_samples: int

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("candidate batch_size must be positive")
        if not math.isfinite(self.native_seconds) or self.native_seconds <= 0:
            raise ValueError("candidate native_seconds must be finite and positive")
        if not math.isfinite(self.ratio_to_training) or self.ratio_to_training <= 0:
            raise ValueError("candidate ratio_to_training must be finite and positive")
        if self.paired_samples < 1:
            raise ValueError("candidate paired_samples must be positive")


@dataclass(frozen=True)
class CalibrationResult:
    """A measured bracket around one native training-service unit."""

    training_batch_size: int
    training_native_seconds: float
    candidates: tuple[CandidateTiming, ...]
    lower_batch_size: int
    upper_batch_size: int

    def __post_init__(self) -> None:
        if self.training_batch_size < 1:
            raise ValueError("training_batch_size must be positive")
        if not math.isfinite(self.training_native_seconds) or self.training_native_seconds <= 0:
            raise ValueError("training_native_seconds must be finite and positive")
        if not self.candidates:
            raise ValueError("calibration requires at least one inference candidate")
        batches = [candidate.batch_size for candidate in self.candidates]
        if len(batches) != len(set(batches)):
            raise ValueError("calibration candidate batch sizes must be unique")
        if self.lower_batch_size not in batches or self.upper_batch_size not in batches:
            raise ValueError("calibration bracket must reference measured candidates")
        if self.lower.ratio_to_training > 1.0 + 1e-12:
            raise ValueError("lower calibration candidate must not exceed training service")
        if self.upper.ratio_to_training < 1.0 - 1e-12:
            raise ValueError("upper calibration candidate must not be below training service")

    @property
    def lower(self) -> CandidateTiming:
        return self.candidate(self.lower_batch_size)

    @property
    def upper(self) -> CandidateTiming:
        return self.candidate(self.upper_batch_size)

    def candidate(self, batch_size: int) -> CandidateTiming:
        for candidate in self.candidates:
            if candidate.batch_size == batch_size:
                return candidate
        raise KeyError(f"inference batch {batch_size} was not calibrated")

    @property
    def upper_cycle_fraction(self) -> float:
        """Ideal asymptotic fraction of cycles assigned to the upper candidate."""

        lower = self.lower.ratio_to_training
        upper = self.upper.ratio_to_training
        if abs(upper - lower) <= 1e-15:
            return 0.0
        return min(1.0, max(0.0, (1.0 - lower) / (upper - lower)))

    @property
    def maximum_single_cycle_service_error(self) -> float:
        return max(1.0 - self.lower.ratio_to_training, self.upper.ratio_to_training - 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CalibrationResult":
        return cls(
            training_batch_size=int(payload["training_batch_size"]),
            training_native_seconds=float(payload["training_native_seconds"]),
            candidates=tuple(CandidateTiming(**candidate) for candidate in payload["candidates"]),
            lower_batch_size=int(payload["lower_batch_size"]),
            upper_batch_size=int(payload["upper_batch_size"]),
        )


def build_calibration(
    *,
    training_batch_size: int,
    paired_seconds: Mapping[int, Sequence[tuple[float, float]]],
) -> CalibrationResult:
    """Build a drift-resistant calibration from paired inference/training timings.

    Each tuple is ``(inference_seconds, training_seconds)`` measured next to one
    another.  The median paired ratio rejects slow thermal drift better than a
    ratio between two long, separately measured phases.
    """

    if training_batch_size < 1:
        raise ValueError("training_batch_size must be positive")
    if len(paired_seconds) < 2:
        raise ValueError("calibration needs at least two inference batch candidates")
    all_training_seconds: list[float] = []
    ratios: dict[int, float] = {}
    sample_counts: dict[int, int] = {}
    for batch_size, samples in paired_seconds.items():
        if batch_size < 1:
            raise ValueError("inference candidate batch sizes must be positive")
        if not samples:
            raise ValueError(f"inference candidate {batch_size} has no timing samples")
        candidate_ratios = []
        for inference_seconds, training_seconds in samples:
            if (
                not math.isfinite(inference_seconds)
                or inference_seconds <= 0
                or not math.isfinite(training_seconds)
                or training_seconds <= 0
            ):
                raise ValueError("all calibration timings must be finite and positive")
            candidate_ratios.append(inference_seconds / training_seconds)
            all_training_seconds.append(training_seconds)
        ratios[int(batch_size)] = statistics.median(candidate_ratios)
        sample_counts[int(batch_size)] = len(samples)

    lower_batches = [batch for batch, ratio in ratios.items() if ratio <= 1.0]
    upper_batches = [batch for batch, ratio in ratios.items() if ratio >= 1.0]
    if not lower_batches or not upper_batches:
        observed = ", ".join(f"{batch}:{ratios[batch]:.4f}" for batch in sorted(ratios))
        raise RuntimeError(
            "inference calibration did not bracket one training-service unit; "
            f"measured inference/training ratios were [{observed}]. Expand the candidate range."
        )
    lower_batch = max(lower_batches, key=lambda batch: ratios[batch])
    upper_batch = min(upper_batches, key=lambda batch: ratios[batch])
    training_native_seconds = statistics.median(all_training_seconds)
    candidates = tuple(
        CandidateTiming(
            batch_size=batch,
            native_seconds=ratios[batch] * training_native_seconds,
            ratio_to_training=ratios[batch],
            paired_samples=sample_counts[batch],
        )
        for batch in sorted(ratios)
    )
    return CalibrationResult(
        training_batch_size=training_batch_size,
        training_native_seconds=training_native_seconds,
        candidates=candidates,
        lower_batch_size=lower_batch,
        upper_batch_size=upper_batch,
    )


def rescale_inference_service(
    calibration: CalibrationResult,
    factor: float,
) -> CalibrationResult:
    """Apply schedule-level feedback to all inferred inference service costs."""

    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("inference service scale factor must be finite and positive")
    candidates = tuple(
        CandidateTiming(
            batch_size=candidate.batch_size,
            native_seconds=candidate.native_seconds * factor,
            ratio_to_training=candidate.ratio_to_training * factor,
            paired_samples=candidate.paired_samples,
        )
        for candidate in calibration.candidates
    )
    lower = [candidate for candidate in candidates if candidate.ratio_to_training <= 1.0]
    upper = [candidate for candidate in candidates if candidate.ratio_to_training >= 1.0]
    if not lower or not upper:
        observed = ", ".join(
            f"{candidate.batch_size}:{candidate.ratio_to_training:.4f}"
            for candidate in candidates
        )
        raise RuntimeError(
            "schedule-level correction moved the 1:1 crossing outside the calibrated range; "
            f"corrected inference/training ratios were [{observed}]"
        )
    return CalibrationResult(
        training_batch_size=calibration.training_batch_size,
        training_native_seconds=calibration.training_native_seconds,
        candidates=candidates,
        lower_batch_size=max(lower, key=lambda candidate: candidate.ratio_to_training).batch_size,
        upper_batch_size=min(upper, key=lambda candidate: candidate.ratio_to_training).batch_size,
    )


def set_training_native_seconds(
    calibration: CalibrationResult,
    training_native_seconds: float,
) -> CalibrationResult:
    """Replace the absolute time scale while preserving calibrated service ratios."""

    if not math.isfinite(training_native_seconds) or training_native_seconds <= 0:
        raise ValueError("training native seconds must be finite and positive")
    return CalibrationResult(
        training_batch_size=calibration.training_batch_size,
        training_native_seconds=training_native_seconds,
        candidates=tuple(
            CandidateTiming(
                batch_size=candidate.batch_size,
                native_seconds=candidate.ratio_to_training * training_native_seconds,
                ratio_to_training=candidate.ratio_to_training,
                paired_samples=candidate.paired_samples,
            )
            for candidate in calibration.candidates
        ),
        lower_batch_size=calibration.lower_batch_size,
        upper_batch_size=calibration.upper_batch_size,
    )


class BalancedServiceController:
    """Keep inference and training native-equivalent service at a 1:1 ratio."""

    def __init__(self, calibration: CalibrationResult) -> None:
        self.calibration = calibration
        self.inference_service_units = 0.0
        self.training_service_units = 0.0
        self.cycles = 0
        self.lower_cycles = 0
        self.upper_cycles = 0

    @property
    def service_debt(self) -> float:
        """Inference service minus training service, in training-batch units."""

        return self.inference_service_units - self.training_service_units

    @property
    def service_ratio(self) -> float:
        if self.training_service_units == 0:
            return 1.0
        return self.inference_service_units / self.training_service_units

    @property
    def relative_imbalance(self) -> float:
        denominator = max(
            (self.inference_service_units + self.training_service_units) / 2.0,
            1e-15,
        )
        return abs(self.service_debt) / denominator

    def choose(self) -> CandidateTiming:
        """Choose the candidate minimizing absolute debt after the next full cycle."""

        projected_training = self.training_service_units + 1.0
        choices = {self.calibration.lower.batch_size: self.calibration.lower}
        choices[self.calibration.upper.batch_size] = self.calibration.upper
        return min(
            choices.values(),
            key=lambda candidate: (
                abs(self.inference_service_units + candidate.ratio_to_training - projected_training),
                candidate.batch_size,
            ),
        )

    def commit(self, candidate: CandidateTiming | int) -> None:
        if isinstance(candidate, int):
            candidate = self.calibration.candidate(candidate)
        if candidate.batch_size not in {
            self.calibration.lower_batch_size,
            self.calibration.upper_batch_size,
        }:
            raise ValueError("controller can commit only a calibrated bracket candidate")
        self.inference_service_units += candidate.ratio_to_training
        self.training_service_units += 1.0
        self.cycles += 1
        if candidate.batch_size == self.calibration.lower_batch_size:
            self.lower_cycles += 1
        if candidate.batch_size == self.calibration.upper_batch_size:
            self.upper_cycles += 1

    def next_batch_size(self) -> int:
        candidate = self.choose()
        self.commit(candidate)
        return candidate.batch_size

    def plan(self, cycles: int) -> list[int]:
        if cycles < 1:
            raise ValueError("cycles must be positive")
        return [self.next_batch_size() for _ in range(cycles)]

    def state_dict(self) -> dict:
        return {
            "cycles": self.cycles,
            "lower_cycles": self.lower_cycles,
            "upper_cycles": self.upper_cycles,
            "inference_service_units": self.inference_service_units,
            "training_service_units": self.training_service_units,
            "service_debt": self.service_debt,
            "service_ratio": self.service_ratio,
            "relative_imbalance": self.relative_imbalance,
        }


def summarize_durations(
    inference_seconds: Iterable[float],
    training_seconds: Iterable[float],
    mixed_seconds: Iterable[float],
) -> dict:
    """Aggregate direct service and slowdown measurements without cherry-picking."""

    inference = tuple(float(value) for value in inference_seconds)
    training = tuple(float(value) for value in training_seconds)
    mixed = tuple(float(value) for value in mixed_seconds)
    if not inference or len(inference) != len(training) or len(inference) != len(mixed):
        raise ValueError("duration groups must be non-empty and have equal lengths")
    if any(not math.isfinite(value) or value <= 0 for value in inference + training + mixed):
        raise ValueError("all durations must be finite and positive")
    inference_total = sum(inference)
    training_total = sum(training)
    mixed_total = sum(mixed)
    slowdowns = {
        "inference": mixed_total / inference_total,
        "training": mixed_total / training_total,
    }
    per_round_slowdowns = {
        "inference": [mixed_value / inference_value for mixed_value, inference_value in zip(mixed, inference)],
        "training": [mixed_value / training_value for mixed_value, training_value in zip(mixed, training)],
    }
    service_ratio = inference_total / training_total
    return {
        "rounds": len(inference),
        "total_seconds": {
            "inference": inference_total,
            "training": training_total,
            "mixed": mixed_total,
        },
        "median_seconds": {
            "inference": statistics.median(inference),
            "training": statistics.median(training),
            "mixed": statistics.median(mixed),
        },
        "native_service_ratio": service_ratio,
        "native_service_relative_imbalance": abs(service_ratio - 1.0),
        "slowdown": slowdowns,
        "max_slowdown": max(slowdowns.values()),
        "per_round_slowdown": per_round_slowdowns,
        "mean_per_round_slowdown": {
            service: statistics.mean(values)
            for service, values in per_round_slowdowns.items()
        },
        "stdev_per_round_slowdown": {
            service: statistics.stdev(values) if len(values) > 1 else 0.0
            for service, values in per_round_slowdowns.items()
        },
    }
