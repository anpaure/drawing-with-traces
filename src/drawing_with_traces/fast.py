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


def interleaved_calibration_commands(
    widths: np.ndarray,
    *,
    repeats: int = 4,
    seed: int = 1729,
) -> np.ndarray:
    """Repeat every width in varied transition contexts, never in blocks."""

    widths = np.unique(np.asarray(widths, dtype=np.int64))
    if widths.ndim != 1 or widths.size < 3:
        raise ValueError("calibration requires at least three unique widths")
    if repeats < 2:
        raise ValueError("calibration repeats must be at least two")
    rng = np.random.default_rng(seed)
    orders = []
    previous = None
    for _ in range(repeats):
        order = rng.permutation(widths)
        if previous is not None and order[0] == previous:
            order = np.roll(order, 1)
        orders.append(order)
        previous = int(order[-1])
    return np.concatenate(orders)


def dense_calibration_widths(
    minimum_width: int,
    maximum_width: int,
    *,
    dense_step: int = 64,
    dense_until: int = 1280,
) -> np.ndarray:
    """Build a dense low/mid-range calibration with sparse high-width anchors."""

    if minimum_width < 32 or minimum_width % 32:
        raise ValueError("minimum calibration width must be a multiple of 32 and at least 32")
    if maximum_width <= minimum_width or maximum_width % 32:
        raise ValueError("maximum calibration width must exceed the minimum and be a multiple of 32")
    if dense_step < 32 or dense_step % 32:
        raise ValueError("dense calibration step must be a multiple of 32 and at least 32")
    dense_stop = min(maximum_width, dense_until)
    dense = np.arange(minimum_width, dense_stop + 1, dense_step, dtype=np.int64)
    anchors = np.asarray([dense_stop, 1536, 2048, 3072, 4096, maximum_width], dtype=np.int64)
    widths = np.unique(np.concatenate([dense, anchors]))
    widths = widths[(widths >= minimum_width) & (widths <= maximum_width)]
    if widths.size < 3:
        raise ValueError("calibration width range must contain at least three levels")
    return widths


def silhouette_partition_commands(
    target: np.ndarray,
    *,
    minimum_width: int = 32,
    maximum_width: int = 1024,
    width_quantum: int = 32,
) -> np.ndarray:
    """Map a normalized silhouette to an exact output-column partition."""

    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)):
        raise ValueError("target must be a finite one-dimensional array")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("target values must be between 0 and 1")
    if width_quantum < 1 or minimum_width < width_quantum:
        raise ValueError("minimum_width must be at least one positive width_quantum")
    if minimum_width % width_quantum or maximum_width % width_quantum:
        raise ValueError("minimum_width and maximum_width must be multiples of width_quantum")
    if maximum_width <= minimum_width:
        raise ValueError("maximum_width must exceed minimum_width")
    continuous = minimum_width + target * (maximum_width - minimum_width)
    commands = np.rint(continuous / width_quantum) * width_quantum
    return np.clip(commands, minimum_width, maximum_width).astype(np.int64)


def project_commands_to_total(
    desired: np.ndarray,
    total_width: int,
    *,
    minimum_width: int = 32,
    maximum_width: int,
    width_quantum: int = 32,
) -> np.ndarray:
    """Find the nearest legal command vector with an exact fixed sum."""

    desired = np.asarray(desired, dtype=np.float64)
    if desired.ndim != 1 or desired.size < 2 or not np.all(np.isfinite(desired)):
        raise ValueError("desired commands must be a finite one-dimensional array")
    if any(value % width_quantum for value in (total_width, minimum_width, maximum_width)):
        raise ValueError("width bounds and total_width must be multiples of width_quantum")
    count = desired.size
    if not count * minimum_width <= total_width <= count * maximum_width:
        raise ValueError(
            "total_width is infeasible for the command count and width bounds: "
            f"{count} * {minimum_width} <= {total_width} <= {count} * {maximum_width}"
        )
    target_units = total_width // width_quantum
    minimum_units = minimum_width // width_quantum
    maximum_units = maximum_width // width_quantum
    desired_units = np.clip(desired / width_quantum, minimum_units, maximum_units)
    units = np.clip(np.rint(desired_units), minimum_units, maximum_units).astype(np.int64)
    while int(units.sum()) < target_units:
        candidates = np.flatnonzero(units < maximum_units)
        marginal_cost = 2 * (units[candidates] - desired_units[candidates]) + 1
        units[candidates[int(np.argmin(marginal_cost))]] += 1
    while int(units.sum()) > target_units:
        candidates = np.flatnonzero(units > minimum_units)
        marginal_cost = -2 * (units[candidates] - desired_units[candidates]) + 1
        units[candidates[int(np.argmin(marginal_cost))]] -= 1
    return units * width_quantum


class TiledLinearTrainingWorkload(Workload):
    """Hand-written exact gradient for ``0.5/B * ||XW-Y||²``.

    Every controlled tile computes both a real forward block ``X @ W[:, j:k]``
    and its real weight-gradient block ``X.T @ residual[:, j:k]``. Repeated
    visits are averaged by output column before the accepted SGD update, so the
    delivered gradient is independent of the tile-width pattern.
    """

    replay_safe = True
    task_label = "chipwhisperer-tiled-linear-training"

    def __init__(
        self,
        commands: np.ndarray,
        *,
        profile_duration_s: float,
        hidden_size: int = 4096,
        output_size: int | None = None,
        batch_size: int = 512,
        learning_rate: float = 0.01,
        validation_batch_size: int = 128,
        lead_s: float = 0.002,
        tail_s: float = 0.002,
        active_lead_width: int = 0,
        seed: int = 1729,
        schedule_mode: str = "timed-repeat",
        tile_repeats: int = 1,
        reference_widths: np.ndarray | None = None,
        reference_repeats: int = 0,
    ):
        output_size = hidden_size if output_size is None else int(output_size)
        commands = np.asarray(commands, dtype=np.int64)
        if commands.ndim != 1 or commands.size < 2:
            raise ValueError("commands must be a one-dimensional array with at least two bins")
        if np.any(commands < 0) or np.any(commands > output_size):
            raise ValueError("tile commands must be between 0 and output_size")
        nonzero = commands[commands > 0]
        if nonzero.size and np.any(nonzero % 32):
            raise ValueError("non-zero tile widths must be multiples of 32")
        if profile_duration_s <= 0 or lead_s < 0 or tail_s < 0:
            raise ValueError("profile duration must be positive and lead/tail cannot be negative")
        if hidden_size < 32 or hidden_size % 32:
            raise ValueError("hidden_size must be a positive multiple of 32")
        if output_size < 32 or output_size % 32:
            raise ValueError("output_size must be a positive multiple of 32")
        if batch_size < 1 or validation_batch_size < 1 or learning_rate <= 0:
            raise ValueError("batch sizes and learning_rate must be positive")
        self.commands = np.ascontiguousarray(commands)
        self.profile_duration_s = float(profile_duration_s)
        self.hidden_size = int(hidden_size)
        self.output_size = output_size
        self.batch_size = int(batch_size)
        self.validation_batch_size = int(validation_batch_size)
        self.learning_rate = float(learning_rate)
        self.lead_s = float(lead_s)
        self.tail_s = float(tail_s)
        if active_lead_width < 0 or active_lead_width > output_size:
            raise ValueError("active_lead_width must be between 0 and output_size")
        if active_lead_width and active_lead_width % 32:
            raise ValueError("active_lead_width must be zero or a multiple of 32")
        self.active_lead_width = int(active_lead_width)
        self.seed = int(seed)
        if schedule_mode not in {"timed-repeat", "single-tile", "exact-once"}:
            raise ValueError("schedule_mode must be timed-repeat, single-tile, or exact-once")
        if schedule_mode == "exact-once" and (
            np.any(commands == 0) or int(commands.sum()) != output_size
        ):
            raise ValueError("exact-once commands must be nonzero and sum exactly to output_size")
        self.schedule_mode = schedule_mode
        if tile_repeats < 1:
            raise ValueError("tile_repeats must be at least one")
        self.tile_repeats = int(tile_repeats)
        if reference_repeats < 0:
            raise ValueError("reference_repeats cannot be negative")
        if reference_widths is None:
            reference_widths = np.asarray([], dtype=np.int64)
        reference_widths = np.unique(np.asarray(reference_widths, dtype=np.int64))
        if reference_repeats == 0:
            reference_widths = np.asarray([], dtype=np.int64)
        if np.any(reference_widths < 32) or np.any(reference_widths > output_size):
            raise ValueError("reference widths must be between 32 and output_size")
        if reference_widths.size and np.any(reference_widths % 32):
            raise ValueError("reference widths must be multiples of 32")
        self.reference_widths = reference_widths
        self.reference_repeats = int(reference_repeats)
        self.torch = None
        self.inputs = None
        self.targets = None
        self.validation_inputs = None
        self.validation_targets = None
        self.weight = None
        self.master_weight = None
        self.gradient_sum = None
        self.column_visits = None
        self.cursor = 0
        self.loss_before = None
        self.loss_after = None
        self.current_performance = None
        self.last_performance = None
        self.accepted_training_steps = 0
        self.gradient_compute_wall_s = None
        self.profile_gradient_compute_wall_s = None
        self.completion_gradient_compute_wall_s = None
        self.baseline_gradient_compute_s = None
        self.executed_tile_width_sum = 0
        self.profile_executed_tile_width_sum = 0
        self.completion_tile_width_sum = 0
        self.last_operation_count = 0
        self.profile_operation_count = 0
        self.completion_operation_count = 0
        self.lead_stabilization_operation_count = 0
        self.lead_stabilization_tile_width_sum = 0
        self.capture_control_wall_s = None
        self.coverage_before_completion = None
        self.coverage_after_completion = None

    @property
    def parameter_count(self) -> int:
        return self.hidden_size * self.output_size

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
        teacher_weight = torch.randn(
            self.hidden_size,
            self.output_size,
            device="cuda",
            dtype=torch.bfloat16,
        ) / math.sqrt(self.hidden_size)
        self.targets = self.inputs @ teacher_weight
        self.validation_inputs = torch.randn(
            self.validation_batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.validation_targets = self.validation_inputs @ teacher_weight
        del teacher_weight
        self.master_weight = torch.randn(
            self.hidden_size,
            self.output_size,
            device="cuda",
            dtype=torch.float32,
        ) / math.sqrt(self.hidden_size)
        self.weight = self.master_weight.to(torch.bfloat16)
        self.gradient_sum = torch.zeros_like(self.master_weight)
        self.column_visits = np.zeros(self.output_size, dtype=np.int64)
        self.current_performance = self.evaluate_performance()
        self.loss_before = self.current_performance["train_half_mse"]
        torch.cuda.synchronize()

    def require_setup(self):
        if self.torch is None or self.weight is None or self.gradient_sum is None:
            raise RuntimeError("TiledLinearTrainingWorkload.setup() has not completed")
        return self.torch

    def half_mse(self, inputs, targets) -> float:
        torch = self.torch
        with torch.no_grad():
            residual = inputs @ self.weight - targets
            loss = 0.5 * residual.float().square().mean()
        torch.cuda.synchronize()
        return float(loss)

    def evaluate_performance(self) -> dict[str, float]:
        train_half_mse = self.half_mse(self.inputs, self.targets)
        validation_half_mse = self.half_mse(self.validation_inputs, self.validation_targets)
        return {
            "train_half_mse": train_half_mse,
            "train_objective": train_half_mse * self.output_size,
            "validation_half_mse": validation_half_mse,
        }

    def performance_before_capture(self) -> dict[str, float]:
        """Return the loss snapshot associated with the gradient being captured."""

        return self.evaluate_performance()

    def full_loss(self) -> float:
        """Backward-compatible normalized training loss."""

        return self.half_mse(self.inputs, self.targets)

    def gradient_tile(
        self,
        start: int,
        width: int,
        *,
        accumulate: bool = True,
        synchronize: bool = True,
    ) -> None:
        torch = self.require_setup()
        end = start + width
        prediction = self.inputs @ self.weight[:, start:end]
        residual = prediction - self.targets[:, start:end]
        gradient = self.inputs.T @ residual
        if accumulate:
            self.gradient_sum[:, start:end].add_(gradient.float() / self.batch_size)
            self.column_visits[start:end] += 1
        if synchronize:
            torch.cuda.synchronize()

    def next_tile(self, requested_width: int) -> None:
        if requested_width == 0:
            return
        width = min(int(requested_width), self.output_size)
        if self.cursor + width > self.output_size:
            self.cursor = 0
        self.gradient_tile(self.cursor, width)
        self.cursor = (self.cursor + width) % self.output_size

    def repeat_partition_tile(self, requested_width: int, *, synchronize: bool = True) -> int:
        """Repeat one block for signal quality, then advance the partition once."""

        width = min(int(requested_width), self.output_size)
        if self.cursor + width > self.output_size:
            self.cursor = 0
        start = self.cursor
        for _ in range(self.tile_repeats):
            self.gradient_tile(start, width, synchronize=synchronize)
        self.cursor = (start + width) % self.output_size
        return self.tile_repeats

    def clear_gradient(self) -> None:
        if self.gradient_sum is not None:
            self.gradient_sum.zero_()
        if self.column_visits is not None:
            self.column_visits.fill(0)
        self.cursor = 0

    def warmup(self, iteration: int) -> None:
        del iteration
        for width in np.unique(self.commands[self.commands > 0]):
            self.gradient_tile(0, int(width), accumulate=False)
        self.clear_gradient()

    def set_schedule(
        self,
        commands: np.ndarray,
        *,
        profile_duration_s: float,
        schedule_mode: str | None = None,
        tile_repeats: int | None = None,
    ) -> None:
        commands = np.asarray(commands, dtype=np.int64)
        if commands.ndim != 1 or commands.size < 2:
            raise ValueError("commands must be a one-dimensional array with at least two bins")
        if np.any(commands < 0) or np.any(commands > self.output_size):
            raise ValueError("tile commands must be between 0 and output_size")
        nonzero = commands[commands > 0]
        if nonzero.size and np.any(nonzero % 32):
            raise ValueError("non-zero tile widths must be multiples of 32")
        if profile_duration_s <= 0:
            raise ValueError("profile_duration_s must be positive")
        schedule_mode = self.schedule_mode if schedule_mode is None else schedule_mode
        if schedule_mode not in {"timed-repeat", "single-tile", "exact-once"}:
            raise ValueError("schedule_mode must be timed-repeat, single-tile, or exact-once")
        if schedule_mode == "exact-once" and (
            np.any(commands == 0) or int(commands.sum()) != self.output_size
        ):
            raise ValueError("exact-once commands must be nonzero and sum exactly to output_size")
        self.commands = np.ascontiguousarray(commands)
        self.profile_duration_s = float(profile_duration_s)
        self.schedule_mode = schedule_mode
        if tile_repeats is not None:
            if tile_repeats < 1:
                raise ValueError("tile_repeats must be at least one")
            self.tile_repeats = int(tile_repeats)
        if self.torch is not None:
            self.warmup(0)

    def run(self, context):
        self.clear_gradient()
        self.current_performance = self.performance_before_capture()
        self.loss_before = self.current_performance["train_half_mse"]
        context.labels.update(
            task=self.task_label,
            profile_duration_s=self.profile_duration_s,
            profile_bins=int(self.commands.size),
            model_parameters=self.parameter_count,
            training_step=self.accepted_training_steps,
            train_half_mse_before=self.current_performance["train_half_mse"],
            validation_half_mse_before=self.current_performance["validation_half_mse"],
            schedule_mode=self.schedule_mode,
        )
        context.add_artifact("tile_width_commands", self.commands.astype(np.int32))
        capture_control_start_ns = time.monotonic_ns()
        lead_deadline_ns = time.monotonic_ns() + int(round(self.lead_s * 1e9))
        self.lead_stabilization_operation_count = 0
        lead_name = "tile.lead_baseline" if self.active_lead_width else "tile.lead_idle"
        with context.region(
            lead_name,
            duration_s=self.lead_s,
            requested_width=self.active_lead_width,
        ):
            if self.active_lead_width:
                while time.monotonic_ns() < lead_deadline_ns:
                    self.gradient_tile(
                        0,
                        self.active_lead_width,
                        accumulate=False,
                    )
                    self.lead_stabilization_operation_count += 1
            else:
                sleep_until(lead_deadline_ns)
        training_compute_start_ns = time.monotonic_ns()
        reference_commands = np.asarray([], dtype=np.int64)
        if self.reference_widths.size:
            if self.schedule_mode not in {"single-tile", "exact-once"}:
                raise RuntimeError("on-trace references require an operation-level schedule")
            reference_commands = interleaved_calibration_commands(
                self.reference_widths,
                repeats=self.reference_repeats,
                seed=self.seed + self.accepted_training_steps,
            )
            with context.cuda_region(
                "tile.reference_profile",
                sync="none",
                references=int(reference_commands.size),
            ):
                for reference_index, width in enumerate(reference_commands):
                    with context.cuda_region(
                        "tile.reference",
                        sync="none",
                        index=reference_index,
                        requested_width=int(width),
                    ):
                        self.gradient_tile(0, int(width), synchronize=False)
            context.add_artifact("reference_width_commands", reference_commands.astype(np.int32))
        bin_ns = int(round(self.bin_duration_s * 1e9))
        operation_counts = np.zeros(self.commands.size, dtype=np.int64)
        overruns = np.zeros(self.commands.size, dtype=np.int64)
        profile_start_ns = time.monotonic_ns()
        with context.region(
            "tile.profile",
            duration_s=self.profile_duration_s,
            bins=int(self.commands.size),
            schedule_mode=self.schedule_mode,
        ):
            for index, width in enumerate(self.commands):
                deadline_ns = profile_start_ns + (index + 1) * bin_ns
                if self.schedule_mode in {"single-tile", "exact-once"}:
                    region = context.cuda_region(
                        "tile.bin",
                        sync="none",
                        index=index,
                        requested_width=int(width),
                    )
                else:
                    region = context.region(
                        "tile.bin",
                        index=index,
                        requested_width=int(width),
                    )
                with region:
                    if self.schedule_mode in {"single-tile", "exact-once"}:
                        if width == 0:
                            sleep_until(deadline_ns)
                        else:
                            operation_counts[index] = self.repeat_partition_tile(
                                int(width),
                                synchronize=False,
                            )
                    elif width == 0:
                        sleep_until(deadline_ns)
                    else:
                        while time.monotonic_ns() < deadline_ns:
                            self.next_tile(int(width))
                            operation_counts[index] += 1
                if self.schedule_mode == "timed-repeat":
                    overruns[index] = max(0, time.monotonic_ns() - deadline_ns)
        if self.schedule_mode in {"single-tile", "exact-once"}:
            self.require_setup().cuda.synchronize()
        profile_end_ns = time.monotonic_ns()
        self.profile_gradient_compute_wall_s = (profile_end_ns - training_compute_start_ns) / 1e9
        self.gradient_compute_wall_s = self.profile_gradient_compute_wall_s
        target_width_sum = int(np.dot(operation_counts, self.commands))
        reference_width_sum = int(reference_commands.sum())
        self.profile_executed_tile_width_sum = target_width_sum + reference_width_sum
        self.lead_stabilization_tile_width_sum = (
            self.lead_stabilization_operation_count * self.active_lead_width
        )
        self.executed_tile_width_sum = (
            self.profile_executed_tile_width_sum + self.lead_stabilization_tile_width_sum
        )
        self.profile_operation_count = int(operation_counts.sum() + reference_commands.size)
        self.last_operation_count = (
            self.profile_operation_count + self.lead_stabilization_operation_count
        )
        self.completion_gradient_compute_wall_s = 0.0
        self.completion_tile_width_sum = 0
        self.completion_operation_count = 0
        self.coverage_after_completion = None
        self.coverage_before_completion = {
            "minimum_visits": int(self.column_visits.min()),
            "maximum_visits": int(self.column_visits.max()),
            "unvisited_columns": int(np.count_nonzero(self.column_visits == 0)),
            "columns_visited_once": int(np.count_nonzero(self.column_visits == 1)),
        }
        expected_visits = np.full(self.output_size, self.tile_repeats, dtype=np.int64)
        for width in reference_commands:
            expected_visits[: int(width)] += 1
        if self.schedule_mode == "exact-once" and not np.array_equal(
            self.column_visits, expected_visits
        ):
            raise RuntimeError(
                "exact partition plus reference schedule produced unexpected gradient visits: "
                f"{self.coverage_before_completion}"
            )
        with context.region("tile.tail_idle", duration_s=self.tail_s):
            sleep_until(time.monotonic_ns() + int(round(self.tail_s * 1e9)))
        self.capture_control_wall_s = (time.monotonic_ns() - capture_control_start_ns) / 1e9
        max_overrun_ns = int(overruns.max(initial=0))
        if self.schedule_mode == "timed-repeat" and max_overrun_ns > max(500_000, bin_ns // 2):
            raise RuntimeError(
                f"a {self.bin_duration_s * 1e3:.3f} ms tile bin overran by "
                f"{max_overrun_ns / 1e6:.3f} ms; rejecting the timing-distorted trace"
            )
        context.add_artifact("tile_operations_per_bin", operation_counts)
        context.add_artifact("gradient_column_visits", self.column_visits.astype(np.int32))
        context.labels.update(
            tiled_gradient_operations=int(operation_counts.sum()),
            max_bin_overrun_us=max_overrun_ns / 1e3,
            gradient_compute_wall_s=self.gradient_compute_wall_s,
            executed_tile_width_sum=self.executed_tile_width_sum,
            gradient_unvisited_columns=self.coverage_before_completion["unvisited_columns"],
            reference_gradient_operations=int(reference_commands.size),
            reference_width_sum=reference_width_sum,
            active_lead_width=self.active_lead_width,
            lead_stabilization_operations=self.lead_stabilization_operation_count,
            loss_before_update=self.loss_before,
        )
        return {
            "tiled_gradient_operations": int(operation_counts.sum()),
            "max_bin_overrun_us": max_overrun_ns / 1e3,
            "loss_before_update": self.loss_before,
            "gradient_compute_wall_s": self.gradient_compute_wall_s,
            "executed_tile_width_sum": self.executed_tile_width_sum,
            "reference_gradient_operations": int(reference_commands.size),
            "reference_width_sum": reference_width_sum,
        }

    def complete_exact_gradient(self) -> None:
        """Fill any unvisited columns and account for all post-profile training work."""

        started_ns = time.monotonic_ns()
        completed_width = 0
        completed_operations = 0
        for start in range(0, self.output_size, 256):
            end = min(self.output_size, start + 256)
            if np.any(self.column_visits[start:end] == 0):
                self.gradient_tile(start, end - start)
                completed_width += end - start
                completed_operations += 1
        self.completion_gradient_compute_wall_s = (time.monotonic_ns() - started_ns) / 1e9
        self.completion_tile_width_sum = completed_width
        self.completion_operation_count = completed_operations
        self.gradient_compute_wall_s += self.completion_gradient_compute_wall_s
        self.executed_tile_width_sum += completed_width
        self.last_operation_count += completed_operations
        self.coverage_after_completion = {
            "minimum_visits": int(self.column_visits.min()),
            "maximum_visits": int(self.column_visits.max()),
            "unvisited_columns": int(np.count_nonzero(self.column_visits == 0)),
            "columns_visited_once": int(np.count_nonzero(self.column_visits == 1)),
        }
        if self.coverage_after_completion["unvisited_columns"]:
            raise RuntimeError(
                "gradient completion left output columns unvisited: "
                f"{self.coverage_after_completion}"
            )

    def on_accept(self, result: Any) -> None:
        del result
        torch = self.require_setup()
        self.complete_exact_gradient()
        visits = torch.from_numpy(self.column_visits).to(device="cuda", dtype=torch.float32)
        average_gradient = self.gradient_sum / visits.clamp_min(1).unsqueeze(0)
        baseline_started_ns = time.monotonic_ns()
        reference_residual = self.inputs @ self.weight - self.targets
        reference_gradient = self.inputs.T @ reference_residual
        reference_gradient = reference_gradient.float() / self.batch_size
        torch.cuda.synchronize()
        self.baseline_gradient_compute_s = (time.monotonic_ns() - baseline_started_ns) / 1e9
        difference = average_gradient - reference_gradient
        reference_norm = float(torch.linalg.vector_norm(reference_gradient).item())
        gradient_relative_l2 = float(torch.linalg.vector_norm(difference).item()) / max(
            reference_norm, 1e-30
        )
        gradient_max_absolute_error = float(difference.abs().max().item())
        update_started_ns = time.monotonic_ns()
        self.master_weight.add_(average_gradient, alpha=-self.learning_rate)
        self.weight.copy_(self.master_weight.to(torch.bfloat16))
        torch.cuda.synchronize()
        optimizer_update_wall_s = (time.monotonic_ns() - update_started_ns) / 1e9
        after = self.evaluate_performance()
        self.loss_after = after["train_half_mse"]
        if not math.isfinite(self.loss_after) or self.loss_after > self.loss_before:
            raise RuntimeError(
                f"exact tiled SGD update did not reduce loss: {self.loss_before} -> {self.loss_after}"
            )
        self.last_performance = {
            "training_step": self.accepted_training_steps,
            "before": dict(self.current_performance),
            "after": dict(after),
            "train_improvement_percent": 100
            * (1 - after["train_half_mse"] / self.current_performance["train_half_mse"]),
            "validation_improvement_percent": 100
            * (
                1
                - after["validation_half_mse"]
                / self.current_performance["validation_half_mse"]
            ),
            "throughput": {
                "controlled_profile_gradient_wall_s": self.profile_gradient_compute_wall_s,
                "gradient_completion_wall_s": self.completion_gradient_compute_wall_s,
                "drawing_gradient_wall_s": self.gradient_compute_wall_s,
                "drawing_control_wall_s": (
                    self.capture_control_wall_s + self.completion_gradient_compute_wall_s
                ),
                "no_drawing_gradient_wall_s": self.baseline_gradient_compute_s,
                "optimizer_update_wall_s": optimizer_update_wall_s,
                "drawing_step_wall_s": self.capture_control_wall_s
                + self.completion_gradient_compute_wall_s
                + optimizer_update_wall_s,
                "no_drawing_step_wall_s": self.baseline_gradient_compute_s
                + optimizer_update_wall_s,
                "drawing_examples_per_s": self.batch_size
                / max(
                    self.capture_control_wall_s
                    + self.completion_gradient_compute_wall_s
                    + optimizer_update_wall_s,
                    1e-15,
                ),
                "no_drawing_examples_per_s": self.batch_size
                / max(self.baseline_gradient_compute_s + optimizer_update_wall_s, 1e-15),
                "drawing_steps_per_s": 1
                / max(
                    self.capture_control_wall_s
                    + self.completion_gradient_compute_wall_s
                    + optimizer_update_wall_s,
                    1e-15,
                ),
                "no_drawing_steps_per_s": 1
                / max(self.baseline_gradient_compute_s + optimizer_update_wall_s, 1e-15),
                "drawing_throughput_fraction": (
                    self.baseline_gradient_compute_s + optimizer_update_wall_s
                )
                / max(
                    self.capture_control_wall_s
                    + self.completion_gradient_compute_wall_s
                    + optimizer_update_wall_s,
                    1e-15,
                ),
                "drawing_slowdown_x": (
                    self.capture_control_wall_s
                    + self.completion_gradient_compute_wall_s
                    + optimizer_update_wall_s
                )
                / max(self.baseline_gradient_compute_s + optimizer_update_wall_s, 1e-15),
            },
            "arithmetic": {
                "tile_operations": self.last_operation_count,
                "controlled_profile_tile_operations": self.profile_operation_count,
                "gradient_completion_tile_operations": self.completion_operation_count,
                "lead_stabilization_tile_operations": self.lead_stabilization_operation_count,
                "executed_tile_width_sum": self.executed_tile_width_sum,
                "controlled_profile_tile_width_sum": self.profile_executed_tile_width_sum,
                "gradient_completion_tile_width_sum": self.completion_tile_width_sum,
                "lead_stabilization_tile_width_sum": self.lead_stabilization_tile_width_sum,
                "useful_output_width": self.output_size,
                "redundancy_ratio": self.executed_tile_width_sum / self.output_size,
                "useful_forward_backward_flops": 4
                * self.batch_size
                * self.hidden_size
                * self.output_size,
                "executed_forward_backward_flops": 4
                * self.batch_size
                * self.hidden_size
                * self.executed_tile_width_sum,
                "coverage_before_completion": dict(self.coverage_before_completion),
                "coverage_after_completion": dict(self.coverage_after_completion),
            },
            "equivalence": {
                "drawing_vs_no_drawing_gradient_relative_l2": gradient_relative_l2,
                "drawing_vs_no_drawing_gradient_max_absolute_error": gradient_max_absolute_error,
            },
        }
        del reference_gradient, reference_residual, difference
        self.current_performance = after
        self.accepted_training_steps += 1
        self.clear_gradient()

    def on_reject(self, error: BaseException) -> None:
        del error
        self.clear_gradient()

    def teardown(self) -> None:
        self.inputs = None
        self.targets = None
        self.validation_inputs = None
        self.validation_targets = None
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
            "schedule_mode": self.schedule_mode,
            "tile_repeats": self.tile_repeats,
            "reference_widths": self.reference_widths.tolist(),
            "reference_repeats": self.reference_repeats,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "parameter_count": self.parameter_count,
            "dtype": "bfloat16 matmuls, float32 gradient accumulation/master weights",
            "optimizer": "exact deferred SGD",
            "learning_rate": self.learning_rate,
            "lead_s": self.lead_s,
            "tail_s": self.tail_s,
            "active_lead_width": self.active_lead_width,
            "seed": self.seed,
            "transactional_update": True,
            "training_task": "teacher-student linear regression with held-out validation batch",
            "accepted_training_steps": self.accepted_training_steps,
        }


class TiledResidualMLPTrainingWorkload(TiledLinearTrainingWorkload):
    """Train a residual MLP while its real weight gradients draw the trace.

    The model contains ``depth`` square residual layers::

        h[l + 1] = h[l] + residual_scale * relu(h[l] @ W[l])

    Forward activations and reverse-mode deltas are prepared once for the
    current optimizer step.  Every controlled operation then computes a true
    block of ``h[l].T @ delta[l]``.  The scheduler walks across both columns
    and layers, repeated visits are averaged, and any unvisited blocks are
    completed after capture.  The resulting update is therefore independent
    of the silhouette command sequence apart from floating-point rounding.
    """

    task_label = "chipwhisperer-tiled-residual-mlp-training"

    def __init__(
        self,
        commands: np.ndarray,
        *,
        profile_duration_s: float,
        hidden_size: int = 4096,
        depth: int = 14,
        batch_size: int = 256,
        learning_rate: float = 0.002,
        validation_batch_size: int = 128,
        residual_scale: float = 0.125,
        verify_autograd: bool = True,
        lead_s: float = 0.002,
        tail_s: float = 0.002,
        active_lead_width: int = 0,
        seed: int = 1729,
        schedule_mode: str = "timed-repeat",
        tile_repeats: int = 1,
        reference_widths: np.ndarray | None = None,
        reference_repeats: int = 0,
    ):
        if depth < 2:
            raise ValueError("residual MLP depth must be at least two")
        if not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be greater than zero and at most one")
        self.depth = int(depth)
        self.residual_scale = float(residual_scale)
        self.verify_autograd = bool(verify_autograd)
        super().__init__(
            commands,
            profile_duration_s=profile_duration_s,
            hidden_size=hidden_size,
            output_size=hidden_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_batch_size=validation_batch_size,
            lead_s=lead_s,
            tail_s=tail_s,
            active_lead_width=active_lead_width,
            seed=seed,
            schedule_mode=schedule_mode,
            tile_repeats=tile_repeats,
            reference_widths=reference_widths,
            reference_repeats=reference_repeats,
        )
        self.layer_cursor = 0
        self.activations = None
        self.deltas = None
        self.state_prepare_wall_s = None
        self.autograd_equivalence = None

    @property
    def parameter_count(self) -> int:
        return self.depth * self.hidden_size * self.hidden_size

    @property
    def useful_gradient_width(self) -> int:
        return self.depth * self.output_size

    def _forward_with_weights(self, inputs, weights, *, retain_state: bool = False):
        torch = self.torch
        hidden = inputs
        activations = []
        masks = []
        for layer in range(self.depth):
            if retain_state:
                activations.append(hidden)
            preactivation = hidden @ weights[layer]
            if retain_state:
                masks.append(preactivation > 0)
            hidden = hidden + torch.relu(preactivation) * self.residual_scale
        return hidden, activations, masks

    def setup(self) -> None:
        try:
            import torch
        except ImportError as error:
            raise ConfigurationError("residual-MLP tiled training requires PyTorch") from error
        if not torch.cuda.is_available():
            raise ConfigurationError("residual-MLP tiled training requires a CUDA GPU")
        self.torch = torch
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.validation_inputs = torch.randn(
            self.validation_batch_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        teacher_weight = torch.randn(
            self.depth,
            self.hidden_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        ) / math.sqrt(self.hidden_size)
        with torch.no_grad():
            self.targets = self._forward_with_weights(self.inputs, teacher_weight)[0]
            self.validation_targets = self._forward_with_weights(
                self.validation_inputs,
                teacher_weight,
            )[0]
        del teacher_weight
        self.master_weight = torch.randn(
            self.depth,
            self.hidden_size,
            self.hidden_size,
            device="cuda",
            dtype=torch.float32,
        ) / math.sqrt(self.hidden_size)
        self.weight = self.master_weight.to(torch.bfloat16)
        self.gradient_sum = torch.zeros_like(self.master_weight)
        self.column_visits = np.zeros(
            (self.depth, self.output_size),
            dtype=np.int64,
        )
        self.current_performance = self.evaluate_performance()
        self.loss_before = self.current_performance["train_half_mse"]
        self._prepare_training_state()
        if self.verify_autograd:
            self.autograd_equivalence = self.verify_manual_gradient_against_autograd()
        torch.cuda.synchronize()

    def half_mse(self, inputs, targets) -> float:
        torch = self.torch
        with torch.no_grad():
            prediction = self._forward_with_weights(inputs, self.weight)[0]
            residual = prediction - targets
            loss = 0.5 * residual.float().square().mean()
        torch.cuda.synchronize()
        return float(loss)

    def performance_before_capture(self) -> dict[str, float]:
        if self.current_performance is None:
            raise RuntimeError("residual MLP has no prepared performance snapshot")
        return dict(self.current_performance)

    def _prepare_training_state(self) -> None:
        torch = self.require_setup()
        started_ns = time.monotonic_ns()
        with torch.no_grad():
            prediction, activations, masks = self._forward_with_weights(
                self.inputs,
                self.weight,
                retain_state=True,
            )
            residual = prediction - self.targets
            incoming = (residual.float() / self.batch_size).to(torch.bfloat16)
            deltas = [None] * self.depth
            for layer in range(self.depth - 1, -1, -1):
                local_delta = incoming * masks[layer]
                local_delta = local_delta * self.residual_scale
                deltas[layer] = local_delta
                incoming = incoming + local_delta @ self.weight[layer].T
        torch.cuda.synchronize()
        self.activations = activations
        self.deltas = deltas
        self.state_prepare_wall_s = (time.monotonic_ns() - started_ns) / 1e9

    def _full_weight_gradient(self):
        torch = self.require_setup()
        gradient = torch.empty_like(self.master_weight)
        for layer in range(self.depth):
            block = self.activations[layer].T @ self.deltas[layer]
            gradient[layer].copy_(block.float())
        torch.cuda.synchronize()
        return gradient

    def verify_manual_gradient_against_autograd(self) -> dict[str, float]:
        """Compare the handwritten reverse pass to PyTorch autograd once at setup."""

        torch = self.require_setup()
        manual = self._full_weight_gradient()
        reference_weight = self.weight.detach().clone().requires_grad_(True)
        prediction = self._forward_with_weights(self.inputs, reference_weight)[0]
        residual = prediction - self.targets
        objective = 0.5 * residual.float().square().sum() / self.batch_size
        (autograd,) = torch.autograd.grad(objective, reference_weight)
        torch.cuda.synchronize()
        difference = manual - autograd.float()
        reference_norm = float(torch.linalg.vector_norm(autograd.float()).item())
        result = {
            "manual_vs_autograd_gradient_relative_l2": float(
                torch.linalg.vector_norm(difference).item()
            )
            / max(reference_norm, 1e-30),
            "manual_vs_autograd_gradient_max_absolute_error": float(
                difference.abs().max().item()
            ),
        }
        del manual, reference_weight, autograd, difference, prediction, residual, objective
        torch.cuda.empty_cache()
        return result

    def gradient_tile(
        self,
        start: int,
        width: int,
        *,
        accumulate: bool = True,
        synchronize: bool = True,
        layer_index: int | None = None,
    ) -> None:
        torch = self.require_setup()
        layer = self.layer_cursor if layer_index is None else int(layer_index)
        if not 0 <= layer < self.depth:
            raise ValueError(f"layer index {layer} is outside [0, {self.depth})")
        end = start + width
        if not 0 <= start < end <= self.output_size:
            raise ValueError("gradient tile falls outside the residual layer")
        gradient = self.activations[layer].T @ self.deltas[layer][:, start:end]
        if accumulate:
            self.gradient_sum[layer, :, start:end].add_(gradient.float())
            self.column_visits[layer, start:end] += 1
        if synchronize:
            torch.cuda.synchronize()

    def _next_block(self, requested_width: int) -> tuple[int, int, int]:
        width = min(int(requested_width), self.output_size)
        if self.cursor + width > self.output_size:
            self.cursor = 0
            self.layer_cursor = (self.layer_cursor + 1) % self.depth
        layer = self.layer_cursor
        start = self.cursor
        self.cursor = start + width
        if self.cursor == self.output_size:
            self.cursor = 0
            self.layer_cursor = (self.layer_cursor + 1) % self.depth
        return layer, start, width

    def next_tile(self, requested_width: int) -> None:
        if requested_width == 0:
            return
        layer, start, width = self._next_block(requested_width)
        self.gradient_tile(start, width, layer_index=layer)

    def repeat_partition_tile(self, requested_width: int, *, synchronize: bool = True) -> int:
        layer, start, width = self._next_block(requested_width)
        for _ in range(self.tile_repeats):
            self.gradient_tile(
                start,
                width,
                synchronize=synchronize,
                layer_index=layer,
            )
        return self.tile_repeats

    def clear_gradient(self) -> None:
        super().clear_gradient()
        self.layer_cursor = 0

    def warmup(self, iteration: int) -> None:
        """First-touch every layer/shape outside the timed capture window."""

        del iteration
        for layer in range(self.depth):
            for width in np.unique(self.commands[self.commands > 0]):
                self.gradient_tile(
                    0,
                    int(width),
                    accumulate=False,
                    layer_index=layer,
                )
        self.clear_gradient()

    def complete_exact_gradient(self) -> None:
        started_ns = time.monotonic_ns()
        completed_width = 0
        completed_operations = 0
        for layer in range(self.depth):
            for start in range(0, self.output_size, 256):
                end = min(self.output_size, start + 256)
                if np.any(self.column_visits[layer, start:end] == 0):
                    self.gradient_tile(
                        start,
                        end - start,
                        layer_index=layer,
                    )
                    completed_width += end - start
                    completed_operations += 1
        self.completion_gradient_compute_wall_s = (time.monotonic_ns() - started_ns) / 1e9
        self.completion_tile_width_sum = completed_width
        self.completion_operation_count = completed_operations
        self.gradient_compute_wall_s += self.completion_gradient_compute_wall_s
        self.executed_tile_width_sum += completed_width
        self.last_operation_count += completed_operations
        self.coverage_after_completion = {
            "minimum_visits": int(self.column_visits.min()),
            "maximum_visits": int(self.column_visits.max()),
            "unvisited_columns": int(np.count_nonzero(self.column_visits == 0)),
            "columns_visited_once": int(np.count_nonzero(self.column_visits == 1)),
        }
        if self.coverage_after_completion["unvisited_columns"]:
            raise RuntimeError(
                "gradient completion left residual-MLP columns unvisited: "
                f"{self.coverage_after_completion}"
            )

    def on_accept(self, result: Any) -> None:
        del result
        torch = self.require_setup()
        common_compute_s = float(self.state_prepare_wall_s)
        self.complete_exact_gradient()
        visits = torch.from_numpy(self.column_visits).to(device="cuda", dtype=torch.float32)
        average_gradient = self.gradient_sum / visits.clamp_min(1).unsqueeze(1)
        baseline_started_ns = time.monotonic_ns()
        reference_gradient = self._full_weight_gradient()
        self.baseline_gradient_compute_s = (time.monotonic_ns() - baseline_started_ns) / 1e9
        difference = average_gradient - reference_gradient
        reference_norm = float(torch.linalg.vector_norm(reference_gradient).item())
        gradient_relative_l2 = float(torch.linalg.vector_norm(difference).item()) / max(
            reference_norm,
            1e-30,
        )
        gradient_max_absolute_error = float(difference.abs().max().item())
        update_started_ns = time.monotonic_ns()
        self.master_weight.add_(average_gradient, alpha=-self.learning_rate)
        self.weight.copy_(self.master_weight.to(torch.bfloat16))
        torch.cuda.synchronize()
        optimizer_update_wall_s = (time.monotonic_ns() - update_started_ns) / 1e9
        after = self.evaluate_performance()
        self.loss_after = after["train_half_mse"]
        if not math.isfinite(self.loss_after) or self.loss_after > self.loss_before:
            raise RuntimeError(
                "exact tiled residual-MLP update did not reduce loss: "
                f"{self.loss_before} -> {self.loss_after}"
            )
        drawing_step_s = (
            common_compute_s
            + self.capture_control_wall_s
            + self.completion_gradient_compute_wall_s
            + optimizer_update_wall_s
        )
        no_drawing_step_s = (
            common_compute_s + self.baseline_gradient_compute_s + optimizer_update_wall_s
        )
        common_forward_backward_flops = (
            4 * self.batch_size * self.hidden_size * self.hidden_size * self.depth
        )
        useful_weight_gradient_flops = (
            2 * self.batch_size * self.hidden_size * self.hidden_size * self.depth
        )
        executed_weight_gradient_flops = (
            2 * self.batch_size * self.hidden_size * self.executed_tile_width_sum
        )
        self.last_performance = {
            "training_step": self.accepted_training_steps,
            "before": dict(self.current_performance),
            "after": dict(after),
            "train_improvement_percent": 100
            * (1 - after["train_half_mse"] / self.current_performance["train_half_mse"]),
            "validation_improvement_percent": 100
            * (
                1
                - after["validation_half_mse"]
                / self.current_performance["validation_half_mse"]
            ),
            "throughput": {
                "common_forward_and_delta_wall_s": common_compute_s,
                "controlled_profile_gradient_wall_s": self.profile_gradient_compute_wall_s,
                "gradient_completion_wall_s": self.completion_gradient_compute_wall_s,
                "drawing_gradient_wall_s": self.gradient_compute_wall_s,
                "drawing_control_wall_s": self.capture_control_wall_s,
                "no_drawing_gradient_wall_s": self.baseline_gradient_compute_s,
                "optimizer_update_wall_s": optimizer_update_wall_s,
                "drawing_step_wall_s": drawing_step_s,
                "no_drawing_step_wall_s": no_drawing_step_s,
                "drawing_examples_per_s": self.batch_size / max(drawing_step_s, 1e-15),
                "no_drawing_examples_per_s": self.batch_size
                / max(no_drawing_step_s, 1e-15),
                "drawing_steps_per_s": 1 / max(drawing_step_s, 1e-15),
                "no_drawing_steps_per_s": 1 / max(no_drawing_step_s, 1e-15),
                "drawing_throughput_fraction": no_drawing_step_s
                / max(drawing_step_s, 1e-15),
                "drawing_slowdown_x": drawing_step_s / max(no_drawing_step_s, 1e-15),
            },
            "arithmetic": {
                "tile_operations": self.last_operation_count,
                "controlled_profile_tile_operations": self.profile_operation_count,
                "gradient_completion_tile_operations": self.completion_operation_count,
                "lead_stabilization_tile_operations": self.lead_stabilization_operation_count,
                "executed_tile_width_sum": self.executed_tile_width_sum,
                "controlled_profile_tile_width_sum": self.profile_executed_tile_width_sum,
                "gradient_completion_tile_width_sum": self.completion_tile_width_sum,
                "lead_stabilization_tile_width_sum": self.lead_stabilization_tile_width_sum,
                "useful_output_width": self.useful_gradient_width,
                "redundancy_ratio": self.executed_tile_width_sum
                / self.useful_gradient_width,
                "useful_forward_backward_flops": common_forward_backward_flops
                + useful_weight_gradient_flops,
                "executed_forward_backward_flops": common_forward_backward_flops
                + executed_weight_gradient_flops,
                "coverage_before_completion": dict(self.coverage_before_completion),
                "coverage_after_completion": dict(self.coverage_after_completion),
            },
            "equivalence": {
                "drawing_vs_no_drawing_gradient_relative_l2": gradient_relative_l2,
                "drawing_vs_no_drawing_gradient_max_absolute_error": (
                    gradient_max_absolute_error
                ),
                **(self.autograd_equivalence or {}),
            },
        }
        del reference_gradient, difference, average_gradient, visits
        self.current_performance = after
        self.accepted_training_steps += 1
        self.clear_gradient()
        self._prepare_training_state()

    def teardown(self) -> None:
        self.activations = None
        self.deltas = None
        super().teardown()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "drawing_with_traces.exact_tiled_residual_mlp_gradient",
            "architecture": (
                "h[l+1] = h[l] + residual_scale * relu(h[l] @ W[l])"
            ),
            "objective": "0.5 / batch * ||MLP(X)-Y||^2",
            "gradient": "h[l].T @ delta[l], tiled over output columns and layers",
            "commands_sha256": array_hash(self.commands, "<i8"),
            "profile_bins": int(self.commands.size),
            "profile_duration_s": self.profile_duration_s,
            "bin_duration_s": self.bin_duration_s,
            "schedule_mode": self.schedule_mode,
            "tile_repeats": self.tile_repeats,
            "reference_widths": self.reference_widths.tolist(),
            "reference_repeats": self.reference_repeats,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "residual_scale": self.residual_scale,
            "batch_size": self.batch_size,
            "validation_batch_size": self.validation_batch_size,
            "parameter_count": self.parameter_count,
            "dtype": "bfloat16 matmuls, float32 gradient accumulation/master weights",
            "optimizer": "exact deferred SGD",
            "learning_rate": self.learning_rate,
            "lead_s": self.lead_s,
            "tail_s": self.tail_s,
            "active_lead_width": self.active_lead_width,
            "seed": self.seed,
            "transactional_update": True,
            "training_task": "teacher-student 14-layer residual MLP regression",
            "accepted_training_steps": self.accepted_training_steps,
            "autograd_equivalence": self.autograd_equivalence,
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

    def commands_for(self, target: np.ndarray, *, width_quantum: int = 32) -> np.ndarray:
        if width_quantum < 1:
            raise ValueError("width_quantum must be positive")
        desired = self.minimum_activity + np.asarray(target) * (
            self.maximum_activity - self.minimum_activity
        )
        activity, first = np.unique(self.monotonic_activity, return_index=True)
        efficient_widths = self.widths[first]
        continuous = np.interp(desired, activity, efficient_widths)
        quantized = np.rint(continuous / width_quantum) * width_quantum
        return np.clip(quantized, 0, self.widths[-1]).astype(np.int64)

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


class FastRefinementController:
    """Adapt tile commands from the best measured SideCapture trace."""

    def __init__(
        self,
        target: np.ndarray,
        calibration: FastCalibration,
        *,
        target_accuracy_percent: float | None = None,
        initial_gain: float = 0.35,
        minimum_gain: float = 0.05,
        gain_decay: float = 0.5,
        gain_growth: float = 1.05,
        improvement_tolerance: float = 1e-4,
        total_width: int | None = None,
        minimum_width: int = 0,
        correction_smoothing_sigma_points: float = 0.0,
    ):
        target = np.asarray(target, dtype=np.float64)
        if target.ndim != 1 or target.size < 2 or not np.all(np.isfinite(target)):
            raise ValueError("target must be a finite one-dimensional array")
        if np.any((target < 0) | (target > 1)):
            raise ValueError("target values must be between 0 and 1")
        if target_accuracy_percent is not None and not 0 < target_accuracy_percent <= 100:
            raise ValueError("target_accuracy_percent must be in (0, 100]")
        if initial_gain <= 0 or minimum_gain <= 0 or minimum_gain > initial_gain:
            raise ValueError("controller gains must satisfy 0 < minimum_gain <= initial_gain")
        if not 0 < gain_decay < 1:
            raise ValueError("gain_decay must be in (0, 1)")
        if gain_growth < 1:
            raise ValueError("gain_growth must be at least 1")
        if improvement_tolerance < 0:
            raise ValueError("improvement_tolerance cannot be negative")
        if correction_smoothing_sigma_points < 0:
            raise ValueError("correction_smoothing_sigma_points cannot be negative")
        self.target = target
        self.calibration = calibration
        self.target_accuracy_percent = target_accuracy_percent
        self.initial_gain = float(initial_gain)
        self.minimum_gain = float(minimum_gain)
        self.gain_decay = float(gain_decay)
        self.gain_growth = float(gain_growth)
        self.improvement_tolerance = float(improvement_tolerance)
        self.total_width = total_width
        self.minimum_width = int(minimum_width)
        self.correction_smoothing_sigma_points = float(
            correction_smoothing_sigma_points
        )
        if total_width is not None:
            project_commands_to_total(
                np.full(target.size, total_width / target.size),
                total_width,
                minimum_width=self.minimum_width,
                maximum_width=int(calibration.widths[-1]),
            )
        self.gain = float(initial_gain)
        self.best_rmse = math.inf
        self.best_accuracy_percent = 0.0
        self.best_commands: np.ndarray | None = None
        self.best_measured: np.ndarray | None = None
        self.observations = 0

    @property
    def reached_target(self) -> bool:
        return (
            self.target_accuracy_percent is not None
            and self.best_accuracy_percent >= self.target_accuracy_percent
        )

    def initial_commands(self) -> np.ndarray:
        return self._constrain(self.calibration.commands_for(self.target))

    def _constrain(self, commands: np.ndarray) -> np.ndarray:
        if self.total_width is None:
            commands = np.rint(np.asarray(commands, dtype=np.float64) / 32) * 32
            return np.clip(
                commands,
                self.minimum_width,
                int(self.calibration.widths[-1]),
            ).astype(np.int64)
        return project_commands_to_total(
            commands,
            self.total_width,
            minimum_width=self.minimum_width,
            maximum_width=int(self.calibration.widths[-1]),
        )

    def observe(
        self,
        commands: np.ndarray,
        measured: np.ndarray,
        *,
        normalized_rmse: float,
        shape_accuracy_percent: float,
    ) -> bool:
        commands = np.asarray(commands, dtype=np.int64)
        measured = np.asarray(measured, dtype=np.float64)
        if commands.shape != self.target.shape or measured.shape != self.target.shape:
            raise ValueError("commands, measured values, and target must have matching shapes")
        if not np.all(np.isfinite(measured)) or not math.isfinite(normalized_rmse):
            raise ValueError("refinement measurements must be finite")
        improved = normalized_rmse < self.best_rmse
        if improved:
            had_best = self.best_commands is not None
            significant = normalized_rmse < self.best_rmse - self.improvement_tolerance
            self.best_rmse = float(normalized_rmse)
            self.best_accuracy_percent = float(shape_accuracy_percent)
            self.best_commands = commands.copy()
            self.best_measured = measured.copy()
            if had_best and significant:
                self.gain = min(self.initial_gain, self.gain * self.gain_growth)
        else:
            self.gain = max(self.minimum_gain, self.gain * self.gain_decay)
        self.observations += 1
        return improved

    def next_commands(self) -> np.ndarray:
        if self.best_commands is None or self.best_measured is None:
            return self.initial_commands()
        error = self.target - self.best_measured
        corrected_target = np.clip(self.target + self.gain * error, 0, 1)
        baseline_commands = self.calibration.commands_for(self.target)
        corrected_commands = self.calibration.commands_for(corrected_target)
        # True iterative learning control: retain the best delivered command and
        # add the calibration-derived correction. Recomputing from the baseline
        # on every round discards prior successful corrections and oscillates.
        correction = (
            self.best_commands - baseline_commands
            + corrected_commands - baseline_commands
        ).astype(np.float64)
        if self.correction_smoothing_sigma_points:
            correction = _smooth_points(
                correction,
                self.correction_smoothing_sigma_points,
            )
        commands = self._constrain(baseline_commands + correction)
        if np.array_equal(commands, self.best_commands) and not self.reached_target:
            # Quantization can otherwise make a nonzero residual a fixed point.
            largest = int(np.argmax(np.abs(error)))
            direction = 32 if error[largest] > 0 else -32
            if self.total_width is None:
                commands[largest] = int(
                    np.clip(
                        self.best_commands[largest] + direction,
                        0,
                        self.calibration.widths[-1],
                    )
                )
            else:
                if direction > 0:
                    donors = np.flatnonzero(commands > self.minimum_width)
                    donors = donors[donors != largest]
                    receiver_ok = commands[largest] < self.calibration.widths[-1]
                    donor = donors[int(np.argmin(error[donors]))] if donors.size else None
                else:
                    donors = np.flatnonzero(commands < self.calibration.widths[-1])
                    donors = donors[donors != largest]
                    receiver_ok = commands[largest] > self.minimum_width
                    donor = donors[int(np.argmax(error[donors]))] if donors.size else None
                if receiver_ok and donor is not None:
                    commands[largest] += direction
                    commands[donor] -= direction
        return commands

    def metadata(self) -> dict[str, Any]:
        return {
            "target_accuracy_percent": self.target_accuracy_percent,
            "initial_gain": self.initial_gain,
            "minimum_gain": self.minimum_gain,
            "current_gain": self.gain,
            "gain_decay": self.gain_decay,
            "gain_growth": self.gain_growth,
            "observations": self.observations,
            "best_normalized_rmse": None if not math.isfinite(self.best_rmse) else self.best_rmse,
            "best_accuracy_percent": self.best_accuracy_percent,
            "reached_target": self.reached_target,
            "total_width": self.total_width,
            "minimum_width": self.minimum_width,
            "correction_smoothing_sigma_points": self.correction_smoothing_sigma_points,
        }


def _trace_and_bins(
    root: str | Path,
    *,
    record_index: int | None = None,
    event_name: str = "tile.bin",
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(root)
    records = DirectoryStore(root, resume=True).records()
    if record_index is None and len(records) != 1:
        raise ConfigurationError(f"expected one ChipWhisperer trace in {root}, found {len(records)}")
    if record_index is None:
        record = records[0]
    else:
        matches = [record for record in records if int(record["index"]) == record_index]
        if len(matches) != 1:
            raise ConfigurationError(
                f"expected capture index {record_index} in {root}, found {len(matches)}"
            )
        record = matches[0]
    descriptor = record["channels"][record["primary_channel"]]
    trace = np.load(root / descriptor["path"], allow_pickle=False).astype(np.float64)
    annotations = json.loads((root / record["annotations"]["path"]).read_text())
    bins = [event for event in annotations if event["name"] == event_name]
    bins.sort(key=lambda event: int(event["metadata"]["index"]))
    if not bins:
        raise ConfigurationError(f"trace has no {event_name} annotations")
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


def measured_bin_features(
    root: str | Path,
    *,
    record_index: int | None = None,
    event_name: str = "tile.bin",
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    trace, bins, record = _trace_and_bins(
        root,
        record_index=record_index,
        event_name=event_name,
    )
    center = float(np.median(trace))
    by_name: dict[str, list[float]] = {}
    widths = []
    for event in bins:
        start, end = int(event["start_sample"]), int(event["end_sample"])
        segment = trace[max(0, start) : min(trace.size, end)]
        if segment.size < 8:
            raise ConfigurationError(
                f"{event_name} event {event['metadata']['index']} has only {segment.size} ADC samples"
            )
        for name, value in _features(segment, center).items():
            by_name.setdefault(name, []).append(value)
        widths.append(int(event["metadata"]["requested_width"]))
    return {name: np.asarray(value) for name, value in by_name.items()}, np.asarray(widths), record


def calibrate_fast(
    root: str | Path,
    *,
    record_index: int | None = None,
    event_name: str = "tile.bin",
    preferred_feature: str | None = None,
) -> FastCalibration:
    features, commands, _ = measured_bin_features(
        root,
        record_index=record_index,
        event_name=event_name,
    )
    widths = np.asarray(sorted(set(commands)), dtype=np.float64)
    x = np.log2(widths + 32)
    candidates = []
    for name, values in features.items():
        medians = np.asarray([np.median(values[commands == width]) for width in widths])
        correlation = float(np.corrcoef(x, medians)[0, 1])
        sign = 1.0 if medians[-1] >= medians[0] else -1.0
        activity = sign * medians
        peak = int(np.argmax(activity))
        candidate_widths = widths[: peak + 1]
        candidate_measured = medians[: peak + 1]
        candidate_activity = activity[: peak + 1]
        monotonic = _isotonic_increasing(candidate_activity)
        within = np.asarray(
            [np.std(values[commands == width]) for width in candidate_widths],
            dtype=np.float64,
        )
        span = float(monotonic[-1] - monotonic[0])
        noise = float(np.median(within))
        snr = span / max(noise, 1e-15)
        score = snr * max(abs(correlation), 0.1) * math.sqrt(candidate_widths.size)
        candidates.append(
            {
                "score": score,
                "name": name,
                "sign": sign,
                "widths": candidate_widths,
                "measured": candidate_measured,
                "monotonic": monotonic,
                "within": within,
                "span": span,
                "noise": noise,
                "snr": snr,
                "correlation": correlation,
            }
        )
    valid = [
        item
        for item in candidates
        if item["widths"].size >= 3 and item["span"] > max(1e-9, 2 * item["noise"])
    ]
    if not valid:
        diagnostics = "; ".join(
            f"{item['name']}: span={item['span']:.4g}, noise={item['noise']:.4g}, "
            f"snr={item['snr']:.2f}, widths={item['widths'].size}"
            for item in candidates
        )
        raise ConfigurationError(
            "tile-width calibration has no separable feature; " + diagnostics
        )
    if preferred_feature is not None:
        preferred = [item for item in valid if item["name"] == preferred_feature]
        if not preferred:
            raise ConfigurationError(
                f"preferred calibration feature {preferred_feature!r} is not separable; "
                + diagnostics
            )
        valid = preferred
    best = max(valid, key=lambda item: item["score"])
    return FastCalibration(
        best["widths"],
        best["name"],
        best["sign"],
        best["measured"],
        best["monotonic"],
        best["within"],
    )


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


def render_training_comparison(
    performances: list[dict[str, Any]],
    output: str | Path,
) -> Path:
    """Plot learning and drawing-versus-no-drawing training throughput."""

    if not performances:
        raise ValueError("at least one training performance record is required")
    output = Path(output)
    steps = [int(item["training_step"]) for item in performances]
    x = np.asarray([steps[0], *[step + 1 for step in steps]], dtype=np.int64)
    train = np.asarray(
        [performances[0]["before"]["train_half_mse"]]
        + [item["after"]["train_half_mse"] for item in performances]
    )
    validation = np.asarray(
        [performances[0]["before"]["validation_half_mse"]]
        + [item["after"]["validation_half_mse"] for item in performances]
    )
    drawing_steps = np.asarray(
        [item["throughput"]["drawing_steps_per_s"] for item in performances]
    )
    no_drawing_steps = np.asarray(
        [item["throughput"]["no_drawing_steps_per_s"] for item in performances]
    )
    drawing_median = float(np.median(drawing_steps))
    no_drawing_median = float(np.median(no_drawing_steps))
    retained = drawing_median / max(no_drawing_median, 1e-15) * 100
    slowdown = no_drawing_median / max(drawing_median, 1e-15)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True, facecolor="white")
    axes[0].plot(x, train, "o-", color="#D92D20", label="Training loss")
    axes[0].plot(x, validation, "o-", color="#475467", label="Held-out loss")
    axes[0].set_xlabel("accepted optimizer steps")
    axes[0].set_ylabel("half mean-squared error")
    axes[0].set_title("The persistent model really learns")
    axes[0].grid(True, color="#E4E7EC", linewidth=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)

    bars = axes[1].bar(
        ["Drawing", "No drawing"],
        [drawing_median, no_drawing_median],
        color=["#D92D20", "#475467"],
        width=0.62,
    )
    axes[1].bar_label(bars, fmt="%.1f steps/s", padding=4)
    axes[1].set_ylabel("optimizer steps per second")
    axes[1].set_title(f"Drawing retains {retained:.1f}% throughput · {slowdown:.2f}× slower")
    axes[1].grid(True, axis="y", color="#E4E7EC", linewidth=0.7)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, max(drawing_median, no_drawing_median) * 1.2)
    figure.suptitle("Training performance with and without power-trace drawing", weight="bold")
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


def normalized_curve_metrics(
    target: np.ndarray,
    measured_raw: np.ndarray,
    measured_smoothed: np.ndarray,
) -> dict[str, Any]:
    """Score native and display-bandwidth curves without hiding bin-scale error."""

    target = np.asarray(target, dtype=np.float64)
    measured_raw = np.asarray(measured_raw, dtype=np.float64)
    measured_smoothed = np.asarray(measured_smoothed, dtype=np.float64)
    if target.ndim != 1 or target.shape != measured_raw.shape or target.shape != measured_smoothed.shape:
        raise ValueError("target, raw measurement, and smoothed measurement must be matching vectors")
    if not all(np.all(np.isfinite(value)) for value in (target, measured_raw, measured_smoothed)):
        raise ValueError("curve metrics require finite values")
    error = measured_smoothed - target
    raw_error = measured_raw - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    raw_rmse = float(np.sqrt(np.mean(np.square(raw_error))))
    multiscale_rmse = float(np.sqrt((rmse**2 + raw_rmse**2) / 2))
    target_variance = float(np.sum(np.square(target - target.mean())))
    return {
        "pearson_r": float(np.corrcoef(target, measured_smoothed)[0, 1]),
        "normalized_mae": float(np.mean(np.abs(error))),
        "normalized_rmse": rmse,
        "r_squared": float(1 - np.sum(np.square(error)) / max(target_variance, 1e-15)),
        "shape_accuracy_percent": float(max(0, 1 - rmse) * 100),
        "raw_pearson_r": float(np.corrcoef(target, measured_raw)[0, 1]),
        "raw_normalized_mae": float(np.mean(np.abs(raw_error))),
        "raw_normalized_rmse": raw_rmse,
        "raw_shape_accuracy_percent": float(max(0, 1 - raw_rmse) * 100),
        "multiscale_normalized_rmse": multiscale_rmse,
        "multiscale_shape_accuracy_percent": float(max(0, 1 - multiscale_rmse) * 100),
        "multiscale_score_definition": (
            "100 * (1 - RMS(raw normalized RMSE, smoothed normalized RMSE))"
        ),
    }


def analyze_fast_drawing(
    root: str | Path,
    target: np.ndarray,
    calibration: FastCalibration,
    *,
    smoothing_sigma_points: float = 0.8,
    record_index: int | None = None,
    feature_name: str | None = None,
    feature_sign: float | None = None,
    normalization: str = "calibration",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features, commands, record = measured_bin_features(root, record_index=record_index)
    _, bins, _ = _trace_and_bins(root, record_index=record_index)
    channel = record["channels"][record["primary_channel"]]
    sample_rate_hz = float(channel["sample_rate_hz"])
    profile_start_sample = min(int(item["start_sample"]) for item in bins)
    profile_end_sample = max(int(item["end_sample"]) for item in bins)
    bin_center_ms = np.asarray(
        [
            ((int(item["start_sample"]) + int(item["end_sample"])) / 2 - profile_start_sample)
            / sample_rate_hz
            * 1e3
            for item in bins
        ],
        dtype=np.float64,
    )
    actual_duration_ms = (profile_end_sample - profile_start_sample) / sample_rate_hz * 1e3
    feature_name = calibration.feature_name if feature_name is None else feature_name
    if feature_name not in features:
        raise ConfigurationError(
            f"trace feature {feature_name!r} is unavailable; choose from {sorted(features)}"
        )
    feature_sign = calibration.feature_sign if feature_sign is None else float(feature_sign)
    if feature_sign not in {-1.0, 1.0}:
        raise ValueError("feature_sign must be -1 or 1")
    measured_feature = features[feature_name]
    activity = feature_sign * measured_feature
    if normalization == "calibration":
        measured = (activity - calibration.minimum_activity) / max(
            calibration.maximum_activity - calibration.minimum_activity,
            1e-15,
        )
        normalization_bounds = [calibration.minimum_activity, calibration.maximum_activity]
    elif normalization == "robust":
        low, high = np.quantile(activity, [0.02, 0.98])
        if high - low <= 1e-15:
            raise ConfigurationError("robust trace normalization has zero activity span")
        measured = np.clip((activity - low) / (high - low), 0, 1)
        normalization_bounds = [float(low), float(high)]
    else:
        raise ValueError("normalization must be 'calibration' or 'robust'")
    measured_smoothed = _smooth_points(measured, smoothing_sigma_points)
    target = np.asarray(target, dtype=np.float64)
    if target.shape != measured.shape:
        raise ConfigurationError(
            f"target has {target.size} points but trace contains {measured.size} tile bins"
        )
    curve_metrics = normalized_curve_metrics(target, measured, measured_smoothed)
    metrics = {
        "requested_duration_ms": float(record["labels"]["profile_duration_s"] * 1e3),
        "actual_profile_duration_ms": actual_duration_ms,
        "bins": int(target.size),
        "feature": feature_name,
        "feature_sign": feature_sign,
        "normalization": normalization,
        "normalization_bounds": normalization_bounds,
        "smoothing_sigma_points": smoothing_sigma_points,
        **curve_metrics,
        "measurement": "ChipWhisperer normalized ADC activity; not calibrated watts",
        "loss_before_update": record["labels"].get("loss_before_update"),
    }
    arrays = {
        "target": target,
        "measured_raw": measured,
        "measured_smoothed": measured_smoothed,
        "commands": commands,
        "feature_raw": measured_feature,
        "bin_center_ms": bin_center_ms,
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
    record_index: int | None = None,
    analysis_root: str | Path | None = None,
    feature_name: str | None = None,
    feature_sign: float | None = None,
    normalization: str = "calibration",
) -> tuple[Path, dict[str, Any]]:
    root = Path(root)
    metrics, arrays = analyze_fast_drawing(
        root,
        target,
        calibration,
        smoothing_sigma_points=smoothing_sigma_points,
        record_index=record_index,
        feature_name=feature_name,
        feature_sign=feature_sign,
        normalization=normalization,
    )
    analysis_root = root if analysis_root is None else Path(analysis_root)
    output = Path(output) if output is not None else analysis_root / "fast_measured_silhouette.png"
    duration_ms = metrics["actual_profile_duration_ms"]
    x = arrays["bin_center_ms"]
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
    target_titles = {
        "height": "Lowered target silhouette",
        "upper-boundary": "Lowered upper boundary",
        "lower-boundary": "Lowered lower boundary",
    }
    axes[0].set_title(target_titles[envelope.extraction_mode], fontsize=13)
    axes[1].plot(x, target, color="#475467", linestyle="--", linewidth=1.4, label="Target")
    axes[1].plot(
        x,
        arrays["measured_raw"],
        color="#F97066",
        linewidth=0.7,
        alpha=0.45,
        label="Measured native bins",
    )
    axes[1].plot(
        x,
        arrays["measured_smoothed"],
        color="#D92D20",
        linewidth=1.8,
        label="Measured scored curve",
    )
    axes[1].fill_between(x, 0, arrays["measured_smoothed"], color="#D0D5DD", alpha=0.65)
    axes[1].set_xlim(0, duration_ms)
    low = min(0.0, float(arrays["measured_smoothed"].min()) - 0.05)
    high = max(1.0, float(arrays["measured_smoothed"].max()) + 0.05)
    axes[1].set_ylim(low, high)
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("normalized activity")
    axes[1].set_title(
        f"{duration_ms:g} ms real tiled-gradient capture · "
        f"fidelity {metrics['multiscale_shape_accuracy_percent']:.1f}% · "
        f"smoothed {metrics['shape_accuracy_percent']:.1f}%"
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
    atomic_json(metrics, analysis_root / "fast_metrics.json")
    for name, values in arrays.items():
        atomic_numpy(np.asarray(values, dtype=np.float32), analysis_root / f"fast_{name}.npy")
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
