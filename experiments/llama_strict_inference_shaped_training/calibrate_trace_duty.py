#!/usr/bin/env python3
"""Calibrate graph operation count using one fixed real Llama dW tile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sidecapture as sc

from drawing_with_traces.fast import (
    calibrate_fast,
    render_fast_calibration,
    save_fast_calibration,
)

from .trace_drawing import TransformerGradientCalibrationWorkload


def operation_levels(
    maximum_operations: int,
    level_count: int | None = None,
) -> np.ndarray:
    if maximum_operations < 32:
        raise ValueError("maximum_operations must be at least 32")
    anchors = [1, 2, 4, 8, 12, 16, 24, 32]
    if level_count is not None:
        if level_count < len(anchors) + 2:
            raise ValueError(f"level_count must be at least {len(anchors) + 2}")
        remaining = level_count - len(anchors)
        sparse = np.rint(np.linspace(40, maximum_operations, remaining)).astype(np.int64)
        return np.unique(np.asarray([*anchors, *sparse, maximum_operations], dtype=np.int64))
    dense = list(range(40, maximum_operations + 1, 8))
    return np.unique(np.asarray([*anchors, *dense, maximum_operations], dtype=np.int64))


def interleaved_levels(levels: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    orders = []
    previous = None
    for _ in range(repeats):
        order = rng.permutation(levels)
        if previous is not None and order[0] == previous:
            order = np.roll(order, 1)
        orders.append(order)
        previous = int(order[-1])
    return np.concatenate(orders)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-width", type=int, default=160)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    parser.add_argument("--training-sequence-length", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--maximum-operations", type=int, default=160)
    parser.add_argument("--level-count", type=int)
    parser.add_argument("--bin-ms", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=6600)
    args = parser.parse_args()
    if args.fixed_width < 32 or args.fixed_width % 32:
        raise ValueError("--fixed-width must be a positive multiple of 32")
    if args.bin_ms <= 0 or args.repeats < 2:
        raise ValueError("--bin-ms must be positive and --repeats must be at least two")
    levels = operation_levels(args.maximum_operations, args.level_count)
    operations = interleaved_levels(levels, args.repeats, args.seed)
    widths = np.full(operations.shape, args.fixed_width, dtype=np.int64)
    workload = TransformerGradientCalibrationWorkload(
        widths,
        operation_commands=operations,
        bin_duration_s=args.bin_ms / 1e3,
        training_batch_size=args.training_batch_size,
        training_sequence_length=args.training_sequence_length,
        tile_rows=args.tile_rows,
        execution_backend="cuda-graph",
        seed=args.seed,
    )
    duration_s = workload.total_duration_s + 0.005
    sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=f"{duration_s * 1e3:.3f}ms",
            sample_rate=args.sample_rate,
            mode="burst",
            bits_per_sample=12,
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(args.output, trace_dtype="float32"),
        retry=sc.RetryPolicy(max_attempts=3, backoff_s=0.5),
        workload_sync="none",
    ) as experiment:
        records = experiment.run(1)
    calibration = calibrate_fast(args.output, command_key="requested_operations")
    save_fast_calibration(args.output, calibration)
    render_fast_calibration(
        calibration,
        args.output / "calibration.png",
        command_label=f"Exact dW GEMMs per graph (fixed tile width {args.fixed_width})",
    )
    summary = {
        "output": str(args.output),
        "record_index": int(records[0]["index"]),
        "fixed_width": args.fixed_width,
        "training_batch_size": args.training_batch_size,
        "training_sequence_length": args.training_sequence_length,
        "tile_rows": args.tile_rows,
        "operation_levels": levels.tolist(),
        "commands": operations.tolist(),
        "bin_ms": args.bin_ms,
        "capture_duration_ms": duration_s * 1e3,
        "calibration": calibration.to_dict(),
        "strict_invariants": workload.config.strict_invariants(),
    }
    (args.output / "duty_calibration_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
