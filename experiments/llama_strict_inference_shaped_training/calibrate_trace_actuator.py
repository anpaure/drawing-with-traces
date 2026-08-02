#!/usr/bin/env python3
"""Physically calibrate Llama gradient-tile width against ChipWhisperer activity."""

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

from .trace_drawing import (
    DEFAULT_ACTUATOR_MODULE,
    TransformerGradientCalibrationWorkload,
)


def calibration_widths() -> np.ndarray:
    return np.unique(
        np.asarray(
            [
                32,
                64,
                96,
                128,
                192,
                256,
                384,
                512,
                768,
                1024,
                1280,
                1536,
                2048,
                3072,
                4096,
                6144,
                8192,
            ],
            dtype=np.int64,
        )
    )


def interleaved_commands(widths: np.ndarray, *, repeats: int, seed: int) -> np.ndarray:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module-name", default=DEFAULT_ACTUATOR_MODULE)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    parser.add_argument("--training-sequence-length", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--bin-ms", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument(
        "--execution-backend",
        choices=("cuda-graph", "python-queued"),
        default="cuda-graph",
    )
    parser.add_argument("--graph-target-utilization", type=float, default=0.82)
    parser.add_argument("--graph-maximum-operations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument(
        "--sync-each-operation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    if args.bin_ms <= 0 or args.repeats < 2:
        raise ValueError("--bin-ms must be positive and --repeats must be at least two")
    widths = calibration_widths()
    commands = interleaved_commands(widths, repeats=args.repeats, seed=args.seed)
    workload = TransformerGradientCalibrationWorkload(
        commands,
        bin_duration_s=args.bin_ms / 1e3,
        module_name=args.module_name,
        training_batch_size=args.training_batch_size,
        training_sequence_length=args.training_sequence_length,
        tile_rows=args.tile_rows,
        synchronize_each_operation=args.sync_each_operation,
        execution_backend=args.execution_backend,
        graph_target_utilization=args.graph_target_utilization,
        graph_maximum_operations=args.graph_maximum_operations,
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
    calibration = calibrate_fast(args.output, command_key="requested_width")
    save_fast_calibration(args.output, calibration)
    render_fast_calibration(calibration, args.output / "calibration.png")
    summary = {
        "output": str(args.output),
        "record_index": int(records[0]["index"]),
        "commands": commands.tolist(),
        "widths": widths.tolist(),
        "bin_ms": args.bin_ms,
        "repeats": args.repeats,
        "capture_duration_ms": duration_s * 1e3,
        "module_name": args.module_name,
        "training_batch_size": args.training_batch_size,
        "training_sequence_length": args.training_sequence_length,
        "tile_rows": args.tile_rows,
        "synchronize_each_operation": args.sync_each_operation,
        "execution_backend": args.execution_backend,
        "graph_target_utilization": args.graph_target_utilization,
        "graph_maximum_operations": args.graph_maximum_operations,
        "calibration": calibration.to_dict(),
        "last_profile": workload.last_profile,
        "strict_invariants": {
            "inference_cover_tokens": 0,
            "secondary_model_instances": 0,
            "measured_operations": "redundant exact current-step dW blocks",
        },
    }
    (args.output / "calibration_capture_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
