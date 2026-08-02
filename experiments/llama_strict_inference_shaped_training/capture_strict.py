#!/usr/bin/env python3
"""Capture unaligned windows from strict inference or no-cover training."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import sidecapture as sc

from .strict_workloads import (
    DEFAULT_MODEL,
    PersistentStrictWorkload,
    StrictWorkloadConfig,
)


class RandomPretriggerDelaySampler:
    """Delay an already-armed sampler to remove deterministic process phase."""

    def __init__(
        self,
        sampler,
        *,
        minimum_delay_seconds: float,
        maximum_delay_seconds: float,
        seed: int,
    ) -> None:
        if minimum_delay_seconds < 0 or maximum_delay_seconds < minimum_delay_seconds:
            raise ValueError("delay bounds must satisfy 0 <= minimum <= maximum")
        self.sampler = sampler
        self.request = sampler.request
        self.resolved = None
        self.minimum_delay_seconds = float(minimum_delay_seconds)
        self.maximum_delay_seconds = float(maximum_delay_seconds)
        self._random = random.Random(seed)
        self._last_delay_seconds: float | None = None

    def open(self) -> None:
        self.sampler.open()

    def plan(self):
        self.resolved = self.sampler.plan()
        return self.resolved

    def arm(self) -> None:
        self.sampler.arm()
        self._last_delay_seconds = self._random.uniform(
            self.minimum_delay_seconds, self.maximum_delay_seconds
        )
        if self._last_delay_seconds:
            time.sleep(self._last_delay_seconds)

    def trigger(self):
        return self.sampler.trigger()

    def observe_trigger(self, anchor) -> None:
        self.sampler.observe_trigger(anchor)

    def finish(self):
        return self.sampler.finish()

    def abort(self) -> None:
        self.sampler.abort()

    def recover(self, error: BaseException) -> bool:
        return bool(self.sampler.recover(error))

    def close(self) -> None:
        self.sampler.close()

    def metadata(self) -> dict[str, Any]:
        return {
            **self.sampler.metadata(),
            "random_pretrigger_delay": {
                "minimum_seconds": self.minimum_delay_seconds,
                "maximum_seconds": self.maximum_delay_seconds,
                "last_delay_seconds": self._last_delay_seconds,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inference", "ordinary-training", "shaped-training"),
        required=True,
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--captures", type=int, default=8)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--minimum-pretrigger-delay-ms", type=float, default=0.0)
    parser.add_argument("--maximum-pretrigger-delay-ms", type=float, default=500.0)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--inference-decode-tokens", type=int, default=64)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument(
        "--shaping-backend",
        choices=("grouped-m1", "tiled-gemm", "shared-carrier"),
        default="tiled-gemm",
    )
    parser.add_argument(
        "--shared-carrier-dw-layout",
        choices=("direct", "inference-balanced", "inference-balanced-strided"),
        default="direct",
    )
    parser.add_argument(
        "--weight-gradient-schedule",
        choices=(
            "inline",
            "round-robin",
            "balanced-round-robin",
            "streaming-round-robin",
            "streaming-inference-cycle",
            "streaming-grouped",
        ),
        default="round-robin",
    )
    parser.add_argument("--streaming-dw-tasks-per-record", type=int, default=32)
    parser.add_argument("--grouped-dw-min-batch", type=int, default=4)
    parser.add_argument("--grouped-dw-max-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture/replay the complete training update as one CUDA graph.",
    )
    parser.add_argument("--actuator-commands", type=Path)
    parser.add_argument("--actuator-width", type=int, default=768)
    parser.add_argument(
        "--actuator-width-commands",
        type=Path,
        help="Optional one-dimensional .npy width schedule matching --actuator-commands.",
    )
    parser.add_argument("--actuator-repetitions-per-update", type=int, default=1)
    parser.add_argument(
        "--actuator-bin-duration-us",
        type=float,
        default=0.0,
        help="Replay each actuator command in a fixed-duration bin; zero replays back-to-back.",
    )
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--kernel-launch-period-us", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.captures < 1:
        raise ValueError("--captures must be positive")
    actuator_operations: tuple[int, ...] = ()
    actuator_width_commands: tuple[int, ...] = ()
    if args.actuator_commands is not None:
        values = np.load(args.actuator_commands, allow_pickle=False)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("--actuator-commands must contain a one-dimensional schedule")
        actuator_operations = tuple(int(value) for value in values)
    if args.actuator_width_commands is not None:
        values = np.load(args.actuator_width_commands, allow_pickle=False)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("--actuator-width-commands must contain a one-dimensional schedule")
        actuator_width_commands = tuple(int(value) for value in values)
    config = StrictWorkloadConfig(
        mode=args.mode,
        session_id=args.session_id,
        model=args.model,
        seed=args.seed,
        training_batch_size=args.training_batch_size,
        training_sequence_length=args.sequence_length,
        inference_batch_size=args.inference_batch_size,
        inference_decode_tokens=args.inference_decode_tokens,
        learning_rate=args.learning_rate,
        tile_rows=args.tile_rows,
        shaping_backend=args.shaping_backend,
        shared_carrier_weight_gradient_layout=args.shared_carrier_dw_layout,
        weight_gradient_schedule=args.weight_gradient_schedule,
        streaming_weight_gradient_tasks_per_record=args.streaming_dw_tasks_per_record,
        grouped_weight_gradient_min_batch=args.grouped_dw_min_batch,
        grouped_weight_gradient_max_batch=args.grouped_dw_max_batch,
        cuda_graph=args.cuda_graph,
        actuator_width=args.actuator_width,
        actuator_width_commands=actuator_width_commands,
        actuator_operations=actuator_operations,
        actuator_repetitions_per_update=args.actuator_repetitions_per_update,
        actuator_bin_duration_us=args.actuator_bin_duration_us,
        optimizer_bucket_size=args.optimizer_bucket_size,
        kernel_launch_period_us=args.kernel_launch_period_us,
    )
    base_sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.duration,
            sample_rate=args.sample_rate,
            mode="burst",
            bits_per_sample=12,
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    sampler = RandomPretriggerDelaySampler(
        base_sampler,
        minimum_delay_seconds=args.minimum_pretrigger_delay_ms / 1e3,
        maximum_delay_seconds=args.maximum_pretrigger_delay_ms / 1e3,
        seed=args.seed,
    )
    workload = PersistentStrictWorkload(
        config,
        startup_timeout_seconds=args.startup_timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
    )
    process = "inference" if args.mode == "inference" else "training"
    session_root = args.output_dir / process / args.session_id
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(session_root, trace_dtype="float32"),
        retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
        workload_sync="none",
    ) as experiment:
        records = experiment.run(args.captures)
    summary = {
        "session_root": str(session_root),
        "accepted_records": len(records),
        "worker": config.metadata(),
        "capture": {
            "duration": args.duration,
            "sample_rate": args.sample_rate,
            "gain_db": args.gain_db,
            "minimum_pretrigger_delay_ms": args.minimum_pretrigger_delay_ms,
            "maximum_pretrigger_delay_ms": args.maximum_pretrigger_delay_ms,
            "actuator_commands": (None if args.actuator_commands is None else str(args.actuator_commands)),
            "actuator_width_commands": (
                None if args.actuator_width_commands is None else str(args.actuator_width_commands)
            ),
            "actuator_bin_duration_us": args.actuator_bin_duration_us,
        },
    }
    (session_root / "session_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
