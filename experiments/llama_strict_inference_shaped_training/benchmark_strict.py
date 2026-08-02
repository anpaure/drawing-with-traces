#!/usr/bin/env python3
"""Benchmark one persistent strict workload, excluding model startup time."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np

from .strict_workloads import (
    DEFAULT_MODEL,
    StrictWorkloadConfig,
    start_workload,
    stop_workload,
    wait_for_ready,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inference", "ordinary-training", "shaped-training"),
        required=True,
    )
    parser.add_argument("--session-id", default="benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--observe-seconds", type=float, default=8.0)
    parser.add_argument("--replays-per-heartbeat", type=int, default=8)
    parser.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--inference-decode-tokens", type=int, default=64)
    parser.add_argument("--tile-rows", type=int, default=128)
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
    parser.add_argument("--shaping-backend", choices=("grouped-m1", "tiled-gemm"), default="tiled-gemm")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actuator-commands", type=Path)
    parser.add_argument("--actuator-width", type=int, default=768)
    parser.add_argument("--actuator-repetitions-per-update", type=int, default=1)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--kernel-launch-period-us", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.observe_seconds <= 0:
        raise SystemExit("--observe-seconds must be positive")
    actuator_operations: tuple[int, ...] = ()
    if args.actuator_commands is not None:
        values = np.load(args.actuator_commands, allow_pickle=False)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("--actuator-commands must contain a one-dimensional schedule")
        actuator_operations = tuple(int(value) for value in values)
    config = StrictWorkloadConfig(
        mode=args.mode,
        session_id=args.session_id,
        model=args.model,
        training_batch_size=args.training_batch_size,
        training_sequence_length=args.sequence_length,
        inference_batch_size=args.inference_batch_size,
        inference_decode_tokens=args.inference_decode_tokens,
        tile_rows=args.tile_rows,
        learning_rate=args.learning_rate,
        shaping_backend=args.shaping_backend,
        weight_gradient_schedule=args.weight_gradient_schedule,
        streaming_weight_gradient_tasks_per_record=args.streaming_dw_tasks_per_record,
        grouped_weight_gradient_min_batch=args.grouped_dw_min_batch,
        grouped_weight_gradient_max_batch=args.grouped_dw_max_batch,
        cuda_graph=args.cuda_graph,
        actuator_width=args.actuator_width,
        actuator_operations=actuator_operations,
        actuator_repetitions_per_update=args.actuator_repetitions_per_update,
        optimizer_bucket_size=args.optimizer_bucket_size,
        replays_per_heartbeat=args.replays_per_heartbeat,
        kernel_launch_period_us=args.kernel_launch_period_us,
    )
    process, stop_event, messages = start_workload(config)
    ready = wait_for_ready(process, messages, timeout_seconds=args.startup_timeout_seconds)
    print(json.dumps(ready, indent=2), flush=True)
    deadline = time.monotonic() + args.observe_seconds
    heartbeats: list[dict] = []
    try:
        while time.monotonic() < deadline:
            try:
                message = messages.get(timeout=min(1.0, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if message.get("event") == "error":
                raise RuntimeError(message.get("traceback") or message.get("error"))
            if message.get("event") == "heartbeat":
                heartbeats.append(message)
                print(json.dumps(message), flush=True)
    finally:
        stop_workload(process, stop_event)
    if not heartbeats:
        raise RuntimeError("workload produced no heartbeat during the observation interval")
    result = {
        "config": config.metadata(),
        "ready": ready,
        "final": heartbeats[-1],
        "observation_seconds": args.observe_seconds,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
