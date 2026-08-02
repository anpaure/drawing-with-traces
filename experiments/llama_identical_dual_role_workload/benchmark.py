#!/usr/bin/env python3
"""Benchmark one role of the identical fused inference/training workload."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

from .workload import (
    DEFAULT_MODEL,
    DualRoleConfig,
    start_workload,
    stop_workload,
    wait_for_ready,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inference", "training"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compute-seed", type=int, default=31_337)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--training-batch-size", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--data-ring-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations-per-heartbeat", type=int, default=1)
    parser.add_argument("--period-profile-output", default="")
    parser.add_argument("--observe-seconds", type=float, default=8.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> DualRoleConfig:
    return DualRoleConfig(
        role=args.role,
        session_id=args.session_id,
        model=args.model,
        compute_seed=args.compute_seed,
        inference_batch_size=args.inference_batch_size,
        training_batch_size=args.training_batch_size,
        sequence_length=args.sequence_length,
        data_ring_size=args.data_ring_size,
        learning_rate=args.learning_rate,
        optimizer_bucket_size=args.optimizer_bucket_size,
        warmup_iterations=args.warmup_iterations,
        iterations_per_heartbeat=args.iterations_per_heartbeat,
        period_profile_output=args.period_profile_output,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.observe_seconds <= 0:
        raise ValueError("observe-seconds must be positive")
    config = config_from_args(args)
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
        stop_workload(
            process,
            stop_event,
            timeout_seconds=args.shutdown_timeout_seconds,
        )
        messages.close()
    if not heartbeats:
        raise RuntimeError("dual-role workload produced no heartbeat")
    result = {
        "config": config.metadata(),
        "ready": ready,
        "final": heartbeats[-1],
        "observation_seconds": args.observe_seconds,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
