#!/usr/bin/env python3
"""Capture the role-blind fused workload with a ChipWhisperer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sidecapture as sc

from ..llama_strict_inference_shaped_training.capture_strict import (
    RandomPretriggerDelaySampler,
)
from .workload import DEFAULT_MODEL, DualRoleConfig, PersistentDualRoleWorkload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inference", "training"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--captures", type=int, default=8)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--minimum-pretrigger-delay-ms", type=float, default=0.0)
    parser.add_argument("--maximum-pretrigger-delay-ms", type=float, default=5000.0)
    parser.add_argument("--compute-seed", type=int, default=31_337)
    parser.add_argument("--capture-seed", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    parser.add_argument("--training-batch-size", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--data-ring-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--iterations-per-heartbeat", type=int, default=1)
    parser.add_argument("--period-profile-output", default="")
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
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
    if args.captures < 1:
        raise ValueError("captures must be positive")
    config = config_from_args(args)
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
        seed=args.capture_seed,
    )
    workload = PersistentDualRoleWorkload(
        config,
        startup_timeout_seconds=args.startup_timeout_seconds,
        shutdown_timeout_seconds=args.shutdown_timeout_seconds,
    )
    session_root = args.output_dir / config.role / config.session_id
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
            "compute_seed": args.compute_seed,
            "capture_seed": args.capture_seed,
        },
    }
    (session_root / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
