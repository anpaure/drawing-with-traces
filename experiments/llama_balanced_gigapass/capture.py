#!/usr/bin/env python3
"""Capture the calibrated 1:1 inference/training service with SideCapture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sidecapture as sc

from ..llama_strict_inference_shaped_training.capture_strict import (
    RandomPretriggerDelaySampler,
)
from .benchmark import parse_candidates
from .workload import (
    DEFAULT_INFERENCE_CANDIDATES,
    DEFAULT_MODEL,
    BalancedGigaPassConfig,
    BalancedRoleConfig,
    PersistentBalancedGigaPassWorkload,
    load_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inference", "training"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--captures", type=int, default=8)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--minimum-pretrigger-delay-ms", type=float, default=0.0)
    parser.add_argument("--maximum-pretrigger-delay-ms", type=float, default=5000.0)
    parser.add_argument("--compute-seed", type=int, default=31_337)
    parser.add_argument("--capture-seed", type=int, default=0)
    parser.add_argument("--training-batch-size", type=int, default=1_024)
    parser.add_argument(
        "--inference-candidates",
        type=parse_candidates,
        default=DEFAULT_INFERENCE_CANDIDATES,
    )
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--data-ring-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-cycles", type=int, default=8)
    parser.add_argument("--calibration-rounds", type=int, default=4)
    parser.add_argument("--calibration-refinement-steps", type=int, default=10)
    parser.add_argument("--calibration-validation-blocks", type=int, default=5)
    parser.add_argument("--calibration-block-cycles", type=int, default=4)
    parser.add_argument("--calibration-target-imbalance", type=float, default=0.01)
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--cycles-per-heartbeat", type=int, default=4)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=30.0)
    return parser


def config_from_args(args: argparse.Namespace) -> BalancedRoleConfig:
    calibration = None if args.calibration_input is None else load_calibration(args.calibration_input)
    computation = BalancedGigaPassConfig(
        model=args.model,
        attention_backend=args.attention_backend,
        compute_seed=args.compute_seed,
        training_batch_size=args.training_batch_size,
        inference_batch_candidates=tuple(args.inference_candidates),
        sequence_length=args.sequence_length,
        data_ring_size=args.data_ring_size,
        learning_rate=args.learning_rate,
        warmup_cycles=args.warmup_cycles,
        calibration_rounds=args.calibration_rounds,
        calibration_refinement_steps=args.calibration_refinement_steps,
        calibration_validation_blocks=args.calibration_validation_blocks,
        calibration_block_cycles=args.calibration_block_cycles,
        calibration_target_imbalance=args.calibration_target_imbalance,
        cycles_per_heartbeat=args.cycles_per_heartbeat,
        calibration_override=calibration,
    )
    return BalancedRoleConfig(
        role=args.role,
        session_id=args.session_id,
        computation=computation,
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
    workload = PersistentBalancedGigaPassWorkload(
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
    session_root.mkdir(parents=True, exist_ok=True)
    (session_root / "session_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
