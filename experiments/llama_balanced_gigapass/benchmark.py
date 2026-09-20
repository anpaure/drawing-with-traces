#!/usr/bin/env python3
"""Certify 1:1 native service and <=2x slowdown on the real H100 workload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .scheduler import BalancedServiceController, summarize_durations
from .workload import (
    DEFAULT_INFERENCE_CANDIDATES,
    DEFAULT_MODEL,
    BalancedGigaPassConfig,
    BalancedGigaPassEngine,
    load_calibration,
)


def parse_candidates(value: str) -> tuple[int, ...]:
    try:
        candidates = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("inference candidates must be comma-separated integers") from exc
    if len(candidates) < 2 or any(candidate < 1 for candidate in candidates):
        raise argparse.ArgumentTypeError("at least two positive inference candidates are required")
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--compute-seed", type=int, default=31_337)
    parser.add_argument("--training-batch-size", type=int, default=1_024)
    parser.add_argument(
        "--inference-candidates",
        type=parse_candidates,
        default=DEFAULT_INFERENCE_CANDIDATES,
        help="Comma-separated calibration candidates (default: 6144..10240 by 256)",
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
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--benchmark-rounds", type=int, default=9)
    parser.add_argument("--cycles-per-block", type=int, default=4)
    parser.add_argument("--maximum-slowdown", type=float, default=2.0)
    parser.add_argument(
        "--slowdown-measurement-tolerance",
        type=float,
        default=0.01,
        help="Relative tolerance around the 2x target for DVFS/measurement variation",
    )
    parser.add_argument("--maximum-service-imbalance", type=float, default=0.02)
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def config_from_args(args: argparse.Namespace) -> BalancedGigaPassConfig:
    calibration = None if args.calibration_input is None else load_calibration(args.calibration_input)
    return BalancedGigaPassConfig(
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
        calibration_override=calibration,
    )


def main() -> None:
    args = build_parser().parse_args()
    for name in ("benchmark_rounds", "cycles_per_block"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.maximum_slowdown) or args.maximum_slowdown <= 0:
        raise ValueError("maximum-slowdown must be finite and positive")
    if (
        not math.isfinite(args.slowdown_measurement_tolerance)
        or args.slowdown_measurement_tolerance < 0
    ):
        raise ValueError("slowdown-measurement-tolerance must be finite and non-negative")
    if (
        not math.isfinite(args.maximum_service_imbalance)
        or args.maximum_service_imbalance < 0
    ):
        raise ValueError("maximum-service-imbalance must be finite and non-negative")

    engine = BalancedGigaPassEngine(config_from_args(args))
    try:
        engine.setup()
        calibration = engine.calibration
        if calibration is None:
            raise RuntimeError("engine setup completed without calibration")
        if args.calibration_output is not None:
            args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            args.calibration_output.write_text(
                json.dumps({"calibration": calibration.to_dict()}, indent=2) + "\n"
            )

        planner = BalancedServiceController(calibration)
        schedules = [
            planner.plan(args.cycles_per_block) for _ in range(args.benchmark_rounds)
        ]
        gpu: dict[str, list[float]] = {"inference": [], "training": [], "mixed": []}
        wall: dict[str, list[float]] = {"inference": [], "training": [], "mixed": []}

        for round_index, schedule in enumerate(schedules):
            def run(mode: str):
                if mode == "inference":
                    return engine.run_dedicated_inference_batches(schedule)
                if mode == "training":
                    return engine.run_dedicated_training_cycles(len(schedule))
                if mode == "mixed":
                    return engine.run_mixed_batches(schedule)
                raise AssertionError(f"unknown benchmark mode: {mode}")

            orders = (
                ("inference", "training", "mixed"),
                ("mixed", "inference", "training"),
                ("training", "mixed", "inference"),
            )
            for mode in orders[round_index % len(orders)]:
                timing = run(mode)
                gpu[mode].append(timing.gpu_seconds)
                wall[mode].append(timing.wall_seconds)

        gpu_summary = summarize_durations(gpu["inference"], gpu["training"], gpu["mixed"])
        wall_summary = summarize_durations(
            wall["inference"], wall["training"], wall["mixed"]
        )
        snapshot = engine.snapshot()
        total_cycles = args.benchmark_rounds * args.cycles_per_block
        inference_tokens = sum(sum(schedule) for schedule in schedules) * engine.config.sequence_length
        training_tokens = (
            total_cycles
            * engine.config.training_batch_size
            * engine.config.sequence_length
        )
        mixed_wall_seconds = wall_summary["total_seconds"]["mixed"]
        useful_inference_flops = 2 * engine.parameter_count * inference_tokens
        useful_training_flops = 6 * engine.parameter_count * training_tokens

        accepted_slowdown = args.maximum_slowdown * (
            1.0 + args.slowdown_measurement_tolerance
        )
        point_target_checks = {
            "gpu_slowdown_at_or_below_target": (
                gpu_summary["max_slowdown"] <= args.maximum_slowdown
            ),
            "wall_slowdown_at_or_below_target": (
                wall_summary["max_slowdown"] <= args.maximum_slowdown
            ),
        }
        checks = {
            "finite_training_loss": math.isfinite(snapshot["last_loss"]),
            "nonzero_parameter_change": snapshot["parameter_probe_delta_linf"] > 0,
            "gpu_service_ratio_within_limit": (
                gpu_summary["native_service_relative_imbalance"]
                <= args.maximum_service_imbalance
            ),
            "wall_service_ratio_within_limit": (
                wall_summary["native_service_relative_imbalance"]
                <= args.maximum_service_imbalance
            ),
            "gpu_slowdown_within_measurement_tolerance": (
                gpu_summary["max_slowdown"] <= accepted_slowdown
            ),
            "wall_slowdown_within_measurement_tolerance": (
                wall_summary["max_slowdown"] <= accepted_slowdown
            ),
        }
        result = {
            "passed": all(checks.values()),
            "checks": checks,
            "point_target_checks": point_target_checks,
            "limits": {
                "target_maximum_slowdown": args.maximum_slowdown,
                "slowdown_measurement_tolerance_fraction": (
                    args.slowdown_measurement_tolerance
                ),
                "accepted_measured_slowdown": accepted_slowdown,
                "maximum_service_imbalance": args.maximum_service_imbalance,
            },
            "config": {
                "model": engine.config.model,
                "attention_backend": engine.config.attention_backend,
                "training_batch_size": engine.config.training_batch_size,
                "sequence_length": engine.config.sequence_length,
                "benchmark_rounds": args.benchmark_rounds,
                "cycles_per_block": args.cycles_per_block,
            },
            "calibration": calibration.to_dict(),
            "planned_service": planner.state_dict(),
            "gpu_timing": gpu_summary,
            "wall_timing": wall_summary,
            "work": {
                "inference_tokens": inference_tokens,
                "training_tokens": training_tokens,
                "training_updates": total_cycles,
                "useful_inference_flops": useful_inference_flops,
                "useful_training_flops": useful_training_flops,
                "useful_total_flops_per_second": (
                    useful_inference_flops + useful_training_flops
                )
                / mixed_wall_seconds,
            },
            "snapshot": snapshot,
            "engine": engine.metadata(),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not result["passed"] and not args.allow_failure:
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            raise RuntimeError(f"balanced giga-pass certification failed: {failed}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
