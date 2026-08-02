#!/usr/bin/env python3
"""Refine a real Llama-gradient power trace toward a measured inference target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sidecapture as sc

from drawing_with_traces.fast import (
    FastCalibration,
    FastRefinementController,
    analyze_fast_drawing,
    normalized_curve_metrics,
)

from .trace_drawing import TransformerGradientCalibrationWorkload


def render_iteration(
    output: Path,
    target: np.ndarray,
    measured_raw: np.ndarray,
    measured_smoothed: np.ndarray,
    metrics: dict,
    bin_ms: float,
) -> None:
    x = (np.arange(target.size) + 0.5) * bin_ms
    figure, axis = plt.subplots(figsize=(12, 4.2), constrained_layout=True, facecolor="white")
    axis.plot(x, target, color="#1D4ED8", linestyle="--", lw=1.4, label="Inference target")
    axis.plot(x, measured_raw, color="#FCA5A5", lw=0.7, alpha=0.65, label="Measured bins")
    axis.plot(x, measured_smoothed, color="#DC2626", lw=1.7, label="Median scored curve")
    axis.set(
        xlim=(0, target.size * bin_ms),
        xlabel="time (ms)",
        ylabel="normalized ChipWhisperer activity",
        title=(
            f"Real Llama-gradient ILC · multiscale fidelity "
            f"{metrics['multiscale_shape_accuracy_percent']:.1f}%"
        ),
    )
    low = min(-0.05, float(measured_smoothed.min()) - 0.05)
    high = max(1.05, float(measured_smoothed.max()) + 0.05)
    axis.set_ylim(low, high)
    axis.grid(color="#E5E7EB", lw=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=3)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--captures-per-iteration", type=int, default=2)
    parser.add_argument("--bin-ms", type=float, default=1.0)
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument(
        "--execution-backend",
        choices=("cuda-graph", "python-queued"),
        default="cuda-graph",
    )
    parser.add_argument("--graph-target-utilization", type=float, default=0.82)
    parser.add_argument("--graph-maximum-operations", type=int, default=256)
    parser.add_argument(
        "--command-mode",
        choices=("width", "operations"),
        default="width",
    )
    parser.add_argument("--fixed-width", type=int, default=160)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    parser.add_argument("--training-sequence-length", type=int, default=1)
    parser.add_argument("--tile-rows", type=int, default=1024)
    parser.add_argument("--operation-command-quantum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6400)
    args = parser.parse_args()
    if args.iterations < 1 or args.captures_per_iteration < 1:
        raise ValueError("iterations and captures per iteration must be positive")
    target = np.load(args.target, allow_pickle=False).astype(np.float64)
    calibration = FastCalibration.from_dict(json.loads(args.calibration.read_text()))
    command_quantum = 32 if args.command_mode == "width" else args.operation_command_quantum
    minimum_command = 32 if args.command_mode == "width" else max(1, int(calibration.widths[0]))
    controller = FastRefinementController(
        target,
        calibration,
        initial_gain=0.30,
        minimum_gain=0.04,
        gain_decay=0.6,
        gain_growth=1.05,
        correction_smoothing_sigma_points=1.0,
        feedback_reference="best",
        minimum_width=minimum_command,
        command_quantum=command_quantum,
    )
    commands = controller.initial_commands()
    width_commands = (
        commands
        if args.command_mode == "width"
        else np.full(commands.shape, args.fixed_width, dtype=np.int64)
    )
    workload = TransformerGradientCalibrationWorkload(
        width_commands,
        operation_commands=(commands if args.command_mode == "operations" else None),
        bin_duration_s=args.bin_ms / 1e3,
        synchronize_each_operation=False,
        execution_backend=args.execution_backend,
        graph_target_utilization=args.graph_target_utilization,
        graph_maximum_operations=args.graph_maximum_operations,
        training_batch_size=args.training_batch_size,
        training_sequence_length=args.training_sequence_length,
        tile_rows=args.tile_rows,
        seed=args.seed,
    )
    duration_ms = (workload.total_duration_s + 0.005) * 1e3
    sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=f"{duration_ms:.3f}ms",
            sample_rate=args.sample_rate,
            mode="burst",
            bits_per_sample=12,
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    captures_root = args.output / "captures"
    history = []
    best = None
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(captures_root, trace_dtype="float32"),
        retry=sc.RetryPolicy(max_attempts=4, backoff_s=0.5),
        workload_sync="none",
    ) as experiment:
        for iteration in range(args.iterations):
            if args.command_mode == "operations":
                workload.set_program_commands(
                    np.full(commands.shape, args.fixed_width, dtype=np.int64),
                    commands,
                )
            else:
                workload.set_commands(commands)
            records = experiment.run(args.captures_per_iteration)
            raw_rows = []
            smooth_rows = []
            per_capture_metrics = []
            for record in records:
                metrics, arrays = analyze_fast_drawing(
                    captures_root,
                    target,
                    calibration,
                    smoothing_sigma_points=1.0,
                    record_index=int(record["index"]),
                )
                raw_rows.append(arrays["measured_raw"])
                smooth_rows.append(arrays["measured_smoothed"])
                per_capture_metrics.append(metrics)
            feedback_raw = np.median(np.stack(raw_rows), axis=0)
            feedback_smoothed = np.median(np.stack(smooth_rows), axis=0)
            metrics = normalized_curve_metrics(target, feedback_raw, feedback_smoothed)
            improved = controller.observe(
                commands,
                feedback_smoothed,
                normalized_rmse=metrics["multiscale_normalized_rmse"],
                shape_accuracy_percent=metrics["multiscale_shape_accuracy_percent"],
            )
            iteration_root = args.output / f"iteration_{iteration:02d}"
            iteration_root.mkdir(parents=True, exist_ok=True)
            np.save(iteration_root / "commands.npy", commands)
            np.save(iteration_root / "measured_raw.npy", feedback_raw)
            np.save(iteration_root / "measured_smoothed.npy", feedback_smoothed)
            (iteration_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (iteration_root / "capture_metrics.json").write_text(json.dumps(per_capture_metrics, indent=2))
            render_iteration(
                iteration_root / "comparison.png",
                target,
                feedback_raw,
                feedback_smoothed,
                metrics,
                args.bin_ms,
            )
            row = {
                "iteration": iteration,
                "record_indices": [int(record["index"]) for record in records],
                "commands": commands.tolist(),
                "command_mode": args.command_mode,
                "metrics": metrics,
                "improved": improved,
                "controller": controller.metadata(),
            }
            history.append(row)
            if improved:
                best = {
                    **row,
                    "iteration_root": str(iteration_root),
                    "measured_raw": feedback_raw.tolist(),
                    "measured_smoothed": feedback_smoothed.tolist(),
                }
            print(json.dumps(row), flush=True)
            next_commands = controller.next_commands()
            progress = {
                "status": "running",
                "last_completed_iteration": iteration,
                "history": history,
                "best": best,
                "next_commands": next_commands.tolist(),
                "completed_training_updates": workload.completed_updates,
            }
            progress_path = args.output / "refinement_progress.json"
            temporary_path = progress_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(progress, indent=2))
            temporary_path.replace(progress_path)
            commands = next_commands
    summary = {
        "target": str(args.target),
        "calibration": str(args.calibration),
        "capture_plan": {
            "duration_ms": duration_ms,
            "sample_rate": args.sample_rate,
            "gain_db": args.gain_db,
            "execution_backend": args.execution_backend,
            "graph_target_utilization": args.graph_target_utilization,
            "graph_maximum_operations": args.graph_maximum_operations,
            "command_mode": args.command_mode,
            "fixed_width": args.fixed_width if args.command_mode == "operations" else None,
            "training_batch_size": args.training_batch_size,
            "training_sequence_length": args.training_sequence_length,
            "tile_rows": args.tile_rows,
            "command_quantum": command_quantum,
        },
        "iterations": args.iterations,
        "captures_per_iteration": args.captures_per_iteration,
        "history": history,
        "best": best,
        "strict_invariants": workload.config.strict_invariants(),
        "completed_training_updates": workload.completed_updates,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "refinement_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
