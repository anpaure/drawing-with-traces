"""CLI for target preview, H100 calibration, capture, and rendering."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sidecapture as sc
from sidecapture.errors import ConfigurationError
from sidecapture.store import atomic_json

from .analysis import (
    calibration_from_run,
    load_calibration,
    render_calibration,
    render_drawing,
    render_target_preview,
    save_calibration,
)
from .envelope import extract_envelope, save_envelope
from .fast import (
    TiledLinearTrainingWorkload,
    analyze_fast_drawing,
    calibrate_fast,
    capture_fast,
    render_fast_calibration,
    render_fast_drawing,
    save_fast_calibration,
)
from .workload import AdaptiveTrainingPowerWorkload, TrainingDutyWorkload


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--feedforward-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-scale", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--quantum-ms", type=float, default=20.0)
    parser.add_argument("--nvml-interval-ms", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=3)


def add_envelope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--points", type=int, default=120)
    parser.add_argument("--foreground-distance", type=float, default=32.0)
    parser.add_argument("--smoothing-sigma", type=float, default=1.5)


def add_adaptive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adaptive", action="store_true", help="use anticipatory NVML feedback")
    parser.add_argument("--control-ms", type=float, default=100.0)
    parser.add_argument("--advance-s", type=float, default=1.0)
    parser.add_argument("--feedback-kp", type=float, default=0.2)
    parser.add_argument("--feedback-ki", type=float, default=0.04)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drawing-with-traces",
        description="Make real model-training power follow a flattened image silhouette.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview = subparsers.add_parser("preview", help="render the lowered one-dimensional target")
    add_envelope_arguments(preview)
    preview.add_argument("--output", type=Path, required=True)

    calibrate = subparsers.add_parser("calibrate", help="measure H100 power versus training duty")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--levels", default="0,0.2,0.4,0.6,0.8,1")
    calibrate.add_argument("--segment-s", type=float, default=2.5)
    calibrate.add_argument("--lead-s", type=float, default=3.0)
    calibrate.add_argument("--tail-s", type=float, default=3.0)
    add_model_arguments(calibrate)

    draw = subparsers.add_parser("draw", help="capture the measured silhouette power curve")
    add_envelope_arguments(draw)
    draw.add_argument("--output", type=Path, required=True)
    draw.add_argument("--calibration", type=Path, required=True)
    draw.add_argument("--duration-s", type=float, default=60.0)
    draw.add_argument("--lead-s", type=float, default=3.0)
    draw.add_argument("--tail-s", type=float, default=3.0)
    draw.add_argument("--display-smoothing-s", type=float, default=0.25)
    add_model_arguments(draw)
    add_adaptive_arguments(draw)

    render = subparsers.add_parser("render", help="rerender an existing measured drawing")
    render.add_argument("--output", type=Path, required=True, help="drawing SideCapture run")
    render.add_argument("--calibration", type=Path, required=True)
    render.add_argument("--display-smoothing-s", type=float, default=0.25)

    run = subparsers.add_parser("run", help="calibrate, draw, and render end-to-end")
    add_envelope_arguments(run)
    run.add_argument("--output", type=Path, required=True, help="parent experiment directory")
    run.add_argument("--duration-s", type=float, default=60.0)
    run.add_argument("--calibration-levels", default="0,0.2,0.4,0.6,0.8,1")
    run.add_argument("--calibration-segment-s", type=float, default=2.5)
    run.add_argument("--lead-s", type=float, default=3.0)
    run.add_argument("--tail-s", type=float, default=3.0)
    run.add_argument("--display-smoothing-s", type=float, default=0.25)
    add_model_arguments(run)
    add_adaptive_arguments(run)

    fast = subparsers.add_parser(
        "fast",
        help="draw a 100 ms or 10 ms envelope with tiled gradients and ChipWhisperer",
    )
    add_envelope_arguments(fast)
    fast.add_argument("--output", type=Path, required=True, help="parent fast experiment directory")
    fast.add_argument("--duration-ms", type=float, default=100.0)
    fast.add_argument("--iterations", type=int, default=5)
    fast.add_argument("--ilc-gain", type=float, default=0.45)
    fast.add_argument("--sample-rate")
    fast.add_argument("--calibration-bin-ms", type=float)
    fast.add_argument("--lead-ms", type=float)
    fast.add_argument("--tail-ms", type=float)
    fast.add_argument("--margin-ms", type=float)
    fast.add_argument("--hidden-size", type=int, default=4096)
    fast.add_argument("--batch-size", type=int, default=512)
    fast.add_argument("--learning-rate", type=float, default=0.01)
    fast.add_argument("--gain-db", type=float, default=10.0)
    fast.add_argument("--max-attempts", type=int, default=4)
    fast.add_argument("--feature-smoothing", type=float, default=0.8)
    fast.add_argument("--seed", type=int, default=1729)
    return parser


def parse_levels(value: str) -> np.ndarray:
    try:
        levels = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float32)
    except ValueError as error:
        raise ConfigurationError("calibration levels must be comma-separated numbers") from error
    if levels.size < 3 or np.any((levels < 0) | (levels > 1)):
        raise ConfigurationError("calibration requires at least three levels between 0 and 1")
    levels = np.unique(levels)
    if levels[0] != 0 or levels[-1] != 1:
        raise ConfigurationError("calibration levels must include 0 and 1")
    return levels


def workload_kwargs(args: argparse.Namespace) -> dict:
    return {
        "quantum_s": args.quantum_ms / 1e3,
        "hidden_size": args.hidden_size,
        "feedforward_size": args.feedforward_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_scale": args.gradient_scale,
        "seed": args.seed,
    }


def capture_once(root: Path, workload: TrainingDutyWorkload, args: argparse.Namespace) -> dict:
    sampler = sc.NVMLSampler(gpu_index=0, interval_s=args.nvml_interval_ms / 1e3)
    store = sc.DirectoryStore(root, trace_dtype="float32")
    started = time.monotonic()
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=store,
        retry=sc.RetryPolicy(max_attempts=args.max_attempts),
        warmup=2,
        workload_sync="cuda",
    ) as experiment:
        plan = experiment.resolved.to_dict()
        records = experiment.run(1)
    return {
        "run": str(root.resolve()),
        "elapsed_s": time.monotonic() - started,
        "capture_plan": plan,
        "record": {
            "index": records[0]["index"],
            "attempt": records[0]["attempt"],
            "labels": records[0]["labels"],
            "health": records[0]["health"],
            "channels": sorted(records[0]["channels"]),
        },
        "model_parameters": workload.parameter_count,
    }


def run_calibration(args: argparse.Namespace) -> dict:
    root = args.output.expanduser().resolve()
    levels = parse_levels(args.levels)
    schedule = np.concatenate([levels, levels[-2::-1]])
    workload = TrainingDutyWorkload(
        schedule,
        schedule,
        profile_duration_s=float(args.segment_s) * schedule.size,
        lead_s=args.lead_s,
        tail_s=args.tail_s,
        **workload_kwargs(args),
    )
    capture = capture_once(root, workload, args)
    curve = calibration_from_run(root)
    calibration_path = save_calibration(root, curve)
    plot = render_calibration(root, curve, root / "calibration.png")
    summary = {
        **capture,
        "schedule": schedule.tolist(),
        "calibration": curve.to_dict(),
        "calibration_file": str(calibration_path),
        "plot": str(plot),
    }
    atomic_json(summary, root / "calibration_summary.json")
    return summary


def run_drawing(args: argparse.Namespace) -> dict:
    root = args.output.expanduser().resolve()
    envelope = extract_envelope(
        args.image,
        points=args.points,
        foreground_distance=args.foreground_distance,
        smoothing_sigma_points=args.smoothing_sigma,
    )
    save_envelope(root, envelope)
    preview = render_target_preview(envelope, root / "target_envelope.png")
    curve = load_calibration(args.calibration)
    duties = curve.duties_for(envelope.values)
    if args.adaptive:
        workload = AdaptiveTrainingPowerWorkload(
            envelope.values,
            calibration_duty=curve.duty,
            calibration_power_w=curve.monotonic_power_w,
            profile_duration_s=args.duration_s,
            control_s=args.control_ms / 1e3,
            advance_s=args.advance_s,
            feedback_kp=args.feedback_kp,
            feedback_ki=args.feedback_ki,
            lead_s=args.lead_s,
            tail_s=args.tail_s,
            **workload_kwargs(args),
        )
    else:
        workload = TrainingDutyWorkload(
            envelope.values,
            duties,
            profile_duration_s=args.duration_s,
            lead_s=args.lead_s,
            tail_s=args.tail_s,
            **workload_kwargs(args),
        )
    capture = capture_once(root, workload, args)
    plot, metrics = render_drawing(
        root,
        curve,
        smooth_s=args.display_smoothing_s,
    )
    summary = {
        **capture,
        "target_preview": str(preview),
        "measured_plot": str(plot),
        "metrics": metrics,
        "calibration": str(args.calibration),
        "duty_min": float(duties.min()),
        "duty_max": float(duties.max()),
        "adaptive_controller": bool(args.adaptive),
    }
    atomic_json(summary, root / "drawing_summary.json")
    return summary


def run_all(args: argparse.Namespace) -> dict:
    parent = args.output.expanduser().resolve()
    calibration_root = parent / "calibration"
    drawing_root = parent / "drawing"
    calibration_args = argparse.Namespace(**vars(args))
    calibration_args.output = calibration_root
    calibration_args.levels = args.calibration_levels
    calibration_args.segment_s = args.calibration_segment_s
    calibration = run_calibration(calibration_args)
    drawing_args = argparse.Namespace(**vars(args))
    drawing_args.output = drawing_root
    drawing_args.calibration = calibration_root / "calibration.json"
    drawing = run_drawing(drawing_args)
    summary = {"calibration": calibration, "drawing": drawing}
    atomic_json(summary, parent / "experiment_summary.json")
    return summary


def run_fast(args: argparse.Namespace) -> dict:
    if args.duration_ms <= 0 or args.duration_ms > 200:
        raise ConfigurationError("fast duration must be positive and at most 200 ms")
    if args.iterations < 1:
        raise ConfigurationError("fast iterations must be at least 1")
    short = args.duration_ms <= 20
    points = args.points
    if points == 120:  # argparse's generic default; choose a realizable fast resolution.
        points = 40 if short else 60
    sample_rate = args.sample_rate or ("10MHz" if short else "1.5MHz")
    calibration_bin_ms = args.calibration_bin_ms or (
        args.duration_ms / points if short else 2.0
    )
    lead_ms = args.lead_ms if args.lead_ms is not None else (1.0 if short else 2.0)
    tail_ms = args.tail_ms if args.tail_ms is not None else (1.0 if short else 2.0)
    margin_ms = args.margin_ms if args.margin_ms is not None else (3.0 if short else 5.0)
    if args.duration_ms / points < 0.2:
        raise ConfigurationError(
            f"{points} points in {args.duration_ms:g} ms leaves less than 0.2 ms per tile bin; "
            "reduce --points"
        )
    parent = args.output.expanduser().resolve()
    envelope = extract_envelope(
        args.image,
        points=points,
        foreground_distance=args.foreground_distance,
        smoothing_sigma_points=args.smoothing_sigma,
    )
    widths = np.asarray([0, 32, 64, 128, 256, 512, 1024, 2048, args.hidden_size], dtype=np.int64)
    widths = np.unique(widths[widths <= args.hidden_size])
    calibration_commands = (
        np.repeat(widths, 4)
        if short
        else np.concatenate([widths, widths[-2::-1]])
    )
    calibration_root = parent / "calibration"
    calibration_workload = TiledLinearTrainingWorkload(
        calibration_commands,
        profile_duration_s=calibration_bin_ms / 1e3 * calibration_commands.size,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lead_s=lead_ms / 1e3,
        tail_s=tail_ms / 1e3,
        seed=args.seed,
    )
    calibration_capture = capture_fast(
        calibration_root,
        calibration_workload,
        sample_rate=sample_rate,
        margin_s=margin_ms / 1e3,
        gain_db=args.gain_db,
        max_attempts=args.max_attempts,
    )
    calibration = calibrate_fast(calibration_root)
    calibration_file = save_fast_calibration(calibration_root, calibration)
    calibration_plot = render_fast_calibration(
        calibration,
        calibration_root / "fast_calibration.png",
    )
    commands = calibration.commands_for(envelope.values)
    history = []
    best = None
    for iteration in range(args.iterations):
        iteration_root = parent / f"iteration_{iteration:02d}"
        save_envelope(iteration_root, envelope)
        workload = TiledLinearTrainingWorkload(
            commands,
            profile_duration_s=args.duration_ms / 1e3,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lead_s=lead_ms / 1e3,
            tail_s=tail_ms / 1e3,
            seed=args.seed,
        )
        capture = capture_fast(
            iteration_root,
            workload,
            sample_rate=sample_rate,
            margin_s=margin_ms / 1e3,
            gain_db=args.gain_db,
            max_attempts=args.max_attempts,
        )
        plot, metrics = render_fast_drawing(
            iteration_root,
            envelope.values,
            envelope,
            calibration,
            smoothing_sigma_points=args.feature_smoothing,
        )
        _, arrays = analyze_fast_drawing(
            iteration_root,
            envelope.values,
            calibration,
            smoothing_sigma_points=args.feature_smoothing,
        )
        result = {
            "iteration": iteration,
            "capture": capture,
            "metrics": metrics,
            "plot": str(plot),
            "commands": commands.tolist(),
        }
        history.append(result)
        if best is None or metrics["normalized_rmse"] < best["metrics"]["normalized_rmse"]:
            best = result
        width_indices = np.asarray(
            [int(np.argmin(np.abs(calibration.widths - width))) for width in commands],
            dtype=np.float64,
        )
        error = envelope.values - arrays["measured_smoothed"]
        corrected = np.clip(
            width_indices + args.ilc_gain * error * (calibration.widths.size - 1),
            0,
            calibration.widths.size - 1,
        )
        commands = calibration.widths[np.rint(corrected).astype(int)].astype(np.int64)
    summary = {
        "duration_ms": args.duration_ms,
        "points": points,
        "sample_rate": sample_rate,
        "model": {
            "type": "exact tiled linear gradient",
            "parameters": args.hidden_size * args.hidden_size,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
        },
        "calibration": {
            "capture": calibration_capture,
            "file": str(calibration_file),
            "plot": str(calibration_plot),
            "curve": calibration.to_dict(),
        },
        "iterations": history,
        "best_iteration": best["iteration"],
        "best_metrics": best["metrics"],
        "best_plot": best["plot"],
    }
    atomic_json(summary, parent / "fast_experiment_summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        envelope = extract_envelope(
            args.image,
            points=args.points,
            foreground_distance=args.foreground_distance,
            smoothing_sigma_points=args.smoothing_sigma,
        )
        result = {
            "preview": str(render_target_preview(envelope, args.output)),
            "target": envelope.metadata(),
        }
    elif args.command == "calibrate":
        result = run_calibration(args)
    elif args.command == "draw":
        result = run_drawing(args)
    elif args.command == "render":
        plot, metrics = render_drawing(
            args.output,
            load_calibration(args.calibration),
            smooth_s=args.display_smoothing_s,
        )
        result = {"measured_plot": str(plot), "metrics": metrics}
    elif args.command == "run":
        result = run_all(args)
    else:
        result = run_fast(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
