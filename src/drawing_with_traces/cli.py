"""CLI for target preview, H100 calibration, capture, and rendering."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import sidecapture as sc
from sidecapture.errors import ConfigurationError
from sidecapture.store import atomic_json, atomic_numpy

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
    FastRefinementController,
    TiledLinearTrainingWorkload,
    TiledResidualMLPTrainingWorkload,
    analyze_fast_drawing,
    calibrate_fast,
    capture_fast,
    dense_calibration_widths,
    interleaved_calibration_commands,
    normalized_curve_metrics,
    render_fast_calibration,
    render_fast_drawing,
    render_training_comparison,
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


def add_envelope_arguments(
    parser: argparse.ArgumentParser,
    *,
    positional_image: bool = False,
    points_default: int = 120,
) -> None:
    if positional_image:
        parser.add_argument("image", type=Path, help="PNG/JPEG silhouette to draw")
    else:
        parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--points", type=int, default=points_default)
    parser.add_argument("--foreground-distance", type=float, default=32.0)
    parser.add_argument("--smoothing-sigma", type=float, default=1.5)
    parser.add_argument(
        "--silhouette-mode",
        choices=("height", "upper-boundary", "lower-boundary"),
        default="height",
        help="column height, top edge, or bottom edge to encode as power",
    )


def add_adaptive_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adaptive", action="store_true", help="use anticipatory NVML feedback")
    parser.add_argument("--control-ms", type=float, default=100.0)
    parser.add_argument("--advance-s", type=float, default=1.0)
    parser.add_argument("--feedback-kp", type=float, default=0.2)
    parser.add_argument("--feedback-ki", type=float, default=0.04)


def add_fast_arguments(
    parser: argparse.ArgumentParser,
    *,
    duration_ms: float,
    iterations: int,
    ilc_gain: float,
    target_accuracy: float | None,
    batch_size: int = 512,
) -> None:
    parser.add_argument("--output", type=Path, required=True, help="parent experiment directory")
    parser.add_argument("--duration-ms", type=float, default=duration_ms)
    parser.add_argument(
        "--iterations",
        "--max-refinements",
        dest="iterations",
        type=int,
        default=iterations,
        help="maximum number of accepted drawing captures",
    )
    parser.add_argument("--target-accuracy", type=float, default=target_accuracy)
    parser.add_argument("--ilc-gain", type=float, default=ilc_gain)
    parser.add_argument("--minimum-ilc-gain", type=float, default=0.05)
    parser.add_argument("--ilc-gain-decay", type=float, default=0.5)
    parser.add_argument("--ilc-gain-growth", type=float, default=1.05)
    parser.add_argument(
        "--ilc-feedback-reference",
        choices=("best", "latest"),
        default="best",
        help="correct from the best trace, or track drift from the latest median trace",
    )
    parser.add_argument(
        "--ilc-correction-smoothing",
        type=float,
        help="Gaussian sigma for the ILC command correction, in control points",
    )
    parser.add_argument("--sample-rate")
    parser.add_argument("--calibration-bin-ms", type=float)
    parser.add_argument("--lead-ms", type=float)
    parser.add_argument("--tail-ms", type=float)
    parser.add_argument("--margin-ms", type=float)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--training-model",
        choices=("linear", "residual-mlp"),
        default="linear",
        help="exact-gradient model used by the tiled drawing engine",
    )
    parser.add_argument("--residual-depth", type=int, default=14)
    parser.add_argument("--residual-scale", type=float, default=0.125)
    parser.add_argument("--residual-learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--drawing-layer-count",
        type=int,
        help=(
            "number of leading residual layers used for repeated in-profile gradient "
            "tiles; all layers are still completed and updated after capture"
        ),
    )
    parser.add_argument(
        "--fresh-training-batches",
        action="store_true",
        help="cycle through a precomputed teacher-labeled batch pool after accepted steps",
    )
    parser.add_argument("--training-batch-pool-size", type=int, default=32)
    parser.add_argument(
        "--skip-autograd-verification",
        action="store_true",
        help="skip the setup-time handwritten-gradient versus autograd check",
    )
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--replicates-per-refinement",
        type=int,
        default=1,
        help="real training captures per ILC command; feedback uses their pointwise median",
    )
    parser.add_argument(
        "--feature-smoothing",
        type=float,
        help="smoothing in control points (default: 2 for timed, 0.8 otherwise)",
    )
    parser.add_argument(
        "--trace-feature",
        choices=("rms", "std", "mean_abs", "diff_rms", "q98_q02_span"),
        help="preselect the ADC feature; timed mode defaults to RMS",
    )
    parser.add_argument("--tile-repeats", type=int, choices=(1, 2), default=1)
    parser.add_argument("--reference-repeats", type=int, choices=range(0, 9), default=0)
    parser.add_argument(
        "--minimum-tile-width",
        type=int,
        help="minimum real gradient tile width (timed default: 128; operation default: 32)",
    )
    parser.add_argument(
        "--maximum-tile-width",
        type=int,
        help="maximum tile width (timed default: 4096; operation default: 256)",
    )
    parser.add_argument("--capture-window-ms", type=float, default=25.0)
    parser.add_argument(
        "--engine",
        choices=("operation", "timed"),
        default="operation",
        help="one block per position, or sustain every position for a fixed time window",
    )
    parser.add_argument("--seed", type=int, default=1729)


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
    add_fast_arguments(
        fast,
        duration_ms=100.0,
        iterations=5,
        ilc_gain=0.45,
        target_accuracy=None,
    )

    draw_png = subparsers.add_parser(
        "draw-png",
        help="calibrate and refine a PNG on real ChipWhisperer hardware until accurate",
    )
    add_envelope_arguments(draw_png, positional_image=True, points_default=120)
    add_fast_arguments(
        draw_png,
        duration_ms=10.0,
        iterations=12,
        ilc_gain=0.35,
        target_accuracy=95.0,
        batch_size=256,
    )
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
        extraction_mode=args.silhouette_mode,
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


def _record_summary(record: dict, performance: dict | None = None) -> dict:
    return {
        "index": record["index"],
        "attempt": record["attempt"],
        "labels": record["labels"],
        "health": record["health"],
        "training_performance": performance,
    }


def _fast_analysis_policy(args: argparse.Namespace, *, engine: str) -> dict:
    """Resolve analysis choices before any drawing trace is captured."""

    timed = engine == "timed"
    return {
        "preferred_feature": args.trace_feature or ("rms" if timed else None),
        "smoothing_sigma_points": (
            float(args.feature_smoothing)
            if args.feature_smoothing is not None
            else (2.0 if timed else 0.8)
        ),
        "normalization": "robust" if timed else "calibration",
        "selection_rmse_key": (
            "multiscale_normalized_rmse" if timed else "normalized_rmse"
        ),
        "selection_accuracy_key": (
            "multiscale_shape_accuracy_percent"
            if timed
            else "shape_accuracy_percent"
        ),
        "selected_before_drawing": True,
    }


def run_draw_png_timed(args: argparse.Namespace) -> dict:
    """Sustain each distinct target position for an equal-duration training window."""

    if args.duration_ms <= 0 or args.iterations < 1:
        raise ConfigurationError("duration and maximum refinements must be positive")
    if args.replicates_per_refinement < 1:
        raise ConfigurationError("replicates per refinement must be at least one")
    if args.target_accuracy is not None and not 0 < args.target_accuracy <= 100:
        raise ConfigurationError("target accuracy must be greater than 0 and at most 100")
    bin_duration_ms = args.duration_ms / args.points
    if bin_duration_ms < 0.2:
        raise ConfigurationError(
            f"{args.points} positions in {args.duration_ms:g} ms leaves only "
            f"{bin_duration_ms:.3f} ms per position; use a longer duration or fewer positions"
        )
    lead_ms = args.lead_ms if args.lead_ms is not None else 2.0
    tail_ms = args.tail_ms if args.tail_ms is not None else 2.0
    required_window_ms = args.duration_ms + lead_ms + tail_ms
    if args.capture_window_ms < required_window_ms:
        raise ConfigurationError(
            f"capture window {args.capture_window_ms:g} ms is shorter than the "
            f"{required_window_ms:g} ms timed workload; increase --capture-window-ms"
        )
    parent = args.output.expanduser().resolve()
    if (parent / "captures" / "manifest.json").exists():
        raise ConfigurationError(
            f"{parent} already contains a hardware run; choose a new --output"
        )
    envelope = extract_envelope(
        args.image,
        points=args.points,
        foreground_distance=args.foreground_distance,
        smoothing_sigma_points=args.smoothing_sigma,
        extraction_mode=args.silhouette_mode,
    )
    save_envelope(parent, envelope)
    target_preview = render_target_preview(envelope, parent / "target_envelope.png")
    analysis_policy = _fast_analysis_policy(args, engine="timed")
    minimum_tile_width = args.minimum_tile_width or 128
    maximum_tile_width = args.maximum_tile_width or min(args.hidden_size, 4096)
    if minimum_tile_width % 32 or maximum_tile_width % 32:
        raise ConfigurationError("tile width bounds must be multiples of 32")
    if not 32 <= minimum_tile_width < maximum_tile_width <= args.hidden_size:
        raise ConfigurationError(
            "tile widths must satisfy 32 <= minimum < maximum <= hidden size"
        )
    correction_smoothing = (
        float(args.ilc_correction_smoothing)
        if args.ilc_correction_smoothing is not None
        else 0.0
    )
    started = time.monotonic()
    sample_rate = args.sample_rate or "1.5MHz"
    widths = dense_calibration_widths(minimum_tile_width, maximum_tile_width)
    calibration_repeats = max(4, int(np.ceil(3.2 / bin_duration_ms)))
    calibration_commands = interleaved_calibration_commands(
        widths,
        repeats=calibration_repeats,
        seed=args.seed,
    )
    common_workload = {
        "profile_duration_s": bin_duration_ms / 1e3 * calibration_commands.size,
        "hidden_size": args.hidden_size,
        "batch_size": args.batch_size,
        "lead_s": lead_ms / 1e3,
        "tail_s": tail_ms / 1e3,
        "active_lead_width": minimum_tile_width,
        "seed": args.seed,
        "schedule_mode": "timed-repeat",
    }
    if args.training_model == "residual-mlp":
        workload = TiledResidualMLPTrainingWorkload(
            calibration_commands,
            depth=args.residual_depth,
            residual_scale=args.residual_scale,
            drawing_layer_count=args.drawing_layer_count,
            fresh_batch_each_step=args.fresh_training_batches,
            training_batch_pool_size=args.training_batch_pool_size,
            verify_autograd=not args.skip_autograd_verification,
            learning_rate=args.residual_learning_rate,
            **common_workload,
        )
    else:
        workload = TiledLinearTrainingWorkload(
            calibration_commands,
            output_size=args.hidden_size,
            learning_rate=args.learning_rate,
            **common_workload,
        )
    captures_root = parent / "captures"
    sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.capture_window_ms / 1e3,
            sample_rate=sample_rate,
            mode="burst",
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    history = []
    refinement_rounds = []
    best = None
    stop_reason = "maximum_refinements_reached"
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(captures_root, trace_dtype="float16"),
        retry=sc.RetryPolicy(max_attempts=args.max_attempts, backoff_s=0.2),
        warmup=2,
        workload_sync="cuda",
    ) as experiment:
        capture_plan = experiment.resolved.to_dict()
        calibration_record = experiment.run(1)[0]
        calibration_performance = workload.last_performance
        calibration = calibrate_fast(
            captures_root,
            record_index=int(calibration_record["index"]),
            preferred_feature=analysis_policy["preferred_feature"],
        )
        calibration_root = parent / "calibration"
        calibration_file = save_fast_calibration(calibration_root, calibration)
        calibration_plot = render_fast_calibration(
            calibration,
            calibration_root / "fast_calibration.png",
        )
        controller = FastRefinementController(
            envelope.values,
            calibration,
            target_accuracy_percent=args.target_accuracy,
            initial_gain=args.ilc_gain,
            minimum_gain=args.minimum_ilc_gain,
            gain_decay=args.ilc_gain_decay,
            gain_growth=args.ilc_gain_growth,
            minimum_width=minimum_tile_width,
            correction_smoothing_sigma_points=correction_smoothing,
            feedback_reference=args.ilc_feedback_reference,
        )
        commands = controller.initial_commands()
        for iteration in range(args.iterations):
            workload.set_schedule(
                commands,
                profile_duration_s=args.duration_ms / 1e3,
                schedule_mode="timed-repeat",
                tile_repeats=1,
            )
            iteration_root = parent / f"iteration_{iteration:02d}"
            save_envelope(iteration_root, envelope)
            objective_key = analysis_policy["selection_rmse_key"]
            accuracy_key = analysis_policy["selection_accuracy_key"]
            replicate_results = []
            measured_raw = []
            measured_smoothed = []
            for replicate in range(args.replicates_per_refinement):
                record = experiment.run(1)[0]
                performance = workload.last_performance
                record_index = int(record["index"])
                analysis_root = (
                    iteration_root
                    if args.replicates_per_refinement == 1
                    else iteration_root / f"replicate_{replicate:02d}"
                )
                save_envelope(analysis_root, envelope)
                plot, metrics = render_fast_drawing(
                    captures_root,
                    envelope.values,
                    envelope,
                    calibration,
                    output=analysis_root / "fast_measured_silhouette.png",
                    smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
                    record_index=record_index,
                    analysis_root=analysis_root,
                    feature_name=calibration.feature_name,
                    normalization=analysis_policy["normalization"],
                )
                _, arrays = analyze_fast_drawing(
                    captures_root,
                    envelope.values,
                    calibration,
                    smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
                    record_index=record_index,
                    feature_name=calibration.feature_name,
                    normalization=analysis_policy["normalization"],
                )
                atomic_json(performance, analysis_root / "training_performance.json")
                result = {
                    "iteration": iteration,
                    "replicate": replicate,
                    "record": _record_summary(record, performance),
                    "metrics": metrics,
                    "plot": str(plot),
                    "commands": commands.tolist(),
                    "controller_gain": controller.gain,
                }
                history.append(result)
                replicate_results.append(result)
                measured_raw.append(arrays["measured_raw"])
                measured_smoothed.append(arrays["measured_smoothed"])
                capture_improved = best is None or metrics[objective_key] < best["metrics"][
                    objective_key
                ]
                result["improved_best_single_trace"] = capture_improved
                if capture_improved:
                    best = result

            feedback_raw = np.median(np.stack(measured_raw), axis=0)
            feedback_smoothed = np.median(np.stack(measured_smoothed), axis=0)
            feedback_metrics = normalized_curve_metrics(
                envelope.values,
                feedback_raw,
                feedback_smoothed,
            )
            atomic_numpy(feedback_raw, iteration_root / "feedback_measured_raw.npy")
            atomic_numpy(feedback_smoothed, iteration_root / "feedback_measured_smoothed.npy")
            atomic_json(feedback_metrics, iteration_root / "feedback_metrics.json")
            controller_improved = controller.observe(
                commands,
                feedback_smoothed,
                normalized_rmse=feedback_metrics[objective_key],
                shape_accuracy_percent=feedback_metrics[accuracy_key],
            )
            round_result = {
                "iteration": iteration,
                "commands": commands.tolist(),
                "replicate_record_indices": [
                    item["record"]["index"] for item in replicate_results
                ],
                "feedback_aggregation": "pointwise median",
                "feedback_metrics": feedback_metrics,
                "controller_improved": controller_improved,
                "controller_after_observation": controller.metadata(),
            }
            refinement_rounds.append(round_result)
            atomic_json(
                {
                    "status": "running",
                    "completed_iterations": len(refinement_rounds),
                    "drawing_captures_completed": len(history),
                    "best_iteration": best["iteration"],
                    "best_replicate": best["replicate"],
                    "best_metrics": best["metrics"],
                    "latest_feedback_metrics": feedback_metrics,
                },
                parent / "experiment_progress.json",
            )
            print(
                f"iteration {iteration}: median_fidelity={feedback_metrics[accuracy_key]:.2f}% "
                f"best_single={best['metrics'][accuracy_key]:.2f}% "
                f"objective_rmse={feedback_metrics[objective_key]:.4f} "
                f"replicates={args.replicates_per_refinement} "
                f"{'[controller improved]' if controller_improved else '[kept prior controller]'}",
                file=sys.stderr,
                flush=True,
            )
            best_reached_target = (
                args.target_accuracy is not None
                and best["metrics"][accuracy_key] >= args.target_accuracy
            )
            if controller.reached_target or best_reached_target:
                stop_reason = "target_accuracy_reached"
                break
            commands = controller.next_commands()

    promoted_plot = parent / "measured_silhouette.png"
    promoted_calibration_plot = parent / "tile_width_calibration.png"
    shutil.copy2(best["plot"], promoted_plot)
    shutil.copy2(calibration_plot, promoted_calibration_plot)
    performances = [item["record"]["training_performance"] for item in history]
    training_comparison_plot = render_training_comparison(
        performances,
        parent / "training_performance.png",
    )
    drawing_steps_per_s = float(
        np.median([item["throughput"]["drawing_steps_per_s"] for item in performances])
    )
    no_drawing_steps_per_s = float(
        np.median([item["throughput"]["no_drawing_steps_per_s"] for item in performances])
    )
    summary = {
        "engine": "persistent timed exact-gradient training",
        "elapsed_s": time.monotonic() - started,
        "requested_profile_duration_ms": args.duration_ms,
        "position_duration_ms": bin_duration_ms,
        "points": args.points,
        "sample_rate": sample_rate,
        "capture_plan": capture_plan,
        "target": envelope.metadata(),
        "target_preview": str(target_preview),
        "model": {
            "type": (
                "teacher-student residual MLP training"
                if args.training_model == "residual-mlp"
                else "teacher-student tiled linear training"
            ),
            "training_model": args.training_model,
            "parameters": workload.parameter_count,
            "input_size": args.hidden_size,
            "output_size": args.hidden_size,
            "batch_size": args.batch_size,
            "persistent_across_captures": True,
            **(
                {
                    "depth": args.residual_depth,
                    "residual_scale": args.residual_scale,
                    "drawing_layer_count": workload.drawing_layer_count,
                    "fresh_batch_each_step": args.fresh_training_batches,
                    "training_batch_pool_size": (
                        args.training_batch_pool_size
                        if args.fresh_training_batches
                        else 1
                    ),
                    "autograd_verification": workload.autograd_equivalence,
                }
                if args.training_model == "residual-mlp"
                else {}
            ),
        },
        "calibration": {
            "context_repeats_per_width": calibration_repeats,
            "record": _record_summary(calibration_record, calibration_performance),
            "file": str(calibration_file),
            "plot": str(calibration_plot),
            "curve": calibration.to_dict(),
        },
        "measurement_policy": {
            **analysis_policy,
            "resolved_feature": calibration.feature_name,
            "feature_sign": calibration.feature_sign,
            "normalization_quantiles": [0.02, 0.98],
            "uses_target_for_feature_selection": False,
            "minimum_tile_width": minimum_tile_width,
            "maximum_requested_tile_width": maximum_tile_width,
        },
        "control_policy": {
            "algorithm": "cumulative iterative learning control",
            "replicates_per_refinement": args.replicates_per_refinement,
            "feedback_aggregation": "pointwise median",
            "promoted_result_is_single_physical_trace": True,
            "active_lead_width": minimum_tile_width,
            "correction_smoothing_sigma_points": correction_smoothing,
            "correction_smoothing_ms": correction_smoothing * bin_duration_ms,
            "feedback_reference": args.ilc_feedback_reference,
            "preserves_unsmoothed_target_baseline": True,
        },
        "captures_dataset": str(captures_root),
        "iterations": history,
        "refinement_rounds": refinement_rounds,
        "iterations_completed": len(refinement_rounds),
        "drawing_captures_completed": len(history),
        "accepted_training_steps": 1 + len(history),
        "stop_reason": stop_reason,
        "controller": controller.metadata(),
        "best_iteration": best["iteration"],
        "best_replicate": best["replicate"],
        "best_record_index": best["record"]["index"],
        "best_metrics": best["metrics"],
        "best_training_performance": best["record"]["training_performance"],
        "best_plot": str(promoted_plot),
        "drawing_vs_no_drawing": {
            "drawing_median_steps_per_s": drawing_steps_per_s,
            "no_drawing_median_steps_per_s": no_drawing_steps_per_s,
            "drawing_throughput_fraction": drawing_steps_per_s / no_drawing_steps_per_s,
            "drawing_slowdown_x": no_drawing_steps_per_s / drawing_steps_per_s,
            "comparison_plot": str(training_comparison_plot),
        },
    }
    atomic_json(summary, parent / "experiment_summary.json")
    atomic_json(
        {
            "status": "complete",
            "stop_reason": stop_reason,
            "best_iteration": best["iteration"],
            "best_metrics": best["metrics"],
        },
        parent / "experiment_progress.json",
    )
    return summary


def run_draw_png(args: argparse.Namespace) -> dict:
    """Run operation-level calibration and persistent-model refinement."""

    if args.engine == "timed":
        return run_draw_png_timed(args)
    if args.training_model != "linear":
        raise ConfigurationError(
            "--training-model residual-mlp currently requires --engine timed; "
            "operation-level scheduling remains available with --training-model linear"
        )
    if args.iterations < 1:
        raise ConfigurationError("maximum refinements must be at least one")
    if args.target_accuracy is not None and not 0 < args.target_accuracy <= 100:
        raise ConfigurationError("target accuracy must be greater than 0 and at most 100")
    maximum_tile_width = args.maximum_tile_width or 256
    minimum_tile_width = args.minimum_tile_width or 32
    if maximum_tile_width < 64 or maximum_tile_width % 32:
        raise ConfigurationError("maximum tile width must be a multiple of 32 and at least 64")
    if minimum_tile_width < 32 or minimum_tile_width % 32 or minimum_tile_width >= maximum_tile_width:
        raise ConfigurationError(
            "minimum tile width must be a multiple of 32 below the maximum"
        )
    if args.capture_window_ms <= 0:
        raise ConfigurationError("capture window must be positive")
    parent = args.output.expanduser().resolve()
    if (parent / "captures" / "manifest.json").exists():
        raise ConfigurationError(
            f"{parent} already contains a hardware run; choose a new --output so raw evidence is not mixed"
        )
    envelope = extract_envelope(
        args.image,
        points=args.points,
        foreground_distance=args.foreground_distance,
        smoothing_sigma_points=args.smoothing_sigma,
        extraction_mode=args.silhouette_mode,
    )
    save_envelope(parent, envelope)
    target_preview = render_target_preview(envelope, parent / "target_envelope.png")
    analysis_policy = _fast_analysis_policy(args, engine="operation")
    correction_smoothing = (
        float(args.ilc_correction_smoothing)
        if args.ilc_correction_smoothing is not None
        else 0.0
    )
    started = time.monotonic()

    widths = np.asarray([32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=np.int64)
    widths = np.unique(widths[(widths >= minimum_tile_width) & (widths <= maximum_tile_width)])
    calibration_commands = interleaved_calibration_commands(widths, repeats=4, seed=args.seed)
    lead_ms = args.lead_ms if args.lead_ms is not None else 1.0
    tail_ms = args.tail_ms if args.tail_ms is not None else 1.0
    sample_rate = args.sample_rate or "10MHz"
    calibration_workload = TiledLinearTrainingWorkload(
        calibration_commands,
        profile_duration_s=args.duration_ms / 1e3,
        hidden_size=args.hidden_size,
        output_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lead_s=lead_ms / 1e3,
        tail_s=tail_ms / 1e3,
        seed=args.seed,
        schedule_mode="single-tile",
        tile_repeats=args.tile_repeats,
    )
    calibration_root = parent / "calibration"
    calibration_captures_root = calibration_root / "captures"
    calibration_sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.capture_window_ms / 1e3,
            sample_rate=sample_rate,
            mode="burst",
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    with sc.Experiment(
        sampler=calibration_sampler,
        workload=calibration_workload,
        store=sc.DirectoryStore(calibration_captures_root, trace_dtype="float16"),
        retry=sc.RetryPolicy(max_attempts=args.max_attempts, backoff_s=0.2),
        warmup=2,
        workload_sync="cuda",
    ) as calibration_experiment:
        calibration_capture_plan = calibration_experiment.resolved.to_dict()
        calibration_record = calibration_experiment.run(1)[0]
        calibration_performance = calibration_workload.last_performance
    calibration = calibrate_fast(
        calibration_captures_root,
        record_index=int(calibration_record["index"]),
        preferred_feature=analysis_policy["preferred_feature"],
    )
    calibration_file = save_fast_calibration(calibration_root, calibration)
    calibration_plot = render_fast_calibration(
        calibration,
        calibration_root / "fast_calibration.png",
    )
    seed_controller = FastRefinementController(envelope.values, calibration)
    seed_commands = seed_controller.initial_commands()
    seed_commands = np.maximum(seed_commands, minimum_tile_width)
    output_size = int(seed_commands.sum())
    controller = FastRefinementController(
        envelope.values,
        calibration,
        target_accuracy_percent=args.target_accuracy,
        initial_gain=args.ilc_gain,
        minimum_gain=args.minimum_ilc_gain,
        gain_decay=args.ilc_gain_decay,
        gain_growth=args.ilc_gain_growth,
        total_width=output_size,
        minimum_width=minimum_tile_width,
        correction_smoothing_sigma_points=correction_smoothing,
        feedback_reference=args.ilc_feedback_reference,
    )
    commands = controller.initial_commands()
    workload = TiledLinearTrainingWorkload(
        commands,
        profile_duration_s=args.duration_ms / 1e3,
        hidden_size=args.hidden_size,
        output_size=output_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lead_s=lead_ms / 1e3,
        tail_s=tail_ms / 1e3,
        seed=args.seed,
        schedule_mode="exact-once",
        tile_repeats=args.tile_repeats,
        reference_widths=(
            calibration.widths.astype(np.int64) if args.reference_repeats else None
        ),
        reference_repeats=args.reference_repeats,
    )
    captures_root = parent / "captures"
    sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.capture_window_ms / 1e3,
            sample_rate=sample_rate,
            mode="burst",
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    history = []
    best = None
    stop_reason = "maximum_refinements_reached"
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(captures_root, trace_dtype="float16"),
        retry=sc.RetryPolicy(max_attempts=args.max_attempts, backoff_s=0.2),
        warmup=2,
        workload_sync="cuda",
    ) as experiment:
        capture_plan = experiment.resolved.to_dict()
        for iteration in range(args.iterations):
            workload.set_schedule(
                commands,
                profile_duration_s=args.duration_ms / 1e3,
                schedule_mode="exact-once",
                tile_repeats=args.tile_repeats,
            )
            record = experiment.run(1)[0]
            performance = workload.last_performance
            record_index = int(record["index"])
            iteration_root = parent / f"iteration_{iteration:02d}"
            save_envelope(iteration_root, envelope)
            if args.reference_repeats:
                try:
                    trace_calibration = calibrate_fast(
                        captures_root,
                        record_index=record_index,
                        event_name="tile.reference",
                        preferred_feature=analysis_policy["preferred_feature"],
                    )
                    trace_calibration_source = "same-trace gradient references"
                except ConfigurationError as error:
                    trace_calibration = calibration
                    trace_calibration_source = f"global calibration fallback: {error}"
            else:
                trace_calibration = calibration
                trace_calibration_source = "global calibration"
            atomic_json(
                trace_calibration.to_dict(),
                iteration_root / "trace_calibration.json",
            )
            trace_calibration_plot = render_fast_calibration(
                trace_calibration,
                iteration_root / "trace_calibration.png",
            )
            plot, metrics = render_fast_drawing(
                captures_root,
                envelope.values,
                envelope,
                trace_calibration,
                output=iteration_root / "fast_measured_silhouette.png",
                smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
                record_index=record_index,
                analysis_root=iteration_root,
            )
            _, arrays = analyze_fast_drawing(
                captures_root,
                envelope.values,
                trace_calibration,
                smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
                record_index=record_index,
            )
            atomic_json(performance, iteration_root / "training_performance.json")
            result = {
                "iteration": iteration,
                "record": _record_summary(record, performance),
                "metrics": metrics,
                "plot": str(plot),
                "commands": commands.tolist(),
                "controller_gain": controller.gain,
                "trace_calibration_source": trace_calibration_source,
                "trace_calibration": trace_calibration.to_dict(),
                "trace_calibration_plot": str(trace_calibration_plot),
            }
            history.append(result)
            objective_key = analysis_policy["selection_rmse_key"]
            accuracy_key = analysis_policy["selection_accuracy_key"]
            if best is None or metrics[objective_key] < best["metrics"][objective_key]:
                best = result
            improved = controller.observe(
                commands,
                arrays["measured_smoothed"],
                normalized_rmse=metrics[objective_key],
                shape_accuracy_percent=metrics[accuracy_key],
            )
            result["improved_best"] = improved
            result["controller_after_observation"] = controller.metadata()
            atomic_json(
                {
                    "status": "running",
                    "completed_iterations": len(history),
                    "latest_record": record_index,
                    "best_iteration": best["iteration"],
                    "best_metrics": best["metrics"],
                    "controller": controller.metadata(),
                },
                parent / "experiment_progress.json",
            )
            print(
                f"iteration {iteration}: accuracy={metrics[accuracy_key]:.2f}% "
                f"rmse={metrics[objective_key]:.4f} "
                f"step={performance['training_step']} "
                f"drawing={performance['throughput']['drawing_steps_per_s']:.2f} steps/s "
                f"no-drawing={performance['throughput']['no_drawing_steps_per_s']:.2f} steps/s "
                f"{'[new best]' if improved else '[kept prior best]'}",
                file=sys.stderr,
                flush=True,
            )
            if controller.reached_target:
                stop_reason = "target_accuracy_reached"
                break
            commands = controller.next_commands()

    promoted_plot = parent / "measured_silhouette.png"
    promoted_calibration_plot = parent / "tile_width_calibration.png"
    promoted_trace_calibration_plot = parent / "best_trace_calibration.png"
    training_comparison_plot = render_training_comparison(
        [item["record"]["training_performance"] for item in history],
        parent / "training_performance.png",
    )
    drawing_steps_per_s = float(
        np.median(
            [
                item["record"]["training_performance"]["throughput"]["drawing_steps_per_s"]
                for item in history
            ]
        )
    )
    no_drawing_steps_per_s = float(
        np.median(
            [
                item["record"]["training_performance"]["throughput"][
                    "no_drawing_steps_per_s"
                ]
                for item in history
            ]
        )
    )
    shutil.copy2(best["plot"], promoted_plot)
    shutil.copy2(calibration_plot, promoted_calibration_plot)
    shutil.copy2(best["trace_calibration_plot"], promoted_trace_calibration_plot)
    summary = {
        "engine": "persistent operation-level exact-gradient training",
        "elapsed_s": time.monotonic() - started,
        "requested_profile_duration_ms": args.duration_ms,
        "points": args.points,
        "sample_rate": sample_rate,
        "capture_plan": capture_plan,
        "target": envelope.metadata(),
        "target_preview": str(target_preview),
        "model": {
            "type": "teacher-student tiled linear training",
            "parameters": workload.parameter_count,
            "input_size": args.hidden_size,
            "output_size": output_size,
            "batch_size": args.batch_size,
            "optimizer": "SGD",
            "learning_rate": args.learning_rate,
            "persistent_across_captures": True,
        },
        "compute_budget": {
            "tile_repeats": args.tile_repeats,
            "target_redundancy_ratio": float(args.tile_repeats),
            "same_trace_reference_repeats": args.reference_repeats,
            "same_trace_reference_widths": calibration.widths.tolist(),
        },
        "drawing_vs_no_drawing": {
            "drawing_median_steps_per_s": drawing_steps_per_s,
            "no_drawing_median_steps_per_s": no_drawing_steps_per_s,
            "drawing_throughput_fraction": drawing_steps_per_s / no_drawing_steps_per_s,
            "drawing_slowdown_x": no_drawing_steps_per_s / drawing_steps_per_s,
            "comparison_plot": str(training_comparison_plot),
        },
        "calibration": {
            "record": _record_summary(calibration_record, calibration_performance),
            "capture_plan": calibration_capture_plan,
            "captures_dataset": str(calibration_captures_root),
            "file": str(calibration_file),
            "plot": str(calibration_plot),
            "curve": calibration.to_dict(),
        },
        "measurement_policy": {
            **analysis_policy,
            "resolved_feature": calibration.feature_name,
            "feature_sign": calibration.feature_sign,
            "uses_target_for_feature_selection": False,
        },
        "captures_dataset": str(captures_root),
        "iterations": history,
        "iterations_completed": len(history),
        "maximum_iterations": args.iterations,
        "accepted_training_steps": len(history),
        "stop_reason": stop_reason,
        "controller": controller.metadata(),
        "best_iteration": best["iteration"],
        "best_record_index": best["record"]["index"],
        "best_metrics": best["metrics"],
        "best_training_performance": best["record"]["training_performance"],
        "best_trace_calibration": best["trace_calibration"],
        "best_trace_calibration_plot": str(promoted_trace_calibration_plot),
        "training_comparison_plot": str(training_comparison_plot),
        "best_plot": str(promoted_plot),
    }
    atomic_json(summary, parent / "experiment_summary.json")
    atomic_json(
        {
            "status": "complete",
            "stop_reason": stop_reason,
            "completed_iterations": len(history),
            "best_iteration": best["iteration"],
            "best_record_index": best["record"]["index"],
            "best_metrics": best["metrics"],
        },
        parent / "experiment_progress.json",
    )
    return summary


def run_fast(args: argparse.Namespace) -> dict:
    if args.training_model != "linear":
        raise ConfigurationError(
            "the legacy fast command supports --training-model linear only; use "
            "draw-png --engine timed --training-model residual-mlp"
        )
    if args.duration_ms <= 0 or args.duration_ms > 200:
        raise ConfigurationError("fast duration must be positive and at most 200 ms")
    if args.iterations < 1:
        raise ConfigurationError("fast iterations must be at least 1")
    if args.target_accuracy is not None and not 0 < args.target_accuracy <= 100:
        raise ConfigurationError("target accuracy must be greater than 0 and at most 100")
    short = args.duration_ms <= 20
    points = args.points
    if points == 120:  # argparse's generic default; choose a realizable fast resolution.
        points = 40 if short else 60
    sample_rate = args.sample_rate or ("10MHz" if short else "1.5MHz")
    analysis_policy = _fast_analysis_policy(args, engine="operation")
    correction_smoothing = (
        float(args.ilc_correction_smoothing)
        if args.ilc_correction_smoothing is not None
        else 0.0
    )
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
        extraction_mode=args.silhouette_mode,
    )
    save_envelope(parent, envelope)
    target_preview = render_target_preview(envelope, parent / "target_envelope.png")
    widths = np.asarray([0, 32, 64, 128, 256, 512, 1024, 2048, args.hidden_size], dtype=np.int64)
    widths = np.unique(widths[widths <= args.hidden_size])
    calibration_commands = interleaved_calibration_commands(widths, repeats=4, seed=args.seed)
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
    calibration = calibrate_fast(
        calibration_root,
        preferred_feature=analysis_policy["preferred_feature"],
    )
    calibration_file = save_fast_calibration(calibration_root, calibration)
    calibration_plot = render_fast_calibration(
        calibration,
        calibration_root / "fast_calibration.png",
    )
    controller = FastRefinementController(
        envelope.values,
        calibration,
        target_accuracy_percent=args.target_accuracy,
        initial_gain=args.ilc_gain,
        minimum_gain=args.minimum_ilc_gain,
        gain_decay=args.ilc_gain_decay,
        gain_growth=args.ilc_gain_growth,
        correction_smoothing_sigma_points=correction_smoothing,
        feedback_reference=args.ilc_feedback_reference,
    )
    commands = controller.initial_commands()
    history = []
    best = None
    stop_reason = "maximum_refinements_reached"
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
            smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
        )
        _, arrays = analyze_fast_drawing(
            iteration_root,
            envelope.values,
            calibration,
            smoothing_sigma_points=analysis_policy["smoothing_sigma_points"],
        )
        result = {
            "iteration": iteration,
            "capture": capture,
            "metrics": metrics,
            "plot": str(plot),
            "commands": commands.tolist(),
            "controller_gain": controller.gain,
        }
        history.append(result)
        objective_key = analysis_policy["selection_rmse_key"]
        accuracy_key = analysis_policy["selection_accuracy_key"]
        if best is None or metrics[objective_key] < best["metrics"][objective_key]:
            best = result
        improved = controller.observe(
            commands,
            arrays["measured_smoothed"],
            normalized_rmse=metrics[objective_key],
            shape_accuracy_percent=metrics[accuracy_key],
        )
        result["improved_best"] = improved
        result["controller_after_observation"] = controller.metadata()
        atomic_json(
            {
                "status": "running",
                "completed_iterations": len(history),
                "latest_iteration": iteration,
                "best_iteration": best["iteration"],
                "best_metrics": best["metrics"],
                "controller": controller.metadata(),
            },
            parent / "fast_experiment_progress.json",
        )
        print(
            f"iteration {iteration}: accuracy={metrics[accuracy_key]:.2f}% "
            f"rmse={metrics[objective_key]:.4f} "
            f"{'[new best]' if improved else '[kept prior best]'}",
            file=sys.stderr,
            flush=True,
        )
        if controller.reached_target:
            stop_reason = "target_accuracy_reached"
            break
        commands = controller.next_commands()
    promoted_plot = parent / "measured_silhouette.png"
    promoted_calibration_plot = parent / "tile_width_calibration.png"
    shutil.copy2(best["plot"], promoted_plot)
    shutil.copy2(calibration_plot, promoted_calibration_plot)
    summary = {
        "duration_ms": args.duration_ms,
        "points": points,
        "sample_rate": sample_rate,
        "target": envelope.metadata(),
        "target_preview": str(target_preview),
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
        "measurement_policy": {
            **analysis_policy,
            "resolved_feature": calibration.feature_name,
            "feature_sign": calibration.feature_sign,
            "uses_target_for_feature_selection": False,
        },
        "iterations": history,
        "iterations_completed": len(history),
        "maximum_iterations": args.iterations,
        "stop_reason": stop_reason,
        "controller": controller.metadata(),
        "best_iteration": best["iteration"],
        "best_run": best["capture"]["run"],
        "best_metrics": best["metrics"],
        "best_plot": str(promoted_plot),
    }
    atomic_json(summary, parent / "fast_experiment_summary.json")
    atomic_json(
        {
            "status": "complete",
            "stop_reason": stop_reason,
            "completed_iterations": len(history),
            "best_iteration": best["iteration"],
            "best_run": best["capture"]["run"],
            "best_metrics": best["metrics"],
        },
        parent / "fast_experiment_progress.json",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        envelope = extract_envelope(
            args.image,
            points=args.points,
            foreground_distance=args.foreground_distance,
            smoothing_sigma_points=args.smoothing_sigma,
            extraction_mode=args.silhouette_mode,
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
    elif args.command == "draw-png":
        result = run_draw_png(args)
    else:
        result = run_fast(args)
    print(json.dumps(result, indent=2))


def draw_png_main() -> None:
    """Installed one-command entry point for automatic hardware refinement."""

    main(["draw-png", *sys.argv[1:]])


if __name__ == "__main__":
    main()
