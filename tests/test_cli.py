import pytest
from sidecapture.errors import ConfigurationError

from drawing_with_traces.cli import _fast_analysis_policy, build_parser, run_draw_png, run_fast


def test_draw_png_defaults_to_short_automatic_hardware_run():
    args = build_parser().parse_args(["draw-png", "logo.png", "--output", "run"])

    assert args.command == "draw-png"
    assert str(args.image) == "logo.png"
    assert args.duration_ms == 10
    assert args.points == 120
    assert args.batch_size == 256
    assert args.target_accuracy == 95
    assert args.iterations == 12
    assert args.silhouette_mode == "height"


def test_draw_png_accepts_upper_boundary_and_refinement_alias():
    args = build_parser().parse_args(
        [
            "draw-png",
            "logo.png",
            "--output",
            "run",
            "--silhouette-mode",
            "upper-boundary",
            "--max-refinements",
            "7",
        ]
    )

    assert args.silhouette_mode == "upper-boundary"
    assert args.iterations == 7


def test_draw_png_exposes_120_position_timed_engine():
    args = build_parser().parse_args(
        [
            "draw-png",
            "logo.png",
            "--output",
            "run",
            "--engine",
            "timed",
            "--duration-ms",
            "100",
            "--capture-window-ms",
            "110",
        ]
    )

    assert args.engine == "timed"
    assert args.points == 120
    assert args.duration_ms == 100
    assert args.training_model == "linear"
    assert _fast_analysis_policy(args, engine="timed") == {
        "preferred_feature": "rms",
        "smoothing_sigma_points": 2.0,
        "normalization": "robust",
        "selection_rmse_key": "multiscale_normalized_rmse",
        "selection_accuracy_key": "multiscale_shape_accuracy_percent",
        "selected_before_drawing": True,
    }


def test_timed_analysis_policy_can_be_overridden_explicitly():
    args = build_parser().parse_args(
        [
            "draw-png",
            "logo.png",
            "--output",
            "run",
            "--engine",
            "timed",
            "--trace-feature",
            "std",
            "--feature-smoothing",
            "1.25",
        ]
    )

    policy = _fast_analysis_policy(args, engine="timed")
    assert policy["preferred_feature"] == "std"
    assert policy["smoothing_sigma_points"] == 1.25


def test_draw_png_exposes_fable_style_residual_mlp_controls():
    args = build_parser().parse_args(
        [
            "draw-png",
            "logo.png",
            "--output",
            "run",
            "--engine",
            "timed",
            "--training-model",
            "residual-mlp",
            "--residual-depth",
            "14",
            "--residual-scale",
            "0.125",
        ]
    )

    assert args.training_model == "residual-mlp"
    assert args.residual_depth == 14
    assert args.residual_scale == pytest.approx(0.125)
    assert args.residual_learning_rate == pytest.approx(0.002)


def test_residual_mlp_rejects_non_timed_engines_before_hardware_access():
    draw_args = build_parser().parse_args(
        [
            "draw-png",
            "logo.png",
            "--output",
            "run",
            "--training-model",
            "residual-mlp",
        ]
    )
    fast_args = build_parser().parse_args(
        [
            "fast",
            "--image",
            "logo.png",
            "--output",
            "run",
            "--training-model",
            "residual-mlp",
        ]
    )

    with pytest.raises(ConfigurationError, match="requires --engine timed"):
        run_draw_png(draw_args)
    with pytest.raises(ConfigurationError, match="legacy fast command"):
        run_fast(fast_args)
