import numpy as np
import pytest
from sidecapture.errors import ConfigurationError

from drawing_with_traces.fast import (
    FastCalibration,
    FastRefinementController,
    TiledLinearTrainingWorkload,
    TiledResidualMLPTrainingWorkload,
    calibrate_fast,
    dense_calibration_widths,
    interleaved_calibration_commands,
    normalized_curve_metrics,
    project_commands_to_total,
    silhouette_partition_commands,
)


def calibration():
    return FastCalibration(
        widths=np.array([0, 32, 128, 512, 4096], dtype=float),
        feature_name="rms",
        feature_sign=-1,
        measured_feature=np.array([5, 4, 3, 2, 1], dtype=float),
        monotonic_activity=np.array([-5, -4, -3, -2, -1], dtype=float),
        within_width_std=np.full(5, 0.01),
    )


def test_fast_calibration_maps_target_to_supported_widths():
    commands = calibration().commands_for(np.array([0, 0.25, 0.5, 0.75, 1]))
    assert commands[0] == 0
    assert commands[-1] == 4096
    assert np.all(commands % 32 == 0)


def test_fast_calibration_interpolates_to_legal_tile_widths():
    commands = calibration().commands_for(np.array([0.125, 0.375, 0.625, 0.875]))
    assert np.all(commands % 32 == 0)
    assert np.all(np.diff(commands) > 0)
    assert any(command not in calibration().widths for command in commands)


def test_refinement_controller_supports_non_width_command_quantum():
    duty = FastCalibration(
        widths=np.array([1, 8, 32, 128], dtype=float),
        feature_name="std",
        feature_sign=1,
        measured_feature=np.array([1, 2, 3, 4], dtype=float),
        monotonic_activity=np.array([1, 2, 3, 4], dtype=float),
        within_width_std=np.full(4, 0.01),
    )
    controller = FastRefinementController(
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        duty,
        minimum_width=1,
        command_quantum=1,
    )
    commands = controller.initial_commands()
    assert commands[0] == 1
    assert commands[-1] == 128
    assert np.any(commands % 32)


def test_calibration_widths_are_repeated_in_interleaved_contexts():
    widths = np.array([0, 32, 64, 128, 256])
    commands = interleaved_calibration_commands(widths, repeats=4, seed=7)

    assert commands.size == widths.size * 4
    assert all(np.count_nonzero(commands == width) == 4 for width in widths)
    assert not any(np.all(chunk == width) for width in widths for chunk in np.split(commands, 4))
    assert not np.any(commands[1:] == commands[:-1])


def test_dense_calibration_avoids_idle_and_resolves_control_range():
    widths = dense_calibration_widths(128, 4096)

    assert widths[0] == 128
    assert widths[-1] == 4096
    assert 0 not in widths
    assert np.all(np.diff(widths[widths <= 1280]) == 64)
    assert {1536, 2048, 3072, 4096}.issubset(set(widths))


def test_calibration_reports_diagnostics_when_preferred_feature_is_not_separable(
    monkeypatch,
):
    commands = np.array([32, 64, 128, 32, 64, 128])
    features = {
        "rms": np.array([1.0, 1.1, 1.0, 1.0, 1.1, 1.0]),
        "std": np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
    }
    monkeypatch.setattr(
        "drawing_with_traces.fast.measured_bin_features",
        lambda *args, **kwargs: (features, commands, {}),
    )

    with pytest.raises(
        ConfigurationError,
        match=r"preferred calibration feature 'rms'.*rms: span=.*std: span=",
    ):
        calibrate_fast("unused", preferred_feature="rms")


def test_silhouette_commands_form_an_exact_quantized_partition():
    target = np.array([0, 0.25, 0.5, 0.75, 1.0])
    commands = silhouette_partition_commands(target, minimum_width=32, maximum_width=1024)

    assert np.all(commands % 32 == 0)
    assert commands[0] == 32
    assert commands[-1] == 1024
    workload = TiledLinearTrainingWorkload(
        commands,
        output_size=int(commands.sum()),
        profile_duration_s=0.01,
        schedule_mode="exact-once",
    )
    assert workload.parameter_count == 4096 * int(commands.sum())


def test_projection_preserves_total_and_legal_bounds():
    projected = project_commands_to_total(
        np.array([32, 128, 512, 1024], dtype=float),
        2048,
        minimum_width=32,
        maximum_width=1024,
    )

    assert projected.sum() == 2048
    assert np.all(projected % 32 == 0)
    assert np.all((projected >= 32) & (projected <= 1024))


def test_exact_once_rejects_incomplete_output_partition():
    with pytest.raises(ValueError, match="sum exactly"):
        TiledLinearTrainingWorkload(
            np.array([32, 64, 128]),
            output_size=256,
            profile_duration_s=0.01,
            schedule_mode="exact-once",
        )


def test_fast_calibration_round_trip():
    original = calibration()
    restored = FastCalibration.from_dict(original.to_dict())
    assert restored.feature_name == "rms"
    assert restored.feature_sign == -1
    assert np.array_equal(restored.widths, original.widths)


def test_tiled_workload_metadata_is_transactional_without_cuda():
    workload = TiledLinearTrainingWorkload(
        np.array([0, 32, 128, 4096]),
        profile_duration_s=0.01,
        hidden_size=4096,
        batch_size=512,
    )
    assert workload.bin_duration_s == pytest.approx(0.0025)
    assert workload.parameter_count == 16_777_216
    assert workload.metadata()["transactional_update"] is True
    assert "X.T @" in workload.metadata()["gradient"]


def test_residual_mlp_metadata_and_parameter_count_without_cuda():
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 512, 4096]),
        profile_duration_s=0.1,
        hidden_size=4096,
        depth=14,
        batch_size=256,
    )

    assert workload.parameter_count == 14 * 4096 * 4096
    assert workload.useful_gradient_width == 14 * 4096
    assert workload.task_label == "chipwhisperer-tiled-residual-mlp-training"
    assert workload.metadata()["depth"] == 14
    assert workload.metadata()["residual_scale"] == pytest.approx(0.125)
    assert workload.metadata()["drawing_layer_count"] == 14
    assert workload.metadata()["drawing_layers"] == list(range(14))
    assert workload.metadata()["fresh_batch_each_step"] is False
    assert workload.metadata()["training_batch_pool_size"] == 1
    assert "layers" in workload.metadata()["gradient"]


def test_residual_mlp_fresh_batch_mode_cycles_precomputed_pool():
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 256]),
        profile_duration_s=0.01,
        hidden_size=256,
        depth=2,
        batch_size=8,
        fresh_batch_each_step=True,
        training_batch_pool_size=3,
    )
    workload.training_batches = [("x0", "y0"), ("x1", "y1"), ("x2", "y2")]
    workload.inputs, workload.targets = workload.training_batches[0]

    workload._advance_training_batch()
    assert (workload.inputs, workload.targets) == ("x1", "y1")
    workload._advance_training_batch()
    assert (workload.inputs, workload.targets) == ("x2", "y2")
    workload._advance_training_batch()
    assert (workload.inputs, workload.targets) == ("x0", "y0")


def test_residual_mlp_rejects_too_small_training_batch_pool():
    with pytest.raises(ValueError, match="pool_size"):
        TiledResidualMLPTrainingWorkload(
            np.array([128, 256]),
            profile_duration_s=0.01,
            hidden_size=256,
            depth=2,
            batch_size=8,
            fresh_batch_each_step=True,
            training_batch_pool_size=1,
        )


def test_residual_mlp_scheduler_walks_distinct_blocks_and_layers(monkeypatch):
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 128]),
        profile_duration_s=0.01,
        hidden_size=256,
        depth=3,
        batch_size=8,
    )
    visited = []

    def fake_gradient_tile(start, width, **kwargs):
        visited.append((kwargs["layer_index"], start, width))

    monkeypatch.setattr(workload, "gradient_tile", fake_gradient_tile)
    for _ in range(7):
        workload.next_tile(128)

    assert visited == [
        (0, 0, 128),
        (0, 128, 128),
        (1, 0, 128),
        (1, 128, 128),
        (2, 0, 128),
        (2, 128, 128),
        (0, 0, 128),
    ]


def test_residual_mlp_scheduler_limits_profile_to_drawing_layers(monkeypatch):
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 128]),
        profile_duration_s=0.01,
        hidden_size=256,
        depth=4,
        drawing_layer_count=2,
        batch_size=8,
    )
    visited = []

    def fake_gradient_tile(start, width, **kwargs):
        visited.append((kwargs["layer_index"], start, width))

    monkeypatch.setattr(workload, "gradient_tile", fake_gradient_tile)
    for _ in range(10):
        workload.next_tile(128)

    assert visited == [
        (0, 0, 128),
        (0, 128, 128),
        (1, 0, 128),
        (1, 128, 128),
        (0, 0, 128),
        (0, 128, 128),
        (1, 0, 128),
        (1, 128, 128),
        (0, 0, 128),
        (0, 128, 128),
    ]


@pytest.mark.parametrize("drawing_layer_count", [0, 4])
def test_residual_mlp_rejects_invalid_drawing_layer_count(drawing_layer_count):
    with pytest.raises(ValueError, match="drawing_layer_count"):
        TiledResidualMLPTrainingWorkload(
            np.array([128, 256]),
            profile_duration_s=0.01,
            hidden_size=256,
            depth=3,
            drawing_layer_count=drawing_layer_count,
            batch_size=8,
        )


def test_residual_mlp_warmup_first_touches_every_layer_and_width(monkeypatch):
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 256, 128]),
        profile_duration_s=0.01,
        hidden_size=256,
        depth=3,
        batch_size=8,
    )
    visited = []
    monkeypatch.setattr(
        workload,
        "gradient_tile",
        lambda start, width, **kwargs: visited.append(
            (kwargs["layer_index"], start, width, kwargs["accumulate"])
        ),
    )
    monkeypatch.setattr(workload, "clear_gradient", lambda: None)

    workload.warmup(0)

    assert visited == [
        (0, 0, 128, False),
        (0, 0, 256, False),
        (1, 0, 128, False),
        (1, 0, 256, False),
        (2, 0, 128, False),
        (2, 0, 256, False),
    ]


def test_residual_mlp_teardown_preserves_scalar_verification_evidence():
    workload = TiledResidualMLPTrainingWorkload(
        np.array([128, 256]),
        profile_duration_s=0.01,
        hidden_size=256,
        depth=2,
        batch_size=8,
    )
    workload.autograd_equivalence = {
        "manual_vs_autograd_gradient_relative_l2": 0.0,
    }

    workload.teardown()

    assert workload.metadata()["autograd_equivalence"] == {
        "manual_vs_autograd_gradient_relative_l2": 0.0,
    }


def test_gradient_completion_is_included_in_drawing_cost(monkeypatch):
    workload = TiledLinearTrainingWorkload(
        np.array([32, 32]),
        profile_duration_s=0.01,
        hidden_size=256,
        output_size=512,
        batch_size=8,
    )
    workload.column_visits = np.concatenate(
        [np.ones(256, dtype=np.int64), np.zeros(256, dtype=np.int64)]
    )
    workload.gradient_compute_wall_s = 0.1
    workload.executed_tile_width_sum = 256
    workload.last_operation_count = 1

    def fake_gradient_tile(start, width):
        workload.column_visits[start : start + width] += 1

    ticks = iter([1_000_000_000, 1_005_000_000])
    monkeypatch.setattr(workload, "gradient_tile", fake_gradient_tile)
    monkeypatch.setattr("drawing_with_traces.fast.time.monotonic_ns", lambda: next(ticks))

    workload.complete_exact_gradient()

    assert workload.completion_gradient_compute_wall_s == pytest.approx(0.005)
    assert workload.gradient_compute_wall_s == pytest.approx(0.105)
    assert workload.completion_tile_width_sum == 256
    assert workload.executed_tile_width_sum == 512
    assert workload.completion_operation_count == 1
    assert workload.last_operation_count == 2
    assert workload.coverage_after_completion["unvisited_columns"] == 0


def test_refinement_stops_after_reaching_requested_accuracy():
    target = np.linspace(0, 1, 5)
    controller = FastRefinementController(
        target,
        calibration(),
        target_accuracy_percent=95,
    )
    commands = controller.initial_commands()
    improved = controller.observe(
        commands,
        target + 0.01,
        normalized_rmse=0.01,
        shape_accuracy_percent=99,
    )
    assert improved
    assert controller.reached_target
    assert controller.metadata()["observations"] == 1


def test_refinement_uses_best_trace_and_reduces_gain_after_regression():
    target = np.linspace(0, 1, 5)
    controller = FastRefinementController(
        target,
        calibration(),
        initial_gain=0.4,
        minimum_gain=0.1,
        gain_decay=0.5,
    )
    commands = controller.initial_commands()
    best_measured = np.clip(target - 0.2, 0, 1)
    controller.observe(
        commands,
        best_measured,
        normalized_rmse=0.15,
        shape_accuracy_percent=85,
    )
    controller.observe(
        commands,
        np.zeros_like(target),
        normalized_rmse=0.5,
        shape_accuracy_percent=50,
    )

    refined = controller.next_commands()

    assert controller.gain == pytest.approx(0.2)
    assert controller.best_rmse == pytest.approx(0.15)
    assert refined[-2] >= commands[-2]


def test_refinement_can_track_latest_trace_after_plant_drift():
    target = np.linspace(0, 1, 5)
    best = FastRefinementController(
        target,
        calibration(),
        initial_gain=0.4,
        feedback_reference="best",
    )
    latest = FastRefinementController(
        target,
        calibration(),
        initial_gain=0.4,
        feedback_reference="latest",
    )
    initial = best.initial_commands()
    first_measured = np.clip(target - 0.1, 0, 1)
    for controller in (best, latest):
        controller.observe(
            initial,
            first_measured,
            normalized_rmse=0.1,
            shape_accuracy_percent=90,
        )
    delivered = latest.next_commands()
    drifted = np.clip(target - 0.35, 0, 1)
    for controller in (best, latest):
        controller.observe(
            delivered,
            drifted,
            normalized_rmse=0.35,
            shape_accuracy_percent=65,
        )

    assert latest.metadata()["feedback_reference"] == "latest"
    assert latest.next_commands().sum() > best.next_commands().sum()


def test_refinement_accumulates_corrections_from_best_delivered_command():
    target = np.linspace(0, 1, 5)
    controller = FastRefinementController(target, calibration(), initial_gain=0.4)
    initial = controller.initial_commands()
    controller.observe(
        initial,
        np.clip(target - 0.2, 0, 1),
        normalized_rmse=0.2,
        shape_accuracy_percent=80,
    )
    first = controller.next_commands()
    controller.observe(
        first,
        np.clip(target - 0.1, 0, 1),
        normalized_rmse=0.1,
        shape_accuracy_percent=90,
    )
    second = controller.next_commands()

    assert second[-2] > first[-2] > initial[-2]


def test_refinement_smooths_only_the_accumulated_correction():
    target = np.linspace(0, 1, 9)
    rough = FastRefinementController(target, calibration(), initial_gain=0.4)
    smooth = FastRefinementController(
        target,
        calibration(),
        initial_gain=0.4,
        correction_smoothing_sigma_points=1.5,
    )
    baseline = rough.initial_commands()
    noisy = baseline + np.array([0, 128, -128, 128, -128, 128, -128, 128, 0])
    measured = np.clip(target - 0.1, 0, 1)
    for controller in (rough, smooth):
        controller.observe(
            noisy,
            measured,
            normalized_rmse=0.1,
            shape_accuracy_percent=90,
        )

    rough_residual = rough.next_commands() - baseline
    smooth_residual = smooth.next_commands() - baseline
    assert np.abs(np.diff(smooth_residual)).sum() < np.abs(np.diff(rough_residual)).sum()


def test_multiscale_fidelity_penalizes_native_bin_ripple():
    target = np.linspace(0, 1, 9)
    raw = np.clip(target + np.array([0, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 0]), 0, 1)
    metrics = normalized_curve_metrics(target, raw, target)

    assert metrics["shape_accuracy_percent"] == pytest.approx(100)
    assert metrics["raw_shape_accuracy_percent"] < 90
    assert metrics["raw_shape_accuracy_percent"] < metrics["multiscale_shape_accuracy_percent"] < 100


@pytest.mark.parametrize("commands", [[16, 32], [-1, 32], [32, 8192]])
def test_tiled_workload_rejects_invalid_widths(commands):
    with pytest.raises(ValueError):
        TiledLinearTrainingWorkload(
            np.asarray(commands),
            profile_duration_s=0.01,
            hidden_size=4096,
        )
