from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from experiments.llama_balanced_gigapass.scheduler import (
    BalancedServiceController,
    CalibrationResult,
    CandidateTiming,
    build_calibration,
    rescale_inference_service,
    set_training_native_seconds,
    summarize_durations,
)
from experiments.llama_balanced_gigapass.workload import (
    BalancedGigaPassConfig,
    BalancedGigaPassEngine,
    BalancedRoleConfig,
    balanced_role_artifact,
)


def calibration(lower: float = 0.96, upper: float = 1.04) -> CalibrationResult:
    return CalibrationResult(
        training_batch_size=1024,
        training_native_seconds=0.1,
        candidates=(
            CandidateTiming(8448, 0.1 * lower, lower, 5),
            CandidateTiming(8512, 0.1 * upper, upper, 5),
        ),
        lower_batch_size=8448,
        upper_batch_size=8512,
    )


def test_error_diffusion_converges_to_exact_one_to_one_service() -> None:
    controller = BalancedServiceController(calibration())
    batches = controller.plan(10_000)

    assert set(batches) == {8448, 8512}
    assert controller.service_ratio == pytest.approx(1.0, abs=1e-12)
    assert abs(controller.service_debt) <= 0.04 + 1e-12
    assert controller.upper_cycles == 5_000
    assert controller.lower_cycles == 5_000


def test_controller_handles_asymmetric_measured_bracket() -> None:
    measured = calibration(lower=0.9662, upper=1.0091)
    controller = BalancedServiceController(measured)
    controller.plan(100_000)

    assert controller.service_ratio == pytest.approx(1.0, abs=5e-7)
    assert controller.upper_cycles / controller.cycles == pytest.approx(
        measured.upper_cycle_fraction,
        abs=2e-5,
    )


def test_paired_calibration_cancels_common_thermal_drift() -> None:
    paired = {
        8448: [(0.096 * scale, 0.100 * scale) for scale in (1.0, 1.1, 1.2, 1.3)],
        8512: [(0.104 * scale, 0.100 * scale) for scale in (1.3, 1.2, 1.1, 1.0)],
    }
    result = build_calibration(training_batch_size=1024, paired_seconds=paired)

    assert result.lower.ratio_to_training == pytest.approx(0.96)
    assert result.upper.ratio_to_training == pytest.approx(1.04)
    assert result.upper_cycle_fraction == pytest.approx(0.5)


def test_calibration_requires_a_true_service_bracket() -> None:
    with pytest.raises(RuntimeError, match="did not bracket"):
        build_calibration(
            training_batch_size=1024,
            paired_seconds={
                8000: [(0.08, 0.1)],
                8100: [(0.09, 0.1)],
            },
        )


def test_calibration_round_trip_is_lossless() -> None:
    original = calibration()
    assert CalibrationResult.from_dict(original.to_dict()) == original


def test_schedule_feedback_moves_the_service_crossing() -> None:
    original = CalibrationResult(
        training_batch_size=1024,
        training_native_seconds=0.1,
        candidates=(
            CandidateTiming(8192, 0.09, 0.90, 4),
            CandidateTiming(8320, 0.095, 0.95, 4),
            CandidateTiming(8448, 0.101, 1.01, 4),
            CandidateTiming(8576, 0.106, 1.06, 4),
        ),
        lower_batch_size=8320,
        upper_batch_size=8448,
    )
    corrected = rescale_inference_service(original, 1.08)

    assert corrected.lower_batch_size == 8192
    assert corrected.upper_batch_size == 8320
    assert corrected.candidate(8320).ratio_to_training == pytest.approx(1.026)


def test_sustained_training_time_replaces_stale_absolute_scale() -> None:
    original = calibration()
    corrected = set_training_native_seconds(original, 0.125)

    assert corrected.training_native_seconds == 0.125
    assert corrected.lower.native_seconds == pytest.approx(0.125 * 0.96)
    assert corrected.upper.native_seconds == pytest.approx(0.125 * 1.04)


def test_duration_summary_uses_aggregate_not_best_round() -> None:
    summary = summarize_durations(
        inference_seconds=[1.0, 1.0, 1.0],
        training_seconds=[1.0, 1.0, 1.0],
        mixed_seconds=[1.8, 1.9, 2.2],
    )

    assert summary["slowdown"]["inference"] == pytest.approx(5.9 / 3.0)
    assert summary["max_slowdown"] == pytest.approx(5.9 / 3.0)
    assert summary["per_round_slowdown"]["inference"] == [1.8, 1.9, 2.2]
    assert summary["per_round_seconds"] == {
        "inference": [1.0, 1.0, 1.0], "training": [1.0, 1.0, 1.0], "mixed": [1.8, 1.9, 2.2]
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_batch_size": 0}, "training_batch_size"),
        ({"inference_batch_candidates": (1,)}, "at least two"),
        ({"inference_batch_candidates": (2, 1)}, "sorted"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"cycles_per_heartbeat": 0}, "cycles_per_heartbeat"),
    ],
)
def test_config_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BalancedGigaPassConfig(**kwargs)


def test_host_role_cannot_change_computation_fingerprint() -> None:
    computation = BalancedGigaPassConfig(calibration_override=calibration())
    inference = BalancedRoleConfig("inference", "i0", computation)
    training = BalancedRoleConfig("training", "t0", computation)

    assert inference.computation.fingerprint == training.computation.fingerprint
    assert inference.metadata()["strict_invariants"]["filler_kernels"] == 0


def test_gpu_engine_ast_cannot_read_role() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(BalancedGigaPassEngine)))
    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}

    assert "role" not in identifiers


def test_optimizer_is_ordered_after_inference_and_before_next_cycle() -> None:
    source = inspect.getsource(BalancedGigaPassEngine.run_mixed_batches)

    assert source.index("wait_event(inference_done)") < source.index("_enqueue_optimizer_step()")
    assert source.index("update_done.record()") < source.index("previous_update = update_done")
    assert "wait_event(previous_update)" in source


def test_training_uses_native_default_stream_to_avoid_accumulategrad_sync() -> None:
    source = inspect.getsource(BalancedGigaPassEngine.setup)

    assert "self.training_stream = torch.cuda.current_stream()" in source
    assert source.count("torch.cuda.Stream()") == 1


def test_both_host_artifacts_report_both_real_products() -> None:
    snapshot = {
        "mixed_inference_tokens": 85_120,
        "mixed_training_updates": 10,
        "inference_token_checksum": 123,
        "parameter_probe_delta_linf": 0.25,
        "balanced_service": {"service_ratio": 1.0},
    }

    inference = balanced_role_artifact("inference", snapshot)
    training = balanced_role_artifact("training", snapshot)

    assert inference["served_token_checksum"] == 123
    assert training["parameter_probe_delta_linf"] == 0.25
    assert inference["co_applied_training_updates"] == 10
    assert training["co_served_inference_tokens"] == 85_120


def test_nested_detector_recovers_signal_and_never_tunes_on_outer_labels() -> None:
    import numpy as np
    from experiments.llama_balanced_gigapass.audit_detector import evaluate_features

    rng = np.random.default_rng(731)
    groups = np.repeat(np.arange(5), 24)
    y = np.tile(np.repeat([-1.0, 1.0], 12), 5)
    x = rng.normal(size=(len(y), 4))
    x[:, 0] += 5 * y
    result = evaluate_features(x, y, groups)
    assert result["balanced_accuracy"] > 0.95
    flipped = y.copy()
    flipped[groups == 0] *= -1
    perturbed = evaluate_features(x, flipped, groups)
    for field in ("selected_ridge_strength", "selected_sign", "validation_accuracy", "training_pairs"):
        assert result["folds"][0][field] == perturbed["folds"][0][field]
    assert 0 not in result["folds"][0]["training_pairs"]


def test_cnn_partition_has_disjoint_complete_pairs() -> None:
    import numpy as np
    from experiments.llama_balanced_gigapass.paired_cnn import pair_partition

    groups = np.repeat(np.arange(12), 8)
    for held in range(12):
        train, validation, test = pair_partition(groups, held)
        assert np.all(train.astype(int) + validation + test == 1)
        assert set(groups[test]) == {held}
        assert len(set(groups[train])) == 10
        assert len(set(groups[validation])) == 1


def test_collection_plan_pairs_share_inputs_but_pairs_are_independent() -> None:
    from experiments.llama_balanced_gigapass.collect_pairs import collection_plan

    plan = collection_plan(12, 77)
    assert plan == collection_plan(12, 77)
    assert len({s["compute_seed"] for s in plan}) == 12
    for pair in range(12):
        sessions = [s for s in plan if s["pair"] == pair]
        assert {s["role"] for s in sessions} == {"inference", "training"}
        assert len({s["compute_seed"] for s in sessions}) == 1
        assert len({s["capture_seed"] for s in sessions}) == 1


def test_record_audit_rejects_stalled_or_nonfinite_training() -> None:
    from experiments.llama_balanced_gigapass.audit_records import check_progress

    first = dict(last_loss=10.0, parameter_probe_delta_linf=0.01,
                 mixed_training_updates=4, mixed_inference_tokens=100)
    second = {**first, "mixed_training_updates": 8, "mixed_inference_tokens": 200}
    assert check_progress([first, second])["last_update"] == 8
    with pytest.raises(ValueError, match="did not advance"):
        check_progress([first, first])
    with pytest.raises(ValueError, match="non-finite"):
        check_progress([first, {**second, "last_loss": float("nan")}])


def test_report_refuses_partial_detector_results() -> None:
    from experiments.llama_balanced_gigapass.render_paired_report import HORIZONS, detector_series

    payload = {"horizons": {str(float(h)): {"pair_count": 12, "folds": [{}] * 12,
                                          "balanced_accuracy": 0.55} for h in HORIZONS}}
    assert detector_series(payload, 12) == [0.55] * 5
    payload["horizons"]["100.0"]["folds"].pop()
    with pytest.raises(ValueError, match="incomplete"):
        detector_series(payload, 12)


def test_raw_plot_selection_is_independent_of_values() -> None:
    from types import SimpleNamespace
    from experiments.llama_balanced_gigapass.render_paired_report import select_pair

    traces = {role: [SimpleNamespace(session_id=f"{role[0]}{pair}", index=i, values=value)
                     for pair in range(3) for i in range(8)]
              for role, value in (("inference", -100), ("training", 100))}
    first, _ = select_pair(traces, seed=20260920)
    for role in traces:
        traces[role].reverse()
        for trace in traces[role]:
            trace.values *= 123
    second, _ = select_pair(traces, seed=20260920)
    assert first == second
