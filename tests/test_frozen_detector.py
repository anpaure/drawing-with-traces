from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.llama_balanced_gigapass.collect_pairs import collection_plan
from experiments.llama_balanced_gigapass.frozen_detector import (
    auc, balanced_accuracy, check_test_contract, digest, evaluate_scores, partition,
    predict, ridge_fit, ridge_predict, select_decision, source_hashes, verify_bundle,
)


def test_direction_and_threshold_can_recover_inverted_validation_signal():
    scores = np.array([12, 13, 14, 2, 3, 4])
    labels = np.array([0, 0, 0, 1, 1, 1])
    decision = select_decision(scores, labels)
    assert decision["sign"] == -1
    assert decision["validation_accuracy"] == 1
    assert balanced_accuracy(predict(scores, decision), labels) == 1


def test_inference_does_not_use_test_labels_or_adapt_decision():
    decision = select_decision(np.array([-2, -1, 1, 2]), np.array([0, 0, 1, 1]))
    before = json.dumps(decision, sort_keys=True)
    scores, labels = np.array([-1, -0.5, 0.5, 1]), np.array([0, 0, 1, 1])
    prediction = predict(scores, decision)
    assert balanced_accuracy(prediction, labels) == 1
    assert balanced_accuracy(prediction, 1 - labels) == 0
    assert json.dumps(decision, sort_keys=True) == before


@pytest.mark.parametrize("scores,expected", [([0, 0, 1, 1], 1), ([1, 1, 0, 0], 0),
                                           ([1, 1, 1, 1], 0.5), ([0, 1, 1, 2], 0.875)])
def test_auc_ties_and_direction(scores, expected):
    assert auc(scores, [0, 0, 1, 1]) == pytest.approx(expected)


def test_ridge_state_reproduces_scores_without_test_fitting(tmp_path):
    x = np.arange(40).reshape(20, 2).astype(float)
    y = np.arange(20) >= 10
    state = ridge_fit(x, y.astype(int), 1)
    np.savez(tmp_path / "ridge.npz", **state)
    with np.load(tmp_path / "ridge.npz", allow_pickle=False) as loaded:
        assert np.array_equal(ridge_predict(state, x), ridge_predict(loaded, x))
    assert np.isfinite(ridge_predict(state, np.ones((2, 2)) * 1000)).all()


def test_partition_keeps_whole_pairs_out_and_is_reproducible():
    training, validation = partition(np.repeat(np.arange(12), 8), 20260921)
    assert len(training) == 8 and len(validation) == 4
    assert not set(training) & set(validation)
    assert set(training + validation) == set(range(12))
    assert (training, validation) == partition(np.arange(12)[::-1], 20260921)


def test_frozen_contract_rejects_changed_config_and_reused_compute_seeds():
    contract = {"pairs": 4, "captures_per_session": 8, "seed": 20000,
                "attention_backend": "eager", "calibration": {"test": 1}}
    manifest = {"test_plan": contract, "development_compute_seeds": [10000]}
    plan = {**contract, "plan": collection_plan(4, 20000)}
    check_test_contract(manifest, plan)
    with pytest.raises(ValueError, match="differs from frozen"):
        check_test_contract(manifest, {**plan, "pairs": 5})
    with pytest.raises(ValueError, match="reuses"):
        check_test_contract({**manifest, "development_compute_seeds": [21000]}, plan)
    with pytest.raises(ValueError, match="order or seeds"):
        check_test_contract(manifest, {**plan, "plan": list(reversed(plan["plan"]))})


def test_bundle_verification_rejects_modified_checkpoint(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"unit-test fixture, not a trained detector")
    manifest = {"status": "frozen", "schema_version": 1,
                "source_sha256": source_hashes(),
                "horizons": {str(h): {"ridge": {"file": "weights.pt"}, "cnn": {"files": ["weights.pt"]}}
                             for h in (5, 10, 20, 50, 100)},
                "file_sha256": {"weights.pt": digest(checkpoint)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path) == manifest
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(tmp_path)


def test_constant_classifier_stays_at_chance_and_bootstraps_pairs():
    labels = np.tile([0, 1], 12)
    groups = np.repeat(np.arange(12), 2)
    protocol = {"seed": 0, "bootstrap_replicates": 1000, "family_wise_alpha": 0.05,
                "primary_comparisons": 10, "accuracy_target": 0.6}
    result = evaluate_scores(np.zeros(24), labels, groups, {"sign": 1, "threshold": 0}, protocol)
    assert result["balanced_accuracy"] == result["pair_bootstrap_upper_bound"] == 0.5
    assert len(result["pairs"]) == 12
    assert result["oriented_roc_auc"] == 0.5
    assert result["confusion_true_rows_predicted_columns"] == [[0, 12], [0, 12]]


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], []])
def test_invalid_scores_cannot_be_frozen(scores):
    with pytest.raises(ValueError, match="finite"):
        select_decision(scores, np.zeros(len(scores)))
