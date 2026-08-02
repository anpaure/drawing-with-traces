from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from experiments.llama_identical_dual_role_workload.workload import (
    ComputationConfig,
    DualRoleConfig,
    IdenticalDualRoleEngine,
    role_artifact,
    split_dual_logits,
)


def test_role_and_session_cannot_change_computation_fingerprint() -> None:
    inference = DualRoleConfig(
        role="inference",
        session_id="inference-session",
        period_profile_output="inference-profile",
    )
    training = DualRoleConfig(
        role="training",
        session_id="training-session",
        period_profile_output="training-profile",
    )

    assert inference.computation == training.computation
    assert inference.computation.fingerprint == training.computation.fingerprint


def test_gpu_engine_ast_cannot_read_role() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(IdenticalDualRoleEngine)))
    identifiers = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    } | {
        node.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.arg)
    }

    assert "role" not in identifiers


def test_combined_forward_has_real_inference_output_and_training_only_gradient() -> None:
    logits = torch.randn(7, 1, 13, requires_grad=True)
    inference_logits, training_logits = split_dual_logits(logits, 3)
    targets = torch.tensor([1, 2, 3, 4])
    loss = torch.nn.functional.cross_entropy(training_logits[:, -1, :], targets)
    loss.backward()

    assert torch.equal(inference_logits, logits[:3])
    assert torch.count_nonzero(logits.grad[:3]) == 0
    assert torch.count_nonzero(logits.grad[3:]) > 0


def test_role_selection_only_relabels_completed_snapshot() -> None:
    snapshot = {
        "updates": 9,
        "inference_token_checksum": 1234,
        "parameter_probe_delta_linf": 0.125,
    }

    inference = role_artifact("inference", snapshot)
    training = role_artifact("training", snapshot)

    assert inference == {
        "kind": "served_logits",
        "served_token_checksum": 1234,
        "updates_computed_but_not_persisted": 9,
    }
    assert training == {
        "kind": "updated_model_state",
        "updates_persisted": 9,
        "parameter_probe_delta_linf": 0.125,
        "logits_computed_but_not_served": 1234,
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"inference_batch_size": 0}, "inference_batch_size"),
        ({"training_batch_size": 0}, "training_batch_size"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"iterations_per_heartbeat": 0}, "iterations_per_heartbeat"),
    ],
)
def test_computation_config_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ComputationConfig(**kwargs)


def test_metadata_states_exact_role_blind_invariants() -> None:
    metadata = DualRoleConfig(role="training", session_id="s0").metadata()
    invariants = metadata["strict_invariants"]

    assert invariants["role_read_by_gpu_program"] is False
    assert invariants["same_forward_backward_update_both_roles"] is True
    assert invariants["same_model_state_both_roles"] is True
    assert invariants["filler_kernels"] == 0
