from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from experiments.llama_shared_kernel_training_carrier.shared_carrier import (
    SharedCarrierConfig,
    SharedCarrierEmbedding,
    SharedCarrierGradientScheduler,
    SharedCarrierLinear,
    carrier_execution_plans,
    execution_plan,
    replace_linear_modules_with_shared_carrier,
    tiled_inference_balanced_weight_gradient,
    tiled_mm,
    tiled_weight_gradient,
)
from experiments.llama_strict_inference_shaped_training.strict_workloads import (
    StrictWorkloadConfig,
)
from experiments.llama_strict_inference_shaped_training.strict_optimizer import InterleavedSGD


def test_tiled_mm_and_weight_gradient_are_exact_in_float64() -> None:
    torch.manual_seed(7)
    x = torch.randn(8, 5, dtype=torch.float64)
    weight = torch.randn(7, 5, dtype=torch.float64)
    grad_output = torch.randn(8, 7, dtype=torch.float64)

    assert torch.equal(tiled_mm(x, weight.T, row_tile=4), x @ weight.T)
    assert torch.equal(
        tiled_weight_gradient(x, grad_output, row_tile=4),
        grad_output.T @ x,
    )


def test_inference_balanced_weight_gradient_is_exact_in_both_orientations() -> None:
    torch.manual_seed(8)
    square_x = torch.randn(8, 8, dtype=torch.float64)
    square_grad_output = torch.randn(8, 7, dtype=torch.float64)
    rectangular_x = torch.randn(8, 5, dtype=torch.float64)
    rectangular_grad_output = torch.randn(8, 7, dtype=torch.float64)

    torch.testing.assert_close(
        tiled_inference_balanced_weight_gradient(
            square_x,
            square_grad_output,
            row_tile=4,
        ),
        square_grad_output.T @ square_x,
        rtol=1e-14,
        atol=1e-14,
    )
    assert torch.equal(
        tiled_inference_balanced_weight_gradient(
            rectangular_x,
            rectangular_grad_output,
            row_tile=4,
        ),
        rectangular_grad_output.T @ rectangular_x,
    )


def test_shared_carrier_linear_matches_ordinary_forward_and_backward() -> None:
    torch.manual_seed(11)
    ordinary = nn.Linear(5, 7, bias=True).double()
    source = copy.deepcopy(ordinary)
    carrier = SharedCarrierLinear(
        source,
        SharedCarrierConfig(row_tile=4, expected_training_rows=8),
    )
    ordinary_x = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    carrier_x = ordinary_x.detach().clone().requires_grad_(True)

    ordinary_loss = ordinary(ordinary_x).square().sum()
    carrier_loss = carrier(carrier_x).square().sum()
    ordinary_loss.backward()
    carrier_loss.backward()

    assert torch.equal(carrier_loss, ordinary_loss)
    assert torch.equal(carrier_x.grad, ordinary_x.grad)
    assert torch.equal(carrier.weight.grad, ordinary.weight.grad)
    assert torch.equal(carrier.bias.grad, ordinary.bias.grad)


def test_shared_carrier_plan_counts_only_useful_training_flops() -> None:
    config = SharedCarrierConfig(row_tile=4, expected_training_rows=8)
    plan = execution_plan(
        input_rows=8,
        input_features=5,
        output_features=7,
        config=config,
    )
    assert plan.forward_launches == 2
    assert plan.input_gradient_launches == 2
    assert plan.weight_gradient_launches == 2
    assert plan.useful_forward_flops == 560
    assert plan.useful_input_gradient_flops == 560
    assert plan.useful_weight_gradient_flops == 560
    assert plan.weight_gradient_layout == "direct"
    assert plan.layout_transform_values == 0
    assert plan.executed_flops == 1_680
    assert plan.redundant_flops == 0


def test_inference_balanced_plan_tiles_transposed_gradient_without_extra_flops() -> None:
    config = SharedCarrierConfig(
        row_tile=4,
        expected_training_rows=8,
        weight_gradient_layout="inference-balanced",
    )
    plan = execution_plan(
        input_rows=8,
        input_features=8,
        output_features=7,
        config=config,
    )
    assert plan.weight_gradient_layout == "transposed"
    assert plan.weight_gradient_launches == 2
    assert plan.layout_transform_values == 56
    assert plan.executed_flops == 2 * 8 * 8 * 7 * 3
    assert plan.redundant_flops == 0


def test_strided_plan_removes_layout_transform_without_adding_flops() -> None:
    config = SharedCarrierConfig(
        row_tile=4,
        expected_training_rows=8,
        weight_gradient_layout="inference-balanced-strided",
    )
    plan = execution_plan(
        input_rows=8,
        input_features=8,
        output_features=7,
        config=config,
    )
    assert plan.weight_gradient_layout == "transposed-strided"
    assert plan.weight_gradient_launches == 2
    assert plan.layout_transform_values == 0
    assert plan.executed_flops == 2 * 8 * 8 * 7 * 3
    assert plan.redundant_flops == 0


def test_replacement_retains_parameter_identity_and_reports_plans() -> None:
    class Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(5, 5, bias=False)
            self.second = nn.Linear(5, 5, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    model = Pair().double()
    parameter = model.first.weight
    names = replace_linear_modules_with_shared_carrier(
        model,
        SharedCarrierConfig(row_tile=4, expected_training_rows=8),
    )
    assert names == ["first", "second"]
    assert isinstance(model.first, SharedCarrierLinear)
    assert isinstance(model.second, SharedCarrierLinear)
    assert model.first.weight is parameter
    assert model.second.weight is parameter

    model(torch.randn(8, 5, dtype=torch.float64)).sum().backward()
    plans = carrier_execution_plans(model)
    assert set(plans) == {"first", "second"}
    assert all(plan.redundant_flops == 0 for plan in plans.values())


def test_deferred_carrier_scheduler_matches_shared_weight_training_step() -> None:
    class Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(5, 5, bias=False)
            self.second = nn.Linear(5, 5, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(x))

    torch.manual_seed(19)
    ordinary = Pair().double()
    carrier = copy.deepcopy(ordinary)
    scheduler = SharedCarrierGradientScheduler(row_tile=4, tasks_per_record=1)
    replace_linear_modules_with_shared_carrier(
        carrier,
        SharedCarrierConfig(row_tile=4, expected_training_rows=8),
        scheduler=scheduler,
    )
    optimizer = InterleavedSGD(
        carrier,
        learning_rate=0.01,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    x = torch.randn(8, 5, dtype=torch.float64)

    ordinary.zero_grad(set_to_none=False)
    ordinary(x).square().sum().backward()
    with torch.no_grad():
        for parameter in ordinary.parameters():
            parameter.add_(parameter.grad, alpha=-0.01)

    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    carrier(x).square().sum().backward()
    audit = scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    optimizer.step_deferred()

    assert audit.recorded_invocations == 2
    assert audit.gemm_launches == 4
    assert audit.redundant_gemm_flops == 0
    assert audit.parameter_updates_deferred == 1
    assert torch.equal(carrier.first.weight.grad, ordinary.first.weight.grad)
    assert torch.equal(carrier.first.weight, ordinary.first.weight)
    optimizer.close()


def test_inference_balanced_scheduler_matches_ordinary_training_step() -> None:
    torch.manual_seed(23)
    ordinary = nn.Sequential(nn.Linear(8, 7, bias=False)).double()
    carrier = copy.deepcopy(ordinary)
    scheduler = SharedCarrierGradientScheduler(
        row_tile=4,
        tasks_per_record=2,
        weight_gradient_layout="inference-balanced",
    )
    config = SharedCarrierConfig(
        row_tile=4,
        expected_training_rows=8,
        weight_gradient_layout="inference-balanced",
    )
    replace_linear_modules_with_shared_carrier(carrier, config, scheduler=scheduler)
    optimizer = InterleavedSGD(
        carrier,
        learning_rate=0.01,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    x = torch.randn(8, 8, dtype=torch.float64)

    ordinary.zero_grad(set_to_none=False)
    ordinary(x).square().sum().backward()
    with torch.no_grad():
        for parameter in ordinary.parameters():
            parameter.add_(parameter.grad, alpha=-0.01)

    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    carrier(x).square().sum().backward()
    audit = scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    optimizer.step_deferred()

    assert audit.direct_gradient_modules == 0
    assert audit.transposed_gradient_modules == 1
    assert audit.layout_transform_values == 56
    assert audit.execution_geometry_counts == {"m4-n7-k8": 2}
    assert audit.redundant_gemm_flops == 0
    torch.testing.assert_close(carrier[0].weight.grad, ordinary[0].weight.grad)
    torch.testing.assert_close(carrier[0].weight, ordinary[0].weight)
    optimizer.close()


def test_strided_scheduler_preserves_tied_embedding_training_step() -> None:
    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.embed(token_ids))

    torch.manual_seed(29)
    ordinary = TiedModel().double()
    carrier = copy.deepcopy(ordinary)
    original_data_ptr = carrier.embed.weight.data_ptr()
    scheduler = SharedCarrierGradientScheduler(
        row_tile=2,
        tasks_per_record=2,
        weight_gradient_layout="inference-balanced-strided",
    )
    config = SharedCarrierConfig(
        row_tile=2,
        expected_training_rows=4,
        weight_gradient_layout="inference-balanced-strided",
    )
    replace_linear_modules_with_shared_carrier(carrier, config, scheduler=scheduler)
    assert isinstance(carrier.embed, SharedCarrierEmbedding)
    assert carrier.embed.weight is carrier.head.weight
    assert carrier.embed.weight.data_ptr() == original_data_ptr
    assert carrier.embed.weight.stride() == (1, 4)

    optimizer = InterleavedSGD(
        carrier,
        learning_rate=0.01,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    token_ids = torch.tensor([0, 1, 2, 3])

    ordinary.zero_grad(set_to_none=False)
    ordinary(token_ids).square().sum().backward()
    with torch.no_grad():
        for parameter in ordinary.parameters():
            parameter.add_(parameter.grad, alpha=-0.01)

    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    carrier(token_ids).square().sum().backward()
    audit = scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    optimizer.step_deferred()

    assert audit.direct_gradient_modules == 0
    assert audit.transposed_gradient_modules == 1
    assert audit.strided_gradient_modules == 1
    assert audit.layout_transform_values == 0
    assert audit.execution_geometry_counts == {"m2-n8-k4": 2}
    assert audit.parameter_updates_deferred == 1
    torch.testing.assert_close(carrier.embed.weight.grad.T, ordinary.embed.weight.grad)
    torch.testing.assert_close(carrier.embed.weight.T, ordinary.embed.weight)
    optimizer.close()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"row_tile": 0}, "row_tile must be positive"),
        ({"expected_training_rows": 0}, "expected_training_rows must be positive"),
        (
            {"row_tile": 3, "expected_training_rows": 8},
            "expected_training_rows must be divisible",
        ),
        (
            {"weight_gradient_layout": "unknown"},
            "weight_gradient_layout must be",
        ),
    ],
)
def test_shared_carrier_config_rejects_invalid_geometry(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SharedCarrierConfig(**kwargs)


def test_shared_carrier_workload_contract_is_strict_and_no_cover() -> None:
    config = StrictWorkloadConfig(
        mode="shaped-training",
        session_id="carrier",
        training_batch_size=2_048,
        training_sequence_length=1,
        tile_rows=1_024,
        shaping_backend="shared-carrier",
        weight_gradient_schedule="inline",
    )
    assert config.strict_invariants()["inference_cover_tokens"] == 0
    assert config.strict_invariants()["filler_kernels"] == 0


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"weight_gradient_schedule": "round-robin"}, "supports only inline"),
        ({"training_batch_size": 2_047}, "must be divisible"),
        ({"kernel_launch_period_us": 1.0}, "does not support host kernel pacing"),
        ({"actuator_operations": (4, 8)}, "does not support appended gradient actuation"),
        (
            {
                "shaping_backend": "tiled-gemm",
                "shared_carrier_weight_gradient_layout": "inference-balanced",
            },
            "requires --shaping-backend shared-carrier",
        ),
    ],
)
def test_shared_carrier_workload_rejects_conflicting_mechanisms(
    overrides: dict,
    message: str,
) -> None:
    arguments = {
        "mode": "shaped-training",
        "session_id": "bad-carrier",
        "training_batch_size": 2_048,
        "training_sequence_length": 1,
        "tile_rows": 1_024,
        "shaping_backend": "shared-carrier",
        "weight_gradient_schedule": "inline",
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        StrictWorkloadConfig(**arguments)
