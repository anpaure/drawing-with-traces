from __future__ import annotations

import math

import pytest
import sidecapture as sc
import torch
from torch import nn

from experiments.llama_strict_inference_shaped_training.build_inference_target import bin_feature
from experiments.llama_strict_inference_shaped_training.calibrate_trace_duty import operation_levels
from experiments.llama_strict_inference_shaped_training.capture_strict import build_parser as capture_parser
from experiments.llama_strict_inference_shaped_training.validate_strict import (
    build_parser as validation_parser,
)

from experiments.llama_strict_inference_shaped_training.strict_shapes import (
    DeferredWeightGradientScheduler,
    StrictM1Linear,
    StrictShapeConfig,
    exact_grouped_m1_weight_gradient,
    exact_row_tiled_weight_gradient,
    execution_plan,
    grouped_shared_rhs_m1,
    replace_linear_modules,
    row_tiled_matmul,
)
from experiments.llama_strict_inference_shaped_training.strict_optimizer import (
    InterleavedSGD,
    shared_parameter_ids,
)
from experiments.llama_strict_inference_shaped_training.strict_workloads import (
    PersistentStrictWorkload,
    StrictWorkloadConfig,
)
from experiments.llama_strict_inference_shaped_training.trace_drawing import (
    GradientTileActuator,
    TransformerGradientCalibrationWorkload,
    plan_operations_per_bin,
)


@pytest.mark.parametrize("rows", [1, 3, 7])
@pytest.mark.parametrize("group", [1, 2, 8])
def test_grouped_shared_rhs_m1_matches_matmul(rows: int, group: int) -> None:
    torch.manual_seed(1)
    lhs = torch.randn(rows, 5, dtype=torch.float64)
    rhs = torch.randn(5, 4, dtype=torch.float64)
    actual = grouped_shared_rhs_m1(lhs, rhs, m1_per_launch=group)
    torch.testing.assert_close(actual, lhs @ rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("rows", [1, 3, 7])
@pytest.mark.parametrize("tile", [1, 2, 8])
def test_row_tiled_matmul_matches_matmul(rows: int, tile: int) -> None:
    torch.manual_seed(11)
    lhs = torch.randn(rows, 5, dtype=torch.float64)
    rhs = torch.randn(5, 4, dtype=torch.float64)
    actual = row_tiled_matmul(lhs, rhs, rows_per_launch=tile)
    torch.testing.assert_close(actual, lhs @ rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("token_rows,reduction_width", [(3, 5), (5, 5), (9, 5)])
@pytest.mark.parametrize("group", [1, 2, 8])
def test_exact_grouped_weight_gradient(token_rows: int, reduction_width: int, group: int) -> None:
    torch.manual_seed(2)
    x = torch.randn(token_rows, 5, dtype=torch.float64)
    grad_output = torch.randn(token_rows, 4, dtype=torch.float64)
    actual = exact_grouped_m1_weight_gradient(
        x,
        grad_output,
        m1_per_launch=group,
        reduction_width=reduction_width,
    )
    torch.testing.assert_close(actual, grad_output.T @ x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("token_rows,reduction_width", [(3, 5), (5, 5), (9, 5)])
@pytest.mark.parametrize("tile", [1, 2, 8])
def test_exact_row_tiled_weight_gradient(token_rows: int, reduction_width: int, tile: int) -> None:
    torch.manual_seed(12)
    x = torch.randn(token_rows, 5, dtype=torch.float64)
    grad_output = torch.randn(token_rows, 4, dtype=torch.float64)
    actual = exact_row_tiled_weight_gradient(
        x,
        grad_output,
        rows_per_launch=tile,
        reduction_width=reduction_width,
    )
    torch.testing.assert_close(actual, grad_output.T @ x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("backend", ["grouped-m1", "tiled-gemm"])
def test_strict_linear_forward_and_backward_match_linear(with_bias: bool, backend: str) -> None:
    torch.manual_seed(3)
    ordinary = nn.Linear(5, 4, bias=with_bias, dtype=torch.float64)
    shaped_source = nn.Linear(5, 4, bias=with_bias, dtype=torch.float64)
    shaped_source.load_state_dict(ordinary.state_dict())
    shaped = StrictM1Linear(
        shaped_source,
        StrictShapeConfig(
            backend=backend,
            forward_m1_per_launch=2,
            input_gradient_m1_per_launch=3,
            weight_gradient_m1_per_launch=2,
        ),
    )
    x_ordinary = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    x_shaped = x_ordinary.detach().clone().requires_grad_(True)
    target = torch.randn(2, 3, 4, dtype=torch.float64)

    ordinary_loss = (ordinary(x_ordinary) - target).square().mean()
    shaped_loss = (shaped(x_shaped) - target).square().mean()
    ordinary_loss.backward()
    shaped_loss.backward()

    torch.testing.assert_close(shaped_loss, ordinary_loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x_shaped.grad, x_ordinary.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shaped.weight.grad, ordinary.weight.grad, rtol=1e-12, atol=1e-12)
    if with_bias:
        torch.testing.assert_close(shaped.bias.grad, ordinary.bias.grad, rtol=1e-12, atol=1e-12)


def test_execution_plan_accounts_for_padding() -> None:
    config = StrictShapeConfig(
        forward_m1_per_launch=4,
        input_gradient_m1_per_launch=3,
        weight_gradient_m1_per_launch=2,
    )
    plan = execution_plan(input_rows=3, input_features=5, output_features=7, config=config)
    assert plan.forward_launches == 1
    assert plan.input_gradient_launches == 1
    assert plan.weight_gradient_launches == 3
    assert plan.weight_gradient_reduction_width == 5
    assert plan.useful_flops == 3 * (2 * 3 * 5 * 7)
    assert plan.executed_flops == 2 * 3 * 5 * 7 + 2 * 3 * 5 * 7 + 2 * 5 * 5 * 7
    assert plan.redundant_flops == 2 * (5 - 3) * 5 * 7


def test_replace_linear_modules_preserves_parameters() -> None:
    model = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Sequential(nn.Linear(4, 3)))
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    names = replace_linear_modules(model, StrictShapeConfig())
    assert names == ["0", "2.0"]
    assert isinstance(model[0], StrictM1Linear)
    assert isinstance(model[2][0], StrictM1Linear)
    assert {id(parameter) for parameter in model.parameters()} == parameter_ids


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_m1_per_launch": 0},
        {"input_gradient_m1_per_launch": 0},
        {"weight_gradient_m1_per_launch": 0},
    ],
)
def test_shape_config_rejects_nonpositive_groups(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        StrictShapeConfig(**kwargs)


def test_shape_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown strict shaping backend"):
        StrictShapeConfig(backend="not-real")  # type: ignore[arg-type]


def test_shape_config_rejects_unknown_weight_gradient_schedule() -> None:
    with pytest.raises(ValueError, match="unknown weight-gradient schedule"):
        StrictShapeConfig(weight_gradient_schedule="not-real")  # type: ignore[arg-type]


def test_shape_config_rejects_nonpositive_streaming_budget() -> None:
    with pytest.raises(ValueError, match="streaming_weight_gradient_tasks_per_record"):
        StrictShapeConfig(streaming_weight_gradient_tasks_per_record=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grouped_weight_gradient_min_batch": 1},
        {
            "grouped_weight_gradient_min_batch": 8,
            "grouped_weight_gradient_max_batch": 4,
        },
    ],
)
def test_shape_config_rejects_invalid_grouped_batches(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="grouped_weight_gradient"):
        StrictShapeConfig(**kwargs)


@pytest.mark.parametrize(
    "schedule,executed_launches,redundant_launches,redundant_flops",
    [
        ("round-robin", 7, 0, 196),
        ("balanced-round-robin", 8, 1, 336),
        ("streaming-round-robin", 7, 0, 196),
        ("streaming-inference-cycle", 7, 0, 196),
        ("streaming-grouped", 7, 0, 196),
    ],
)
def test_round_robin_weight_gradients_and_updates_match_ordinary_sgd(
    schedule: str,
    executed_launches: int,
    redundant_launches: int,
    redundant_flops: int,
) -> None:
    torch.manual_seed(19)
    ordinary = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    shaped = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    shaped.load_state_dict(ordinary.state_dict())
    scheduler = DeferredWeightGradientScheduler()
    replace_linear_modules(
        shaped,
        StrictShapeConfig(
            backend="tiled-gemm",
            forward_m1_per_launch=2,
            input_gradient_m1_per_launch=2,
            weight_gradient_m1_per_launch=2,
            weight_gradient_schedule=schedule,  # type: ignore[arg-type]
        ),
        scheduler=scheduler,
    )
    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    learning_rate = 0.03

    ordinary_loss = (ordinary(x) - target).square().mean()
    ordinary_loss.backward()
    with torch.no_grad():
        for parameter in ordinary.parameters():
            parameter.add_(parameter.grad, alpha=-learning_rate)

    optimizer = InterleavedSGD(
        shaped,
        learning_rate=learning_rate,
        manual_parameter_ids=scheduler.parameter_ids,
        manual_update_bucket_size=2,
    )
    optimizer.zero_grad(set_to_none=False)
    before_backward = {
        id(parameter): parameter.detach().clone()
        for parameter in shaped.parameters()
        if id(parameter) in scheduler.parameter_ids
    }
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    shaped_loss = (shaped(x) - target).square().mean()
    shaped_loss.backward()
    if schedule == "streaming-round-robin":
        assert all(
            not torch.equal(parameter, before_backward[id(parameter)])
            for parameter in shaped.parameters()
            if id(parameter) in scheduler.parameter_ids
        )
    else:
        assert all(
            torch.equal(parameter, before_backward[id(parameter)])
            for parameter in shaped.parameters()
            if id(parameter) in scheduler.parameter_ids
        )
    audit = scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    optimizer.step_deferred()

    torch.testing.assert_close(shaped_loss, ordinary_loss, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(shaped.parameters(), ordinary.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert audit.recorded_invocations == 2
    assert audit.gemm_launches == 7
    assert audit.executed_gemm_launches == executed_launches
    assert audit.physical_gemm_launches == executed_launches
    assert audit.grouped_gemm_launches == 0
    assert audit.grouped_gemm_tasks == 0
    assert audit.redundant_gemm_launches == redundant_launches
    assert audit.redundant_gemm_flops == redundant_flops
    assert audit.executed_gemm_flops == audit.useful_gemm_flops + redundant_flops
    assert audit.parameter_updates_interleaved == 2
    assert audit.parameter_updates_deferred == 0
    assert optimizer.audit().manual_parameter_tensors == 2
    assert optimizer.audit().manual_update_bucket_size == 2
    assert optimizer.audit().fused_manual_update_flushes == 1
    assert optimizer.audit().fused_manual_update_tensors == 2
    optimizer.close()


def test_inference_cycle_scheduler_uses_forward_projection_grammar() -> None:
    torch.manual_seed(393)

    class ProjectionBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.k_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.v_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.o_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.gate_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.up_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)
            self.down_proj = nn.Linear(3, 3, bias=False, dtype=torch.float64)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            for module in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj,
                self.gate_proj,
                self.up_proj,
                self.down_proj,
            ):
                value = module(value)
            return value

    model = ProjectionBlock()
    scheduler = DeferredWeightGradientScheduler()
    replace_linear_modules(
        model,
        StrictShapeConfig(
            backend="tiled-gemm",
            forward_m1_per_launch=2,
            input_gradient_m1_per_launch=2,
            weight_gradient_m1_per_launch=3,
            weight_gradient_schedule="streaming-inference-cycle",
            streaming_weight_gradient_tasks_per_record=7,
        ),
        scheduler=scheduler,
    )
    optimizer = InterleavedSGD(
        model,
        learning_rate=0.01,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    model(torch.randn(2, 3, dtype=torch.float64)).square().mean().backward()
    scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    optimizer.step_deferred()

    assert scheduler.execution_family_trace() == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    optimizer.close()


@pytest.mark.parametrize("bucket_size", [1, 2, 3, 8])
def test_fused_manual_optimizer_matches_scalar_updates_across_partial_buckets(
    bucket_size: int,
) -> None:
    torch.manual_seed(391)
    scalar = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 3), nn.Linear(3, 2))
    fused = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 3), nn.Linear(3, 2))
    fused.load_state_dict(scalar.state_dict())
    scalar_parameters = list(scalar.parameters())
    fused_parameters = list(fused.parameters())
    learning_rate = 0.03125
    scalar_optimizer = InterleavedSGD(
        scalar,
        learning_rate=learning_rate,
        manual_parameter_ids={id(parameter) for parameter in scalar_parameters},
        manual_update_bucket_size=1,
    )
    fused_optimizer = InterleavedSGD(
        fused,
        learning_rate=learning_rate,
        manual_parameter_ids={id(parameter) for parameter in fused_parameters},
        manual_update_bucket_size=bucket_size,
    )
    for index, (scalar_parameter, fused_parameter) in enumerate(
        zip(scalar_parameters, fused_parameters, strict=True)
    ):
        gradient = torch.randn_like(scalar_parameter) * (index + 1)
        scalar_parameter.grad = gradient.clone()
        fused_parameter.grad = gradient.clone()
        scalar_optimizer.step_manual(scalar_parameter)
        fused_optimizer.step_manual(fused_parameter)
    scalar_optimizer.step_deferred()
    fused_optimizer.step_deferred()
    for scalar_parameter, fused_parameter in zip(scalar_parameters, fused_parameters, strict=True):
        assert torch.equal(fused_parameter, scalar_parameter)
    audit = fused_optimizer.audit()
    assert audit.fused_manual_update_flushes == math.ceil(len(fused_parameters) / bucket_size)
    assert audit.fused_manual_update_tensors == len(fused_parameters)
    scalar_optimizer.close()
    fused_optimizer.close()


def test_interleaved_sgd_matches_conventional_sgd() -> None:
    torch.manual_seed(21)
    ordinary = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    interleaved = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    interleaved.load_state_dict(ordinary.state_dict())
    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    learning_rate = 0.03

    ordinary_loss = (ordinary(x) - target).square().mean()
    ordinary_loss.backward()
    with torch.no_grad():
        for parameter in ordinary.parameters():
            parameter.add_(parameter.grad, alpha=-learning_rate)

    optimizer = InterleavedSGD(interleaved, learning_rate=learning_rate)
    optimizer.zero_grad(set_to_none=False)
    interleaved_loss = (interleaved(x) - target).square().mean()
    interleaved_loss.backward()
    optimizer.step_deferred()

    torch.testing.assert_close(interleaved_loss, ordinary_loss, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(interleaved.parameters(), ordinary.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert optimizer.audit().interleaved_parameter_tensors == 4
    assert optimizer.audit().deferred_shared_parameter_tensors == 0
    optimizer.close()


@pytest.mark.parametrize("bucket_size", [1, 2, 3, 8])
def test_fused_interleaved_optimizer_matches_scalar_hook_updates(bucket_size: int) -> None:
    torch.manual_seed(392)
    scalar = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    fused = nn.Sequential(nn.Linear(5, 7), nn.SiLU(), nn.Linear(7, 3)).double()
    fused.load_state_dict(scalar.state_dict())
    x = torch.randn(4, 5, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    scalar_optimizer = InterleavedSGD(
        scalar,
        learning_rate=0.03,
        manual_update_bucket_size=1,
    )
    fused_optimizer = InterleavedSGD(
        fused,
        learning_rate=0.03,
        manual_update_bucket_size=bucket_size,
    )
    scalar_optimizer.zero_grad(set_to_none=False)
    fused_optimizer.zero_grad(set_to_none=False)
    scalar_loss = (scalar(x) - target).square().mean()
    fused_loss = (fused(x) - target).square().mean()
    scalar_loss.backward()
    fused_loss.backward()
    scalar_optimizer.step_deferred()
    fused_optimizer.step_deferred()
    assert torch.equal(fused_loss, scalar_loss)
    for scalar_parameter, fused_parameter in zip(scalar.parameters(), fused.parameters(), strict=True):
        assert torch.equal(fused_parameter.grad, scalar_parameter.grad)
        assert torch.equal(fused_parameter, scalar_parameter)
    audit = fused_optimizer.audit()
    assert audit.interleaved_update_bucket_size == bucket_size
    assert audit.fused_interleaved_update_flushes == math.ceil(len(list(fused.parameters())) / bucket_size)
    assert audit.fused_interleaved_update_tensors == len(list(fused.parameters()))
    scalar_optimizer.close()
    fused_optimizer.close()


class _TiedLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 3, bias=False, dtype=torch.float64)
        self.second = nn.Linear(3, 3, bias=False, dtype=torch.float64)
        self.second.weight = self.first.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.first(x) + self.second(x)


def test_interleaved_sgd_defers_shared_parameter_until_backward_finishes() -> None:
    torch.manual_seed(22)
    ordinary = _TiedLinearModel()
    interleaved = _TiedLinearModel()
    interleaved.load_state_dict(ordinary.state_dict())
    x = torch.randn(4, 3, dtype=torch.float64)
    learning_rate = 0.02

    ordinary(x).square().mean().backward()
    with torch.no_grad():
        ordinary.first.weight.add_(ordinary.first.weight.grad, alpha=-learning_rate)

    assert shared_parameter_ids(interleaved) == {id(interleaved.first.weight)}
    optimizer = InterleavedSGD(interleaved, learning_rate=learning_rate)
    optimizer.zero_grad(set_to_none=False)
    interleaved(x).square().mean().backward()
    before_deferred = interleaved.first.weight.detach().clone()
    optimizer.step_deferred()

    assert not torch.equal(before_deferred, interleaved.first.weight)
    torch.testing.assert_close(interleaved.first.weight, ordinary.first.weight, rtol=1e-12, atol=1e-12)
    assert optimizer.audit().interleaved_parameter_tensors == 0
    assert optimizer.audit().deferred_shared_parameter_tensors == 1
    optimizer.close()


@pytest.mark.parametrize("schedule", ["round-robin", "streaming-round-robin"])
def test_round_robin_scheduler_accumulates_shared_weight_before_one_update(
    schedule: str,
) -> None:
    torch.manual_seed(23)
    ordinary = _TiedLinearModel()
    shaped = _TiedLinearModel()
    shaped.load_state_dict(ordinary.state_dict())
    scheduler = DeferredWeightGradientScheduler()
    replace_linear_modules(
        shaped,
        StrictShapeConfig(
            backend="tiled-gemm",
            forward_m1_per_launch=2,
            input_gradient_m1_per_launch=2,
            weight_gradient_m1_per_launch=2,
            weight_gradient_schedule=schedule,  # type: ignore[arg-type]
        ),
        scheduler=scheduler,
    )
    x = torch.randn(4, 3, dtype=torch.float64)
    learning_rate = 0.02

    ordinary(x).square().mean().backward()
    with torch.no_grad():
        ordinary.first.weight.add_(ordinary.first.weight.grad, alpha=-learning_rate)

    optimizer = InterleavedSGD(
        shaped,
        learning_rate=learning_rate,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    shaped(x).square().mean().backward()
    before_deferred = shaped.first.weight.detach().clone()
    audit = scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    assert torch.equal(shaped.first.weight, before_deferred)
    optimizer.step_deferred()

    torch.testing.assert_close(
        shaped.first.weight.grad,
        ordinary.first.weight.grad,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        shaped.first.weight,
        ordinary.first.weight,
        rtol=1e-12,
        atol=1e-12,
    )
    assert audit.recorded_invocations == 2
    assert audit.gemm_launches == 8
    assert audit.parameter_updates_interleaved == 0
    assert audit.parameter_updates_deferred == 1
    assert optimizer.audit().manual_parameter_tensors == 0
    assert optimizer.audit().deferred_shared_parameter_tensors == 1
    optimizer.close()


def test_strict_workload_has_no_cover_or_second_model_controls() -> None:
    config = StrictWorkloadConfig(mode="shaped-training", session_id="strict-test")
    fields = set(config.__dataclass_fields__)
    assert all("cover" not in field for field in fields)
    assert all("secondary" not in field for field in fields)
    assert config.strict_invariants() == {
        "inference_cover_tokens": 0,
        "secondary_model_instances": 0,
        "filler_kernels": 0,
        "all_extra_flops_are_accounted_training_arithmetic": True,
        "reduction_padding_flops_are_accounted": True,
        "redundant_gradient_recomputation_flops_are_accounted": True,
        "optimizer_updates_use_real_current_gradients": True,
    }


def test_persistent_workload_implements_sidecapture_protocol() -> None:
    workload = PersistentStrictWorkload(
        StrictWorkloadConfig(mode="shaped-training", session_id="protocol-test")
    )
    assert isinstance(workload, sc.Workload)
    assert workload.snapshot() is None


def test_gradient_tile_actuator_recomputes_current_real_dw_block() -> None:
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(5, 4, bias=False, dtype=torch.float64))
    scheduler = DeferredWeightGradientScheduler()
    replace_linear_modules(
        model,
        StrictShapeConfig(
            backend="tiled-gemm",
            forward_m1_per_launch=2,
            input_gradient_m1_per_launch=2,
            weight_gradient_m1_per_launch=2,
            weight_gradient_schedule="round-robin",
        ),
        scheduler=scheduler,
    )
    optimizer = InterleavedSGD(
        model,
        learning_rate=0.01,
        manual_parameter_ids=scheduler.parameter_ids,
    )
    x = torch.randn(3, 5, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=False)
    scheduler.begin_step()
    model(x).square().mean().backward()
    scheduler.finish_step(
        update_parameter=optimizer.step_manual,
        deferred_parameter_ids=optimizer.deferred_parameter_ids,
    )
    grad_x, grad_output = scheduler.gradient_operands("0")
    actuator = GradientTileActuator(scheduler, "0", width_quantum=1)
    actuator.execute(2, synchronize=False)

    torch.testing.assert_close(actuator.last_result, grad_x[:, :2].T @ grad_output)
    assert actuator.maximum_width == 5
    assert actuator.operations == 1
    assert actuator.executed_flops == 2 * 2 * 3 * 4
    optimizer.close()


def test_graph_operation_planner_is_bounded_and_conservative() -> None:
    assert plan_operations_per_bin(0.02, 1.0) == 41
    assert plan_operations_per_bin(0.001, 1.0) == 256
    assert plan_operations_per_bin(2.0, 1.0) == 1


@pytest.mark.parametrize(
    "operation_ms,bin_ms,target_utilization,maximum_operations",
    [
        (0.0, 1.0, 0.5, 32),
        (1.0, 0.0, 0.5, 32),
        (1.0, 1.0, 1.0, 32),
        (1.0, 1.0, 0.5, 0),
    ],
)
def test_graph_operation_planner_rejects_invalid_inputs(
    operation_ms: float,
    bin_ms: float,
    target_utilization: float,
    maximum_operations: int,
) -> None:
    with pytest.raises(ValueError):
        plan_operations_per_bin(
            operation_ms,
            bin_ms,
            target_utilization=target_utilization,
            maximum_operations=maximum_operations,
        )


@pytest.mark.parametrize("feature", ["rms", "std", "mean_abs", "diff_rms", "q98_q02_span"])
def test_inference_target_features_are_finite(feature: str) -> None:
    trace = torch.linspace(-1, 1, 40, dtype=torch.float64).numpy()
    values = bin_feature(trace, 10, feature)
    assert values.shape == (4,)
    assert torch.isfinite(torch.from_numpy(values)).all()


def test_explicit_operation_commands_are_validated_without_cuda_setup() -> None:
    commands = torch.tensor([160, 160, 160]).numpy()
    workload = TransformerGradientCalibrationWorkload(
        commands,
        operation_commands=torch.tensor([1, 16, 64]).numpy(),
        bin_duration_s=0.001,
    )
    assert workload.operation_commands.tolist() == [1, 16, 64]
    assert workload.metadata()["operation_commands"] == [1, 16, 64]
    with pytest.raises(ValueError, match="same shape"):
        TransformerGradientCalibrationWorkload(
            commands,
            operation_commands=torch.tensor([1, 2]).numpy(),
            bin_duration_s=0.001,
        )


def test_duty_calibration_levels_include_requested_safe_maximum() -> None:
    levels = operation_levels(104)
    assert levels[0] == 1
    assert levels[-1] == 104
    assert {32, 40, 64, 96}.issubset(set(levels))
    sparse = operation_levels(480, 12)
    assert len(sparse) == 12
    assert sparse[-1] == 480


@pytest.mark.parametrize("mode", ["inference", "ordinary-training", "shaped-training"])
def test_strict_workload_accepts_only_explicit_modes(mode: str) -> None:
    config = StrictWorkloadConfig(mode=mode, session_id="mode-test")  # type: ignore[arg-type]
    assert config.mode == mode


def test_strict_workload_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown workload mode"):
        StrictWorkloadConfig(mode="covered-training", session_id="bad")  # type: ignore[arg-type]


def test_strict_workload_rejects_nonpositive_streaming_budget() -> None:
    with pytest.raises(ValueError, match="streaming_weight_gradient_tasks_per_record"):
        StrictWorkloadConfig(
            mode="shaped-training",
            session_id="bad-streaming-budget",
            streaming_weight_gradient_tasks_per_record=0,
        )


def test_strict_workload_rejects_invalid_grouped_batches() -> None:
    with pytest.raises(ValueError, match="grouped_weight_gradient"):
        StrictWorkloadConfig(
            mode="shaped-training",
            session_id="bad-grouped-batch",
            grouped_weight_gradient_min_batch=8,
            grouped_weight_gradient_max_batch=4,
        )


def test_capture_cli_can_disable_cuda_graphs() -> None:
    args = capture_parser().parse_args(
        [
            "--mode",
            "shaped-training",
            "--session-id",
            "eager-test",
            "--output-dir",
            "captures",
            "--no-cuda-graph",
        ]
    )
    assert args.cuda_graph is False


def test_kernel_launch_pacing_requires_eager_deferred_execution() -> None:
    with pytest.raises(ValueError, match="requires --no-cuda-graph"):
        StrictWorkloadConfig(
            mode="shaped-training",
            session_id="paced-graph",
            kernel_launch_period_us=800.0,
        )
    with pytest.raises(ValueError, match="deferred weight-gradient"):
        StrictWorkloadConfig(
            mode="shaped-training",
            session_id="paced-inline",
            cuda_graph=False,
            weight_gradient_schedule="inline",
            kernel_launch_period_us=800.0,
        )
    config = StrictWorkloadConfig(
        mode="shaped-training",
        session_id="paced-eager",
        cuda_graph=False,
        kernel_launch_period_us=800.0,
    )
    assert config.kernel_launch_period_us == 800.0


def test_full_model_validator_accepts_selected_grouped_backend() -> None:
    args = validation_parser().parse_args(
        [
            "--output",
            "validation.json",
            "--shaping-backend",
            "grouped-m1",
        ]
    )
    assert args.shaping_backend == "grouped-m1"


def test_continuous_gradient_actuation_is_no_cover_and_accounted() -> None:
    config = StrictWorkloadConfig(
        mode="shaped-training",
        session_id="actuated",
        actuator_width=768,
        actuator_operations=(160, 176, 184),
    )
    assert config.actuator_operations == (160, 176, 184)
    assert config.strict_invariants()["inference_cover_tokens"] == 0
    assert config.strict_invariants()["filler_kernels"] == 0
    assert config.strict_invariants()["redundant_gradient_recomputation_flops_are_accounted"]


@pytest.mark.parametrize("mode", ["inference", "ordinary-training"])
def test_gradient_actuation_rejects_non_shaped_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="only valid for shaped-training"):
        StrictWorkloadConfig(
            mode=mode,  # type: ignore[arg-type]
            session_id="bad-actuation",
            actuator_operations=(8, 16),
        )
