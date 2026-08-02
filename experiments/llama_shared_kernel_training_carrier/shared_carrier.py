"""Exact forward, dX, and dW through one fixed-row high-MFU GEMM carrier.

This is deliberately implemented as a custom autograd schedule over ``torch.mm``.
The carrier therefore retains cuBLAS performance while forcing all three linear
training paths through the same operator and a small inference-derived geometry
set. It is the feasibility layer before a direct cuBLASLt/CUTLASS implementation;
it does not claim that different GEMM shapes resolve to one vendor kernel binary.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SharedCarrierConfig:
    """Geometry contract for one shared forward/dX/dW carrier."""

    row_tile: int = 1024
    expected_training_rows: int = 2048
    require_exact_training_rows: bool = True
    weight_gradient_layout: Literal[
        "direct",
        "inference-balanced",
        "inference-balanced-strided",
    ] = "direct"
    gemm_backend: Literal["torch-mm", "identical-triton"] = "torch-mm"

    def __post_init__(self) -> None:
        if self.row_tile < 1:
            raise ValueError("row_tile must be positive")
        if self.expected_training_rows < 1:
            raise ValueError("expected_training_rows must be positive")
        if self.expected_training_rows % self.row_tile:
            raise ValueError("expected_training_rows must be divisible by row_tile")
        if self.weight_gradient_layout not in {
            "direct",
            "inference-balanced",
            "inference-balanced-strided",
        }:
            raise ValueError(
                "weight_gradient_layout must be direct, inference-balanced, "
                "or inference-balanced-strided"
            )
        if self.gemm_backend not in {"torch-mm", "identical-triton"}:
            raise ValueError("gemm_backend must be torch-mm or identical-triton")

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "carrier": (
                "one runtime-shape/stride Triton GEMM binary"
                if self.gemm_backend == "identical-triton"
                else "custom autograd schedule over torch.mm/cuBLAS"
            ),
            "single_vendor_kernel_binary_claimed": False,
            "single_compiled_kernel_binary_requires_runtime_audit": (
                self.gemm_backend == "identical-triton"
            ),
            "all_carrier_gemms_are_useful_training_arithmetic": True,
        }


@dataclass(frozen=True)
class CarrierExecutionPlan:
    """Exact launch and useful-FLOP accounting for one linear invocation."""

    input_rows: int
    input_features: int
    output_features: int
    row_tile: int
    gemm_backend: str
    weight_gradient_layout: str
    forward_launches: int
    input_gradient_launches: int
    weight_gradient_launches: int
    useful_forward_flops: int
    useful_input_gradient_flops: int
    useful_weight_gradient_flops: int
    layout_transform_values: int
    executed_flops: int
    redundant_flops: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharedCarrierGradientAudit:
    """Exact work and schedule audit for deferred carrier dW."""

    schedule: str
    weight_gradient_layout: str
    gemm_backend: str
    registered_modules: int
    registered_parameter_tensors: int
    recorded_invocations: int
    gemm_launches: int
    useful_gemm_flops: int
    redundant_gemm_flops: int
    parameter_updates_interleaved: int
    parameter_updates_deferred: int
    direct_gradient_modules: int
    transposed_gradient_modules: int
    strided_gradient_modules: int
    layout_transform_values: int
    execution_family_counts: dict[str, int]
    execution_family_transitions: int
    execution_geometry_counts: dict[str, int]
    execution_geometry_transitions: int


@dataclass
class _GradientRecord:
    module_name: str
    order: int
    weight: Tensor
    weight_is_transposed: bool
    x: Tensor
    grad_output: Tensor


@dataclass
class _GradientState:
    record: _GradientRecord
    tasks: list[tuple[Tensor, Tensor, Tensor]]
    layout: Literal["direct", "transposed", "transposed-strided"]
    transposed_gradient: Tensor | None = None
    next_task: int = 0
    committed: bool = False


class SharedCarrierGradientScheduler:
    """Stream exact dW tiles among dX operations in inference-family order."""

    _INFERENCE_PROJECTION_CYCLE = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    )

    def __init__(
        self,
        *,
        row_tile: int,
        tasks_per_record: int = 5,
        weight_gradient_layout: Literal[
            "direct",
            "inference-balanced",
            "inference-balanced-strided",
        ] = "direct",
        gemm_backend: Literal["torch-mm", "identical-triton"] = "torch-mm",
    ) -> None:
        if row_tile < 1:
            raise ValueError("row_tile must be positive")
        if tasks_per_record < 1:
            raise ValueError("tasks_per_record must be positive")
        if weight_gradient_layout not in {
            "direct",
            "inference-balanced",
            "inference-balanced-strided",
        }:
            raise ValueError(
                "weight_gradient_layout must be direct, inference-balanced, "
                "or inference-balanced-strided"
            )
        if gemm_backend not in {"torch-mm", "identical-triton"}:
            raise ValueError("gemm_backend must be torch-mm or identical-triton")
        self.row_tile = int(row_tile)
        self.tasks_per_record = int(tasks_per_record)
        self.weight_gradient_layout = weight_gradient_layout
        self.gemm_backend = gemm_backend
        self._registrations: dict[str, tuple[int, Tensor, bool]] = {}
        self._records: list[_GradientRecord] = []
        self._states: list[_GradientState] = []
        self._update_parameter: Callable[[Tensor], None] | None = None
        self._deferred_parameter_ids: set[int] = set()
        self._cycle_cursor = 0
        self._execution_families: list[str] = []
        self._execution_geometries: list[str] = []
        self._gemm_launches = 0
        self._useful_flops = 0
        self._layout_transform_values = 0
        self._interleaved_updates = 0
        self._last_audit = SharedCarrierGradientAudit(
            schedule="streaming-inference-cycle",
            weight_gradient_layout=self.weight_gradient_layout,
            gemm_backend=self.gemm_backend,
            registered_modules=0,
            registered_parameter_tensors=0,
            recorded_invocations=0,
            gemm_launches=0,
            useful_gemm_flops=0,
            redundant_gemm_flops=0,
            parameter_updates_interleaved=0,
            parameter_updates_deferred=0,
            direct_gradient_modules=0,
            transposed_gradient_modules=0,
            strided_gradient_modules=0,
            layout_transform_values=0,
            execution_family_counts={},
            execution_family_transitions=0,
            execution_geometry_counts={},
            execution_geometry_transitions=0,
        )

    def register(
        self,
        module_name: str,
        weight: Tensor,
        *,
        weight_is_transposed: bool,
    ) -> None:
        if module_name in self._registrations:
            raise ValueError(f"duplicate shared-carrier module name: {module_name}")
        self._registrations[module_name] = (
            len(self._registrations),
            weight,
            weight_is_transposed,
        )

    @property
    def parameter_ids(self) -> set[int]:
        return {id(weight) for _order, weight, _transposed in self._registrations.values()}

    def begin_step(
        self,
        *,
        update_parameter: Callable[[Tensor], None] | None = None,
        deferred_parameter_ids: set[int] | None = None,
    ) -> None:
        if (update_parameter is None) != (deferred_parameter_ids is None):
            raise ValueError("streaming dW requires both update callback and deferred parameter IDs")
        self._records.clear()
        self._states.clear()
        self._update_parameter = update_parameter
        self._deferred_parameter_ids = set(deferred_parameter_ids or ())
        self._cycle_cursor = 0
        self._execution_families.clear()
        self._execution_geometries.clear()
        self._gemm_launches = 0
        self._useful_flops = 0
        self._layout_transform_values = 0
        self._interleaved_updates = 0

    @classmethod
    def _projection_family(cls, module_name: str) -> str:
        for family in cls._INFERENCE_PROJECTION_CYCLE:
            if module_name.endswith(family):
                return family
        return "other"

    def _build_state(self, record: _GradientRecord) -> _GradientState:
        if record.weight.grad is None:
            raise RuntimeError(f"missing fixed gradient buffer for {record.module_name}")
        tasks: list[tuple[Tensor, Tensor, Tensor]] = []
        if record.weight_is_transposed:
            if self.weight_gradient_layout != "inference-balanced-strided":
                raise RuntimeError(
                    "transposed parameter storage requires inference-balanced-strided dW"
                )
            transposed_x = record.x.transpose(0, 1)
            for start in range(0, transposed_x.shape[0], self.row_tile):
                end = min(start + self.row_tile, transposed_x.shape[0])
                tasks.append(
                    (
                        record.weight.grad[start:end],
                        transposed_x[start:end],
                        record.grad_output,
                    )
                )
            return _GradientState(
                record=record,
                tasks=tasks,
                layout="transposed-strided",
            )

        use_transposed = (
            self.weight_gradient_layout == "inference-balanced"
            and record.x.shape[0] == record.weight.shape[1]
        )
        if use_transposed:
            transposed_gradient = record.x.new_empty(
                (record.weight.shape[1], record.weight.shape[0])
            )
            transposed_x = record.x.transpose(0, 1)
            for start in range(0, transposed_x.shape[0], self.row_tile):
                end = min(start + self.row_tile, transposed_x.shape[0])
                tasks.append(
                    (
                        transposed_gradient[start:end],
                        transposed_x[start:end],
                        record.grad_output,
                    )
                )
            return _GradientState(
                record=record,
                tasks=tasks,
                layout="transposed",
                transposed_gradient=transposed_gradient,
            )

        for start in range(0, record.grad_output.shape[1], self.row_tile):
            end = min(start + self.row_tile, record.grad_output.shape[1])
            tasks.append(
                (
                    record.weight.grad[start:end],
                    record.grad_output[:, start:end].transpose(0, 1),
                    record.x,
                )
            )
        return _GradientState(record=record, tasks=tasks, layout="direct")

    def _commit(self, state: _GradientState) -> None:
        if state.committed:
            return
        state.committed = True
        parameter_id = id(state.record.weight)
        if state.transposed_gradient is not None:
            if state.record.weight.grad is None:
                raise RuntimeError(
                    f"missing fixed gradient buffer for {state.record.module_name}"
                )
            transposed = state.transposed_gradient.transpose(0, 1)
            if parameter_id in self._deferred_parameter_ids:
                state.record.weight.grad.add_(transposed)
            else:
                state.record.weight.grad.copy_(transposed)
            self._layout_transform_values += state.record.weight.numel()
        if parameter_id in self._deferred_parameter_ids:
            return
        if self._update_parameter is None:
            raise RuntimeError("shared-carrier dW completed without an optimizer callback")
        self._update_parameter(state.record.weight)
        self._interleaved_updates += 1

    @torch.no_grad()
    def _execute(self, state: _GradientState) -> None:
        destination, left, right = state.tasks[state.next_task]
        accumulate = (
            state.transposed_gradient is None
            and id(state.record.weight) in self._deferred_parameter_ids
            and self.gemm_backend == "torch-mm"
        )
        _matrix_multiply_into(
            left,
            right,
            destination,
            backend=self.gemm_backend,
            accumulate=accumulate,
            semantic_id=(
                f"{state.record.module_name}:dw:{state.layout}:tile{state.next_task}"
            ),
        )
        family = self._projection_family(state.record.module_name)
        self._execution_families.append(family)
        self._execution_geometries.append(
            f"m{left.shape[0]}-n{right.shape[1]}-k{left.shape[1]}"
        )
        self._gemm_launches += 1
        self._useful_flops += 2 * left.shape[0] * left.shape[1] * right.shape[1]
        state.next_task += 1
        if state.next_task == len(state.tasks):
            self._commit(state)

    def _run_tasks(self, budget: int) -> None:
        for _ in range(budget):
            active = [state for state in self._states if state.next_task < len(state.tasks)]
            if not active:
                return
            selected = None
            selected_cycle = None
            for offset in range(len(self._INFERENCE_PROJECTION_CYCLE)):
                cycle = (self._cycle_cursor + offset) % len(self._INFERENCE_PROJECTION_CYCLE)
                family = self._INFERENCE_PROJECTION_CYCLE[cycle]
                candidates = [
                    state for state in active if self._projection_family(state.record.module_name) == family
                ]
                if candidates:
                    selected = min(candidates, key=lambda state: state.record.order)
                    selected_cycle = cycle
                    break
            if selected is None:
                selected = min(active, key=lambda state: state.record.order)
            elif selected_cycle is not None:
                self._cycle_cursor = (selected_cycle + 1) % len(self._INFERENCE_PROJECTION_CYCLE)
            self._execute(selected)

    def record(
        self,
        *,
        module_name: str,
        weight: Tensor,
        weight_is_transposed: bool,
        x: Tensor,
        grad_output: Tensor,
    ) -> None:
        registration = self._registrations.get(module_name)
        if registration is None:
            raise RuntimeError(f"unregistered shared-carrier module: {module_name}")
        order, registered_weight, registered_transposed = registration
        if registered_weight is not weight:
            raise RuntimeError(f"weight identity changed for {module_name}")
        if registered_transposed != weight_is_transposed:
            raise RuntimeError(f"weight storage layout changed for {module_name}")
        record = _GradientRecord(
            module_name,
            order,
            weight,
            weight_is_transposed,
            x,
            grad_output,
        )
        self._records.append(record)
        self._states.append(self._build_state(record))
        self._run_tasks(self.tasks_per_record)

    def _validate_records(self) -> None:
        if len(self._records) != len(self._registrations):
            raise RuntimeError(
                "shared-carrier dW expected exactly one invocation per module; "
                f"recorded {len(self._records)} for {len(self._registrations)}"
            )
        names = [record.module_name for record in self._records]
        duplicates = [name for name, count in Counter(names).items() if count != 1]
        if duplicates:
            raise RuntimeError(f"shared-carrier modules were invoked more than once: {duplicates[:5]}")

    def finish_step(
        self,
        *,
        update_parameter: Callable[[Tensor], None],
        deferred_parameter_ids: set[int],
    ) -> SharedCarrierGradientAudit:
        self._validate_records()
        if self._update_parameter is None:
            self._update_parameter = update_parameter
            self._deferred_parameter_ids = set(deferred_parameter_ids)
        remaining = sum(len(state.tasks) - state.next_task for state in self._states)
        self._run_tasks(remaining)
        if any(not state.committed for state in self._states):
            raise RuntimeError("shared-carrier dW scheduler did not commit every state")
        family_counts = dict(Counter(self._execution_families))
        transitions = sum(
            left != right
            for left, right in zip(
                self._execution_families,
                self._execution_families[1:],
                strict=False,
            )
        )
        geometry_counts = dict(Counter(self._execution_geometries))
        geometry_transitions = sum(
            left != right
            for left, right in zip(
                self._execution_geometries,
                self._execution_geometries[1:],
                strict=False,
            )
        )
        self._last_audit = SharedCarrierGradientAudit(
            schedule="streaming-inference-cycle",
            weight_gradient_layout=self.weight_gradient_layout,
            gemm_backend=self.gemm_backend,
            registered_modules=len(self._registrations),
            registered_parameter_tensors=len(self.parameter_ids),
            recorded_invocations=len(self._records),
            gemm_launches=self._gemm_launches,
            useful_gemm_flops=sum(
                2 * record.x.shape[0] * record.x.shape[1] * record.grad_output.shape[1]
                for record in self._records
            ),
            redundant_gemm_flops=0,
            parameter_updates_interleaved=self._interleaved_updates,
            parameter_updates_deferred=len(self.parameter_ids.intersection(self._deferred_parameter_ids)),
            direct_gradient_modules=sum(state.layout == "direct" for state in self._states),
            transposed_gradient_modules=sum(
                state.layout != "direct" for state in self._states
            ),
            strided_gradient_modules=sum(
                state.layout == "transposed-strided" for state in self._states
            ),
            layout_transform_values=self._layout_transform_values,
            execution_family_counts=family_counts,
            execution_family_transitions=transitions,
            execution_geometry_counts=geometry_counts,
            execution_geometry_transitions=geometry_transitions,
        )
        if self._last_audit.useful_gemm_flops != self._useful_flops:
            raise RuntimeError("shared-carrier dW FLOP accounting disagrees with executed tasks")
        return self._last_audit

    def audit(self) -> SharedCarrierGradientAudit:
        return self._last_audit

    def release_step_tensors(self) -> None:
        self._records.clear()
        self._states.clear()


def _validate_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {tuple(value.shape)}")


def _matrix_multiply_into(
    lhs: Tensor,
    rhs: Tensor,
    output: Tensor,
    *,
    backend: Literal["torch-mm", "identical-triton"],
    accumulate: bool = False,
    semantic_id: str = "unlabeled",
) -> None:
    if backend == "torch-mm":
        if accumulate:
            output.addmm_(lhs, rhs)
        else:
            torch.mm(lhs, rhs, out=output)
        return
    if backend != "identical-triton":
        raise ValueError(f"unknown GEMM backend: {backend!r}")
    if accumulate:
        raise RuntimeError(
            "identical Triton GEMM deliberately has one overwrite-only epilogue; "
            "shared linear parameters must have one carrier contribution"
        )
    from ..llama_identical_microkernel_carrier.backend import triton_mm_into

    triton_mm_into(lhs, rhs, output, semantic_id=semantic_id)


def tiled_mm(
    lhs: Tensor,
    rhs: Tensor,
    *,
    row_tile: int,
    backend: Literal["torch-mm", "identical-triton"] = "torch-mm",
    semantic_prefix: str = "unlabeled",
) -> Tensor:
    """Compute ``lhs @ rhs`` as independent fixed-row cuBLAS launches."""

    _validate_matrix("lhs", lhs)
    _validate_matrix("rhs", rhs)
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"incompatible matrix shapes: {tuple(lhs.shape)} and {tuple(rhs.shape)}")
    if row_tile < 1:
        raise ValueError("row_tile must be positive")
    output = lhs.new_empty((lhs.shape[0], rhs.shape[1]))
    for tile_index, start in enumerate(range(0, lhs.shape[0], row_tile)):
        end = min(start + row_tile, lhs.shape[0])
        _matrix_multiply_into(
            lhs[start:end],
            rhs,
            output[start:end],
            backend=backend,
            semantic_id=f"{semantic_prefix}:tile{tile_index}",
        )
    return output


def tiled_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    row_tile: int,
    backend: Literal["torch-mm", "identical-triton"] = "torch-mm",
    semantic_prefix: str = "unlabeled:dw",
) -> Tensor:
    """Compute exact ``dW = dY.T @ X`` in fixed output-row tiles."""

    _validate_matrix("x", x)
    _validate_matrix("grad_output", grad_output)
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("x and grad_output must have the same row count")
    if row_tile < 1:
        raise ValueError("row_tile must be positive")
    gradient = x.new_empty((grad_output.shape[1], x.shape[1]))
    for tile_index, start in enumerate(range(0, grad_output.shape[1], row_tile)):
        end = min(start + row_tile, grad_output.shape[1])
        _matrix_multiply_into(
            grad_output[:, start:end].transpose(0, 1),
            x,
            gradient[start:end],
            backend=backend,
            semantic_id=f"{semantic_prefix}:tile{tile_index}",
        )
    return gradient


def tiled_inference_balanced_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    row_tile: int,
    backend: Literal["torch-mm", "identical-triton"] = "torch-mm",
    semantic_prefix: str = "unlabeled:dw",
) -> Tensor:
    """Compute exact dW while preferring forward-inference GEMM geometry.

    When the flattened training-row count equals the linear input width, compute
    ``dW.T = X.T @ dY`` in input-row tiles.  Each GEMM then has the same M/N/K
    dimensions as that linear's inference forward.  Otherwise retain the direct
    ``dY.T @ X`` orientation; this makes a Llama down projection consume the
    same geometry family as its gate/up projections without padding arithmetic.
    """

    _validate_matrix("x", x)
    _validate_matrix("grad_output", grad_output)
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("x and grad_output must have the same row count")
    if x.shape[0] != x.shape[1]:
        return tiled_weight_gradient(
            x,
            grad_output,
            row_tile=row_tile,
            backend=backend,
            semantic_prefix=semantic_prefix,
        )
    transposed = tiled_mm(
        x.transpose(0, 1),
        grad_output,
        row_tile=row_tile,
        backend=backend,
        semantic_prefix=semantic_prefix,
    )
    return transposed.transpose(0, 1).contiguous()


def execution_plan(
    *,
    input_rows: int,
    input_features: int,
    output_features: int,
    config: SharedCarrierConfig,
) -> CarrierExecutionPlan:
    """Return exact carrier launch and FLOP counts for one linear."""

    if min(input_rows, input_features, output_features) < 1:
        raise ValueError("linear dimensions must be positive")
    if config.require_exact_training_rows and input_rows != config.expected_training_rows:
        raise ValueError(
            "shared carrier expected exactly "
            f"{config.expected_training_rows} flattened rows, got {input_rows}"
        )
    flops = 2 * input_rows * input_features * output_features
    use_transposed_gradient = (
        config.weight_gradient_layout
        in {"inference-balanced", "inference-balanced-strided"}
        and input_rows == input_features
    )
    gradient_tile_extent = input_features if use_transposed_gradient else output_features
    return CarrierExecutionPlan(
        input_rows=input_rows,
        input_features=input_features,
        output_features=output_features,
        row_tile=config.row_tile,
        gemm_backend=config.gemm_backend,
        weight_gradient_layout=(
            (
                "transposed-strided"
                if config.weight_gradient_layout == "inference-balanced-strided"
                else "transposed"
            )
            if use_transposed_gradient
            else "direct"
        ),
        forward_launches=math.ceil(input_rows / config.row_tile),
        input_gradient_launches=math.ceil(input_rows / config.row_tile),
        weight_gradient_launches=math.ceil(gradient_tile_extent / config.row_tile),
        useful_forward_flops=flops,
        useful_input_gradient_flops=flops,
        useful_weight_gradient_flops=flops,
        layout_transform_values=(
            input_features * output_features
            if use_transposed_gradient
            and config.weight_gradient_layout == "inference-balanced"
            else 0
        ),
        executed_flops=3 * flops,
        redundant_flops=0,
    )


class _SharedCarrierLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        weight_is_transposed: bool,
        config: SharedCarrierConfig,
        scheduler: SharedCarrierGradientScheduler | None,
        module_name: str,
    ) -> Tensor:
        input_shape = x.shape
        flat_x = x.reshape(-1, input_shape[-1])
        if config.require_exact_training_rows and flat_x.shape[0] != config.expected_training_rows:
            raise RuntimeError(
                "shared carrier requires exactly "
                f"{config.expected_training_rows} flattened rows; got {flat_x.shape[0]}"
            )
        matrix = weight if weight_is_transposed else weight.transpose(0, 1)
        output = tiled_mm(
            flat_x,
            matrix,
            row_tile=config.row_tile,
            backend=config.gemm_backend,
            semantic_prefix=f"{module_name}:forward",
        )
        if bias is not None:
            output.add_(bias)
        ctx.save_for_backward(flat_x, weight)
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.weight_is_transposed = weight_is_transposed
        ctx.config = config
        ctx.scheduler = scheduler
        ctx.module_name = module_name
        return output.reshape(*input_shape[:-1], matrix.shape[1])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        flat_x, weight = ctx.saved_tensors
        config: SharedCarrierConfig = ctx.config
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        grad_x = None
        if ctx.needs_input_grad[0]:
            input_matrix = weight.transpose(0, 1) if ctx.weight_is_transposed else weight
            grad_x = tiled_mm(
                flat_grad_output,
                input_matrix,
                row_tile=config.row_tile,
                backend=config.gemm_backend,
                semantic_prefix=f"{ctx.module_name}:dx",
            ).reshape(ctx.input_shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            if ctx.scheduler is None:
                if ctx.weight_is_transposed:
                    grad_weight = tiled_mm(
                        flat_x.transpose(0, 1),
                        flat_grad_output,
                        row_tile=config.row_tile,
                        backend=config.gemm_backend,
                        semantic_prefix=f"{ctx.module_name}:dw:transposed",
                    )
                elif config.weight_gradient_layout == "inference-balanced":
                    grad_weight = tiled_inference_balanced_weight_gradient(
                        flat_x,
                        flat_grad_output,
                        row_tile=config.row_tile,
                        backend=config.gemm_backend,
                        semantic_prefix=f"{ctx.module_name}:dw:balanced",
                    )
                else:
                    grad_weight = tiled_weight_gradient(
                        flat_x,
                        flat_grad_output,
                        row_tile=config.row_tile,
                        backend=config.gemm_backend,
                        semantic_prefix=f"{ctx.module_name}:dw:direct",
                    )
            else:
                ctx.scheduler.record(
                    module_name=ctx.module_name,
                    weight=weight,
                    weight_is_transposed=ctx.weight_is_transposed,
                    x=flat_x,
                    grad_output=flat_grad_output,
                )

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = flat_grad_output.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None, None, None


class SharedCarrierLinear(nn.Module):
    """Drop-in linear whose forward, dX, and dW share one tiled GEMM carrier."""

    def __init__(
        self,
        source: nn.Linear,
        config: SharedCarrierConfig,
        *,
        weight: nn.Parameter | None = None,
        weight_is_transposed: bool = False,
        scheduler: SharedCarrierGradientScheduler | None = None,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError(f"source must be nn.Linear, got {type(source).__name__}")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight if weight is None else weight
        self.bias = source.bias
        self.weight_is_transposed = bool(weight_is_transposed)
        expected_shape = (
            (self.in_features, self.out_features)
            if self.weight_is_transposed
            else (self.out_features, self.in_features)
        )
        if tuple(self.weight.shape) != expected_shape:
            raise ValueError(
                f"shared-carrier weight shape {tuple(self.weight.shape)} does not match "
                f"storage layout {expected_shape}"
            )
        self.config = config
        self.scheduler = scheduler
        self.module_name = module_name
        self.last_input_rows: int | None = None
        if scheduler is not None:
            if scheduler.weight_gradient_layout != config.weight_gradient_layout:
                raise ValueError(
                    "scheduler and shared-carrier weight-gradient layouts must match"
                )
            if scheduler.gemm_backend != config.gemm_backend:
                raise ValueError("scheduler and shared-carrier GEMM backends must match")
            if not module_name:
                raise ValueError("deferred shared-carrier dW requires a module name")
            scheduler.register(
                module_name,
                self.weight,
                weight_is_transposed=self.weight_is_transposed,
            )

    def forward(self, x: Tensor) -> Tensor:
        self.last_input_rows = x.numel() // x.shape[-1]
        return _SharedCarrierLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.weight_is_transposed,
            self.config,
            self.scheduler,
            self.module_name,
        )

    def plan(self, input_rows: int | None = None) -> CarrierExecutionPlan:
        rows = self.last_input_rows if input_rows is None else input_rows
        if rows is None:
            raise RuntimeError("input_rows is required before the first forward call")
        return execution_plan(
            input_rows=rows,
            input_features=self.in_features,
            output_features=self.out_features,
            config=self.config,
        )


class SharedCarrierEmbedding(nn.Module):
    """Embedding view over a shared ``[hidden, vocabulary]`` parameter."""

    def __init__(self, source: nn.Embedding, weight: nn.Parameter) -> None:
        super().__init__()
        expected = (source.embedding_dim, source.num_embeddings)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"transposed embedding weight has shape {tuple(weight.shape)}, expected {expected}"
            )
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse
        self.weight = weight

    def forward(self, input_ids: Tensor) -> Tensor:
        return torch.nn.functional.embedding(
            input_ids,
            self.weight.transpose(0, 1),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def replace_linear_modules_with_shared_carrier(
    model: nn.Module,
    config: SharedCarrierConfig,
    *,
    scheduler: SharedCarrierGradientScheduler | None = None,
) -> list[str]:
    """Replace every dense linear recursively while retaining Parameter identity."""

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    embeddings: list[tuple[nn.Module, str, nn.Embedding]] = []

    def collect(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                replacements.append((parent, child_name, qualified, child))
            elif isinstance(child, nn.Embedding):
                embeddings.append((parent, child_name, child))
            else:
                collect(child, qualified)

    collect(model, "")
    transposed_parameters: dict[int, nn.Parameter] = {}
    for parent, child_name, qualified, child in replacements:
        weight_is_transposed = (
            config.weight_gradient_layout == "inference-balanced-strided"
            and child.in_features == config.expected_training_rows
        )
        weight = child.weight
        if weight_is_transposed:
            source_id = id(child.weight)
            if source_id not in transposed_parameters:
                transposed_parameters[source_id] = nn.Parameter(
                    child.weight.detach().transpose(0, 1),
                    requires_grad=child.weight.requires_grad,
                )
            weight = transposed_parameters[source_id]
        setattr(
            parent,
            child_name,
            SharedCarrierLinear(
                child,
                config,
                weight=weight,
                weight_is_transposed=weight_is_transposed,
                scheduler=scheduler,
                module_name=qualified,
            ),
        )
    for parent, child_name, child in embeddings:
        replacement_weight = transposed_parameters.get(id(child.weight))
        if replacement_weight is not None:
            setattr(parent, child_name, SharedCarrierEmbedding(child, replacement_weight))
    return [qualified for _parent, _name, qualified, _child in replacements]


def carrier_execution_plans(model: nn.Module) -> dict[str, CarrierExecutionPlan]:
    """Return plans for all carrier modules after at least one forward call."""

    return {
        name: module.plan()
        for name, module in model.named_modules()
        if isinstance(module, SharedCarrierLinear)
    }


def transposed_carrier_parameter_ids(model: nn.Module) -> set[int]:
    """Return parameter IDs whose carrier storage is logical-weight transposed."""

    return {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, SharedCarrierLinear) and module.weight_is_transposed
    }
