"""Exact shape-controlled linear algebra for strict no-cover training.

Forward and input-gradient products use either groups of shared-RHS ``M=1``
products or ordinary row tiles. Deferred weight gradients use exact row tiles;
compatible tiles can be submitted through one physical grouped GEMM. Every matrix
multiplication contributes to forward, ``dX``, exact ``dW``, or reported redundant
gradient arithmetic.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor, nn


class KernelLaunchPacer:
    """Host-pace real CUDA work without emitting filler or timing kernels."""

    def __init__(self, period_us: float) -> None:
        if not math.isfinite(period_us) or period_us <= 0:
            raise ValueError("kernel launch period must be finite and positive")
        self.period_ns = int(round(period_us * 1e3))
        self._deadline_ns: int | None = None
        self.launches = 0
        self.waits = 0
        self.overruns = 0
        self.sleep_ns = 0

    def begin_step(self) -> None:
        self._deadline_ns = None

    def before_launch(self) -> None:
        if self._deadline_ns is None:
            self._deadline_ns = time.monotonic_ns() + self.period_ns

    @staticmethod
    def _wait_until(deadline_ns: int) -> None:
        while True:
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return
            if remaining_ns > 200_000:
                time.sleep((remaining_ns - 100_000) / 1e9)

    def after_launch(self) -> None:
        if self._deadline_ns is None:
            raise RuntimeError("kernel pacer observed completion before launch")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        completed_ns = time.monotonic_ns()
        if completed_ns < self._deadline_ns:
            self._wait_until(self._deadline_ns)
            self.waits += 1
            self.sleep_ns += self._deadline_ns - completed_ns
            completed_ns = self._deadline_ns
        else:
            self.overruns += 1
        self.launches += 1
        self._deadline_ns = completed_ns + self.period_ns

    def finish_step(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def metadata(self) -> dict[str, Any]:
        return {
            "mechanism": "host wait after synchronized real training GEMM; no filler kernel",
            "period_us": self.period_ns / 1e3,
            "launches": self.launches,
            "waits": self.waits,
            "overruns": self.overruns,
            "host_wait_seconds": self.sleep_ns / 1e9,
        }


@dataclass(frozen=True)
class StrictShapeConfig:
    """Launch grouping and exact weight-gradient reduction controls."""

    backend: Literal["grouped-m1", "tiled-gemm"] = "tiled-gemm"
    forward_m1_per_launch: int = 32
    input_gradient_m1_per_launch: int = 32
    weight_gradient_m1_per_launch: int = 32
    pad_weight_gradient_reduction_to_input_width: bool = True
    weight_gradient_schedule: Literal[
        "inline",
        "round-robin",
        "balanced-round-robin",
        "streaming-round-robin",
        "streaming-inference-cycle",
        "streaming-grouped",
    ] = "inline"
    streaming_weight_gradient_tasks_per_record: int = 32
    grouped_weight_gradient_min_batch: int = 4
    grouped_weight_gradient_max_batch: int = 16
    launch_pacer: KernelLaunchPacer | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"grouped-m1", "tiled-gemm"}:
            raise ValueError(f"unknown strict shaping backend: {self.backend!r}")
        if self.weight_gradient_schedule not in {
            "inline",
            "round-robin",
            "balanced-round-robin",
            "streaming-round-robin",
            "streaming-inference-cycle",
            "streaming-grouped",
        }:
            raise ValueError(f"unknown weight-gradient schedule: {self.weight_gradient_schedule!r}")
        for name in (
            "forward_m1_per_launch",
            "input_gradient_m1_per_launch",
            "weight_gradient_m1_per_launch",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.streaming_weight_gradient_tasks_per_record < 1:
            raise ValueError("streaming_weight_gradient_tasks_per_record must be positive")
        if self.grouped_weight_gradient_min_batch < 2:
            raise ValueError("grouped_weight_gradient_min_batch must be at least two")
        if self.grouped_weight_gradient_max_batch < self.grouped_weight_gradient_min_batch:
            raise ValueError("grouped_weight_gradient_max_batch must be at least the minimum batch")


@dataclass(frozen=True)
class LinearExecutionPlan:
    """Auditable useful/executed work for one shaped linear invocation."""

    input_rows: int
    input_features: int
    output_features: int
    forward_launches: int
    input_gradient_launches: int
    weight_gradient_launches: int
    weight_gradient_reduction_width: int
    weight_gradient_reduction_chunks: int
    useful_flops: int
    executed_flops: int
    redundant_flops: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeferredWeightGradientAudit:
    schedule: str
    registered_modules: int
    registered_parameter_tensors: int
    recorded_invocations: int
    gemm_launches: int
    executed_gemm_launches: int
    physical_gemm_launches: int
    grouped_gemm_launches: int
    grouped_gemm_tasks: int
    useful_gemm_flops: int
    redundant_gemm_launches: int
    redundant_gemm_flops: int
    executed_gemm_flops: int
    parameter_updates_interleaved: int
    parameter_updates_deferred: int
    grouped_packing_bytes: int
    grouped_accumulation_tensors: int


@dataclass
class _DeferredWeightGradientRecord:
    module_name: str
    order: int
    weight: Tensor
    x: Tensor
    grad_output: Tensor
    config: StrictShapeConfig


@dataclass
class _DeferredWeightGradientState:
    record: _DeferredWeightGradientRecord
    gradient_t: Tensor
    tasks: list[tuple[Tensor, Tensor, Tensor]]
    scratch: dict[tuple[int, int], Tensor]
    next_task: int = 0
    committed: bool = False


class DeferredWeightGradientScheduler:
    """Compute exact linear dW tiles in auditable temporal schedules.

    Deferring dW is algebraically valid because dX depends on the old weight but
    not on dW. Streaming schedules execute bounded real dW tiles as reverse-pass
    operands become dependency-ready, then drain the remainder. The
    ``streaming-inference-cycle`` policy selects ready projection families in the
    measured Llama inference order instead of registration order; it never creates
    unavailable work or changes the accumulated gradient.
    """

    _INFERENCE_PROJECTION_CYCLE = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __init__(self) -> None:
        self._registrations: dict[str, tuple[int, Tensor]] = {}
        self._records: list[_DeferredWeightGradientRecord] = []
        self._retained_tensors: list[Tensor] = []
        self._streaming_states: list[_DeferredWeightGradientState] = []
        self._update_parameter: Callable[[Tensor], None] | None = None
        self._deferred_parameter_ids: set[int] = set()
        self._remaining_records: dict[int, int] = {}
        self._pending_update_weights: list[Tensor] = []
        self._deferred_updates: set[int] = set()
        self._streaming_cursor_order = 0
        self._inference_cycle_cursor = 0
        self._inference_cycle_started = False
        self._execution_family_trace: list[str] = []
        self._launches = 0
        self._physical_gemm_launches = 0
        self._grouped_gemm_launches = 0
        self._grouped_gemm_tasks = 0
        self._grouped_packing_bytes = 0
        self._grouped_accumulation_tensors = 0
        self._grouped_right_cache: dict[tuple[int, ...], Tensor] = {}
        self._grouped_offsets_cache: dict[tuple[int, ...], Tensor] = {}
        self._useful_gemm_flops = 0
        self._redundant_launches = 0
        self._redundant_gemm_flops = 0
        self._interleaved_updates = 0
        self._last_audit = DeferredWeightGradientAudit(
            schedule="round-robin",
            registered_modules=0,
            registered_parameter_tensors=0,
            recorded_invocations=0,
            gemm_launches=0,
            executed_gemm_launches=0,
            physical_gemm_launches=0,
            grouped_gemm_launches=0,
            grouped_gemm_tasks=0,
            useful_gemm_flops=0,
            redundant_gemm_launches=0,
            redundant_gemm_flops=0,
            executed_gemm_flops=0,
            parameter_updates_interleaved=0,
            parameter_updates_deferred=0,
            grouped_packing_bytes=0,
            grouped_accumulation_tensors=0,
        )

    def register(self, module_name: str, weight: Tensor) -> None:
        if module_name in self._registrations:
            raise ValueError(f"duplicate deferred-gradient module name: {module_name}")
        self._registrations[module_name] = (len(self._registrations), weight)

    @property
    def parameter_ids(self) -> set[int]:
        return {id(weight) for _order, weight in self._registrations.values()}

    def begin_step(
        self,
        *,
        update_parameter: Callable[[Tensor], None] | None = None,
        deferred_parameter_ids: set[int] | None = None,
    ) -> None:
        if (update_parameter is None) != (deferred_parameter_ids is None):
            raise ValueError("streaming dW requires both update callback and deferred parameter IDs")
        self._records.clear()
        self._retained_tensors.clear()
        self._streaming_states.clear()
        self._update_parameter = update_parameter
        self._deferred_parameter_ids = set(deferred_parameter_ids or ())
        self._remaining_records = dict(Counter(id(weight) for _order, weight in self._registrations.values()))
        self._pending_update_weights.clear()
        self._deferred_updates.clear()
        self._streaming_cursor_order = 0
        self._inference_cycle_cursor = 0
        self._inference_cycle_started = False
        self._execution_family_trace.clear()
        self._launches = 0
        self._physical_gemm_launches = 0
        self._grouped_gemm_launches = 0
        self._grouped_gemm_tasks = 0
        self._grouped_packing_bytes = 0
        self._grouped_accumulation_tensors = 0
        self._grouped_right_cache.clear()
        self._useful_gemm_flops = 0
        self._redundant_launches = 0
        self._redundant_gemm_flops = 0
        self._interleaved_updates = 0

    @torch.no_grad()
    def _build_state(self, record: _DeferredWeightGradientRecord) -> _DeferredWeightGradientState:
        token_rows, input_features = record.x.shape
        output_features = record.grad_output.shape[1]
        reduction_width = (
            input_features if record.config.pad_weight_gradient_reduction_to_input_width else token_rows
        )
        gradient_t = record.x.new_zeros((input_features, output_features))
        tasks: list[tuple[Tensor, Tensor, Tensor]] = []
        for token_start in range(0, token_rows, reduction_width):
            token_end = min(token_start + reduction_width, token_rows)
            chunk_rows = token_end - token_start
            left = record.x[token_start:token_end].transpose(0, 1)
            right = record.grad_output[token_start:token_end]
            if chunk_rows < reduction_width:
                left = torch.nn.functional.pad(left, (0, reduction_width - chunk_rows))
                right = torch.nn.functional.pad(
                    right,
                    (0, 0, 0, reduction_width - chunk_rows),
                )
            self._retained_tensors.extend((left, right))
            for feature_start in range(
                0,
                input_features,
                record.config.weight_gradient_m1_per_launch,
            ):
                feature_end = min(
                    feature_start + record.config.weight_gradient_m1_per_launch,
                    input_features,
                )
                tasks.append(
                    (
                        gradient_t[feature_start:feature_end],
                        left[feature_start:feature_end],
                        right,
                    )
                )
        self._retained_tensors.append(gradient_t)
        return _DeferredWeightGradientState(record, gradient_t, tasks, {})

    @torch.no_grad()
    def _commit_state(self, state: _DeferredWeightGradientState) -> None:
        if state.committed:
            return
        record = state.record
        if record.weight.grad is None:
            raise RuntimeError(f"missing fixed gradient buffer for {record.module_name}")
        record.weight.grad.add_(state.gradient_t.transpose(0, 1))
        state.committed = True
        parameter_id = id(record.weight)
        self._remaining_records[parameter_id] -= 1
        if self._remaining_records[parameter_id] != 0:
            return
        if self._update_parameter is None:
            self._pending_update_weights.append(record.weight)
        elif parameter_id in self._deferred_parameter_ids:
            self._deferred_updates.add(parameter_id)
        else:
            self._update_parameter(record.weight)
            self._interleaved_updates += 1

    @torch.no_grad()
    def _execute_useful_task(
        self,
        state: _DeferredWeightGradientState,
        task_index: int,
    ) -> None:
        destination, left, right = state.tasks[task_index]
        pacer = state.record.config.launch_pacer
        if pacer is not None:
            pacer.before_launch()
        destination.addmm_(left, right)
        if pacer is not None:
            pacer.after_launch()
        self._execution_family_trace.append(self._projection_family(state.record.module_name))
        self._launches += 1
        self._physical_gemm_launches += 1
        self._useful_gemm_flops += 2 * left.shape[0] * left.shape[1] * right.shape[1]
        state.next_task = max(state.next_task, task_index + 1)
        if task_index + 1 == len(state.tasks):
            self._commit_state(state)

    @torch.no_grad()
    def _run_streaming_tasks(self, budget: int) -> None:
        for _ in range(budget):
            active = sorted(
                (state for state in self._streaming_states if state.next_task < len(state.tasks)),
                key=lambda state: state.record.order,
            )
            if not active:
                return
            state = next(
                (candidate for candidate in active if candidate.record.order >= self._streaming_cursor_order),
                active[0],
            )
            self._execute_useful_task(state, state.next_task)
            self._streaming_cursor_order = (state.record.order + 1) % max(len(self._registrations), 1)

    @classmethod
    def _projection_family(cls, module_name: str) -> str:
        for family in cls._INFERENCE_PROJECTION_CYCLE:
            if module_name.endswith(family):
                return family
        if module_name.endswith("lm_head"):
            return "lm_head"
        return "other"

    @torch.no_grad()
    def _run_inference_cycle_tasks(self, budget: int, *, flush: bool = False) -> None:
        """Execute dependency-ready dW tiles in Llama forward projection order."""

        consumed = 0
        while consumed < budget:
            active = [state for state in self._streaming_states if state.next_task < len(state.tasks)]
            if not active:
                return
            active_families = {self._projection_family(state.record.module_name) for state in active}
            if not self._inference_cycle_started:
                if not set(self._INFERENCE_PROJECTION_CYCLE).issubset(active_families):
                    if not flush:
                        return
                else:
                    self._inference_cycle_started = True

            selected: _DeferredWeightGradientState | None = None
            selected_cycle_index: int | None = None
            for offset in range(len(self._INFERENCE_PROJECTION_CYCLE)):
                cycle_index = (self._inference_cycle_cursor + offset) % len(self._INFERENCE_PROJECTION_CYCLE)
                family = self._INFERENCE_PROJECTION_CYCLE[cycle_index]
                candidates = [
                    state for state in active if self._projection_family(state.record.module_name) == family
                ]
                if candidates:
                    selected = min(candidates, key=lambda state: state.record.order)
                    selected_cycle_index = cycle_index
                    break
            if selected is None:
                if not flush:
                    return
                selected = min(active, key=lambda state: state.record.order)
            elif selected_cycle_index is not None:
                self._inference_cycle_cursor = (selected_cycle_index + 1) % len(
                    self._INFERENCE_PROJECTION_CYCLE
                )
            self._execute_useful_task(selected, selected.next_task)
            consumed += 1

    @staticmethod
    def _grouped_task_key(state: _DeferredWeightGradientState) -> tuple[Any, ...]:
        _destination, left, right = state.tasks[state.next_task]
        return (left.device, left.dtype, left.shape[1], right.shape[1])

    def _grouped_cohort(
        self,
        state: _DeferredWeightGradientState,
        maximum_batch: int,
    ) -> tuple[tuple[Any, ...], int]:
        weight_shape = tuple(state.record.weight.shape)
        matching_orders = sorted(
            order for order, weight in self._registrations.values() if tuple(weight.shape) == weight_shape
        )
        rank = matching_orders.index(state.record.order)
        cohort_index = rank // maximum_batch
        cohort_start = cohort_index * maximum_batch
        expected_size = min(maximum_batch, len(matching_orders) - cohort_start)
        return (*self._grouped_task_key(state), weight_shape, cohort_index), expected_size

    @torch.no_grad()
    def _execute_grouped_tasks(self, states: list[_DeferredWeightGradientState]) -> None:
        if len(states) < 2 or states[0].tasks[states[0].next_task][1].device.type != "cuda":
            for state in states:
                self._execute_useful_task(state, state.next_task)
            return
        grouped_mm = getattr(torch, "_grouped_mm", None)
        if grouped_mm is None:
            raise RuntimeError(
                "streaming-grouped requires torch._grouped_mm on CUDA; use streaming-round-robin"
            )

        task_rows = [state.tasks[state.next_task] for state in states]
        destinations = [destination for destination, _left, _right in task_rows]
        lefts = [left for _destination, left, _right in task_rows]
        rights = [right for _destination, _left, right in task_rows]
        left_pack = torch.cat(lefts, dim=0)
        right_key = tuple(id(right) for right in rights)
        right_pack = self._grouped_right_cache.get(right_key)
        if right_pack is None:
            right_pack = torch.stack(rights, dim=0)
            self._grouped_right_cache[right_key] = right_pack
            self._retained_tensors.append(right_pack)
            self._grouped_packing_bytes += right_pack.numel() * right_pack.element_size()

        row_sizes = tuple(left.shape[0] for left in lefts)
        offsets = self._grouped_offsets_cache.get(row_sizes)
        if offsets is None:
            cumulative = []
            total = 0
            for rows in row_sizes:
                total += rows
                cumulative.append(total)
            offsets = torch.tensor(cumulative, device=left_pack.device, dtype=torch.int32)
            self._grouped_offsets_cache[row_sizes] = offsets
            self._retained_tensors.append(offsets)
        pacer = states[0].record.config.launch_pacer
        if pacer is not None:
            pacer.before_launch()
        result = grouped_mm(left_pack, right_pack, offs=offsets)
        result_parts = list(result.split(row_sizes, dim=0))
        torch._foreach_add_(destinations, result_parts)
        if pacer is not None:
            pacer.after_launch()
        self._retained_tensors.extend((left_pack, result))
        self._grouped_packing_bytes += left_pack.numel() * left_pack.element_size()
        self._grouped_accumulation_tensors += len(states)
        self._grouped_gemm_launches += 1
        self._grouped_gemm_tasks += len(states)
        self._physical_gemm_launches += 1

        for state, (_destination, left, right) in zip(states, task_rows, strict=True):
            task_index = state.next_task
            self._execution_family_trace.append(self._projection_family(state.record.module_name))
            self._launches += 1
            self._useful_gemm_flops += 2 * left.shape[0] * left.shape[1] * right.shape[1]
            state.next_task = task_index + 1
            if state.next_task == len(state.tasks):
                self._commit_state(state)

    @torch.no_grad()
    def _run_streaming_grouped_tasks(
        self,
        budget: int,
        *,
        minimum_batch: int,
        maximum_batch: int,
        flush: bool = False,
    ) -> None:
        consumed = 0
        while consumed < budget:
            active = [state for state in self._streaming_states if state.next_task < len(state.tasks)]
            if not active:
                return
            groups: dict[tuple[Any, ...], list[_DeferredWeightGradientState]] = {}
            expected_sizes: dict[tuple[Any, ...], int] = {}
            for state in active:
                key, expected_size = self._grouped_cohort(state, maximum_batch)
                groups.setdefault(key, []).append(state)
                expected_sizes[key] = expected_size
            eligible = [
                sorted(states, key=lambda state: state.record.order)
                for key, states in groups.items()
                if flush or (len(states) == expected_sizes[key] and len(states) >= minimum_batch)
            ]
            if not eligible:
                return

            def group_distance(states: list[_DeferredWeightGradientState]) -> tuple[int, int]:
                orders = [state.record.order for state in states]
                ahead = [order for order in orders if order >= self._streaming_cursor_order]
                first = min(ahead) if ahead else min(orders)
                distance = (first - self._streaming_cursor_order) % max(len(self._registrations), 1)
                return distance, -len(states)

            selected_group = min(eligible, key=group_distance)
            ahead = [state for state in selected_group if state.record.order >= self._streaming_cursor_order]
            behind = [state for state in selected_group if state.record.order < self._streaming_cursor_order]
            selected = ahead + behind
            self._execute_grouped_tasks(selected)
            consumed += len(selected)
            self._streaming_cursor_order = (selected[-1].record.order + 1) % max(len(self._registrations), 1)

    def record(
        self,
        *,
        module_name: str,
        weight: Tensor,
        x: Tensor,
        grad_output: Tensor,
        config: StrictShapeConfig,
    ) -> None:
        registration = self._registrations.get(module_name)
        if registration is None:
            raise RuntimeError(f"unregistered deferred-gradient module: {module_name}")
        order, registered_weight = registration
        if registered_weight is not weight:
            raise RuntimeError(f"weight identity changed for deferred-gradient module {module_name}")
        record = _DeferredWeightGradientRecord(
            module_name=module_name,
            order=order,
            weight=weight,
            x=x,
            grad_output=grad_output,
            config=config,
        )
        self._records.append(record)
        if config.weight_gradient_schedule in {
            "streaming-round-robin",
            "streaming-inference-cycle",
            "streaming-grouped",
        }:
            self._streaming_states.append(self._build_state(record))
            if config.weight_gradient_schedule == "streaming-grouped":
                self._run_streaming_grouped_tasks(
                    config.streaming_weight_gradient_tasks_per_record,
                    minimum_batch=config.grouped_weight_gradient_min_batch,
                    maximum_batch=config.grouped_weight_gradient_max_batch,
                )
            elif config.weight_gradient_schedule == "streaming-round-robin":
                self._run_streaming_tasks(config.streaming_weight_gradient_tasks_per_record)
            else:
                self._run_inference_cycle_tasks(config.streaming_weight_gradient_tasks_per_record)

    def _validate_records(self) -> str:
        if len(self._records) != len(self._registrations):
            raise RuntimeError(
                "deferred dW expected exactly one invocation per shaped module; "
                f"recorded {len(self._records)} for {len(self._registrations)} modules"
            )
        names = [record.module_name for record in self._records]
        duplicates = [name for name, count in Counter(names).items() if count != 1]
        if duplicates:
            raise RuntimeError(f"shaped modules were invoked more than once: {duplicates[:5]}")
        schedules = {record.config.weight_gradient_schedule for record in self._records}
        if len(schedules) != 1:
            raise RuntimeError(f"mixed deferred dW schedules are unsupported: {sorted(schedules)}")
        return schedules.pop()

    @torch.no_grad()
    def _flush_pending_updates(self) -> None:
        if self._pending_update_weights and self._update_parameter is None:
            raise RuntimeError("streaming dW completed before an optimizer callback was supplied")
        while self._pending_update_weights:
            weight = self._pending_update_weights.pop(0)
            if id(weight) in self._deferred_parameter_ids:
                self._deferred_updates.add(id(weight))
            else:
                self._update_parameter(weight)  # type: ignore[misc]
                self._interleaved_updates += 1

    def _finalize_audit(self, schedule: str) -> DeferredWeightGradientAudit:
        useful_gemm_flops = sum(
            2 * record.x.shape[0] * record.x.shape[1] * record.grad_output.shape[1]
            for record in self._records
        )
        padded_gemm_flops = sum(
            2
            * math.ceil(
                record.x.shape[0]
                / (
                    record.x.shape[1]
                    if record.config.pad_weight_gradient_reduction_to_input_width
                    else record.x.shape[0]
                )
            )
            * (
                record.x.shape[1]
                if record.config.pad_weight_gradient_reduction_to_input_width
                else record.x.shape[0]
            )
            * record.x.shape[1]
            * record.grad_output.shape[1]
            for record in self._records
        )
        padding_flops = padded_gemm_flops - useful_gemm_flops
        if padded_gemm_flops != self._useful_gemm_flops:
            raise RuntimeError(
                "deferred dW FLOP accounting disagrees with executed exact task geometry: "
                f"planned={padded_gemm_flops}, observed={self._useful_gemm_flops}"
            )
        redundant_gemm_flops = padding_flops + self._redundant_gemm_flops
        self._last_audit = DeferredWeightGradientAudit(
            schedule=schedule,
            registered_modules=len(self._registrations),
            registered_parameter_tensors=len(self.parameter_ids),
            recorded_invocations=len(self._records),
            gemm_launches=self._launches,
            executed_gemm_launches=self._launches + self._redundant_launches,
            physical_gemm_launches=self._physical_gemm_launches,
            grouped_gemm_launches=self._grouped_gemm_launches,
            grouped_gemm_tasks=self._grouped_gemm_tasks,
            useful_gemm_flops=useful_gemm_flops,
            redundant_gemm_launches=self._redundant_launches,
            redundant_gemm_flops=redundant_gemm_flops,
            executed_gemm_flops=useful_gemm_flops + redundant_gemm_flops,
            parameter_updates_interleaved=self._interleaved_updates,
            parameter_updates_deferred=len(self._deferred_updates),
            grouped_packing_bytes=self._grouped_packing_bytes,
            grouped_accumulation_tensors=self._grouped_accumulation_tensors,
        )
        return self._last_audit

    @torch.no_grad()
    def finish_step(
        self,
        *,
        update_parameter: Callable[[Tensor], None],
        deferred_parameter_ids: set[int],
    ) -> DeferredWeightGradientAudit:
        schedule = self._validate_records()
        if self._update_parameter is None:
            self._update_parameter = update_parameter
            self._deferred_parameter_ids = set(deferred_parameter_ids)

        if schedule == "streaming-round-robin":
            remaining = sum(len(state.tasks) - state.next_task for state in self._streaming_states)
            self._run_streaming_tasks(remaining)
            self._flush_pending_updates()
            return self._finalize_audit(schedule)
        if schedule == "streaming-inference-cycle":
            remaining = sum(len(state.tasks) - state.next_task for state in self._streaming_states)
            self._run_inference_cycle_tasks(remaining, flush=True)
            self._flush_pending_updates()
            return self._finalize_audit(schedule)
        if schedule == "streaming-grouped":
            remaining = sum(len(state.tasks) - state.next_task for state in self._streaming_states)
            config = self._records[0].config
            self._run_streaming_grouped_tasks(
                remaining,
                minimum_batch=config.grouped_weight_gradient_min_batch,
                maximum_batch=config.grouped_weight_gradient_max_batch,
                flush=True,
            )
            self._flush_pending_updates()
            return self._finalize_audit(schedule)

        balanced = schedule == "balanced-round-robin"
        states = [self._build_state(record) for record in sorted(self._records, key=lambda item: item.order)]
        rounds = max(len(state.tasks) for state in states)
        for round_index in range(rounds):
            for state in states:
                if round_index >= len(state.tasks):
                    if not balanced:
                        continue
                    _destination, left, right = state.tasks[round_index % len(state.tasks)]
                    shape = (left.shape[0], right.shape[1])
                    scratch = state.scratch.get(shape)
                    if scratch is None:
                        scratch = left.new_empty(shape)
                        state.scratch[shape] = scratch
                        self._retained_tensors.append(scratch)
                    torch.mm(left, right, out=scratch)
                    self._redundant_launches += 1
                    self._physical_gemm_launches += 1
                    self._redundant_gemm_flops += 2 * left.shape[0] * left.shape[1] * right.shape[1]
                    continue
                self._execute_useful_task(state, round_index)
        self._flush_pending_updates()
        return self._finalize_audit(schedule)

    def audit(self) -> DeferredWeightGradientAudit:
        return self._last_audit

    def execution_family_trace(self) -> tuple[str, ...]:
        """Return logical dW family order for schedule validation and profiling."""

        return tuple(self._execution_family_trace)

    def release_step_tensors(self) -> None:
        """Release completed eager-step packing buffers before CUDA graph capture."""

        if any(remaining != 0 for remaining in self._remaining_records.values()):
            raise RuntimeError("cannot release deferred-gradient tensors before the step completes")
        self._records.clear()
        self._retained_tensors.clear()
        self._streaming_states.clear()
        self._grouped_right_cache.clear()

    def gradient_operands(self, module_name: str) -> tuple[Tensor, Tensor]:
        """Return the saved activation and output-gradient matrices for one step."""

        matches = [record for record in self._records if record.module_name == module_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one deferred-gradient record for {module_name!r}, found {len(matches)}"
            )
        record = matches[0]
        return record.x, record.grad_output


def _validate_matrix(name: str, value: Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {tuple(value.shape)}")


def grouped_shared_rhs_m1(
    lhs: Tensor,
    rhs: Tensor,
    *,
    m1_per_launch: int,
    launch_pacer: KernelLaunchPacer | None = None,
) -> Tensor:
    """Compute ``lhs @ rhs`` as grouped batches of genuine ``1 x K @ K x N`` GEMMs.

    ``rhs`` is expanded with a zero batch stride, so it is shared rather than copied.
    The returned value is mathematically identical to the ordinary matrix product,
    modulo the backend's floating-point reduction order.
    """

    _validate_matrix("lhs", lhs)
    _validate_matrix("rhs", rhs)
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"incompatible matrix shapes: {tuple(lhs.shape)} and {tuple(rhs.shape)}")
    if m1_per_launch < 1:
        raise ValueError(f"m1_per_launch must be positive, got {m1_per_launch}")
    outputs: list[Tensor] = []
    shared_rhs = rhs.unsqueeze(0)
    for start in range(0, lhs.shape[0], m1_per_launch):
        rows = min(m1_per_launch, lhs.shape[0] - start)
        left = lhs[start : start + rows].unsqueeze(1)
        right = shared_rhs.expand(rows, -1, -1)
        if launch_pacer is not None:
            launch_pacer.before_launch()
        outputs.append(torch.bmm(left, right).squeeze(1))
        if launch_pacer is not None:
            launch_pacer.after_launch()
    if not outputs:
        return lhs.new_empty((0, rhs.shape[1]))
    return torch.cat(outputs, dim=0)


def row_tiled_matmul(
    lhs: Tensor,
    rhs: Tensor,
    *,
    rows_per_launch: int,
    launch_pacer: KernelLaunchPacer | None = None,
) -> Tensor:
    """Compute ``lhs @ rhs`` as exact row tiles matching batched inference GEMMs."""

    _validate_matrix("lhs", lhs)
    _validate_matrix("rhs", rhs)
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"incompatible matrix shapes: {tuple(lhs.shape)} and {tuple(rhs.shape)}")
    if rows_per_launch < 1:
        raise ValueError(f"rows_per_launch must be positive, got {rows_per_launch}")
    outputs = []
    for start in range(0, lhs.shape[0], rows_per_launch):
        if launch_pacer is not None:
            launch_pacer.before_launch()
        outputs.append(lhs[start : start + rows_per_launch] @ rhs)
        if launch_pacer is not None:
            launch_pacer.after_launch()
    if not outputs:
        return lhs.new_empty((0, rhs.shape[1]))
    return torch.cat(outputs, dim=0)


def exact_grouped_m1_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    m1_per_launch: int,
    reduction_width: int,
) -> Tensor:
    """Compute an exact ``dW`` using grouped inference-shaped row GEMMs.

    For ``y = x W^T``, each row of ``dW^T`` is
    ``x[:, j]^T @ grad_output``.  That is a real ``M=1`` GEMM.  If the token
    reduction is shorter than ``reduction_width``, both operands are zero padded;
    if it is longer, exact partial products are summed.  Padding FLOPs are explicit
    redundant training arithmetic and are reported by :func:`execution_plan`.
    """

    _validate_matrix("x", x)
    _validate_matrix("grad_output", grad_output)
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("x and grad_output must have the same row count")
    if reduction_width < 1:
        raise ValueError(f"reduction_width must be positive, got {reduction_width}")
    if m1_per_launch < 1:
        raise ValueError(f"m1_per_launch must be positive, got {m1_per_launch}")

    token_rows, input_features = x.shape
    output_features = grad_output.shape[1]
    gradient_t = x.new_zeros((input_features, output_features))

    for token_start in range(0, token_rows, reduction_width):
        token_end = min(token_start + reduction_width, token_rows)
        chunk_rows = token_end - token_start
        if chunk_rows == reduction_width:
            rhs = grad_output[token_start:token_end]
        else:
            rhs = grad_output.new_zeros((reduction_width, output_features))
            rhs[:chunk_rows].copy_(grad_output[token_start:token_end])

        shared_rhs = rhs.unsqueeze(0)
        for feature_start in range(0, input_features, m1_per_launch):
            feature_end = min(feature_start + m1_per_launch, input_features)
            features = feature_end - feature_start
            left = x.new_zeros((features, 1, reduction_width))
            left[:, 0, :chunk_rows].copy_(x[token_start:token_end, feature_start:feature_end].transpose(0, 1))
            partial = torch.bmm(left, shared_rhs.expand(features, -1, -1)).squeeze(1)
            gradient_t[feature_start:feature_end].add_(partial)
    return gradient_t.transpose(0, 1)


def exact_row_tiled_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    rows_per_launch: int,
    reduction_width: int,
) -> Tensor:
    """Compute exact ``dW`` as row-tiled GEMMs shaped like batched inference.

    Each tile computes rows of ``dW^T`` as ``tile x K @ K x N``.  With a
    matching inference batch, this is the same GEMM geometry as a real batched
    one-token projection.  Reduction padding/chunking follows the grouped-M=1
    implementation and remains explicit in execution accounting.
    """

    _validate_matrix("x", x)
    _validate_matrix("grad_output", grad_output)
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("x and grad_output must have the same row count")
    if reduction_width < 1:
        raise ValueError(f"reduction_width must be positive, got {reduction_width}")
    if rows_per_launch < 1:
        raise ValueError(f"rows_per_launch must be positive, got {rows_per_launch}")

    token_rows, input_features = x.shape
    output_features = grad_output.shape[1]
    gradient_t = x.new_zeros((input_features, output_features))
    for token_start in range(0, token_rows, reduction_width):
        token_end = min(token_start + reduction_width, token_rows)
        chunk_rows = token_end - token_start
        if chunk_rows == reduction_width:
            rhs = grad_output[token_start:token_end]
        else:
            rhs = grad_output.new_zeros((reduction_width, output_features))
            rhs[:chunk_rows].copy_(grad_output[token_start:token_end])
        for feature_start in range(0, input_features, rows_per_launch):
            feature_end = min(feature_start + rows_per_launch, input_features)
            left = x.new_zeros((feature_end - feature_start, reduction_width))
            left[:, :chunk_rows].copy_(x[token_start:token_end, feature_start:feature_end].transpose(0, 1))
            gradient_t[feature_start:feature_end].add_(left @ rhs)
    return gradient_t.transpose(0, 1)


def _shaped_matmul(
    lhs: Tensor,
    rhs: Tensor,
    *,
    rows: int,
    backend: str,
    launch_pacer: KernelLaunchPacer | None = None,
) -> Tensor:
    if backend == "grouped-m1":
        return grouped_shared_rhs_m1(
            lhs,
            rhs,
            m1_per_launch=rows,
            launch_pacer=launch_pacer,
        )
    if backend == "tiled-gemm":
        return row_tiled_matmul(
            lhs,
            rhs,
            rows_per_launch=rows,
            launch_pacer=launch_pacer,
        )
    raise ValueError(f"unknown strict shaping backend: {backend!r}")


def execution_plan(
    *,
    input_rows: int,
    input_features: int,
    output_features: int,
    config: StrictShapeConfig,
) -> LinearExecutionPlan:
    """Return exact useful, redundant, and launch accounting for one linear."""

    reduction_width = input_features if config.pad_weight_gradient_reduction_to_input_width else input_rows
    reduction_chunks = math.ceil(input_rows / reduction_width)
    forward_flops = 2 * input_rows * input_features * output_features
    input_gradient_flops = forward_flops
    useful_weight_gradient_flops = forward_flops
    executed_weight_gradient_flops = 2 * reduction_chunks * reduction_width * input_features * output_features
    useful_flops = forward_flops + input_gradient_flops + useful_weight_gradient_flops
    executed_flops = forward_flops + input_gradient_flops + executed_weight_gradient_flops
    return LinearExecutionPlan(
        input_rows=input_rows,
        input_features=input_features,
        output_features=output_features,
        forward_launches=math.ceil(input_rows / config.forward_m1_per_launch),
        input_gradient_launches=math.ceil(input_rows / config.input_gradient_m1_per_launch),
        weight_gradient_launches=(
            reduction_chunks * math.ceil(input_features / config.weight_gradient_m1_per_launch)
        ),
        weight_gradient_reduction_width=reduction_width,
        weight_gradient_reduction_chunks=reduction_chunks,
        useful_flops=useful_flops,
        executed_flops=executed_flops,
        redundant_flops=executed_flops - useful_flops,
    )


class _StrictM1LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        config: StrictShapeConfig,
        scheduler: DeferredWeightGradientScheduler | None,
        module_name: str,
    ) -> Tensor:
        input_shape = x.shape
        flat_x = x.reshape(-1, input_shape[-1])
        output = _shaped_matmul(
            flat_x,
            weight.transpose(0, 1),
            rows=config.forward_m1_per_launch,
            backend=config.backend,
            launch_pacer=config.launch_pacer,
        )
        if bias is not None:
            output = output + bias
        ctx.save_for_backward(flat_x, weight)
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.config = config
        ctx.scheduler = scheduler
        ctx.module_name = module_name
        return output.reshape(*input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        flat_x, weight = ctx.saved_tensors
        config: StrictShapeConfig = ctx.config
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])

        grad_x = None
        if ctx.needs_input_grad[0]:
            flat_grad_x = _shaped_matmul(
                flat_grad_output,
                weight,
                rows=config.input_gradient_m1_per_launch,
                backend=config.backend,
                launch_pacer=config.launch_pacer,
            )
            grad_x = flat_grad_x.reshape(ctx.input_shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            if config.weight_gradient_schedule != "inline":
                if ctx.scheduler is None:
                    raise RuntimeError("round-robin dW requires a deferred-gradient scheduler")
                ctx.scheduler.record(
                    module_name=ctx.module_name,
                    weight=weight,
                    x=flat_x,
                    grad_output=flat_grad_output,
                    config=config,
                )
            else:
                reduction_width = (
                    flat_x.shape[1]
                    if config.pad_weight_gradient_reduction_to_input_width
                    else flat_x.shape[0]
                )
                if config.backend == "grouped-m1":
                    grad_weight = exact_grouped_m1_weight_gradient(
                        flat_x,
                        flat_grad_output,
                        m1_per_launch=config.weight_gradient_m1_per_launch,
                        reduction_width=reduction_width,
                    )
                else:
                    grad_weight = exact_row_tiled_weight_gradient(
                        flat_x,
                        flat_grad_output,
                        rows_per_launch=config.weight_gradient_m1_per_launch,
                        reduction_width=reduction_width,
                    )

        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = flat_grad_output.sum(dim=0)
        return grad_x, grad_weight, grad_bias, None, None, None


class StrictM1Linear(nn.Module):
    """Drop-in linear with shaped forward/dX and exact inline or deferred dW."""

    def __init__(
        self,
        source: nn.Linear,
        config: StrictShapeConfig,
        *,
        scheduler: DeferredWeightGradientScheduler | None = None,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError(f"source must be nn.Linear, got {type(source).__name__}")
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.config = config
        self.scheduler = scheduler
        self.module_name = module_name
        self.last_input_rows: int | None = None
        if config.weight_gradient_schedule != "inline":
            if scheduler is None:
                raise ValueError("round-robin dW requires a DeferredWeightGradientScheduler")
            if not module_name:
                raise ValueError("round-robin dW requires a non-empty module name")
            scheduler.register(module_name, self.weight)

    def forward(self, x: Tensor) -> Tensor:
        self.last_input_rows = x.numel() // x.shape[-1]
        return _StrictM1LinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.config,
            self.scheduler,
            self.module_name,
        )

    def plan(self, input_rows: int | None = None) -> LinearExecutionPlan:
        rows = self.last_input_rows if input_rows is None else input_rows
        if rows is None:
            raise RuntimeError("input_rows is required before the first forward call")
        return execution_plan(
            input_rows=rows,
            input_features=self.in_features,
            output_features=self.out_features,
            config=self.config,
        )


def replace_linear_modules(
    model: nn.Module,
    config: StrictShapeConfig,
    *,
    scheduler: DeferredWeightGradientScheduler | None = None,
) -> list[str]:
    """Replace every dense linear recursively and return qualified names."""

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []

    def collect(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                replacements.append((parent, child_name, qualified, child))
            else:
                collect(child, qualified)

    collect(model, "")
    for parent, child_name, qualified, child in replacements:
        setattr(
            parent,
            child_name,
            StrictM1Linear(
                child,
                config,
                scheduler=scheduler,
                module_name=qualified,
            ),
        )
    return [qualified for _parent, _name, qualified, _child in replacements]


def model_execution_plans(model: nn.Module) -> dict[str, LinearExecutionPlan]:
    """Return plans for all shaped modules after at least one forward call."""

    plans: dict[str, LinearExecutionPlan] = {}
    for name, module in model.named_modules():
        if isinstance(module, StrictM1Linear):
            plans[name] = module.plan()
    return plans
