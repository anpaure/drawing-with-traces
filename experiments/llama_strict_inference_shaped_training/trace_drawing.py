"""Closed-loop actuator built from genuine Llama weight-gradient tiles."""

from __future__ import annotations

import math
import random
import time
from typing import Any, Literal

import numpy as np
import sidecapture as sc

from .strict_optimizer import InterleavedSGD
from .strict_shapes import (
    DeferredWeightGradientScheduler,
    StrictShapeConfig,
    replace_linear_modules,
)
from .strict_workloads import DEFAULT_MODEL, StrictWorkloadConfig, _load_model


DEFAULT_ACTUATOR_MODULE = "model.layers.0.mlp.down_proj"


def plan_operations_per_bin(
    operation_duration_ms: float,
    bin_duration_ms: float,
    *,
    target_utilization: float = 0.82,
    maximum_operations: int = 256,
) -> int:
    """Plan a bounded amount of real gradient work for one timed bin."""

    if not math.isfinite(operation_duration_ms) or operation_duration_ms <= 0:
        raise ValueError("operation_duration_ms must be finite and positive")
    if not math.isfinite(bin_duration_ms) or bin_duration_ms <= 0:
        raise ValueError("bin_duration_ms must be finite and positive")
    if not 0 < target_utilization < 1:
        raise ValueError("target_utilization must be between zero and one")
    if maximum_operations < 1:
        raise ValueError("maximum_operations must be positive")
    planned = int(math.floor(bin_duration_ms * target_utilization / operation_duration_ms))
    return min(maximum_operations, max(1, planned))


def sleep_until_ns(deadline_ns: int) -> None:
    """Sleep most of a deadline and spin only for the final scheduling margin."""

    while True:
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 250_000:
            time.sleep((remaining_ns - 150_000) / 1e9)


class GradientTileActuator:
    """Recompute real rows of one current-step dW with a controllable width."""

    def __init__(
        self,
        scheduler: DeferredWeightGradientScheduler,
        module_name: str,
        *,
        width_quantum: int = 32,
    ) -> None:
        if not module_name:
            raise ValueError("module_name cannot be empty")
        if width_quantum < 1:
            raise ValueError("width_quantum must be positive")
        self.scheduler = scheduler
        self.module_name = module_name
        self.width_quantum = int(width_quantum)
        self.cursor = 0
        self.operations = 0
        self.executed_flops = 0
        self._last_result = None

    def operands(self):
        return self.scheduler.gradient_operands(self.module_name)

    @property
    def maximum_width(self) -> int:
        x, _grad_output = self.operands()
        return int(x.shape[1])

    @property
    def last_result(self):
        return self._last_result

    def reset(self) -> None:
        self.cursor = 0
        self.operations = 0
        self.executed_flops = 0
        self._last_result = None

    def execute(self, width: int, *, synchronize: bool = True) -> None:
        import torch

        width = int(width)
        if width < self.width_quantum or width % self.width_quantum:
            raise ValueError(
                f"gradient tile width must be a positive multiple of {self.width_quantum}, got {width}"
            )
        x, grad_output = self.operands()
        if width > x.shape[1]:
            raise ValueError(f"gradient tile width {width} exceeds {x.shape[1]} input features")
        if self.cursor + width > x.shape[1]:
            self.cursor = 0
        start = self.cursor
        end = start + width
        # This is an exact block of dW.T = X.T @ dY from the current real
        # training step.  The full dW was already accumulated for the accepted
        # SGD update, so this measured operation is explicitly redundant
        # gradient recomputation rather than inference cover.
        with torch.no_grad():
            self._last_result = x[:, start:end].transpose(0, 1) @ grad_output
        if synchronize:
            torch.cuda.synchronize()
        self.cursor = 0 if end == x.shape[1] else end
        self.operations += 1
        self.executed_flops += 2 * width * x.shape[0] * grad_output.shape[1]

    def metadata(self) -> dict[str, Any]:
        x, grad_output = self.operands()
        return {
            "module_name": self.module_name,
            "operation": "redundant exact dW.T row block: X[:, j:k].T @ dY",
            "token_rows": int(x.shape[0]),
            "input_features": int(x.shape[1]),
            "output_features": int(grad_output.shape[1]),
            "width_quantum": self.width_quantum,
            "operations": self.operations,
            "executed_flops": self.executed_flops,
        }


class CudaGraphGradientTileActuator:
    """Replay CUDA graphs containing only exact redundant dW block GEMMs.

    Operands live in stable graph-owned buffers. ``refresh_operands`` copies
    the latest real training step's X and dY into those buffers before the next
    capture, so graph replay never points at stale allocator storage.
    """

    def __init__(
        self,
        scheduler: DeferredWeightGradientScheduler,
        module_name: str,
        operations_per_width: dict[int, int],
        *,
        width_quantum: int = 32,
        copy_operands: bool = True,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-graph gradient actuation requires a CUDA GPU")
        if not operations_per_width:
            raise ValueError("operations_per_width cannot be empty")
        self.scheduler = scheduler
        self.module_name = module_name
        self.width_quantum = int(width_quantum)
        self.copy_operands = bool(copy_operands)
        x, grad_output = scheduler.gradient_operands(module_name)
        self.static_x = x.detach().clone() if self.copy_operands else x.detach()
        self.static_grad_output = grad_output.detach().clone() if self.copy_operands else grad_output.detach()
        self._source_x_pointer = int(x.data_ptr())
        self._source_grad_output_pointer = int(grad_output.data_ptr())
        self._capture_stream = torch.cuda.Stream()
        self._graphs: dict[tuple[int, int], Any] = {}
        self._outputs: dict[tuple[int, int], Any] = {}
        self._operations_per_width: dict[int, int] = {}
        self.operations = 0
        self.executed_flops = 0
        self.replays = 0
        self._last_result = None
        self.add_programs(operations_per_width)

    @property
    def maximum_width(self) -> int:
        return int(self.static_x.shape[1])

    @property
    def last_result(self):
        return self._last_result

    def _validate_program(self, width: int, operations: int) -> None:
        if width < self.width_quantum or width % self.width_quantum:
            raise ValueError(
                f"gradient tile width must be a positive multiple of {self.width_quantum}, got {width}"
            )
        if width > self.maximum_width:
            raise ValueError(f"gradient tile width {width} exceeds {self.maximum_width}")
        if operations < 1:
            raise ValueError("a CUDA-graph program must contain at least one gradient GEMM")

    def _capture_program(self, width: int, operations: int) -> None:
        import torch

        self._validate_program(width, operations)
        lhs = self.static_x[:, :width].transpose(0, 1)
        output = torch.empty(
            (width, self.static_grad_output.shape[1]),
            device=self.static_x.device,
            dtype=self.static_x.dtype,
        )
        self._capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._capture_stream), torch.no_grad():
            for _ in range(3):
                torch.mm(lhs, self.static_grad_output, out=output)
        self._capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._capture_stream), torch.no_grad():
            for _ in range(operations):
                torch.mm(lhs, self.static_grad_output, out=output)
        self._capture_stream.synchronize()
        key = (width, operations)
        self._graphs[key] = graph
        self._outputs[key] = output

    def add_programs(self, operations_per_width: dict[int, int]) -> None:
        for width, operations in sorted(operations_per_width.items()):
            width = int(width)
            operations = int(operations)
            if self._operations_per_width.get(width) == operations:
                continue
            self._capture_program(width, operations)
            self._operations_per_width[width] = operations

    def add_explicit_programs(self, programs: set[tuple[int, int]]) -> None:
        for width, operations in sorted(programs):
            width = int(width)
            operations = int(operations)
            if (width, operations) not in self._graphs:
                self._capture_program(width, operations)

    def operations_per_replay(self, width: int) -> int:
        try:
            return self._operations_per_width[int(width)]
        except KeyError as error:
            raise ValueError(f"no CUDA-graph gradient program exists for width {width}") from error

    def refresh_operands(self) -> None:
        import torch

        x, grad_output = self.scheduler.gradient_operands(self.module_name)
        if (
            x.shape != self.static_x.shape
            or x.dtype != self.static_x.dtype
            or x.device != self.static_x.device
            or grad_output.shape != self.static_grad_output.shape
            or grad_output.dtype != self.static_grad_output.dtype
            or grad_output.device != self.static_grad_output.device
        ):
            raise RuntimeError("real gradient operands changed shape, dtype, or device")
        if self.copy_operands:
            self.static_x.copy_(x.detach())
            self.static_grad_output.copy_(grad_output.detach())
        elif (
            int(x.data_ptr()) != self._source_x_pointer
            or int(grad_output.data_ptr()) != self._source_grad_output_pointer
        ):
            raise RuntimeError("direct CUDA-graph gradient operand storage changed")
        torch.cuda.synchronize()

    def execute(
        self,
        width: int,
        *,
        operations: int | None = None,
        synchronize: bool = True,
    ) -> None:
        import torch

        width = int(width)
        if operations is None:
            operations = self.operations_per_replay(width)
        operations = int(operations)
        key = (width, operations)
        try:
            graph = self._graphs[key]
        except KeyError as error:
            raise ValueError(
                f"no CUDA-graph gradient program exists for width {width} and {operations} operations"
            ) from error
        graph.replay()
        if synchronize:
            torch.cuda.synchronize()
        self._last_result = self._outputs[key]
        self.replays += 1
        self.operations += operations
        self.executed_flops += (
            operations * 2 * width * self.static_x.shape[0] * self.static_grad_output.shape[1]
        )

    def reset(self) -> None:
        self.operations = 0
        self.executed_flops = 0
        self.replays = 0
        self._last_result = None

    def metadata(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "operation": "CUDA-graph replay of redundant exact dW.T blocks: X[:, :k].T @ dY",
            "token_rows": int(self.static_x.shape[0]),
            "input_features": int(self.static_x.shape[1]),
            "output_features": int(self.static_grad_output.shape[1]),
            "width_quantum": self.width_quantum,
            "copy_operands": self.copy_operands,
            "programs": dict(self._operations_per_width),
            "captured_programs": [
                {"width": width, "operations": operations} for width, operations in sorted(self._graphs)
            ],
            "replays": self.replays,
            "operations": self.operations,
            "executed_flops": self.executed_flops,
        }


class TransformerGradientCalibrationWorkload(sc.Workload):
    """Measure width→ADC activity using one model and real Llama gradients.

    A complete exact optimizer step is prepared before the scope is armed.  The
    measured profile only recomputes blocks from that step's real dW.  After a
    capture is durably committed, another complete optimizer step prepares fresh
    operands for the next accepted capture.
    """

    replay_safe = True

    def __init__(
        self,
        commands: np.ndarray,
        *,
        bin_duration_s: float,
        model: str = DEFAULT_MODEL,
        module_name: str = DEFAULT_ACTUATOR_MODULE,
        training_batch_size: int = 1024,
        training_sequence_length: int = 1,
        tile_rows: int = 1024,
        learning_rate: float = 3e-4,
        lead_s: float = 0.005,
        tail_s: float = 0.005,
        synchronize_each_operation: bool = False,
        execution_backend: Literal["cuda-graph", "python-queued"] = "cuda-graph",
        graph_target_utilization: float = 0.82,
        graph_maximum_operations: int = 256,
        operation_commands: np.ndarray | None = None,
        seed: int = 6200,
    ) -> None:
        commands = np.asarray(commands, dtype=np.int64)
        if commands.ndim != 1 or commands.size < 3:
            raise ValueError("commands must be a one-dimensional array with at least three widths")
        if np.any(commands < 32) or np.any(commands % 32):
            raise ValueError("every actuator command must be a positive multiple of 32")
        if bin_duration_s <= 0 or lead_s < 0 or tail_s < 0:
            raise ValueError("bin duration must be positive and lead/tail cannot be negative")
        self.commands = np.ascontiguousarray(commands)
        self.bin_duration_s = float(bin_duration_s)
        self.model_name = model
        self.module_name = module_name
        self.training_batch_size = int(training_batch_size)
        self.training_sequence_length = int(training_sequence_length)
        self.tile_rows = int(tile_rows)
        self.learning_rate = float(learning_rate)
        self.lead_s = float(lead_s)
        self.tail_s = float(tail_s)
        self.synchronize_each_operation = bool(synchronize_each_operation)
        if execution_backend not in {"cuda-graph", "python-queued"}:
            raise ValueError("execution_backend must be cuda-graph or python-queued")
        if synchronize_each_operation and execution_backend == "cuda-graph":
            raise ValueError("synchronize_each_operation is incompatible with cuda-graph")
        if not 0 < graph_target_utilization < 1:
            raise ValueError("graph_target_utilization must be between zero and one")
        if graph_maximum_operations < 1:
            raise ValueError("graph_maximum_operations must be positive")
        self.execution_backend = execution_backend
        self.graph_target_utilization = float(graph_target_utilization)
        self.graph_maximum_operations = int(graph_maximum_operations)
        if operation_commands is not None:
            operation_commands = np.asarray(operation_commands, dtype=np.int64)
            if operation_commands.shape != commands.shape:
                raise ValueError("operation_commands must have the same shape as commands")
            if np.any(operation_commands < 1):
                raise ValueError("every explicit operation command must be positive")
            if execution_backend != "cuda-graph":
                raise ValueError("explicit operation commands require the cuda-graph backend")
            self.operation_commands = np.ascontiguousarray(operation_commands)
        else:
            self.operation_commands = None
        self.seed = int(seed)
        self.config = StrictWorkloadConfig(
            mode="shaped-training",
            session_id="gradient-actuator-calibration",
            model=model,
            seed=seed,
            training_batch_size=training_batch_size,
            training_sequence_length=training_sequence_length,
            learning_rate=learning_rate,
            tile_rows=tile_rows,
            weight_gradient_schedule="round-robin",
            cuda_graph=False,
        )
        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.actuator = None
        self.input_ids = None
        self.target_ids = None
        self.last_loss = None
        self.completed_updates = 0
        self.last_profile: dict[str, Any] | None = None
        self.operation_duration_ms: dict[int, float] = {}
        self.operation_host_duration_ms: dict[int, float] = {}
        self.operation_cuda_duration_ms: dict[int, float] = {}
        self.program_duration_ms: dict[int, float] = {}
        self.planned_operations_per_bin: dict[int, int] = {}

    @property
    def profile_duration_s(self) -> float:
        return self.bin_duration_s * self.commands.size

    @property
    def total_duration_s(self) -> float:
        return self.lead_s + self.profile_duration_s + self.tail_s

    def setup(self) -> None:
        import torch

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model = _load_model(self.config).train()
        self.model.config.use_cache = False
        self.scheduler = DeferredWeightGradientScheduler()
        names = replace_linear_modules(
            self.model,
            StrictShapeConfig(
                backend="tiled-gemm",
                forward_m1_per_launch=self.tile_rows,
                input_gradient_m1_per_launch=self.tile_rows,
                weight_gradient_m1_per_launch=self.tile_rows,
                pad_weight_gradient_reduction_to_input_width=True,
                weight_gradient_schedule="round-robin",
            ),
            scheduler=self.scheduler,
        )
        if self.module_name not in names:
            raise ValueError(
                f"actuator module {self.module_name!r} was not found; available examples: {names[:8]}"
            )
        self.optimizer = InterleavedSGD(
            self.model,
            learning_rate=self.learning_rate,
            manual_parameter_ids=self.scheduler.parameter_ids,
        )
        generator = torch.Generator(device="cuda").manual_seed(self.seed + 1)
        tokens = torch.randint(
            0,
            min(int(self.model.config.vocab_size), 32_000),
            (self.training_batch_size, self.training_sequence_length + 1),
            generator=generator,
            device="cuda",
        )
        self.input_ids = tokens[:, :-1].contiguous()
        self.target_ids = tokens[:, 1:].contiguous()
        self._prepare_training_step()
        self._prepare_training_step()
        probe_actuator = GradientTileActuator(self.scheduler, self.module_name)
        if int(self.commands.max()) > probe_actuator.maximum_width:
            raise ValueError(
                f"maximum command {int(self.commands.max())} exceeds actuator width "
                f"{probe_actuator.maximum_width}"
            )
        self.actuator = probe_actuator
        if self.execution_backend == "cuda-graph":
            self._ensure_cuda_graph_programs(np.unique(self.commands))
            self._ensure_explicit_graph_programs()
        elif not self.synchronize_each_operation:
            self._plan_python_queued_operations(np.unique(self.commands))
        self.actuator.reset()

    def _measure_operation_ms(self, width: int, repeats: int = 20) -> float:
        import torch

        if self.scheduler is None:
            raise RuntimeError("gradient actuator is unavailable")
        actuator = GradientTileActuator(self.scheduler, self.module_name)
        for _ in range(2):
            actuator.execute(width, synchronize=False)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        for _ in range(repeats):
            actuator.execute(width, synchronize=False)
        end_event.record()
        end_event.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1e3 / repeats
        cuda_ms = float(start_event.elapsed_time(end_event)) / repeats
        self.operation_host_duration_ms[int(width)] = max(wall_ms, 1e-6)
        self.operation_cuda_duration_ms[int(width)] = max(cuda_ms, 1e-6)
        if self.execution_backend == "cuda-graph":
            return max(cuda_ms, 1e-6)
        return max(wall_ms, cuda_ms, 1e-6)

    def _plan_python_queued_operations(self, widths: np.ndarray) -> None:
        bin_ms = self.bin_duration_s * 1e3
        for raw_width in widths:
            width = int(raw_width)
            if width not in self.operation_duration_ms:
                self.operation_duration_ms[width] = self._measure_operation_ms(width)
            self.planned_operations_per_bin[width] = plan_operations_per_bin(
                self.operation_duration_ms[width],
                bin_ms,
                target_utilization=self.graph_target_utilization,
                maximum_operations=self.graph_maximum_operations,
            )

    def _measure_program_ms(
        self,
        width: int,
        *,
        operations: int | None = None,
        repeats: int = 8,
    ) -> float:
        import torch

        if not isinstance(self.actuator, CudaGraphGradientTileActuator):
            raise RuntimeError("CUDA-graph actuator is unavailable")
        for _ in range(2):
            self.actuator.execute(width, operations=operations, synchronize=False)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(repeats):
            self.actuator.execute(width, operations=operations, synchronize=False)
        end_event.record()
        end_event.synchronize()
        return max(float(start_event.elapsed_time(end_event)) / repeats, 1e-6)

    def _ensure_explicit_graph_programs(self) -> None:
        if self.operation_commands is None:
            return
        if not isinstance(self.actuator, CudaGraphGradientTileActuator):
            raise RuntimeError("explicit operation commands require a CUDA-graph actuator")
        programs = {
            (int(width), int(operations))
            for width, operations in zip(self.commands, self.operation_commands, strict=True)
        }
        self.actuator.add_explicit_programs(programs)
        # Explicit duty-control graphs intentionally occupy nearly the full bin.
        # The run-time overrun audit below remains the authoritative trace gate.
        maximum_program_ms = self.bin_duration_s * 1e3 * 0.98
        for width, operations in sorted(programs):
            duration_ms = self._measure_program_ms(width, operations=operations)
            if duration_ms > maximum_program_ms:
                raise RuntimeError(
                    f"explicit width-{width} x {operations} gradient graph takes "
                    f"{duration_ms:.3f} ms, exceeding the {maximum_program_ms:.3f} ms "
                    "control-bin safety budget"
                )
        self.actuator.reset()

    def _ensure_cuda_graph_programs(self, widths: np.ndarray) -> None:
        if self.scheduler is None:
            raise RuntimeError("gradient scheduler is unavailable")
        bin_ms = self.bin_duration_s * 1e3
        requested: dict[int, int] = {}
        for raw_width in widths:
            width = int(raw_width)
            if width not in self.operation_duration_ms:
                self.operation_duration_ms[width] = self._measure_operation_ms(width)
            requested[width] = plan_operations_per_bin(
                self.operation_duration_ms[width],
                bin_ms,
                target_utilization=self.graph_target_utilization,
                maximum_operations=self.graph_maximum_operations,
            )
        if isinstance(self.actuator, CudaGraphGradientTileActuator):
            self.actuator.add_programs(requested)
        else:
            self.actuator = CudaGraphGradientTileActuator(
                self.scheduler,
                self.module_name,
                requested,
            )
        target_program_ms = bin_ms * self.graph_target_utilization
        maximum_program_ms = bin_ms * 0.95
        for width in requested:
            duration_ms = self._measure_program_ms(width)
            for _ in range(4):
                old_operations = self.actuator.operations_per_replay(width)
                ratio = target_program_ms / duration_ms
                if 0.94 <= ratio <= 1.06:
                    break
                new_operations = min(
                    self.graph_maximum_operations,
                    max(1, int(round(old_operations * ratio))),
                )
                if new_operations == old_operations:
                    break
                self.actuator.add_programs({width: new_operations})
                duration_ms = self._measure_program_ms(width)
            if duration_ms > maximum_program_ms:
                raise RuntimeError(
                    f"width-{width} gradient graph takes {duration_ms:.3f} ms, exceeding the "
                    f"{maximum_program_ms:.3f} ms safety budget for a {bin_ms:.3f} ms bin"
                )
            self.planned_operations_per_bin[width] = self.actuator.operations_per_replay(width)
            self.program_duration_ms[width] = duration_ms
        self.actuator.reset()

    def set_commands(self, commands: np.ndarray) -> None:
        self.set_program_commands(commands, None)

    def set_program_commands(
        self,
        commands: np.ndarray,
        operation_commands: np.ndarray | None,
    ) -> None:
        commands = np.asarray(commands, dtype=np.int64)
        if commands.ndim != 1 or commands.size < 3:
            raise ValueError("commands must be a one-dimensional array with at least three widths")
        if np.any(commands < 32) or np.any(commands % 32):
            raise ValueError("every actuator command must be a positive multiple of 32")
        if self.actuator is not None and int(commands.max()) > self.actuator.maximum_width:
            raise ValueError("an actuator command exceeds the controlled parameter width")
        if commands.size != self.commands.size:
            raise ValueError("closed-loop command updates must preserve the profile point count")
        if operation_commands is not None:
            operation_commands = np.asarray(operation_commands, dtype=np.int64)
            if operation_commands.shape != commands.shape:
                raise ValueError("operation_commands must have the same shape as commands")
            if np.any(operation_commands < 1):
                raise ValueError("every explicit operation command must be positive")
            if self.execution_backend != "cuda-graph":
                raise ValueError("explicit operation commands require the cuda-graph backend")
            self.operation_commands = np.ascontiguousarray(operation_commands)
        else:
            self.operation_commands = None
        self.commands = np.ascontiguousarray(commands)
        if self.actuator is not None and self.execution_backend == "cuda-graph":
            self._ensure_cuda_graph_programs(np.unique(commands))
            self._ensure_explicit_graph_programs()
        elif self.actuator is not None and not self.synchronize_each_operation:
            self._plan_python_queued_operations(np.unique(commands))
            self.actuator.reset()

    def _prepare_training_step(self) -> None:
        import torch

        if self.model is None or self.scheduler is None or self.optimizer is None:
            raise RuntimeError("gradient calibration workload is not set up")
        self.optimizer.zero_grad(set_to_none=False)
        self.scheduler.begin_step()
        output = self.model(input_ids=self.input_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            self.target_ids.reshape(-1),
        )
        loss.backward()
        self.scheduler.finish_step(
            update_parameter=self.optimizer.step_manual,
            deferred_parameter_ids=self.optimizer.deferred_parameter_ids,
        )
        self.optimizer.step_deferred()
        torch.cuda.synchronize()
        self.last_loss = float(loss.detach())
        if not math.isfinite(self.last_loss):
            raise RuntimeError(f"training loss became non-finite: {self.last_loss}")
        self.completed_updates += 1

    def run(self, context: sc.CaptureContext) -> dict[str, Any]:
        if self.actuator is None:
            raise RuntimeError("gradient calibration workload is not set up")
        self.actuator.reset()
        context.labels.update(
            process="training",
            training_variant="gradient-actuator-calibration",
            model=self.model_name,
            module_name=self.module_name,
            completed_training_updates=self.completed_updates,
            loss_before_profile=self.last_loss,
            loss_before_update=self.last_loss,
            profile_duration_s=self.profile_duration_s,
            inference_cover_tokens=0,
            secondary_model_instances=0,
            filler_kernels=0,
            measured_operations="redundant real weight-gradient arithmetic",
            actuator_execution_backend=self.execution_backend,
        )
        context.add_artifact("tile_width_commands", self.commands.astype(np.int32))
        if self.operation_commands is not None:
            context.add_artifact(
                "tile_operation_commands",
                self.operation_commands.astype(np.int32),
            )
        if self.lead_s:
            with context.region("tile.lead", duration_s=self.lead_s):
                time.sleep(self.lead_s)
        bin_ns = int(round(self.bin_duration_s * 1e9))
        profile_start = time.monotonic_ns()
        operation_counts = np.zeros(self.commands.size, dtype=np.int64)
        overruns = np.zeros(self.commands.size, dtype=np.int64)
        with context.region(
            "tile.profile",
            bins=int(self.commands.size),
            duration_s=self.profile_duration_s,
        ):
            for index, width in enumerate(self.commands):
                deadline = profile_start + (index + 1) * bin_ns
                with context.region(
                    "tile.bin",
                    index=index,
                    requested_width=int(width),
                    requested_operations=(
                        None if self.operation_commands is None else int(self.operation_commands[index])
                    ),
                ):
                    if self.synchronize_each_operation:
                        while time.monotonic_ns() < deadline:
                            self.actuator.execute(int(width), synchronize=True)
                            operation_counts[index] += 1
                    elif self.execution_backend == "cuda-graph":
                        import torch

                        if not isinstance(self.actuator, CudaGraphGradientTileActuator):
                            raise RuntimeError("CUDA-graph actuator was not initialized")
                        repeats = (
                            self.actuator.operations_per_replay(int(width))
                            if self.operation_commands is None
                            else int(self.operation_commands[index])
                        )
                        self.actuator.execute(
                            int(width),
                            operations=repeats,
                            synchronize=False,
                        )
                        torch.cuda.synchronize()
                        operation_counts[index] = repeats
                        sleep_until_ns(deadline)
                    else:
                        import torch

                        repeats = self.planned_operations_per_bin[int(width)]
                        for _ in range(repeats):
                            self.actuator.execute(int(width), synchronize=False)
                        torch.cuda.synchronize()
                        operation_counts[index] = repeats
                        remaining_ns = deadline - time.monotonic_ns()
                        if remaining_ns > 0:
                            sleep_until_ns(deadline)
                overruns[index] = max(0, time.monotonic_ns() - deadline)
        if self.tail_s:
            with context.region("tile.tail", duration_s=self.tail_s):
                time.sleep(self.tail_s)
        max_overrun_ns = int(overruns.max(initial=0))
        if max_overrun_ns > max(500_000, bin_ns // 2):
            raise RuntimeError(
                f"a {self.bin_duration_s * 1e3:.3f} ms calibration bin overran by "
                f"{max_overrun_ns / 1e6:.3f} ms"
            )
        context.add_artifact("tile_operations_per_bin", operation_counts)
        self.last_profile = {
            "operations": int(operation_counts.sum()),
            "operations_per_bin": operation_counts.tolist(),
            "maximum_overrun_us": max_overrun_ns / 1e3,
            "actuator": self.actuator.metadata(),
        }
        return self.last_profile

    def on_accept(self, result: Any) -> None:
        del result
        self._prepare_training_step()
        if isinstance(self.actuator, CudaGraphGradientTileActuator):
            self.actuator.refresh_operands()
        else:
            self.actuator = GradientTileActuator(self.scheduler, self.module_name)

    def teardown(self) -> None:
        if self.optimizer is not None:
            self.optimizer.close()
        self.model = None
        self.scheduler = None
        self.optimizer = None
        self.actuator = None
        self.input_ids = None
        self.target_ids = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "llama_real_gradient_tile_actuator",
            "commands": self.commands.tolist(),
            "operation_commands": (
                None if self.operation_commands is None else self.operation_commands.tolist()
            ),
            "bin_duration_s": self.bin_duration_s,
            "profile_duration_s": self.profile_duration_s,
            "lead_s": self.lead_s,
            "tail_s": self.tail_s,
            "worker": self.config.metadata(),
            "module_name": self.module_name,
            "synchronize_each_operation": self.synchronize_each_operation,
            "execution_backend": self.execution_backend,
            "graph_target_utilization": self.graph_target_utilization,
            "graph_maximum_operations": self.graph_maximum_operations,
            "operation_duration_ms": self.operation_duration_ms,
            "operation_host_duration_ms": self.operation_host_duration_ms,
            "operation_cuda_duration_ms": self.operation_cuda_duration_ms,
            "program_duration_ms": self.program_duration_ms,
            "planned_operations_per_bin": self.planned_operations_per_bin,
            "measured_kernel_claim": (
                "every controlled GEMM recomputes an exact block of the current real dW"
            ),
            "completed_updates": self.completed_updates,
            "last_profile": self.last_profile,
        }
