"""Persistent no-cover inference and training workloads for physical capture."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
import gc
from dataclasses import asdict, dataclass
from typing import Any, Literal

import sidecapture as sc

from .strict_optimizer import InterleavedSGD
from .strict_shapes import (
    DeferredWeightGradientScheduler,
    KernelLaunchPacer,
    StrictShapeConfig,
    model_execution_plans,
    replace_linear_modules,
)


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
WorkloadMode = Literal["inference", "ordinary-training", "shaped-training"]


@dataclass(frozen=True)
class StrictWorkloadConfig:
    """Configuration with no representable inference-cover or second-model path."""

    mode: WorkloadMode
    session_id: str
    model: str = DEFAULT_MODEL
    seed: int = 0
    local_files_only: bool = True
    training_batch_size: int = 1
    training_sequence_length: int = 2048
    inference_batch_size: int = 128
    inference_prompt_tokens: int = 1
    inference_decode_tokens: int = 64
    learning_rate: float = 3e-4
    tile_rows: int = 128
    shaping_backend: Literal["grouped-m1", "tiled-gemm"] = "tiled-gemm"
    weight_gradient_schedule: Literal[
        "inline",
        "round-robin",
        "balanced-round-robin",
        "streaming-round-robin",
        "streaming-inference-cycle",
        "streaming-grouped",
    ] = "round-robin"
    streaming_weight_gradient_tasks_per_record: int = 32
    grouped_weight_gradient_min_batch: int = 4
    grouped_weight_gradient_max_batch: int = 16
    cuda_graph: bool = True
    warmup_updates: int = 2
    warmup_inference_requests: int = 2
    replays_per_heartbeat: int = 8
    actuator_module: str = "model.layers.0.mlp.down_proj"
    actuator_width: int = 768
    actuator_operations: tuple[int, ...] = ()
    actuator_repetitions_per_update: int = 1
    optimizer_bucket_size: int = 8
    kernel_launch_period_us: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"inference", "ordinary-training", "shaped-training"}:
            raise ValueError(f"unknown workload mode: {self.mode!r}")
        if self.weight_gradient_schedule not in {
            "inline",
            "round-robin",
            "balanced-round-robin",
            "streaming-round-robin",
            "streaming-inference-cycle",
            "streaming-grouped",
        }:
            raise ValueError(f"unknown weight-gradient schedule: {self.weight_gradient_schedule!r}")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        for name in (
            "training_sequence_length",
            "training_batch_size",
            "inference_batch_size",
            "inference_prompt_tokens",
            "inference_decode_tokens",
            "tile_rows",
            "warmup_inference_requests",
            "replays_per_heartbeat",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_updates < 1:
            raise ValueError("warmup_updates must be positive for CUDA graph allocation stability")
        if self.actuator_width < 32 or self.actuator_width % 32:
            raise ValueError("actuator_width must be a positive multiple of 32")
        if any(operations < 1 for operations in self.actuator_operations):
            raise ValueError("every actuator operation count must be positive")
        if self.actuator_repetitions_per_update < 1:
            raise ValueError("actuator_repetitions_per_update must be positive")
        if self.optimizer_bucket_size < 1:
            raise ValueError("optimizer_bucket_size must be positive")
        if self.streaming_weight_gradient_tasks_per_record < 1:
            raise ValueError("streaming_weight_gradient_tasks_per_record must be positive")
        if self.grouped_weight_gradient_min_batch < 2:
            raise ValueError("grouped_weight_gradient_min_batch must be at least two")
        if self.grouped_weight_gradient_max_batch < self.grouped_weight_gradient_min_batch:
            raise ValueError("grouped_weight_gradient_max_batch must be at least the minimum batch")
        if not math.isfinite(self.kernel_launch_period_us) or self.kernel_launch_period_us < 0:
            raise ValueError("kernel_launch_period_us must be finite and nonnegative")
        if self.kernel_launch_period_us and self.cuda_graph:
            raise ValueError("kernel launch pacing requires --no-cuda-graph")
        if self.kernel_launch_period_us and self.mode != "shaped-training":
            raise ValueError("kernel launch pacing is only valid for shaped-training")
        if self.kernel_launch_period_us and self.weight_gradient_schedule == "inline":
            raise ValueError("kernel launch pacing requires a deferred weight-gradient schedule")
        if self.actuator_operations:
            if self.mode != "shaped-training":
                raise ValueError("gradient actuation is only valid for shaped-training")
            if self.weight_gradient_schedule == "inline":
                raise ValueError("gradient actuation requires a deferred weight-gradient schedule")
            if not self.cuda_graph:
                raise ValueError("gradient actuation requires CUDA graphs")

    @property
    def is_training(self) -> bool:
        return self.mode != "inference"

    def strict_invariants(self) -> dict[str, Any]:
        return {
            "inference_cover_tokens": 0,
            "secondary_model_instances": 0,
            "filler_kernels": 0,
            "all_extra_flops_are_accounted_training_arithmetic": True,
            "reduction_padding_flops_are_accounted": True,
            "redundant_gradient_recomputation_flops_are_accounted": True,
            "optimizer_updates_use_real_current_gradients": True,
        }

    def metadata(self) -> dict[str, Any]:
        return {**asdict(self), "strict_invariants": self.strict_invariants()}


def _send(messages: mp.Queue, payload: dict[str, Any]) -> None:
    try:
        messages.put_nowait(payload)
    except queue.Full:
        pass


def _load_model(config: StrictWorkloadConfig):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        local_files_only=config.local_files_only,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model.cuda()


def _aggregate_plans(model) -> dict[str, Any]:
    plans = model_execution_plans(model)
    return {
        "shaped_linear_modules": len(plans),
        "forward_launches_per_update": sum(plan.forward_launches for plan in plans.values()),
        "input_gradient_launches_per_update": sum(plan.input_gradient_launches for plan in plans.values()),
        "weight_gradient_launches_per_update": sum(plan.weight_gradient_launches for plan in plans.values()),
        "useful_linear_flops_per_update": sum(plan.useful_flops for plan in plans.values()),
        "executed_linear_flops_per_update": sum(plan.executed_flops for plan in plans.values()),
        "redundant_padding_flops_per_update": sum(plan.redundant_flops for plan in plans.values()),
        "per_module": {name: plan.to_dict() for name, plan in plans.items()},
    }


def _training_process(
    config: StrictWorkloadConfig,
    stop_event: mp.Event,
    messages: mp.Queue,
) -> None:
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = _load_model(config).train()
    model.config.use_cache = False
    shaped_names: list[str] = []
    gradient_scheduler: DeferredWeightGradientScheduler | None = None
    launch_pacer = (
        KernelLaunchPacer(config.kernel_launch_period_us) if config.kernel_launch_period_us else None
    )
    if config.mode == "shaped-training":
        if config.weight_gradient_schedule != "inline":
            gradient_scheduler = DeferredWeightGradientScheduler()
        shape_config = StrictShapeConfig(
            backend=config.shaping_backend,
            forward_m1_per_launch=config.tile_rows,
            input_gradient_m1_per_launch=config.tile_rows,
            weight_gradient_m1_per_launch=config.tile_rows,
            pad_weight_gradient_reduction_to_input_width=True,
            weight_gradient_schedule=config.weight_gradient_schedule,
            streaming_weight_gradient_tasks_per_record=(config.streaming_weight_gradient_tasks_per_record),
            grouped_weight_gradient_min_batch=config.grouped_weight_gradient_min_batch,
            grouped_weight_gradient_max_batch=config.grouped_weight_gradient_max_batch,
            launch_pacer=launch_pacer,
        )
        shaped_names = replace_linear_modules(
            model,
            shape_config,
            scheduler=gradient_scheduler,
        )

    parameter_tensors = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = InterleavedSGD(
        model,
        learning_rate=config.learning_rate,
        manual_parameter_ids=None if gradient_scheduler is None else gradient_scheduler.parameter_ids,
        manual_update_bucket_size=config.optimizer_bucket_size,
    )
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 101)
    token_ids = torch.randint(
        low=0,
        high=min(int(model.config.vocab_size), 32_000),
        size=(config.training_batch_size, config.training_sequence_length + 1),
        generator=generator,
        device="cuda",
        dtype=torch.long,
    )
    input_ids = token_ids[:, :-1].contiguous()
    target_ids = token_ids[:, 1:].contiguous()

    def forward_loss():
        output = model(input_ids=input_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            target_ids.reshape(-1),
        )
        return output, loss

    def begin_gradient_step() -> None:
        if launch_pacer is not None:
            launch_pacer.begin_step()
        if gradient_scheduler is not None:
            gradient_scheduler.begin_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )

    def finish_gradient_step() -> None:
        if gradient_scheduler is not None:
            gradient_scheduler.finish_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )
        optimizer.step_deferred()
        if launch_pacer is not None:
            launch_pacer.finish_step()

    last_loss = None
    for _ in range(config.warmup_updates):
        optimizer.zero_grad(set_to_none=False)
        begin_gradient_step()
        warmup_output, last_loss = forward_loss()
        last_loss.backward()
        finish_gradient_step()
    torch.cuda.synchronize()
    warmup_loss = float(last_loss.detach())
    del warmup_output, last_loss
    if gradient_scheduler is not None:
        gradient_scheduler.release_step_tensors()
    gc.collect()
    torch.cuda.empty_cache()

    graph = None
    if config.cuda_graph:
        optimizer.zero_grad(set_to_none=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimizer.zero_grad(set_to_none=False)
            begin_gradient_step()
            _graph_output, graph_loss = forward_loss()
            graph_loss.backward()
            finish_gradient_step()
        last_loss = graph_loss
    else:
        optimizer.zero_grad(set_to_none=False)
        begin_gradient_step()
        _eager_output, last_loss = forward_loss()
        last_loss.backward()
        finish_gradient_step()
        torch.cuda.synchronize()

    actuator = None
    actuator_profile_flops = 0
    if config.actuator_operations:
        if gradient_scheduler is None:
            raise RuntimeError("gradient actuation requires the deferred gradient scheduler")
        from .trace_drawing import CudaGraphGradientTileActuator

        unique_operations = sorted(set(config.actuator_operations))
        actuator = CudaGraphGradientTileActuator(
            gradient_scheduler,
            config.actuator_module,
            {config.actuator_width: unique_operations[0]},
            copy_operands=False,
        )
        actuator.add_explicit_programs(
            {(config.actuator_width, operations) for operations in unique_operations}
        )
        x_operand, grad_output_operand = gradient_scheduler.gradient_operands(config.actuator_module)
        actuator_profile_flops = (
            config.actuator_repetitions_per_update
            * sum(config.actuator_operations)
            * 2
            * config.actuator_width
            * int(x_operand.shape[0])
            * int(grad_output_operand.shape[1])
        )
        actuator.reset()

    shape_audit = _aggregate_plans(model) if shaped_names else None
    optimizer_audit = asdict(optimizer.audit())
    common = {
        "session_id": config.session_id,
        "mode": config.mode,
        "model": config.model,
        "layers": len(model.model.layers),
        "parameter_tensors": len(parameter_tensors),
        "parameters": sum(parameter.numel() for parameter in parameter_tensors),
        "dtype": str(next(model.parameters()).dtype),
        "training_batch_size": config.training_batch_size,
        "training_sequence_length": config.training_sequence_length,
        "externally_shifted_next_token_targets": True,
        "model_instances_loaded": 1,
        "shaped_linear_modules": len(shaped_names),
        "strict_invariants": config.strict_invariants(),
        "cuda_graph": graph is not None,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gradient_actuation": actuator is not None,
        "actuator_module": config.actuator_module if actuator is not None else None,
        "actuator_width": config.actuator_width if actuator is not None else None,
        "actuator_profile_points": len(config.actuator_operations),
        "actuator_repetitions_per_update": config.actuator_repetitions_per_update,
        "actuator_redundant_flops_per_update": actuator_profile_flops,
        "kernel_launch_pacing": None if launch_pacer is None else launch_pacer.metadata(),
    }
    _send(
        messages,
        {
            "event": "ready",
            **common,
            "shape_audit": shape_audit,
            "deferred_weight_gradient_audit": (
                None if gradient_scheduler is None else asdict(gradient_scheduler.audit())
            ),
            "optimizer_audit": optimizer_audit,
            "actuator_audit": None if actuator is None else actuator.metadata(),
            "warmup_updates": config.warmup_updates,
            "last_loss": warmup_loss,
            "gradient_tensors": sum(parameter.grad is not None for parameter in parameter_tensors),
        },
    )

    updates = 0
    started = time.monotonic()
    while not stop_event.is_set():
        for _ in range(config.replays_per_heartbeat):
            if graph is None:
                optimizer.zero_grad(set_to_none=False)
                begin_gradient_step()
                _eager_output, last_loss = forward_loss()
                last_loss.backward()
                finish_gradient_step()
            else:
                graph.replay()
            if actuator is not None:
                for _ in range(config.actuator_repetitions_per_update):
                    for operations in config.actuator_operations:
                        actuator.execute(
                            config.actuator_width,
                            operations=operations,
                            synchronize=False,
                        )
            updates += 1
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        useful_targets = updates * config.training_batch_size * config.training_sequence_length
        _send(
            messages,
            {
                "event": "heartbeat",
                **common,
                "updates": updates,
                "useful_loss_targets": useful_targets,
                "useful_loss_targets_per_second": useful_targets / max(elapsed, 1e-9),
                "last_loss": float(last_loss.detach()),
                "gradient_tensors": sum(parameter.grad is not None for parameter in parameter_tensors),
                "elapsed_seconds": elapsed,
                "actuator_replays": None if actuator is None else actuator.replays,
                "actuator_redundant_flops": (None if actuator is None else actuator.executed_flops),
                "kernel_launch_pacing": (None if launch_pacer is None else launch_pacer.metadata()),
            },
        )
    torch.cuda.synchronize()
    optimizer.close()


def _inference_process(
    config: StrictWorkloadConfig,
    stop_event: mp.Event,
    messages: mp.Queue,
) -> None:
    import torch

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _load_model(config).eval()
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 211)

    def run_request() -> tuple[int, int]:
        input_ids = torch.randint(
            low=0,
            high=min(int(model.config.vocab_size), 32_000),
            size=(config.inference_batch_size, config.inference_prompt_tokens),
            generator=generator,
            device="cuda",
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            cache = output.past_key_values
            next_token = output.logits[:, -1:].argmax(dim=-1)
            for _ in range(config.inference_decode_tokens):
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((config.inference_batch_size, 1))),
                    dim=1,
                )
                output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                next_token = output.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        return config.inference_decode_tokens * config.inference_batch_size, int(next_token.sum())

    warmup_checksum = 0
    for _ in range(config.warmup_inference_requests):
        _, warmup_checksum = run_request()
    torch.cuda.synchronize()

    requests = 0
    generated_tokens = 0
    started = time.monotonic()
    common = {
        "session_id": config.session_id,
        "mode": config.mode,
        "model": config.model,
        "layers": len(model.model.layers),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dtype": str(next(model.parameters()).dtype),
        "model_instances_loaded": 1,
        "decode_batch_size": config.inference_batch_size,
        "warmup_inference_requests": config.warmup_inference_requests,
        "strict_invariants": config.strict_invariants(),
    }
    _send(messages, {"event": "ready", **common, "warmup_checksum": warmup_checksum})

    while not stop_event.is_set():
        generated, last_token_checksum = run_request()
        generated_tokens += generated
        requests += 1
        elapsed = time.monotonic() - started
        _send(
            messages,
            {
                "event": "heartbeat",
                **common,
                "requests": requests,
                "generated_tokens": generated_tokens,
                "generated_tokens_per_second": generated_tokens / max(elapsed, 1e-9),
                "last_token_checksum": last_token_checksum,
                "elapsed_seconds": elapsed,
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            },
        )


def workload_process(
    config: StrictWorkloadConfig,
    stop_event: mp.Event,
    messages: mp.Queue,
) -> None:
    """Spawn target that guarantees exactly one model instance per process."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        if config.mode == "inference":
            _inference_process(config, stop_event, messages)
        else:
            _training_process(config, stop_event, messages)
    except BaseException as exc:
        _send(
            messages,
            {
                "event": "error",
                "mode": config.mode,
                "session_id": config.session_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def start_workload(
    config: StrictWorkloadConfig,
) -> tuple[mp.Process, mp.Event, mp.Queue]:
    context = mp.get_context("spawn")
    stop_event = context.Event()
    messages = context.Queue(maxsize=32)
    process = context.Process(
        target=workload_process,
        args=(config, stop_event, messages),
        daemon=False,
    )
    process.start()
    return process, stop_event, messages


def wait_for_ready(
    process: mp.Process,
    messages: mp.Queue,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"workload exited before ready with code {process.exitcode}")
        try:
            message = messages.get(timeout=min(1.0, deadline - time.monotonic()))
        except queue.Empty:
            continue
        if message.get("event") == "error":
            raise RuntimeError(message.get("traceback") or message.get("error"))
        if message.get("event") == "ready":
            return message
    raise TimeoutError(f"workload did not become ready within {timeout_seconds:.1f} seconds")


def stop_workload(
    process: mp.Process,
    stop_event: mp.Event,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    stop_event.set()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
    if process.exitcode not in {0, None}:
        raise RuntimeError(f"workload exited with code {process.exitcode}")


class PersistentStrictWorkload(sc.Workload):
    """SideCapture-compatible wrapper around one continuously running worker."""

    replay_safe = True

    def __init__(
        self,
        config: StrictWorkloadConfig,
        *,
        startup_timeout_seconds: float = 600.0,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        if startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("worker startup and shutdown timeouts must be positive")
        self.config = config
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)
        self._process = None
        self._stop_event = None
        self._messages = None
        self._latest: dict[str, Any] = {}

    def _accept(self, message: dict[str, Any]) -> None:
        if message.get("event") == "error":
            raise RuntimeError(message.get("traceback") or message.get("error"))
        self._latest = message

    def _assert_alive(self) -> None:
        if self._process is None:
            raise RuntimeError("strict workload has not been started")
        if not self._process.is_alive():
            raise RuntimeError(f"strict workload exited unexpectedly: {self._process.exitcode}")

    def _drain(self) -> None:
        if self._messages is None:
            return
        while True:
            try:
                self._accept(self._messages.get_nowait())
            except queue.Empty:
                return

    def setup(self) -> None:
        self._process, self._stop_event, self._messages = start_workload(self.config)
        self._latest = wait_for_ready(
            self._process,
            self._messages,
            timeout_seconds=self.startup_timeout_seconds,
        )

    def run(self, context) -> dict[str, Any]:
        self._drain()
        self._assert_alive()
        process = "inference" if self.config.mode == "inference" else "training"
        context.labels.update(
            process=process,
            training_variant=(
                None
                if process == "inference"
                else ("gradient-actuated-training" if self.config.actuator_operations else self.config.mode)
            ),
            session_id=self.config.session_id,
            model=self.config.model,
            continuous=True,
            workload_started_before_scope_arm=True,
            attacker_observable="power_trace_only",
            inference_cover_tokens=0,
            secondary_model_instances=0,
            filler_kernels=0,
        )
        return {key: value for key, value in self._latest.items() if key != "traceback"}

    def teardown(self) -> None:
        if self._process is not None and self._stop_event is not None:
            stop_workload(
                self._process,
                self._stop_event,
                timeout_seconds=self.shutdown_timeout_seconds,
            )
        if self._messages is not None:
            self._messages.close()
        self._process = None
        self._stop_event = None
        self._messages = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "persistent_strict_single_model_process",
            "worker": self.config.metadata(),
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "capture_alignment": "worker starts and warms before scope arm",
            "replay_safe": True,
        }
