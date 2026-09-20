"""A real 1:1 inference/training service loop with no filler computation.

Every mixed cycle launches one ordinary inference forward and one ordinary
forward/loss/backward/SGD step on separate CUDA streams.  Both read the same
model version.  The optimizer update is ordered after inference has finished,
and the next cycle starts only after that update.  Calibration and a pure
service-debt controller choose inference batch sizes that equal one native
training-service unit over time.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import sidecapture as sc

from .scheduler import (
    BalancedServiceController,
    CalibrationResult,
    build_calibration,
    rescale_inference_service,
    set_training_native_seconds,
)


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_INFERENCE_CANDIDATES = tuple(range(6_144, 10_241, 256))
Role = Literal["inference", "training"]


@dataclass(frozen=True)
class BlockTiming:
    gpu_seconds: float
    wall_seconds: float
    cycles: int

    def __post_init__(self) -> None:
        if self.cycles < 1:
            raise ValueError("timed block cycles must be positive")
        if self.gpu_seconds <= 0 or self.wall_seconds <= 0:
            raise ValueError("timed block durations must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BalancedGigaPassConfig:
    """Every option allowed to affect the GPU program; no semantic role exists."""

    model: str = DEFAULT_MODEL
    compute_seed: int = 31_337
    local_files_only: bool = True
    attention_backend: str = "sdpa"
    training_batch_size: int = 1_024
    inference_batch_candidates: tuple[int, ...] = DEFAULT_INFERENCE_CANDIDATES
    sequence_length: int = 1
    data_ring_size: int = 8
    learning_rate: float = 3e-4
    warmup_cycles: int = 8
    calibration_rounds: int = 4
    calibration_refinement_steps: int = 10
    calibration_validation_blocks: int = 5
    calibration_block_cycles: int = 4
    calibration_target_imbalance: float = 0.01
    cycles_per_heartbeat: int = 4
    calibration_override: CalibrationResult | None = None

    def __post_init__(self) -> None:
        for name in (
            "training_batch_size",
            "sequence_length",
            "data_ring_size",
            "warmup_cycles",
            "calibration_rounds",
            "calibration_refinement_steps",
            "calibration_validation_blocks",
            "calibration_block_cycles",
            "cycles_per_heartbeat",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.model:
            raise ValueError("model cannot be empty")
        if self.attention_backend not in {"sdpa", "eager"}:
            raise ValueError("attention_backend must be sdpa or eager")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            not math.isfinite(self.calibration_target_imbalance)
            or self.calibration_target_imbalance < 0
        ):
            raise ValueError("calibration_target_imbalance must be finite and non-negative")
        candidates = self.effective_inference_candidates
        if len(candidates) < 2:
            raise ValueError("at least two inference batch candidates are required")
        if tuple(sorted(set(candidates))) != candidates or any(batch < 1 for batch in candidates):
            raise ValueError("inference batch candidates must be positive, sorted, and unique")
        if (
            self.calibration_override is not None
            and self.calibration_override.training_batch_size != self.training_batch_size
        ):
            raise ValueError("calibration training batch does not match configuration")

    @property
    def effective_inference_candidates(self) -> tuple[int, ...]:
        if self.calibration_override is None:
            return tuple(self.inference_batch_candidates)
        return tuple(candidate.batch_size for candidate in self.calibration_override.candidates)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class BalancedRoleConfig:
    """Host-only capture label wrapped around one role-blind mixed computation."""

    role: Role
    session_id: str
    computation: BalancedGigaPassConfig = BalancedGigaPassConfig()

    def __post_init__(self) -> None:
        if self.role not in {"inference", "training"}:
            raise ValueError("role must be inference or training")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")

    def metadata(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "session_id": self.session_id,
            "computation": asdict(self.computation),
            "computation_fingerprint": self.computation.fingerprint,
            "strict_invariants": {
                "role_read_by_gpu_program": False,
                "real_inference_outputs_consumed": True,
                "real_training_gradients_and_updates": True,
                "filler_kernels": 0,
                "optimizer_waits_for_inference": True,
                "next_cycle_waits_for_optimizer": True,
                "target_native_service_ratio": "1:1",
            },
        }


def load_calibration(path: Path) -> CalibrationResult:
    payload = json.loads(path.read_text())
    if "calibration" in payload:
        payload = payload["calibration"]
    return CalibrationResult.from_dict(payload)


def balanced_role_artifact(role: Role, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Expose one product while accurately reporting that both were produced."""

    common = {
        "co_served_inference_tokens": snapshot["mixed_inference_tokens"],
        "co_applied_training_updates": snapshot["mixed_training_updates"],
        "native_service_ratio": snapshot["balanced_service"]["service_ratio"],
    }
    if role == "inference":
        return {
            "kind": "served_logits_from_balanced_mixed_service",
            "served_token_checksum": snapshot["inference_token_checksum"],
            **common,
        }
    if role == "training":
        return {
            "kind": "updated_model_from_balanced_mixed_service",
            "parameter_probe_delta_linf": snapshot["parameter_probe_delta_linf"],
            **common,
        }
    raise ValueError(f"unknown role: {role!r}")


class BalancedGigaPassEngine:
    """GPU engine for concurrent, dependency-safe inference and training."""

    def __init__(self, config: BalancedGigaPassConfig) -> None:
        self.config = config
        self.model = None
        self.optimizer = None
        self.inference_stream = None
        self.training_stream = None
        self.inference_ids = None
        self.training_ids = None
        self.training_targets = None
        self.parameter_probe = None
        self.initial_parameter_probe = None
        self.parameter_probe_name = ""
        self.parameter_count = 0
        self.calibration: CalibrationResult | None = None
        self.controller: BalancedServiceController | None = None
        self.total_inference_outputs = 0
        self.total_training_updates = 0
        self.mixed_cycles = 0
        self.mixed_inference_tokens = 0
        self.mixed_training_tokens = 0
        self.mixed_gpu_seconds = 0.0
        self.mixed_wall_seconds = 0.0
        self.calibration_refinement_history: list[dict[str, Any]] = []
        self._inference_ring_index = 0
        self._training_ring_index = 0
        self._gradient_buffers_ready = False
        self._last_inference_checksum = None
        self._last_loss = None
        self._started = 0.0

    def setup(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        random.seed(self.config.compute_seed)
        torch.manual_seed(self.config.compute_seed)
        torch.cuda.manual_seed_all(self.config.compute_seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            local_files_only=self.config.local_files_only,
            dtype=torch.bfloat16,
            attn_implementation=self.config.attention_backend,
        ).cuda()
        self.model.train()
        self.model.config.use_cache = False
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            foreach=True,
        )
        self.inference_stream = torch.cuda.Stream()
        # Parameter AccumulateGrad nodes are created on PyTorch's default
        # stream. Keep the genuine backward there; moving it to another stream
        # adds an implicit synchronization and depresses native training rate.
        self.training_stream = torch.cuda.current_stream()

        generator = torch.Generator(device="cuda").manual_seed(self.config.compute_seed + 101)
        vocabulary = min(int(self.model.config.vocab_size), 32_000)
        maximum_inference_batch = max(self.config.effective_inference_candidates)
        self.inference_ids = torch.randint(
            0,
            vocabulary,
            (
                self.config.data_ring_size,
                maximum_inference_batch,
                self.config.sequence_length,
            ),
            generator=generator,
            device="cuda",
            dtype=torch.long,
        )
        training_tokens = torch.randint(
            0,
            vocabulary,
            (
                self.config.data_ring_size,
                self.config.training_batch_size,
                self.config.sequence_length + 1,
            ),
            generator=generator,
            device="cuda",
            dtype=torch.long,
        )
        self.training_ids = training_tokens[:, :, :-1].contiguous()
        self.training_targets = training_tokens[:, :, 1:].contiguous()
        self.parameter_probe_name, parameter = next(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if name.endswith("mlp.down_proj.weight") and parameter.requires_grad
        )
        self.parameter_probe = parameter.reshape(-1)[:4096]
        self.initial_parameter_probe = self.parameter_probe.detach().clone()
        torch.cuda.synchronize()

        warmup_batch = self.config.effective_inference_candidates[
            len(self.config.effective_inference_candidates) // 2
        ]
        for _ in range(self.config.warmup_cycles):
            self.run_mixed_batches([warmup_batch], account_service=False)
        if self.config.calibration_override is None:
            self.calibration = self._refine_calibration(self._measure_calibration())
        else:
            self.calibration = self.config.calibration_override
        self.controller = BalancedServiceController(self.calibration)
        self._started = time.monotonic()

    def _require_setup(self) -> None:
        if (
            self.model is None
            or self.optimizer is None
            or self.inference_stream is None
            or self.training_stream is None
        ):
            raise RuntimeError("balanced giga-pass engine is not set up")

    def _enqueue_inference(self, batch_size: int) -> None:
        import torch

        index = self._inference_ring_index % self.config.data_ring_size
        self._inference_ring_index += 1
        with torch.inference_mode():
            logits = self.model(
                input_ids=self.inference_ids[index, :batch_size],
                use_cache=False,
            ).logits
            checksum = logits[:, -1, :].argmax(dim=-1).sum(dtype=torch.int64)
        self._last_inference_checksum = checksum.detach()
        self.total_inference_outputs += batch_size * self.config.sequence_length

    def _enqueue_training_backward(self) -> None:
        import torch

        self.optimizer.zero_grad(set_to_none=not self._gradient_buffers_ready)
        index = self._training_ring_index % self.config.data_ring_size
        self._training_ring_index += 1
        logits = self.model(input_ids=self.training_ids[index], use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            self.training_targets[index].reshape(-1),
        )
        loss.backward()
        self._gradient_buffers_ready = True
        self._last_loss = loss.detach()

    def _enqueue_optimizer_step(self) -> None:
        self.optimizer.step()
        self.total_training_updates += 1

    def run_dedicated_inference_batches(self, batch_sizes: list[int]) -> BlockTiming:
        import torch

        self._require_setup()
        if not batch_sizes:
            raise ValueError("inference timing block cannot be empty")
        allowed = set(self.config.effective_inference_candidates)
        if any(batch not in allowed for batch in batch_sizes):
            raise ValueError("inference timing requested an unallocated candidate batch")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        with torch.cuda.stream(self.inference_stream):
            start.record()
            for batch_size in batch_sizes:
                self._enqueue_inference(batch_size)
            end.record()
        end.synchronize()
        return BlockTiming(
            gpu_seconds=start.elapsed_time(end) / 1e3,
            wall_seconds=time.perf_counter() - wall_started,
            cycles=len(batch_sizes),
        )

    def run_dedicated_training_cycles(self, cycles: int) -> BlockTiming:
        import torch

        self._require_setup()
        if cycles < 1:
            raise ValueError("training timing cycles must be positive")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        with torch.cuda.stream(self.training_stream):
            start.record()
            for _ in range(cycles):
                self._enqueue_training_backward()
                self._enqueue_optimizer_step()
            end.record()
        end.synchronize()
        return BlockTiming(
            gpu_seconds=start.elapsed_time(end) / 1e3,
            wall_seconds=time.perf_counter() - wall_started,
            cycles=cycles,
        )

    def run_mixed_batches(
        self,
        batch_sizes: list[int],
        *,
        account_service: bool = True,
    ) -> BlockTiming:
        """Run real inference and real training concurrently with safe update order."""

        import torch

        self._require_setup()
        if not batch_sizes:
            raise ValueError("mixed timing block cannot be empty")
        allowed = set(self.config.effective_inference_candidates)
        if any(batch not in allowed for batch in batch_sizes):
            raise ValueError("mixed block requested an unallocated candidate batch")
        if account_service and self.controller is None:
            raise RuntimeError("mixed service cannot be accounted before calibration")
        torch.cuda.synchronize()
        origin = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(origin)
        self.inference_stream.wait_event(start)
        self.training_stream.wait_event(start)
        previous_update = None
        wall_started = time.perf_counter()
        for batch_size in batch_sizes:
            if previous_update is not None:
                self.inference_stream.wait_event(previous_update)
                self.training_stream.wait_event(previous_update)
            inference_done = torch.cuda.Event()
            update_done = torch.cuda.Event()
            with torch.cuda.stream(self.inference_stream):
                self._enqueue_inference(batch_size)
                inference_done.record()
            with torch.cuda.stream(self.training_stream):
                self._enqueue_training_backward()
                self.training_stream.wait_event(inference_done)
                self._enqueue_optimizer_step()
                update_done.record()
            previous_update = update_done
            if account_service:
                self.controller.commit(batch_size)
                self.mixed_cycles += 1
                self.mixed_inference_tokens += batch_size * self.config.sequence_length
                self.mixed_training_tokens += (
                    self.config.training_batch_size * self.config.sequence_length
                )
        origin.wait_event(previous_update)
        end.record(origin)
        end.synchronize()
        timing = BlockTiming(
            gpu_seconds=start.elapsed_time(end) / 1e3,
            wall_seconds=time.perf_counter() - wall_started,
            cycles=len(batch_sizes),
        )
        if account_service:
            self.mixed_gpu_seconds += timing.gpu_seconds
            self.mixed_wall_seconds += timing.wall_seconds
        return timing

    def run_mixed_cycles(self, cycles: int) -> BlockTiming:
        if cycles < 1:
            raise ValueError("mixed cycles must be positive")
        if self.controller is None:
            raise RuntimeError("balanced controller is unavailable before setup")
        # Commit occurs inside run_mixed_batches, so choices after the first must
        # account for preceding choices. Generate with a temporary clone.
        planner = BalancedServiceController(self.calibration)
        planner.inference_service_units = self.controller.inference_service_units
        planner.training_service_units = self.controller.training_service_units
        planner.cycles = self.controller.cycles
        batch_sizes = planner.plan(cycles)
        return self.run_mixed_batches(batch_sizes)

    def _measure_calibration(self) -> CalibrationResult:
        paired: dict[int, list[tuple[float, float]]] = {
            batch: [] for batch in self.config.effective_inference_candidates
        }
        candidates = list(self.config.effective_inference_candidates)
        for round_index in range(self.config.calibration_rounds):
            ordered = candidates if round_index % 2 == 0 else list(reversed(candidates))
            for candidate_index, batch_size in enumerate(ordered):
                if (round_index + candidate_index) % 2 == 0:
                    inference = self.run_dedicated_inference_batches([batch_size])
                    training = self.run_dedicated_training_cycles(1)
                else:
                    training = self.run_dedicated_training_cycles(1)
                    inference = self.run_dedicated_inference_batches([batch_size])
                paired[batch_size].append((inference.gpu_seconds, training.gpu_seconds))
        return build_calibration(
            training_batch_size=self.config.training_batch_size,
            paired_seconds=paired,
        )

    def _refine_calibration(self, calibration: CalibrationResult) -> CalibrationResult:
        """Correct one-shot timings against sustained native service blocks."""

        for step in range(self.config.calibration_refinement_steps):
            planner = BalancedServiceController(calibration)
            inference_seconds: list[float] = []
            training_seconds: list[float] = []
            for block_index in range(self.config.calibration_validation_blocks):
                schedule = planner.plan(self.config.calibration_block_cycles)
                if (step + block_index) % 2 == 0:
                    inference = self.run_dedicated_inference_batches(schedule)
                    training = self.run_dedicated_training_cycles(len(schedule))
                else:
                    training = self.run_dedicated_training_cycles(len(schedule))
                    inference = self.run_dedicated_inference_batches(schedule)
                inference_seconds.append(inference.gpu_seconds)
                training_seconds.append(training.gpu_seconds)
            measured_ratio = sum(inference_seconds) / sum(training_seconds)
            training_native_seconds = sum(training_seconds) / (
                self.config.calibration_validation_blocks
                * self.config.calibration_block_cycles
            )
            predicted_ratio = planner.service_ratio
            correction = measured_ratio / predicted_ratio
            applied_correction = math.sqrt(correction)
            record = {
                "step": step,
                "lower_batch_size": calibration.lower_batch_size,
                "upper_batch_size": calibration.upper_batch_size,
                "predicted_service_ratio": predicted_ratio,
                "measured_service_ratio": measured_ratio,
                "correction": correction,
                "applied_damped_correction": applied_correction,
                "training_native_seconds_per_cycle": training_native_seconds,
            }
            self.calibration_refinement_history.append(record)
            if abs(measured_ratio - 1.0) <= self.config.calibration_target_imbalance:
                return set_training_native_seconds(calibration, training_native_seconds)
            calibration = rescale_inference_service(calibration, applied_correction)
        raise RuntimeError(
            "sustained service calibration did not converge to the requested 1:1 ratio; "
            f"last measured ratio was {measured_ratio:.6f} after "
            f"{self.config.calibration_refinement_steps} refinement steps; "
            f"history={json.dumps(self.calibration_refinement_history)}"
        )

    def run_iterations(self, count: int) -> dict[str, Any]:
        self.run_mixed_cycles(count)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        import torch

        if self._last_loss is None or self._last_inference_checksum is None:
            raise RuntimeError("balanced engine has not completed a mixed cycle")
        if self.controller is None or self.calibration is None:
            raise RuntimeError("balanced engine has no service calibration")
        torch.cuda.synchronize()
        elapsed = max(time.monotonic() - self._started, 1e-9)
        probe_delta = (
            self.parameter_probe.float() - self.initial_parameter_probe.float()
        ).abs().max().detach()
        training_service_seconds = (
            self.controller.training_service_units * self.calibration.training_native_seconds
        )
        inference_service_seconds = (
            self.controller.inference_service_units * self.calibration.training_native_seconds
        )
        inference_flops = 2 * self.parameter_count * self.mixed_inference_tokens
        training_flops = 6 * self.parameter_count * self.mixed_training_tokens
        mixed_elapsed = max(self.mixed_wall_seconds, 1e-9)
        return {
            "mixed_cycles": self.mixed_cycles,
            "mixed_training_updates": self.mixed_cycles,
            "total_training_updates_including_calibration": self.total_training_updates,
            "mixed_inference_tokens": self.mixed_inference_tokens,
            "mixed_training_tokens": self.mixed_training_tokens,
            "last_loss": float(self._last_loss),
            "inference_token_checksum": int(self._last_inference_checksum),
            "parameter_probe_delta_linf": float(probe_delta),
            "elapsed_seconds": elapsed,
            "mixed_gpu_seconds": self.mixed_gpu_seconds,
            "mixed_wall_seconds": self.mixed_wall_seconds,
            "balanced_service": {
                **self.controller.state_dict(),
                "inference_native_equivalent_seconds": inference_service_seconds,
                "training_native_equivalent_seconds": training_service_seconds,
                "estimated_inference_slowdown": mixed_elapsed
                / max(inference_service_seconds, 1e-15),
                "estimated_training_slowdown": mixed_elapsed
                / max(training_service_seconds, 1e-15),
            },
            "useful_inference_flops": inference_flops,
            "useful_training_flops": training_flops,
            "useful_total_flops_per_second": (inference_flops + training_flops)
            / mixed_elapsed,
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def metadata(self) -> dict[str, Any]:
        if self.model is None or self.calibration is None:
            raise RuntimeError("balanced engine is not set up")
        return {
            "computation_fingerprint": self.config.fingerprint,
            "model": self.config.model,
            "layers": len(self.model.model.layers),
            "parameters": self.parameter_count,
            "dtype": str(next(self.model.parameters()).dtype),
            "training_batch_size": self.config.training_batch_size,
            "sequence_length": self.config.sequence_length,
            "attention_backend": self.config.attention_backend,
            "calibration": self.calibration.to_dict(),
            "calibration_refinement_history": self.calibration_refinement_history,
            "optimizer": {
                "type": "torch.optim.SGD",
                "learning_rate": self.config.learning_rate,
                "foreach": True,
            },
            "gpu_program": (
                "concurrent inference forward + training forward/loss/backward; "
                "optimizer waits for inference; next cycle waits for optimizer"
            ),
            "dependency_order": [
                "both streams wait for previous optimizer update",
                "inference forward and training forward/backward may overlap",
                "optimizer update waits for inference completion",
                "next cycle waits for optimizer completion",
            ],
            "role_visible_to_engine": False,
            "filler_kernels": 0,
        }

    def close(self) -> None:
        if self.model is not None:
            import torch

            torch.cuda.synchronize()
        self.optimizer = None
        self.model = None
        self.inference_ids = None
        self.training_ids = None
        self.training_targets = None
        gc.collect()


def _send(messages: mp.Queue, payload: dict[str, Any]) -> None:
    try:
        messages.put_nowait(payload)
    except queue.Full:
        pass


def workload_process(
    config: BalancedRoleConfig,
    stop_event: mp.Event,
    messages: mp.Queue,
) -> None:
    """Spawn target; the host label is never passed into the GPU engine."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    engine = BalancedGigaPassEngine(config.computation)
    try:
        engine.setup()
        snapshot = engine.run_iterations(config.computation.cycles_per_heartbeat)
        common = {
            "session_id": config.session_id,
            "role": config.role,
            "engine": engine.metadata(),
        }
        _send(
            messages,
            {
                "event": "ready",
                **common,
                "snapshot": snapshot,
                "role_artifact": balanced_role_artifact(config.role, snapshot),
            },
        )
        while not stop_event.is_set():
            snapshot = engine.run_iterations(config.computation.cycles_per_heartbeat)
            _send(
                messages,
                {
                    "event": "heartbeat",
                    **common,
                    "snapshot": snapshot,
                    "role_artifact": balanced_role_artifact(config.role, snapshot),
                },
            )
    except BaseException as exc:
        _send(
            messages,
            {
                "event": "error",
                "role": config.role,
                "session_id": config.session_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        engine.close()


def start_workload(config: BalancedRoleConfig) -> tuple[mp.Process, mp.Event, mp.Queue]:
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
            raise RuntimeError(f"balanced workload exited before ready: {process.exitcode}")
        try:
            message = messages.get(timeout=min(1.0, deadline - time.monotonic()))
        except queue.Empty:
            continue
        if message.get("event") == "error":
            raise RuntimeError(message.get("traceback") or message.get("error"))
        if message.get("event") == "ready":
            return message
    raise TimeoutError(f"balanced workload was not ready within {timeout_seconds:.1f} seconds")


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
        raise RuntimeError(f"balanced workload exited with code {process.exitcode}")


class PersistentBalancedGigaPassWorkload(sc.Workload):
    """SideCapture wrapper around the continuously mixed process."""

    replay_safe = True

    def __init__(
        self,
        config: BalancedRoleConfig,
        *,
        startup_timeout_seconds: float = 900.0,
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
        if self._process is None or not self._process.is_alive():
            exitcode = None if self._process is None else self._process.exitcode
            raise RuntimeError(f"balanced workload is not alive: {exitcode}")
        context.labels.update(
            process=self.config.role,
            training_variant="balanced-gigapass",
            session_id=self.config.session_id,
            model=self.config.computation.model,
            continuous=True,
            workload_started_before_scope_arm=True,
            attacker_observable="power_trace_only",
            computation_fingerprint=self.config.computation.fingerprint,
            target_native_service_ratio="1:1",
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
            "type": "persistent_balanced_real_inference_and_training",
            "worker": self.config.metadata(),
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "capture_alignment": "calibrated mixed worker starts before scope arm",
            "replay_safe": True,
        }
