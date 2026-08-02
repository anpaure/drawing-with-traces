"""One GPU computation with two host-side semantic roles.

Both roles execute the same model state, inputs, forward, loss, backward, and
optimizer update.  The role is intentionally absent from ``ComputationConfig``
and ``IdenticalDualRoleEngine``.  Only the host-side sink chooses whether the
already-computed logits or already-computed updated state is useful.
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
from typing import Any, Literal

import sidecapture as sc

from ..llama_strict_inference_shaped_training.strict_optimizer import InterleavedSGD


DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
Role = Literal["inference", "training"]


@dataclass(frozen=True)
class ComputationConfig:
    """Every field that is allowed to affect GPU execution.

    There is deliberately no role or session identifier here.
    """

    model: str = DEFAULT_MODEL
    compute_seed: int = 31_337
    local_files_only: bool = True
    inference_batch_size: int = 1024
    training_batch_size: int = 2048
    sequence_length: int = 1
    data_ring_size: int = 64
    learning_rate: float = 3e-4
    optimizer_bucket_size: int = 8
    warmup_iterations: int = 2
    iterations_per_heartbeat: int = 1
    profile_iteration: bool = False

    def __post_init__(self) -> None:
        for name in (
            "inference_batch_size",
            "training_batch_size",
            "sequence_length",
            "data_ring_size",
            "optimizer_bucket_size",
            "warmup_iterations",
            "iterations_per_heartbeat",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")

    @property
    def combined_batch_size(self) -> int:
        return self.inference_batch_size + self.training_batch_size

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class DualRoleConfig:
    role: Role
    session_id: str
    model: str = DEFAULT_MODEL
    compute_seed: int = 31_337
    local_files_only: bool = True
    inference_batch_size: int = 1024
    training_batch_size: int = 2048
    sequence_length: int = 1
    data_ring_size: int = 64
    learning_rate: float = 3e-4
    optimizer_bucket_size: int = 8
    warmup_iterations: int = 2
    iterations_per_heartbeat: int = 1
    period_profile_output: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"inference", "training"}:
            raise ValueError("role must be inference or training")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if self.period_profile_output and not self.period_profile_output.strip():
            raise ValueError("period_profile_output cannot contain only whitespace")
        self.computation  # Run role-independent validation.

    @property
    def computation(self) -> ComputationConfig:
        return ComputationConfig(
            model=self.model,
            compute_seed=self.compute_seed,
            local_files_only=self.local_files_only,
            inference_batch_size=self.inference_batch_size,
            training_batch_size=self.training_batch_size,
            sequence_length=self.sequence_length,
            data_ring_size=self.data_ring_size,
            learning_rate=self.learning_rate,
            optimizer_bucket_size=self.optimizer_bucket_size,
            warmup_iterations=self.warmup_iterations,
            iterations_per_heartbeat=self.iterations_per_heartbeat,
            profile_iteration=bool(self.period_profile_output),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "computation_fingerprint": self.computation.fingerprint,
            "strict_invariants": {
                "role_read_by_gpu_program": False,
                "same_forward_backward_update_both_roles": True,
                "same_model_state_both_roles": True,
                "same_inputs_targets_rng_both_roles": True,
                "role_selection_after_gpu_synchronization": True,
                "secondary_model_instances": 0,
                "filler_kernels": 0,
            },
        }


def split_dual_logits(logits: Any, inference_batch_size: int) -> tuple[Any, Any]:
    """Split one fused forward without changing either partition."""

    if inference_batch_size < 1 or inference_batch_size >= int(logits.shape[0]):
        raise ValueError("inference_batch_size must split a non-empty combined batch")
    return logits[:inference_batch_size], logits[inference_batch_size:]


def role_artifact(role: Role, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Choose semantics after all GPU work has already completed."""

    if role == "inference":
        return {
            "kind": "served_logits",
            "served_token_checksum": snapshot["inference_token_checksum"],
            "updates_computed_but_not_persisted": snapshot["updates"],
        }
    if role == "training":
        return {
            "kind": "updated_model_state",
            "updates_persisted": snapshot["updates"],
            "parameter_probe_delta_linf": snapshot["parameter_probe_delta_linf"],
            "logits_computed_but_not_served": snapshot["inference_token_checksum"],
        }
    raise ValueError(f"unknown role: {role!r}")


class IdenticalDualRoleEngine:
    """GPU engine that cannot observe the host-side role."""

    def __init__(self, config: ComputationConfig) -> None:
        self.config = config
        self.model = None
        self.optimizer = None
        self.input_ids = None
        self.target_ids = None
        self.parameter_probe = None
        self.parameter_probe_name = ""
        self.initial_parameter_probe = None
        self.parameter_count = 0
        self.updates = 0
        self._last_loss = None
        self._last_inference_checksum = None
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
            attn_implementation="sdpa",
        ).cuda()
        self.model.train()
        self.model.config.use_cache = False
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.optimizer = InterleavedSGD(
            self.model,
            learning_rate=self.config.learning_rate,
            manual_update_bucket_size=self.config.optimizer_bucket_size,
        )

        generator = torch.Generator(device="cuda").manual_seed(self.config.compute_seed + 101)
        vocabulary = min(int(self.model.config.vocab_size), 32_000)
        inference_input_ids = torch.randint(
            0,
            vocabulary,
            (
                self.config.data_ring_size,
                self.config.inference_batch_size,
                self.config.sequence_length,
            ),
            generator=generator,
            device="cuda",
            dtype=torch.long,
        )
        training_token_ids = torch.randint(
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
        training_input_ids = training_token_ids[:, :, :-1].contiguous()
        self.target_ids = training_token_ids[:, :, 1:].contiguous()
        self.input_ids = torch.cat((inference_input_ids, training_input_ids), dim=1)

        self.parameter_probe_name, probe_parameter = next(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if name.endswith("mlp.down_proj.weight") and parameter.requires_grad
        )
        self.parameter_probe = probe_parameter.reshape(-1)[:4096]
        self.initial_parameter_probe = self.parameter_probe.detach().clone()
        self._started = time.monotonic()

    def run_iteration(self) -> None:
        """Execute the complete role-blind fused forward/backward/update."""

        import torch

        if self.model is None or self.optimizer is None:
            raise RuntimeError("dual-role engine is not set up")
        self.optimizer.zero_grad(set_to_none=False)
        batch_index = self.updates % self.config.data_ring_size
        output = self.model(input_ids=self.input_ids[batch_index], use_cache=False)
        inference_logits, training_logits = split_dual_logits(
            output.logits,
            self.config.inference_batch_size,
        )
        self._last_inference_checksum = (
            inference_logits[:, -1, :].argmax(dim=-1).sum(dtype=torch.int64).detach()
        )
        self._last_loss = torch.nn.functional.cross_entropy(
            training_logits.reshape(-1, training_logits.shape[-1]).float(),
            self.target_ids[batch_index].reshape(-1),
        )
        self._last_loss.backward()
        self.optimizer.step_deferred()
        self.updates += 1

    def run_iterations(self, count: int) -> dict[str, Any]:
        if count < 1:
            raise ValueError("iteration count must be positive")
        for _ in range(count):
            self.run_iteration()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        import torch

        if self._last_loss is None or self._last_inference_checksum is None:
            raise RuntimeError("dual-role engine has not completed an iteration")
        probe_delta = (self.parameter_probe.float() - self.initial_parameter_probe.float()).abs().max()
        probe_checksum = self.parameter_probe.float().sum()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - self._started
        executed_tokens = self.config.combined_batch_size * self.config.sequence_length
        useful_training_tokens = self.config.training_batch_size * self.config.sequence_length
        useful_inference_tokens = self.config.inference_batch_size * self.config.sequence_length
        return {
            "updates": self.updates,
            "last_loss": float(self._last_loss.detach()),
            "inference_token_checksum": int(self._last_inference_checksum),
            "parameter_probe_delta_linf": float(probe_delta),
            "parameter_probe_checksum": float(probe_checksum),
            "elapsed_seconds": elapsed,
            "executed_tokens": self.updates * executed_tokens,
            "useful_training_tokens": self.updates * useful_training_tokens,
            "useful_inference_tokens": self.updates * useful_inference_tokens,
            "updates_per_second": self.updates / max(elapsed, 1e-9),
            "approx_executed_model_flops_per_second": (
                6 * self.parameter_count * self.updates * executed_tokens / max(elapsed, 1e-9)
            ),
            "approx_useful_training_flops_per_second": (
                6 * self.parameter_count * self.updates * useful_training_tokens
                / max(elapsed, 1e-9)
            ),
            "approx_useful_inference_flops_per_second": (
                2 * self.parameter_count * self.updates * useful_inference_tokens
                / max(elapsed, 1e-9)
            ),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def metadata(self) -> dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("dual-role engine is not set up")
        return {
            "computation_fingerprint": self.config.fingerprint,
            "model": self.config.model,
            "layers": len(self.model.model.layers),
            "parameters": self.parameter_count,
            "dtype": str(next(self.model.parameters()).dtype),
            "combined_batch_size": self.config.combined_batch_size,
            "inference_batch_size": self.config.inference_batch_size,
            "training_batch_size": self.config.training_batch_size,
            "sequence_length": self.config.sequence_length,
            "data_ring_size": self.config.data_ring_size,
            "optimizer": asdict(self.optimizer.audit()),
            "parameter_probe_name": self.parameter_probe_name,
            "gpu_program": (
                "one combined forward; inference-logit reduction; training-row loss; "
                "one backward; one SGD update"
            ),
            "role_visible_to_engine": False,
        }

    def close(self) -> None:
        if self.optimizer is not None:
            self.optimizer.close()
        self.optimizer = None
        self.model = None
        gc.collect()


def _send(messages: mp.Queue, payload: dict[str, Any]) -> None:
    try:
        messages.put_nowait(payload)
    except queue.Full:
        pass


def workload_process(
    config: DualRoleConfig,
    stop_event: mp.Event,
    messages: mp.Queue,
) -> None:
    """Spawn target; role is used only to label already-computed snapshots."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    engine = IdenticalDualRoleEngine(config.computation)
    try:
        engine.setup()
        warmup = engine.run_iterations(config.computation.warmup_iterations)
        profile = None
        if config.computation.profile_iteration:
            from ..llama_cyclic_scs_carrier.profile_events import (
                capture_cuda_period,
                compact_profile_metadata,
            )

            _, profile = capture_cuda_period(
                "identical_dual_role_iteration",
                engine.run_iteration,
                Path(config.period_profile_output),
            )
            profile = compact_profile_metadata(profile)
            warmup = engine.snapshot()
        common = {
            "session_id": config.session_id,
            "role": config.role,
            "engine": engine.metadata(),
            "period_profile": profile,
        }
        _send(
            messages,
            {
                "event": "ready",
                **common,
                "snapshot": warmup,
                "role_artifact": role_artifact(config.role, warmup),
            },
        )
        while not stop_event.is_set():
            snapshot = engine.run_iterations(config.computation.iterations_per_heartbeat)
            _send(
                messages,
                {
                    "event": "heartbeat",
                    **common,
                    "snapshot": snapshot,
                    "role_artifact": role_artifact(config.role, snapshot),
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


def start_workload(config: DualRoleConfig) -> tuple[mp.Process, mp.Event, mp.Queue]:
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
            raise RuntimeError(f"dual-role workload exited before ready: {process.exitcode}")
        try:
            message = messages.get(timeout=min(1.0, deadline - time.monotonic()))
        except queue.Empty:
            continue
        if message.get("event") == "error":
            raise RuntimeError(message.get("traceback") or message.get("error"))
        if message.get("event") == "ready":
            return message
    raise TimeoutError(f"dual-role workload was not ready within {timeout_seconds:.1f} seconds")


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
        raise RuntimeError(f"dual-role workload exited with code {process.exitcode}")


class PersistentDualRoleWorkload(sc.Workload):
    """SideCapture wrapper around the exact same GPU process for either role."""

    replay_safe = True

    def __init__(
        self,
        config: DualRoleConfig,
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
            raise RuntimeError(f"dual-role workload is not alive: {exitcode}")
        context.labels.update(
            process=self.config.role,
            training_variant=(
                None if self.config.role == "inference" else "identical-dual-role"
            ),
            session_id=self.config.session_id,
            model=self.config.model,
            continuous=True,
            workload_started_before_scope_arm=True,
            attacker_observable="power_trace_only",
            computation_fingerprint=self.config.computation.fingerprint,
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
            "type": "persistent_identical_dual_role_single_model_process",
            "worker": self.config.metadata(),
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "capture_alignment": "role-blind worker starts and warms before scope arm",
            "replay_safe": True,
        }
