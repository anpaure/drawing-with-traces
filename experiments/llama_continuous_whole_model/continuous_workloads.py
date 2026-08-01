"""Persistent whole-model serving and training workers.

The GPU workload runs in a spawned child process before the scope is armed.
The capture process therefore cannot accidentally align model execution to the
ChipWhisperer trigger, and USB polling cannot starve Python-side CUDA launches.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import sidecapture as sc

try:
    from .training_shapes import LinearShaping, shape_linear_modules
except ImportError:  # pragma: no cover - direct hardware invocation.
    from training_shapes import LinearShaping, shape_linear_modules


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_CORPUS = """
Side-channel measurements expose physical activity rather than semantic operation names.
Language-model inference repeatedly performs prompt prefill and autoregressive cached decode.
Language-model training performs a causal forward pass, reverse-mode differentiation, and an optimizer update.
An external current monitor sees only a continuous waveform and does not know operation boundaries.
Changing matrix multiplication geometry and temporal scheduling can alter that observable waveform.
Independent sessions and long observation horizons are required for a defensible detector evaluation.
"""


def cyclic_slice(values: list[int], start: int, length: int) -> list[int]:
    """Return a fixed-length cyclic slice without padding tokens."""

    if not values:
        raise ValueError("cyclic_slice requires at least one value")
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    return [values[(start + offset) % len(values)] for offset in range(length)]


@dataclass(frozen=True)
class WorkerConfig:
    """Serializable configuration for one persistent GPU process."""

    mode: Literal["inference", "training"]
    session_id: str
    model: str = DEFAULT_MODEL
    seed: int = 0
    corpus: str = DEFAULT_CORPUS
    local_files_only: bool = True
    prompt_tokens: int = 32
    decode_tokens: int = 32
    training_sequence_length: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    gradient_checkpointing: bool = False
    inference_quantization: Literal["nf4", "none"] = "nf4"
    optimizer: Literal["adamw8bit", "adamw", "adamw_fused", "sgd"] = "adamw8bit"
    gradient_accumulation_steps: int = 1
    linear_shaping: LinearShaping = "none"
    cover_decode_tokens_per_microbatch: int = 0
    cover_decode_token_jitter: int = 0
    cover_backward_layer_interval: int = 0
    cover_prompt_tokens: int = 32
    cover_reset_tokens: int = 32

    def __post_init__(self) -> None:
        if self.mode not in {"inference", "training"}:
            raise ValueError(f"mode must be inference or training, got {self.mode!r}")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        for name in ("prompt_tokens", "decode_tokens", "training_sequence_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")
        if self.inference_quantization not in {"nf4", "none"}:
            raise ValueError(
                "inference_quantization must be 'nf4' or 'none', got "
                f"{self.inference_quantization!r}"
            )
        if self.optimizer not in {"adamw8bit", "adamw", "adamw_fused", "sgd"}:
            raise ValueError(
                "optimizer must be adamw8bit, adamw, adamw_fused, or sgd, got "
                f"{self.optimizer!r}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be positive, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.linear_shaping not in {"none", "token-row", "hybrid"}:
            raise ValueError(
                "linear_shaping must be none, token-row, or hybrid, got "
                f"{self.linear_shaping!r}"
            )
        if self.cover_decode_tokens_per_microbatch < 0:
            raise ValueError(
                "cover_decode_tokens_per_microbatch cannot be negative, got "
                f"{self.cover_decode_tokens_per_microbatch}"
            )
        if not 0 <= self.cover_decode_token_jitter <= self.cover_decode_tokens_per_microbatch:
            raise ValueError(
                "cover_decode_token_jitter must be between 0 and "
                "cover_decode_tokens_per_microbatch; got "
                f"{self.cover_decode_token_jitter} and "
                f"{self.cover_decode_tokens_per_microbatch}"
            )
        if self.cover_backward_layer_interval < 0:
            raise ValueError(
                "cover_backward_layer_interval cannot be negative, got "
                f"{self.cover_backward_layer_interval}"
            )
        if self.cover_prompt_tokens < 1 or self.cover_reset_tokens < 1:
            raise ValueError("cover_prompt_tokens and cover_reset_tokens must be positive")

    def metadata(self) -> dict[str, Any]:
        result = asdict(self)
        corpus = result.pop("corpus")
        result["corpus_sha256"] = hashlib.sha256(corpus.encode()).hexdigest()
        result["corpus_characters"] = len(corpus)
        return result


def _send_latest(messages, payload: dict[str, Any]) -> None:
    payload = {**payload, "host_monotonic_ns": time.monotonic_ns()}
    try:
        messages.put_nowait(payload)
        return
    except queue.Full:
        pass
    try:
        messages.get_nowait()
    except queue.Empty:
        pass
    try:
        messages.put_nowait(payload)
    except queue.Full:
        pass


def _optimizer_state_bytes(optimizer) -> int:
    import torch

    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def _load_model(config: WorkerConfig):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "local_files_only": config.local_files_only,
        "dtype": torch.bfloat16,
        "device_map": torch.cuda.current_device(),
    }
    if config.mode == "inference" and config.inference_quantization == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(config.model, **kwargs)


class _DecodeCover:
    """Maintain a real quantized autoregressive decode stream between microbatches."""

    def __init__(self, config: WorkerConfig, token_ids: list[int]) -> None:
        import torch

        inference_config = replace(config, mode="inference", inference_quantization="nf4")
        self.model = _load_model(inference_config)
        self.model.eval()
        self.token_ids = token_ids
        self.prompt_tokens = config.cover_prompt_tokens
        self.reset_tokens = config.cover_reset_tokens
        self.token_offset = (config.seed + 7919) % len(token_ids)
        self.generated_since_reset = 0
        self.total_decode_tokens = 0
        self.total_prefill_tokens = 0
        self.cache = None
        self.next_token = None
        self.attention_mask = None
        self._torch = torch
        self._reset()

    def _reset(self) -> None:
        torch = self._torch
        prompt = cyclic_slice(self.token_ids, self.token_offset, self.prompt_tokens)
        input_ids = torch.tensor(prompt, device="cuda", dtype=torch.long).unsqueeze(0)
        self.attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=self.attention_mask,
                use_cache=True,
            )
        self.cache = output.past_key_values
        self.next_token = output.logits[:, -1:].argmax(dim=-1)
        self.token_offset = (self.token_offset + self.prompt_tokens) % len(self.token_ids)
        self.generated_since_reset = 0
        self.total_prefill_tokens += self.prompt_tokens

    def decode(self, count: int) -> int:
        torch = self._torch
        checksum = 0
        for _ in range(count):
            if self.generated_since_reset >= self.reset_tokens:
                self._reset()
            assert self.attention_mask is not None
            assert self.cache is not None
            assert self.next_token is not None
            self.attention_mask = torch.cat(
                (self.attention_mask, self.attention_mask.new_ones((1, 1))), dim=1
            )
            with torch.inference_mode():
                output = self.model(
                    input_ids=self.next_token,
                    attention_mask=self.attention_mask,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            self.cache = output.past_key_values
            self.next_token = output.logits[:, -1:].argmax(dim=-1)
            checksum ^= int(self.next_token.sum())
            self.generated_since_reset += 1
            self.total_decode_tokens += 1
        return checksum


def cover_layer_indices(layer_count: int, interval: int) -> list[int]:
    """Return layers whose completed gradients should be followed by decode."""

    if layer_count < 1:
        raise ValueError(f"layer_count must be positive, got {layer_count}")
    if interval < 1:
        raise ValueError(f"interval must be positive, got {interval}")
    return list(range(0, layer_count, interval))


def sample_cover_token_count(base: int, jitter: int, generator: random.Random) -> int:
    """Draw an inclusive, deterministic-per-session cover-token count."""

    if base < 0 or not 0 <= jitter <= base:
        raise ValueError(f"expected base >= jitter >= 0, got {base=} and {jitter=}")
    return generator.randint(base - jitter, base + jitter)


def _register_layer_cover_hooks(model, cover: _DecodeCover, interval: int):
    """Insert a real decode token at regular layer boundaries in backward."""

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError(
            "backward-layer cover requires a Transformers-style model.model.layers sequence"
        )
    handles = []
    for index in cover_layer_indices(len(layers), interval):
        parameter = layers[index].input_layernorm.weight

        def decode_after_layer(gradient, *, _cover=cover):
            _cover.decode(1)
            return gradient

        handles.append(parameter.register_hook(decode_after_layer))
    return handles


def _worker_main(config: WorkerConfig, stop_event, messages) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import torch
        from transformers import AutoTokenizer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            local_files_only=config.local_files_only,
        )
        token_ids = tokenizer(config.corpus, add_special_tokens=False)["input_ids"]
        if len(token_ids) < 2:
            raise ValueError("The workload corpus must tokenize to at least two tokens")
        model = _load_model(config)
        parameter_tensors = list(model.parameters())
        common = {
            "pid": os.getpid(),
            "mode": config.mode,
            "model": config.model,
            "layers": int(model.config.num_hidden_layers),
            "parameter_tensors": len(parameter_tensors),
            "parameters": sum(parameter.numel() for parameter in parameter_tensors),
            "dtype": "bfloat16",
        }

        if config.mode == "inference":
            model.eval()
            request = 0
            token_offset = config.seed % len(token_ids)
            while not stop_event.is_set():
                prompt_ids = cyclic_slice(token_ids, token_offset, config.prompt_tokens)
                input_ids = torch.tensor(prompt_ids, device="cuda", dtype=torch.long).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    start.record()
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                    cache = output.past_key_values
                    next_token = output.logits[:, -1:].argmax(dim=-1)
                    for _ in range(config.decode_tokens):
                        attention_mask = torch.cat(
                            (attention_mask, attention_mask.new_ones((1, 1))), dim=1
                        )
                        output = model(
                            input_ids=next_token,
                            attention_mask=attention_mask,
                            past_key_values=cache,
                            use_cache=True,
                        )
                        cache = output.past_key_values
                        next_token = output.logits[:, -1:].argmax(dim=-1)
                    end.record()
                    end.synchronize()
                    checksum = int(next_token.sum())
                request += 1
                token_offset = (token_offset + config.prompt_tokens) % len(token_ids)
                event = "ready" if request == 1 else "heartbeat"
                _send_latest(
                    messages,
                    {
                        "event": event,
                        **common,
                        "requests": request,
                        "prefill_tokens": request * config.prompt_tokens,
                        "decode_tokens": request * config.decode_tokens,
                        "last_request_cuda_ms": float(start.elapsed_time(end)),
                        "last_token_checksum": checksum,
                        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    },
                )
        else:
            model.train()
            model.config.use_cache = False
            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            shaped_modules = shape_linear_modules(model, config.linear_shaping)
            parameter_tensors = list(model.parameters())
            common["parameter_tensors"] = len(parameter_tensors)
            common["parameters"] = sum(parameter.numel() for parameter in parameter_tensors)
            if config.optimizer == "adamw8bit":
                import bitsandbytes as bnb

                optimizer = bnb.optim.AdamW8bit(
                    parameter_tensors,
                    lr=config.learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=config.weight_decay,
                    min_8bit_size=4096,
                )
            elif config.optimizer in {"adamw", "adamw_fused"}:
                optimizer = torch.optim.AdamW(
                    parameter_tensors,
                    lr=config.learning_rate,
                    betas=(0.9, 0.95),
                    weight_decay=config.weight_decay,
                    foreach=False if config.optimizer == "adamw" else None,
                    fused=config.optimizer == "adamw_fused",
                )
            else:
                optimizer = torch.optim.SGD(
                    parameter_tensors,
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    momentum=0,
                    foreach=False,
                    fused=False,
                )
            cover = (
                _DecodeCover(config, token_ids)
                if config.cover_decode_tokens_per_microbatch
                or config.cover_backward_layer_interval
                else None
            )
            cover_hook_handles = (
                _register_layer_cover_hooks(
                    model,
                    cover,
                    config.cover_backward_layer_interval,
                )
                if cover is not None and config.cover_backward_layer_interval
                else []
            )
            step = 0
            microbatch = 0
            training_input_tokens = 0
            training_loss_tokens = 0
            cover_checksum = 0
            cover_random = random.Random(config.seed ^ 0xC0FEBABE)
            token_offset = config.seed % len(token_ids)
            while not stop_event.is_set():
                optimizer.zero_grad(set_to_none=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                losses = []
                for _ in range(config.gradient_accumulation_steps):
                    batch_ids = cyclic_slice(
                        token_ids,
                        token_offset,
                        config.training_sequence_length,
                    )
                    input_ids = torch.tensor(
                        batch_ids, device="cuda", dtype=torch.long
                    ).unsqueeze(0)
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                        use_cache=False,
                    )
                    (output.loss / config.gradient_accumulation_steps).backward()
                    losses.append(float(output.loss.detach()))
                    del output
                    microbatch += 1
                    training_input_tokens += config.training_sequence_length
                    training_loss_tokens += max(0, config.training_sequence_length - 1)
                    token_offset = (
                        token_offset + config.training_sequence_length
                    ) % len(token_ids)
                    if cover is not None:
                        cover_checksum ^= cover.decode(
                            sample_cover_token_count(
                                config.cover_decode_tokens_per_microbatch,
                                config.cover_decode_token_jitter,
                                cover_random,
                            )
                        )
                gradient_tensors = sum(parameter.grad is not None for parameter in parameter_tensors)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                end.record()
                end.synchronize()
                loss = sum(losses) / len(losses)
                step += 1
                event = "ready" if step == 1 else "heartbeat"
                _send_latest(
                    messages,
                    {
                        "event": event,
                        **common,
                        "optimizer": config.optimizer,
                        "linear_shaping": config.linear_shaping,
                        "shaped_linear_modules": len(shaped_modules),
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "optimizer_parameter_tensors": len(optimizer.param_groups[0]["params"]),
                        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
                        "steps": step,
                        "microbatches": microbatch,
                        "training_input_tokens": training_input_tokens,
                        "training_loss_tokens": training_loss_tokens,
                        "cover_decode_tokens": 0 if cover is None else cover.total_decode_tokens,
                        "cover_decode_token_jitter": config.cover_decode_token_jitter,
                        "cover_prefill_tokens": 0 if cover is None else cover.total_prefill_tokens,
                        "cover_backward_layer_interval": config.cover_backward_layer_interval,
                        "cover_backward_hooks": len(cover_hook_handles),
                        "cover_checksum": cover_checksum,
                        "last_loss": loss,
                        "last_step_cuda_ms": float(start.elapsed_time(end)),
                        "gradient_tensors": gradient_tensors,
                        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    },
                )
    except BaseException as error:
        _send_latest(
            messages,
            {
                "event": "error",
                "pid": os.getpid(),
                "mode": config.mode,
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


class PersistentGpuWorkload(sc.Workload):
    """Keep a GPU process running continuously across arbitrary scope windows."""

    replay_safe = True

    def __init__(
        self,
        config: WorkerConfig,
        *,
        startup_timeout_s: float = 180.0,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        self.config = config
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        if self.startup_timeout_s <= 0 or self.shutdown_timeout_s <= 0:
            raise ValueError("worker startup and shutdown timeouts must be positive")
        self._process = None
        self._stop_event = None
        self._messages = None
        self._latest: dict[str, Any] = {}

    def _accept_message(self, message: dict[str, Any]) -> None:
        if message.get("event") == "error":
            raise RuntimeError(
                f"Persistent {self.config.mode} worker failed: {message.get('error_type')}: "
                f"{message.get('error')}\n{message.get('traceback', '')}"
            )
        self._latest = message

    def _assert_alive(self) -> None:
        if self._process is None:
            raise RuntimeError("Persistent GPU worker was not started")
        if not self._process.is_alive():
            raise RuntimeError(
                f"Persistent GPU worker exited unexpectedly with code {self._process.exitcode}"
            )

    def _drain(self) -> None:
        if self._messages is None:
            return
        while True:
            try:
                self._accept_message(self._messages.get_nowait())
            except queue.Empty:
                return

    def setup(self) -> None:
        context = mp.get_context("spawn")
        self._stop_event = context.Event()
        self._messages = context.Queue(maxsize=128)
        self._process = context.Process(
            target=_worker_main,
            args=(self.config, self._stop_event, self._messages),
            name=f"continuous-{self.config.mode}-{self.config.session_id}",
        )
        self._process.start()
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            self._assert_alive()
            try:
                message = self._messages.get(timeout=0.25)
            except queue.Empty:
                continue
            self._accept_message(message)
            if message.get("event") == "ready":
                return
        raise TimeoutError(
            f"Persistent {self.config.mode} worker did not complete warmup within "
            f"{self.startup_timeout_s:.1f} seconds"
        )

    def run(self, context: sc.CaptureContext) -> dict[str, Any]:
        self._drain()
        self._assert_alive()
        context.labels.update(
            process=self.config.mode,
            session_id=self.config.session_id,
            model=self.config.model,
            continuous=True,
            workload_started_before_scope_arm=True,
            attacker_observable="power_trace_only",
        )
        return {
            key: value
            for key, value in self._latest.items()
            if key not in {"traceback"}
        }

    def teardown(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._process is not None:
            self._process.join(self.shutdown_timeout_s)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(5.0)
        if self._messages is not None:
            self._messages.close()
        self._process = None
        self._stop_event = None
        self._messages = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "persistent_whole_model_gpu_process",
            "worker": self.config.metadata(),
            "startup_timeout_s": self.startup_timeout_s,
            "shutdown_timeout_s": self.shutdown_timeout_s,
            "capture_alignment": "worker starts and warms before scope arm",
            "replay_safe": True,
        }
