#!/usr/bin/env python3
"""Profile complete CUDA kernel streams for Llama inference or training."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

try:
    from .continuous_workloads import (
        DEFAULT_CORPUS,
        DEFAULT_MODEL,
        _DecodeCover,
        _load_model,
        _register_layer_cover_hooks,
        cyclic_slice,
        sample_cover_token_count,
        WorkerConfig,
    )
    from .training_shapes import shape_linear_modules
except ImportError:  # pragma: no cover - direct hardware invocation.
    from continuous_workloads import (
        DEFAULT_CORPUS,
        DEFAULT_MODEL,
        _DecodeCover,
        _load_model,
        _register_layer_cover_hooks,
        cyclic_slice,
        sample_cover_token_count,
        WorkerConfig,
    )
    from training_shapes import shape_linear_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "training"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--linear-shaping", choices=("none", "token-row", "hybrid"), default="none")
    parser.add_argument("--cover-decode-tokens-per-microbatch", type=int, default=0)
    parser.add_argument("--cover-decode-token-jitter", type=int, default=0)
    parser.add_argument("--cover-backward-layer-interval", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("adamw8bit", "adamw", "adamw_fused", "sgd"),
        default="adamw8bit",
    )
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def kernel_family(name: str) -> str:
    """Collapse implementation-specific symbols into observable work families."""

    lowered = name.lower()
    if any(token in lowered for token in ("kgemm_4bit", "gemm_4bit", "matmul4bit", "quant")):
        return "quantized_gemm"
    if any(token in lowered for token in ("cublas", "cutlass", "gemm", "nvjet_")):
        return "dense_gemm"
    if "flash" in lowered and "attention" in lowered:
        return "flash_attention"
    if "softmax" in lowered:
        return "softmax"
    if any(token in lowered for token in ("adam", "optimizer", "percentile_clipping")):
        return "optimizer"
    if "reduce" in lowered:
        return "reduction"
    if any(token in lowered for token in ("elementwise", "vectorized")):
        return "elementwise"
    if "layer_norm" in lowered or "rms" in lowered:
        return "normalization"
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "index" in lowered or "scatter" in lowered:
        return "index_scatter"
    return re.split(r"[<(]", name, maxsplit=1)[0][:100]


def summarize(profiler, phase: str, wall_ms: float) -> dict[str, Any]:
    raw_events = list(profiler.events())
    cuda_events = [event for event in raw_events if str(event.device_type).endswith("CUDA")]
    sequence = []
    exact = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0})
    families = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for event in cuda_events:
        duration = float(
            getattr(event, "device_time_total", 0.0)
            or getattr(event, "cuda_time_total", 0.0)
            or 0.0
        )
        name = str(event.key)
        if name == phase or name.startswith(
            ("Optimizer.step#", "aten::", "autograd::", "torch::", "PyTorch")
        ):
            continue
        family = kernel_family(name)
        time_range = getattr(event, "time_range", None)
        row = {
            "name": name,
            "family": family,
            "duration_us": duration,
            "start_us": None if time_range is None else float(time_range.start),
        }
        sequence.append(row)
        exact[name]["count"] += 1
        exact[name]["total_us"] += duration
        exact[name]["max_us"] = max(exact[name]["max_us"], duration)
        families[family]["count"] += 1
        families[family]["total_us"] += duration
    sequence.sort(key=lambda row: math.inf if row["start_us"] is None else row["start_us"])

    gemm_shapes = collections.Counter()
    for event in raw_events:
        name = str(event.key)
        if name not in {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear"}:
            continue
        shapes = getattr(event, "input_shapes", None)
        gemm_shapes[(name, json.dumps(shapes))] += 1

    top_exact = sorted(
        ({"name": name, **values} for name, values in exact.items()),
        key=lambda row: row["total_us"],
        reverse=True,
    )
    top_families = sorted(
        ({"family": family, **values} for family, values in families.items()),
        key=lambda row: row["total_us"],
        reverse=True,
    )
    return {
        "phase": phase,
        "wall_ms": wall_ms,
        "cuda_kernel_count": len(sequence),
        "cuda_busy_ms": sum(row["duration_us"] for row in sequence) / 1e3,
        "top_kernel_families": top_families,
        "top_cuda_kernels": top_exact[:150],
        "aten_gemm_shapes": [
            {"name": name, "input_shapes": json.loads(shapes), "count": count}
            for (name, shapes), count in gemm_shapes.most_common()
        ],
        "cuda_kernel_sequence": sequence,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def profile_call(phase: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        with record_function(phase):
            value = function()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - start) * 1e3
    return value, summarize(profiler, phase, wall_ms)


def make_optimizer(model, name: str):
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            min_8bit_size=4096,
        )
    if name in {"adamw", "adamw_fused"}:
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            foreach=False if name == "adamw" else None,
            fused=name == "adamw_fused",
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.1,
            momentum=0,
            foreach=False,
            fused=False,
        )
    raise ValueError(f"unknown optimizer: {name}")


def profile_inference(config: WorkerConfig, token_ids: list[int]) -> dict[str, Any]:
    model = _load_model(config)
    model.eval()
    prompt = torch.tensor(
        cyclic_slice(token_ids, config.seed, config.prompt_tokens),
        device="cuda",
        dtype=torch.long,
    ).unsqueeze(0)
    attention_mask = torch.ones_like(prompt)
    with torch.inference_mode():
        prefill = model(input_ids=prompt, attention_mask=attention_mask, use_cache=True)
        cache = prefill.past_key_values
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
        warmup = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = warmup.past_key_values
        next_token = warmup.logits[:, -1:].argmax(dim=-1)
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
    torch.cuda.synchronize()

    def decode_once():
        with torch.inference_mode():
            return model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

    output, summary = profile_call("quantized_cached_decode", decode_once)
    return {
        "config": config.metadata(),
        "profile": summary,
        "checksum": int(output.logits[:, -1:].argmax(dim=-1).sum()),
        "layers": int(model.config.num_hidden_layers),
        "parameter_tensors": len(list(model.parameters())),
        "stored_parameter_elements": sum(parameter.numel() for parameter in model.parameters()),
    }


def profile_training(config: WorkerConfig, token_ids: list[int]) -> dict[str, Any]:
    model = _load_model(config)
    model.train()
    model.config.use_cache = False
    shaped_modules = shape_linear_modules(model, config.linear_shaping)
    parameters = list(model.parameters())
    optimizer = make_optimizer(model, config.optimizer)
    cover = (
        _DecodeCover(config, token_ids)
        if config.cover_decode_tokens_per_microbatch or config.cover_backward_layer_interval
        else None
    )
    cover_hook_handles = (
        _register_layer_cover_hooks(model, cover, config.cover_backward_layer_interval)
        if cover is not None and config.cover_backward_layer_interval
        else []
    )
    token_offset = config.seed % len(token_ids)
    cover_random = random.Random(config.seed ^ 0xC0FEBABE)

    def update() -> tuple[float, int]:
        nonlocal token_offset
        optimizer.zero_grad(set_to_none=True)
        losses = []
        cover_checksum = 0
        for _ in range(config.gradient_accumulation_steps):
            ids = torch.tensor(
                cyclic_slice(token_ids, token_offset, config.training_sequence_length),
                device="cuda",
                dtype=torch.long,
            ).unsqueeze(0)
            token_offset = (token_offset + config.training_sequence_length) % len(token_ids)
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                labels=ids,
                use_cache=False,
            )
            (output.loss / config.gradient_accumulation_steps).backward()
            losses.append(float(output.loss.detach()))
            if cover is not None:
                cover_checksum ^= cover.decode(
                    sample_cover_token_count(
                        config.cover_decode_tokens_per_microbatch,
                        config.cover_decode_token_jitter,
                        cover_random,
                    )
                )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return sum(losses) / len(losses), cover_checksum

    update()  # Warm kernels and initialize every optimizer state tensor.
    torch.cuda.synchronize()
    (loss, cover_checksum), summary = profile_call("complete_training_update", update)
    return {
        "config": config.metadata(),
        "profile": summary,
        "loss": loss,
        "cover_checksum": cover_checksum,
        "layers": int(model.config.num_hidden_layers),
        "parameter_tensors": len(parameters),
        "parameters": sum(parameter.numel() for parameter in parameters),
        "gradient_tensors_before_zero": "verified by persistent-worker benchmark",
        "optimizer_parameter_tensors": len(optimizer.param_groups[0]["params"]),
        "optimizer_state_tensors": len(optimizer.state),
        "shaped_linear_modules": len(shaped_modules),
        "cover_backward_hooks": len(cover_hook_handles),
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    token_ids = tokenizer(DEFAULT_CORPUS, add_special_tokens=False)["input_ids"]
    config = WorkerConfig(
        mode=args.mode,
        session_id=f"profile-{args.mode}",
        model=args.model,
        seed=args.seed,
        prompt_tokens=args.prompt_tokens,
        training_sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        linear_shaping=args.linear_shaping,
        cover_decode_tokens_per_microbatch=args.cover_decode_tokens_per_microbatch,
        cover_decode_token_jitter=args.cover_decode_token_jitter,
        cover_backward_layer_interval=args.cover_backward_layer_interval,
        optimizer=args.optimizer,
    )
    result = (
        profile_inference(config, token_ids)
        if args.mode == "inference"
        else profile_training(config, token_ids)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    compact = {
        "output": str(args.output),
        "mode": args.mode,
        "wall_ms": result["profile"]["wall_ms"],
        "cuda_kernel_count": result["profile"]["cuda_kernel_count"],
        "cuda_busy_ms": result["profile"]["cuda_busy_ms"],
        "top_kernel_families": result["profile"]["top_kernel_families"][:12],
    }
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
