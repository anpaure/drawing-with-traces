#!/usr/bin/env python3
"""Profile complete strict inference or no-cover training CUDA streams."""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inference", "ordinary-training", "shaped-training"),
        required=True,
    )
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--tile-rows", type=int, default=128)
    parser.add_argument(
        "--shaping-backend",
        choices=("grouped-m1", "tiled-gemm"),
        default="tiled-gemm",
    )
    parser.add_argument(
        "--weight-gradient-schedule",
        choices=(
            "inline",
            "round-robin",
            "balanced-round-robin",
            "streaming-round-robin",
            "streaming-inference-cycle",
            "streaming-grouped",
        ),
        default="round-robin",
    )
    parser.add_argument("--streaming-dw-tasks-per-record", type=int, default=32)
    parser.add_argument("--grouped-dw-min-batch", type=int, default=4)
    parser.add_argument("--grouped-dw-max-batch", type=int, default=16)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def kernel_family(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("cublas", "cutlass", "gemm", "nvjet_")):
        return "dense_gemm"
    if "flash" in lowered and ("attention" in lowered or "sdpa" in lowered):
        return "flash_attention"
    if "softmax" in lowered:
        return "softmax"
    if "reduce" in lowered:
        return "reduction"
    if any(token in lowered for token in ("elementwise", "vectorized", "pointwise")):
        return "elementwise"
    if "layer_norm" in lowered or "rms" in lowered:
        return "normalization"
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "index" in lowered or "scatter" in lowered or "embedding" in lowered:
        return "index_scatter"
    return re.split(r"[<(]", name, maxsplit=1)[0][:100]


def summarize(profiler, *, phase: str, wall_ms: float) -> dict[str, Any]:
    raw_events = list(profiler.events())
    cuda_events = [event for event in raw_events if str(event.device_type).endswith("CUDA")]
    sequence = []
    exact = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0})
    families = collections.defaultdict(lambda: {"count": 0, "total_us": 0.0})
    for event in cuda_events:
        duration = float(
            getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or 0.0
        )
        name = str(event.key)
        if name == phase or name.startswith(("aten::", "autograd::", "torch::", "PyTorch")):
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
        "peak_allocated_bytes": int(__import__("torch").cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(__import__("torch").cuda.max_memory_reserved()),
    }


def profile_call(phase: str, function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        with record_function(phase):
            value = function()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - started) * 1e3
    return value, summarize(profiler, phase=phase, wall_ms=wall_ms)


def profile_inference(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    prompt = torch.randint(
        0,
        min(int(model.config.vocab_size), 32_000),
        (args.inference_batch_size, 1),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(prompt)
    with torch.inference_mode():
        prefill = model(input_ids=prompt, attention_mask=attention_mask, use_cache=True)
        cache = prefill.past_key_values
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        attention_mask = torch.cat(
            (attention_mask, attention_mask.new_ones((args.inference_batch_size, 1))), dim=1
        )
        warm = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = warm.past_key_values
        next_token = warm.logits[:, -1:].argmax(dim=-1)
        attention_mask = torch.cat(
            (attention_mask, attention_mask.new_ones((args.inference_batch_size, 1))), dim=1
        )
    torch.cuda.synchronize()

    def decode_once():
        with torch.inference_mode():
            return model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

    output, profile_summary = profile_call("batched_cached_decode", decode_once)
    return {
        "mode": args.mode,
        "model": args.model,
        "inference_batch_size": args.inference_batch_size,
        "profile": profile_summary,
        "checksum": int(output.logits[:, -1:].argmax(dim=-1).sum()),
        "model_instances_loaded": 1,
        "strict_invariants": {"inference_cover_tokens": 0, "filler_kernels": 0},
    }


def profile_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    from .strict_optimizer import InterleavedSGD
    from .strict_shapes import (
        DeferredWeightGradientScheduler,
        StrictShapeConfig,
        model_execution_plans,
        replace_linear_modules,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .train()
    )
    model.config.use_cache = False
    shaped_names = []
    scheduler = None
    if args.mode == "shaped-training":
        if args.weight_gradient_schedule != "inline":
            scheduler = DeferredWeightGradientScheduler()
        shaped_names = replace_linear_modules(
            model,
            StrictShapeConfig(
                backend=args.shaping_backend,
                forward_m1_per_launch=args.tile_rows,
                input_gradient_m1_per_launch=args.tile_rows,
                weight_gradient_m1_per_launch=args.tile_rows,
                pad_weight_gradient_reduction_to_input_width=True,
                weight_gradient_schedule=args.weight_gradient_schedule,
                streaming_weight_gradient_tasks_per_record=args.streaming_dw_tasks_per_record,
                grouped_weight_gradient_min_batch=args.grouped_dw_min_batch,
                grouped_weight_gradient_max_batch=args.grouped_dw_max_batch,
            ),
            scheduler=scheduler,
        )
    optimizer = InterleavedSGD(
        model,
        learning_rate=args.learning_rate,
        manual_parameter_ids=None if scheduler is None else scheduler.parameter_ids,
        manual_update_bucket_size=args.optimizer_bucket_size,
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    token_ids = torch.randint(
        0,
        min(int(model.config.vocab_size), 32_000),
        (args.training_batch_size, args.sequence_length + 1),
        generator=generator,
        device="cuda",
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
        if scheduler is not None:
            scheduler.begin_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )

    def finish_gradient_step() -> None:
        if scheduler is not None:
            scheduler.finish_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )
        optimizer.step_deferred()

    output = None
    loss = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=False)
        begin_gradient_step()
        output, loss = forward_loss()
        loss.backward()
        finish_gradient_step()
    torch.cuda.synchronize()
    del output, loss
    if scheduler is not None:
        scheduler.release_step_tensors()
    gc.collect()
    torch.cuda.empty_cache()

    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=False)
    with torch.cuda.graph(graph):
        optimizer.zero_grad(set_to_none=False)
        begin_gradient_step()
        graph_output, graph_loss = forward_loss()
        graph_loss.backward()
        finish_gradient_step()

    def replay_once():
        graph.replay()
        return graph_output

    output, profile_summary = profile_call("complete_training_graph", replay_once)
    plans = model_execution_plans(model) if shaped_names else {}
    result = {
        "mode": args.mode,
        "model": args.model,
        "training_batch_size": args.training_batch_size,
        "sequence_length": args.sequence_length,
        "tile_rows": args.tile_rows if shaped_names else None,
        "shaping_backend": args.shaping_backend if shaped_names else None,
        "weight_gradient_schedule": args.weight_gradient_schedule if shaped_names else None,
        "profile": profile_summary,
        "loss": float(graph_loss.detach()),
        "useful_loss_targets_per_update": args.training_batch_size * args.sequence_length,
        "parameter_tensors": len(list(model.parameters())),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_instances_loaded": 1,
        "optimizer_audit": vars(optimizer.audit()),
        "deferred_weight_gradient_audit": (None if scheduler is None else vars(scheduler.audit())),
        "shape_audit": None
        if not plans
        else {
            "modules": len(plans),
            "forward_launches": sum(plan.forward_launches for plan in plans.values()),
            "input_gradient_launches": sum(plan.input_gradient_launches for plan in plans.values()),
            "weight_gradient_launches": sum(plan.weight_gradient_launches for plan in plans.values()),
            "useful_flops": sum(plan.useful_flops for plan in plans.values()),
            "executed_flops": sum(plan.executed_flops for plan in plans.values()),
            "redundant_flops": sum(plan.redundant_flops for plan in plans.values()),
        },
        "strict_invariants": {
            "inference_cover_tokens": 0,
            "secondary_model_instances": 0,
            "filler_kernels": 0,
        },
    }
    optimizer.close()
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "inference":
        result = profile_inference(args)
    else:
        result = profile_training(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                **result,
                "profile": {k: v for k, v in result["profile"].items() if k != "cuda_kernel_sequence"},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
