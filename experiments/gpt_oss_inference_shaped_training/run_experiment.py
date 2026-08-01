#!/usr/bin/env python3
"""Profile real GPT-OSS decode and exact inference-shaped fine-tuning.

The experiment trains the layer-2 attention output projection against the real
causal-LM loss.  All other parameters are frozen so ordinary and shaped runs
perform the same optimization problem.  The shaped backward replaces only the
weight-gradient reduction; no gradient values are approximated.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.profiler import ProfilerActivity, profile, record_function

from inference_shaped_linear import InferenceShapeConfig, InferenceShapedLinear


DEFAULT_MODEL = "openai/gpt-oss-20b"
TARGET_LAYER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Training can imitate inference")
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--layer", type=int, default=TARGET_LAYER)
    parser.add_argument("--weight-gradient-rows", type=int, default=1)
    parser.add_argument("--weight-gradient-inference-rows", type=int)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-chrome-traces", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def kernel_family(name: str) -> str:
    """Normalize volatile implementation names into workload-level families."""

    if name.startswith("_matmul_ogs_"):
        return "mxfp4_moe_matmul"
    if name.startswith("_finalize_matmul_scatter"):
        return "moe_finalize_scatter"
    if name.startswith("_finalize_matmul"):
        return "moe_finalize"
    if name.startswith(("_combined_routing", "_sum_bitmatrix", "_topk")):
        return "moe_routing"
    if any(token in name for token in ("nvjet_", "cutlass::Kernel", "cublasLt::")):
        return "dense_gemm"
    if re.search(r"gem[mv]", name, re.IGNORECASE):
        return "dense_gemm"
    if "softmax" in name.lower():
        return "softmax"
    if "reduce_kernel" in name:
        return "reduction"
    if "elementwise_kernel" in name:
        return "elementwise"
    if name.startswith("Memcpy"):
        return "memcpy"
    if name.startswith("Memset"):
        return "memset"
    if "CatArray" in name:
        return "concat_copy"
    if "index" in name.lower() or "scatter" in name.lower():
        return "index_scatter"
    return name.split("<", 1)[0][:100]


def event_summary(profiler, phase: str, wall_ms: float) -> dict[str, Any]:
    raw_events = list(profiler.events())
    cuda_events = [event for event in raw_events if str(event.device_type).endswith("CUDA")]
    kernel_sequence = []
    by_kernel: dict[str, dict[str, float | int]] = collections.defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0}
    )
    for event in cuda_events:
        duration = float(
            getattr(event, "device_time_total", 0.0)
            or getattr(event, "cuda_time_total", 0.0)
            or 0.0
        )
        name = str(event.key)
        row = by_kernel[name]
        row["count"] = int(row["count"]) + 1
        row["total_us"] = float(row["total_us"]) + duration
        row["max_us"] = max(float(row["max_us"]), duration)
        time_range = getattr(event, "time_range", None)
        kernel_sequence.append(
            {
                "name": name,
                "family": kernel_family(name),
                "duration_us": duration,
                "start_us": None if time_range is None else float(time_range.start),
            }
        )
    kernel_sequence.sort(key=lambda row: math.inf if row["start_us"] is None else row["start_us"])
    actual_kernels = [row for row in kernel_sequence if row["name"] != phase]
    top_kernels = sorted(
        ({"name": name, **values} for name, values in by_kernel.items() if name != phase),
        key=lambda row: float(row["total_us"]),
        reverse=True,
    )
    family_duration = collections.Counter()
    for row in actual_kernels:
        family_duration[row["family"]] += row["duration_us"]

    aten_gemms = []
    for event in raw_events:
        name = str(event.key)
        if name not in {"aten::mm", "aten::addmm", "aten::bmm", "aten::linear"}:
            continue
        aten_gemms.append(
            {
                "name": name,
                "input_shapes": getattr(event, "input_shapes", None),
                "device_time_us": float(getattr(event, "device_time_total", 0.0) or 0.0),
            }
        )
    return {
        "phase": phase,
        "wall_ms": wall_ms,
        "actual_cuda_kernel_count": len(actual_kernels),
        "actual_cuda_busy_ms": sum(row["duration_us"] for row in actual_kernels) / 1e3,
        "top_cuda_kernels": top_kernels[:100],
        "family_duration_us": dict(family_duration),
        "cuda_kernel_sequence": actual_kernels,
        "aten_gemms": aten_gemms,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def profile_call(
    output_dir: Path,
    phase: str,
    function: Callable[[], Any],
    *,
    enabled: bool,
    save_chrome_trace: bool,
) -> tuple[Any, dict[str, Any]]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if not enabled:
        start = time.perf_counter()
        value = function()
        torch.cuda.synchronize()
        return value, {"phase": phase, "wall_ms": (time.perf_counter() - start) * 1e3}

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
    summary = event_summary(profiler, phase, wall_ms)
    (output_dir / f"{phase}.summary.json").write_text(json.dumps(summary, indent=2))
    if save_chrome_trace:
        profiler.export_chrome_trace(str(output_dir / f"{phase}.chrome.json"))
    print(
        f"PROFILE {phase}: wall={wall_ms:.3f} ms, "
        f"kernels={summary['actual_cuda_kernel_count']}, "
        f"busy={summary['actual_cuda_busy_ms']:.3f} ms",
        flush=True,
    )
    return value, summary


def weighted_jaccard(
    left: list[dict[str, Any]], right: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]
) -> float:
    left_duration = collections.Counter()
    right_duration = collections.Counter()
    for row in left:
        left_duration[key(row)] += row["duration_us"]
    for row in right:
        right_duration[key(row)] += row["duration_us"]
    keys = left_duration.keys() | right_duration.keys()
    denominator = sum(max(left_duration[item], right_duration[item]) for item in keys)
    if denominator == 0:
        return 1.0
    return sum(min(left_duration[item], right_duration[item]) for item in keys) / denominator


def compare_profiles(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    left_sequence = left["cuda_kernel_sequence"]
    right_sequence = right["cuda_kernel_sequence"]
    left_families = [row["family"] for row in left_sequence]
    right_families = [row["family"] for row in right_sequence]
    return {
        "exact_kernel_duration_jaccard": weighted_jaccard(left_sequence, right_sequence, lambda row: row["name"]),
        "family_duration_jaccard": weighted_jaccard(left_sequence, right_sequence, lambda row: row["family"]),
        "family_sequence_match": difflib.SequenceMatcher(
            None, left_families, right_families, autojunk=False
        ).ratio(),
    }


@contextmanager
def capture_linear_boundary(module: nn.Module) -> Iterator[dict[str, Tensor]]:
    captured: dict[str, Tensor] = {}

    def hook(_module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        captured["input"] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()

        def save_gradient(gradient: Tensor) -> None:
            captured["grad_output"] = gradient.detach().reshape(-1, gradient.shape[-1]).clone()

        output.register_hook(save_gradient)

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def gradient_error(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    denominator = expected.float().norm().clamp_min(torch.finfo(torch.float32).tiny)
    return {
        "max_absolute": float(difference.max().cpu()),
        "mean_absolute": float(difference.mean().cpu()),
        "relative_l2": float((difference.norm() / denominator).cpu()),
        "equal_fraction": float((actual == expected).float().mean().cpu()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading cached model {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    model.config.use_cache = False

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, : args.tokens].to("cuda")
    if input_ids.shape[1] < 2:
        raise RuntimeError(
            f"The prompt produced only {input_ids.shape[1]} token(s); at least two are required for causal loss"
        )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent = model.model.layers[args.layer].self_attn
    ordinary_projection = parent.o_proj
    if not isinstance(ordinary_projection, nn.Linear):
        raise TypeError(
            f"Expected layer {args.layer} o_proj to be nn.Linear, got {type(ordinary_projection).__name__}"
        )
    ordinary_projection.weight.requires_grad_(True)
    if ordinary_projection.bias is not None:
        ordinary_projection.bias.requires_grad_(False)

    target_name = f"model.layers.{args.layer}.self_attn.o_proj.weight"
    print(
        f"Training only {target_name}, shape={tuple(ordinary_projection.weight.shape)}, "
        f"dtype={ordinary_projection.weight.dtype}",
        flush=True,
    )

    # Real one-token cached decode target.
    model.eval()
    with torch.inference_mode():
        prefill = model(
            input_ids=input_ids[:, :1],
            attention_mask=attention_mask[:, :1],
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = prefill.past_key_values
        decode_id = input_ids[:, 1:2]
        decode_mask = attention_mask[:, :2]
        for _ in range(2):
            model(
                input_ids=decode_id,
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )

    decode_output, decode_profile = profile_call(
        args.output_dir,
        "decode",
        lambda: model(
            input_ids=decode_id,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        ),
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )

    def causal_loss() -> Tensor:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        ).loss

    # Ordinary selective fine-tuning baseline.
    model.train()
    parent.o_proj = ordinary_projection
    ordinary_projection.weight.grad = None

    def ordinary_step() -> Tensor:
        ordinary_projection.weight.grad = None
        loss = causal_loss()
        loss.backward()
        return loss

    with capture_linear_boundary(ordinary_projection) as ordinary_boundary:
        ordinary_loss, ordinary_profile = profile_call(
            args.output_dir,
            "ordinary_training_step",
            ordinary_step,
            enabled=args.profile,
            save_chrome_trace=args.save_chrome_traces,
        )
    ordinary_gradient = ordinary_projection.weight.grad.detach().clone()
    ordinary_direct_gradient = (
        ordinary_boundary["grad_output"].transpose(0, 1) @ ordinary_boundary["input"]
    )
    ordinary_gradient_error = gradient_error(ordinary_gradient, ordinary_direct_gradient)

    # Install the exact inference-shaped decomposition around the same Parameter.
    shaped_projection = InferenceShapedLinear(
        ordinary_projection,
        InferenceShapeConfig(
            forward_rows=1,
            input_gradient_rows=1,
            weight_gradient_rows=args.weight_gradient_rows,
            weight_gradient_inference_rows=args.weight_gradient_inference_rows,
            reduction_k=ordinary_projection.in_features,
            bias_epilogue=True,
        ),
    )
    parent.o_proj = shaped_projection
    ordinary_projection.weight.grad = None

    def shaped_step() -> Tensor:
        ordinary_projection.weight.grad = None
        loss = causal_loss()
        loss.backward()
        return loss

    # Warm up all kernels once before timing and profiling.
    shaped_warmup_loss = shaped_step()
    ordinary_projection.weight.grad = None
    torch.cuda.synchronize()

    with capture_linear_boundary(shaped_projection) as shaped_boundary:
        shaped_loss, shaped_profile = profile_call(
            args.output_dir,
            "inference_shaped_training_step",
            shaped_step,
            enabled=args.profile,
            save_chrome_trace=args.save_chrome_traces,
        )
    shaped_gradient = ordinary_projection.weight.grad.detach().clone()
    shaped_direct_gradient = shaped_boundary["grad_output"].transpose(0, 1) @ shaped_boundary["input"]
    shaped_gradient_error = gradient_error(shaped_gradient, shaped_direct_gradient)

    # Region-level controls: ordinary dW, shaped dW, and repeated real inference
    # projection.  The latter makes kernel-shape matching independently testable
    # from the rest of the transformer step.
    ordinary_dw, ordinary_dw_profile = profile_call(
        args.output_dir,
        "ordinary_weight_gradient",
        lambda: shaped_boundary["grad_output"].transpose(0, 1) @ shaped_boundary["input"],
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )

    from inference_shaped_linear import inference_shaped_weight_gradient

    shaped_dw, shaped_dw_profile = profile_call(
        args.output_dir,
        "inference_shaped_weight_gradient",
        lambda: inference_shaped_weight_gradient(
            shaped_boundary["input"],
            shaped_boundary["grad_output"],
            reduction_k=ordinary_projection.in_features,
            rows=args.weight_gradient_rows,
            inference_rows=args.weight_gradient_inference_rows,
            bias_epilogue=True,
        ),
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )

    projection_input = shaped_boundary["input"][:1]
    shaped_feature_rows = (
        ordinary_projection.in_features
        if args.weight_gradient_inference_rows is None
        else args.weight_gradient_inference_rows
    )
    replay_count = math.ceil(shaped_feature_rows / args.weight_gradient_rows)

    def projection_replay() -> Tensor:
        result = None
        for _ in range(replay_count):
            result = F.linear(projection_input, ordinary_projection.weight, ordinary_projection.bias)
        if result is None:
            return F.linear(projection_input, ordinary_projection.weight, ordinary_projection.bias)
        return result

    projection_output, projection_profile = profile_call(
        args.output_dir,
        "inference_projection_replay",
        projection_replay,
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )

    # Prove an effective model update despite the BF16 model parameter by using
    # an FP32 master copy, as mixed-precision optimizers do.  A short deterministic
    # line search avoids claiming training progress from a numerically zero step.
    original_weight = ordinary_projection.weight.detach().clone()
    master_weight = original_weight.float()
    before_update_loss = float(shaped_loss.detach().float().cpu())
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    update_trials = []
    parent.o_proj = ordinary_projection
    model.eval()
    with torch.no_grad():
        for learning_rate in learning_rates:
            candidate = master_weight - learning_rate * shaped_gradient.float()
            ordinary_projection.weight.copy_(candidate.to(dtype=ordinary_projection.weight.dtype))
            candidate_loss = float(causal_loss().detach().float().cpu())
            changed = ordinary_projection.weight != original_weight
            update_trials.append(
                {
                    "learning_rate": learning_rate,
                    "loss": candidate_loss,
                    "changed_elements": int(changed.count_nonzero().cpu()),
                    "max_parameter_delta": float(
                        (ordinary_projection.weight.float() - original_weight.float()).abs().max().cpu()
                    ),
                }
            )
        best_trial = min(update_trials, key=lambda row: row["loss"])
        best_candidate = master_weight - best_trial["learning_rate"] * shaped_gradient.float()
        ordinary_projection.weight.copy_(best_candidate.to(dtype=ordinary_projection.weight.dtype))
        after_update_loss = float(causal_loss().detach().float().cpu())

    comparisons = {}
    if args.profile:
        profiles = {
            "decode": decode_profile,
            "ordinary_training_step": ordinary_profile,
            "inference_shaped_training_step": shaped_profile,
            "ordinary_weight_gradient": ordinary_dw_profile,
            "inference_shaped_weight_gradient": shaped_dw_profile,
            "inference_projection_replay": projection_profile,
        }
        for left, right in (
            ("decode", "ordinary_training_step"),
            ("decode", "inference_shaped_training_step"),
            ("inference_projection_replay", "ordinary_weight_gradient"),
            ("inference_projection_replay", "inference_shaped_weight_gradient"),
        ):
            comparisons[f"{left}__vs__{right}"] = compare_profiles(profiles[left], profiles[right])

    result = {
        "model": args.model,
        "target_parameter": target_name,
        "target_shape": list(ordinary_projection.weight.shape),
        "tokens": input_ids.detach().cpu().tolist(),
        "weight_gradient_rows": args.weight_gradient_rows,
        "weight_gradient_inference_rows": shaped_feature_rows,
        "reduction_k": ordinary_projection.in_features,
        "decode_checksum": float(decode_output.logits.float().sum().cpu()),
        "projection_replay_checksum": float(projection_output.float().sum().cpu()),
        "ordinary_weight_gradient_checksum": float(ordinary_dw.float().sum().cpu()),
        "shaped_weight_gradient_checksum": float(shaped_dw.float().sum().cpu()),
        "losses": {
            "ordinary": float(ordinary_loss.detach().float().cpu()),
            "shaped_warmup": float(shaped_warmup_loss.detach().float().cpu()),
            "shaped": before_update_loss,
            "after_update": after_update_loss,
        },
        "ordinary_gradient_vs_direct": ordinary_gradient_error,
        "shaped_gradient_vs_direct": shaped_gradient_error,
        "ordinary_vs_shaped_gradient": gradient_error(shaped_gradient, ordinary_gradient),
        "weight_gradient_region_equivalence": gradient_error(shaped_dw, ordinary_dw),
        "update_trials": update_trials,
        "selected_update": best_trial,
        "comparisons": comparisons,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
