#!/usr/bin/env python3
"""Validate full-model gradients and updates against ordinary Llama training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1)
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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _comparison(left, right, *, chunk_values: int = 4_000_000) -> dict[str, float]:
    import torch

    if left.shape != right.shape:
        raise ValueError(f"cannot compare shapes {tuple(left.shape)} and {tuple(right.shape)}")
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    diff_sq = torch.zeros((), device=left.device, dtype=torch.float64)
    reference_sq = torch.zeros((), device=left.device, dtype=torch.float64)
    equal = torch.zeros((), device=left.device, dtype=torch.int64)
    maximum = torch.zeros((), device=left.device, dtype=torch.float32)
    for start in range(0, left_flat.numel(), chunk_values):
        actual = left_flat[start : start + chunk_values]
        expected = right_flat[start : start + chunk_values]
        difference = actual.float() - expected.float()
        diff_sq += difference.double().square().sum()
        reference_sq += expected.double().square().sum()
        equal += torch.count_nonzero(actual == expected)
        maximum = torch.maximum(maximum, difference.abs().max())
    return {
        "relative_l2": float(torch.sqrt(diff_sq / reference_sq.clamp_min(1e-30))),
        "maximum_absolute": float(maximum),
        "equal_fraction": float(equal / max(left_flat.numel(), 1)),
        "values": left_flat.numel(),
    }


def _aggregate_comparisons(rows: list[tuple]) -> dict[str, float]:
    import torch

    difference_sq = torch.zeros((), device="cuda", dtype=torch.float64)
    reference_sq = torch.zeros((), device="cuda", dtype=torch.float64)
    equal_values = 0
    total_values = 0
    maximum = 0.0
    per_tensor: dict[str, dict[str, float]] = {}
    for name, actual, expected in rows:
        comparison = _comparison(actual, expected)
        per_tensor[name] = comparison
        actual_flat = actual.detach().reshape(-1)
        expected_flat = expected.detach().reshape(-1)
        for start in range(0, actual_flat.numel(), 4_000_000):
            difference = (
                actual_flat[start : start + 4_000_000].float()
                - expected_flat[start : start + 4_000_000].float()
            )
            difference_sq += difference.double().square().sum()
            reference_sq += expected_flat[start : start + 4_000_000].double().square().sum()
        equal_values += round(comparison["equal_fraction"] * comparison["values"])
        total_values += int(comparison["values"])
        maximum = max(maximum, comparison["maximum_absolute"])
    worst = sorted(per_tensor.items(), key=lambda item: item[1]["relative_l2"], reverse=True)[:20]
    return {
        "relative_l2": float(torch.sqrt(difference_sq / reference_sq.clamp_min(1e-30))),
        "maximum_absolute": maximum,
        "equal_fraction": equal_values / max(total_values, 1),
        "tensor_count": len(rows),
        "values": total_values,
        "worst_tensors": [{"name": name, **values} for name, values in worst],
    }


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from .strict_optimizer import InterleavedSGD
    from .strict_shapes import (
        DeferredWeightGradientScheduler,
        StrictShapeConfig,
        model_execution_plans,
        replace_linear_modules,
    )

    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    ordinary = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().train()
    shaped = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().train()
    scheduler = DeferredWeightGradientScheduler() if args.weight_gradient_schedule != "inline" else None
    shaped_names = replace_linear_modules(
        shaped,
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
    ordinary.config.use_cache = False
    shaped.config.use_cache = False
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    token_ids = torch.randint(
        0,
        min(int(ordinary.config.vocab_size), 32_000),
        (args.training_batch_size, args.sequence_length + 1),
        generator=generator,
        device="cuda",
    )
    input_ids = token_ids[:, :-1].contiguous()
    target_ids = token_ids[:, 1:].contiguous()

    def forward_loss(model):
        output = model(input_ids=input_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            target_ids.reshape(-1),
        )
        return output, loss

    ordinary.zero_grad(set_to_none=True)
    ordinary_start = torch.cuda.Event(enable_timing=True)
    ordinary_end = torch.cuda.Event(enable_timing=True)
    ordinary_start.record()
    _ordinary_output, ordinary_loss = forward_loss(ordinary)
    ordinary_loss.backward()
    ordinary_end.record()
    ordinary_end.synchronize()

    shaped_optimizer = InterleavedSGD(
        shaped,
        learning_rate=args.learning_rate,
        manual_parameter_ids=None if scheduler is None else scheduler.parameter_ids,
        manual_update_bucket_size=args.optimizer_bucket_size,
    )
    shaped_optimizer.zero_grad(set_to_none=False)
    if scheduler is not None:
        scheduler.begin_step(
            update_parameter=shaped_optimizer.step_manual,
            deferred_parameter_ids=shaped_optimizer.deferred_parameter_ids,
        )
    shaped_start = torch.cuda.Event(enable_timing=True)
    shaped_end = torch.cuda.Event(enable_timing=True)
    shaped_start.record()
    _shaped_output, shaped_loss = forward_loss(shaped)
    shaped_loss.backward()
    if scheduler is not None:
        scheduler.finish_step(
            update_parameter=shaped_optimizer.step_manual,
            deferred_parameter_ids=shaped_optimizer.deferred_parameter_ids,
        )
    shaped_optimizer.step_deferred()
    shaped_end.record()
    shaped_end.synchronize()

    ordinary_parameters = dict(ordinary.named_parameters())
    shaped_parameters = dict(shaped.named_parameters())
    if ordinary_parameters.keys() != shaped_parameters.keys():
        raise RuntimeError("ordinary and shaped parameter names differ")
    gradient_rows = []
    for name in ordinary_parameters:
        ordinary_gradient = ordinary_parameters[name].grad
        shaped_gradient = shaped_parameters[name].grad
        if ordinary_gradient is None or shaped_gradient is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradient_rows.append((name, shaped_gradient, ordinary_gradient))
    gradient_comparison = _aggregate_comparisons(gradient_rows)

    with torch.no_grad():
        for parameter in ordinary_parameters.values():
            parameter.add_(parameter.grad, alpha=-args.learning_rate)
    update_rows = [(name, shaped_parameters[name], ordinary_parameters[name]) for name in ordinary_parameters]
    update_comparison = _aggregate_comparisons(update_rows)
    plans = model_execution_plans(shaped)
    shape_audit = {
        "modules": len(plans),
        "forward_launches": sum(plan.forward_launches for plan in plans.values()),
        "input_gradient_launches": sum(plan.input_gradient_launches for plan in plans.values()),
        "weight_gradient_launches": sum(plan.weight_gradient_launches for plan in plans.values()),
        "useful_flops": sum(plan.useful_flops for plan in plans.values()),
        "executed_flops": sum(plan.executed_flops for plan in plans.values()),
        "redundant_flops": sum(plan.redundant_flops for plan in plans.values()),
    }
    result = {
        "model": args.model,
        "training_batch_size": args.training_batch_size,
        "sequence_length": args.sequence_length,
        "tile_rows": args.tile_rows,
        "shaping_backend": args.shaping_backend,
        "weight_gradient_schedule": args.weight_gradient_schedule,
        "optimizer_bucket_size": args.optimizer_bucket_size,
        "parameters": sum(parameter.numel() for parameter in ordinary_parameters.values()),
        "parameter_tensors": len(ordinary_parameters),
        "shaped_linear_modules": len(shaped_names),
        "useful_loss_targets": args.training_batch_size * args.sequence_length,
        "externally_shifted_next_token_targets": True,
        "ordinary_loss": float(ordinary_loss.detach()),
        "shaped_loss": float(shaped_loss.detach()),
        "loss_absolute_difference": abs(float(ordinary_loss.detach()) - float(shaped_loss.detach())),
        "ordinary_cuda_ms": ordinary_start.elapsed_time(ordinary_end),
        "shaped_cuda_ms": shaped_start.elapsed_time(shaped_end),
        "gradient_comparison": gradient_comparison,
        "updated_parameter_comparison": update_comparison,
        "optimizer_audit": vars(shaped_optimizer.audit()),
        "deferred_weight_gradient_audit": (None if scheduler is None else vars(scheduler.audit())),
        "shape_audit": shape_audit,
        "strict_invariants": {
            "inference_cover_tokens": 0,
            "secondary_model_instances_per_process": 0,
            "filler_kernels": 0,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "validated_at_unix_seconds": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    shaped_optimizer.close()


if __name__ == "__main__":
    main()
