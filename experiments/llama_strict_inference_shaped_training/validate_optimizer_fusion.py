#!/usr/bin/env python3
"""Prove fused manual SGD equals scalar updates in strict shaped training.

Both sides execute the same strict forward, dX, and deferred exact dW schedule.
The only changed variable is how many completed manual parameter updates are
submitted together through ``torch._foreach_add_``.
"""

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
    parser.add_argument("--tile-rows", type=int, default=64)
    parser.add_argument(
        "--shaping-backend",
        choices=("grouped-m1", "tiled-gemm"),
        default="grouped-m1",
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
        default="inline",
    )
    parser.add_argument("--streaming-dw-tasks-per-record", type=int, default=32)
    parser.add_argument("--grouped-dw-min-batch", type=int, default=4)
    parser.add_argument("--grouped-dw-max-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reference-bucket-size", type=int, default=1)
    parser.add_argument("--fused-bucket-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from .strict_optimizer import InterleavedSGD
    from .strict_shapes import DeferredWeightGradientScheduler, StrictShapeConfig, replace_linear_modules
    from .validate_strict import _aggregate_comparisons

    args = build_parser().parse_args()
    for name in ("reference_bucket_size", "fused_bucket_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    def load_shaped():
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().train()
        model.config.use_cache = False
        scheduler = None if args.weight_gradient_schedule == "inline" else DeferredWeightGradientScheduler()
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
        return model, scheduler, shaped_names

    reference, reference_scheduler, reference_names = load_shaped()
    fused, fused_scheduler, fused_names = load_shaped()
    if reference_names != fused_names:
        raise RuntimeError("strict model replacements differ between validation sides")

    reference_parameters = dict(reference.named_parameters())
    fused_parameters = dict(fused.named_parameters())
    if reference_parameters.keys() != fused_parameters.keys():
        raise RuntimeError("parameter names differ between validation sides")
    initial_comparison = _aggregate_comparisons(
        [(name, fused_parameters[name], parameter) for name, parameter in reference_parameters.items()]
    )
    if initial_comparison["equal_fraction"] != 1.0:
        raise RuntimeError("validation models did not start from bitwise-identical parameters")

    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    token_ids = torch.randint(
        0,
        min(int(reference.config.vocab_size), 32_000),
        (args.training_batch_size, args.sequence_length + 1),
        generator=generator,
        device="cuda",
    )
    input_ids = token_ids[:, :-1].contiguous()
    target_ids = token_ids[:, 1:].contiguous()

    def execute_step(model, scheduler, *, bucket_size: int):
        optimizer = InterleavedSGD(
            model,
            learning_rate=args.learning_rate,
            manual_parameter_ids=None if scheduler is None else scheduler.parameter_ids,
            manual_update_bucket_size=bucket_size,
        )
        optimizer.zero_grad(set_to_none=False)
        if scheduler is not None:
            scheduler.begin_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        output = model(input_ids=input_ids, use_cache=False)
        loss = torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            target_ids.reshape(-1),
        )
        loss.backward()
        if scheduler is not None:
            scheduler.finish_step(
                update_parameter=optimizer.step_manual,
                deferred_parameter_ids=optimizer.deferred_parameter_ids,
            )
        optimizer.step_deferred()
        finished.record()
        finished.synchronize()
        result = {
            "loss": float(loss.detach()),
            "cuda_ms": started.elapsed_time(finished),
            "optimizer_audit": vars(optimizer.audit()),
            "scheduler_audit": None if scheduler is None else vars(scheduler.audit()),
        }
        optimizer.close()
        del output, loss
        return result

    # Reset RNG before each side so future stochastic model components stay controlled.
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 2)
    reference_result = execute_step(
        reference,
        reference_scheduler,
        bucket_size=args.reference_bucket_size,
    )
    torch.manual_seed(args.seed + 2)
    torch.cuda.manual_seed_all(args.seed + 2)
    fused_result = execute_step(fused, fused_scheduler, bucket_size=args.fused_bucket_size)

    gradient_rows = []
    update_rows = []
    for name, reference_parameter in reference_parameters.items():
        fused_parameter = fused_parameters[name]
        if reference_parameter.grad is None or fused_parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradient_rows.append((name, fused_parameter.grad, reference_parameter.grad))
        update_rows.append((name, fused_parameter, reference_parameter))
    gradient_comparison = _aggregate_comparisons(gradient_rows)
    update_comparison = _aggregate_comparisons(update_rows)
    loss_absolute_difference = abs(reference_result["loss"] - fused_result["loss"])
    bitwise_equivalent = (
        loss_absolute_difference == 0.0
        and gradient_comparison["equal_fraction"] == 1.0
        and update_comparison["equal_fraction"] == 1.0
    )

    result = {
        "experiment": "strict_shaped_optimizer_fusion_equivalence",
        "model": args.model,
        "training_batch_size": args.training_batch_size,
        "sequence_length": args.sequence_length,
        "tile_rows": args.tile_rows,
        "shaping_backend": args.shaping_backend,
        "weight_gradient_schedule": args.weight_gradient_schedule,
        "learning_rate": args.learning_rate,
        "reference_bucket_size": args.reference_bucket_size,
        "fused_bucket_size": args.fused_bucket_size,
        "parameters": sum(parameter.numel() for parameter in reference_parameters.values()),
        "parameter_tensors": len(reference_parameters),
        "shaped_linear_modules": len(reference_names),
        "initial_parameter_comparison": initial_comparison,
        "reference": reference_result,
        "fused": fused_result,
        "loss_absolute_difference": loss_absolute_difference,
        "gradient_comparison": gradient_comparison,
        "updated_parameter_comparison": update_comparison,
        "bitwise_equivalent": bitwise_equivalent,
        "strict_invariants": {
            "inference_cover_tokens": 0,
            "secondary_model_instances_per_training_process": 0,
            "filler_kernels": 0,
            "changed_variable": "manual_optimizer_update_bucket_size_only",
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "validated_at_unix_seconds": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if not bitwise_equivalent:
        raise SystemExit("fused optimizer did not match scalar-update reference bitwise")


if __name__ == "__main__":
    main()
