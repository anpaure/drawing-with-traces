#!/usr/bin/env python3
"""Validate the full shared-carrier update against ordinary BF16 training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .shared_carrier import (
    SharedCarrierConfig,
    SharedCarrierGradientScheduler,
    carrier_execution_plans,
    replace_linear_modules_with_shared_carrier,
    transposed_carrier_parameter_ids,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--training-batch-size", type=int, default=2_048)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--row-tile", type=int, default=1_024)
    parser.add_argument(
        "--weight-gradient-schedule",
        choices=("inline", "streaming-inference-cycle"),
        default="inline",
    )
    parser.add_argument(
        "--weight-gradient-layout",
        choices=("direct", "inference-balanced", "inference-balanced-strided"),
        default="direct",
    )
    parser.add_argument("--streaming-dw-tasks-per-record", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer-bucket-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from ..llama_strict_inference_shaped_training.strict_optimizer import InterleavedSGD
    from ..llama_strict_inference_shaped_training.validate_strict import _aggregate_comparisons

    args = build_parser().parse_args()
    flattened_rows = args.training_batch_size * args.sequence_length
    config = SharedCarrierConfig(
        row_tile=args.row_tile,
        expected_training_rows=flattened_rows,
        weight_gradient_layout=args.weight_gradient_layout,
    )
    scheduler = (
        None
        if args.weight_gradient_schedule == "inline"
        else SharedCarrierGradientScheduler(
            row_tile=args.row_tile,
            tasks_per_record=args.streaming_dw_tasks_per_record,
            weight_gradient_layout=args.weight_gradient_layout,
        )
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_kwargs = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    ordinary = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().train()
    carrier = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).cuda().train()
    carrier_names = replace_linear_modules_with_shared_carrier(
        carrier,
        config,
        scheduler=scheduler,
    )
    ordinary.config.use_cache = False
    carrier.config.use_cache = False

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
        return torch.nn.functional.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]).float(),
            target_ids.reshape(-1),
        )

    ordinary.zero_grad(set_to_none=True)
    ordinary_start = torch.cuda.Event(enable_timing=True)
    ordinary_end = torch.cuda.Event(enable_timing=True)
    ordinary_start.record()
    ordinary_loss = forward_loss(ordinary)
    ordinary_loss.backward()
    ordinary_end.record()
    ordinary_end.synchronize()

    carrier_optimizer = InterleavedSGD(
        carrier,
        learning_rate=args.learning_rate,
        manual_parameter_ids=None if scheduler is None else scheduler.parameter_ids,
        manual_update_bucket_size=args.optimizer_bucket_size,
    )
    carrier_optimizer.zero_grad(set_to_none=False)
    carrier_start = torch.cuda.Event(enable_timing=True)
    carrier_end = torch.cuda.Event(enable_timing=True)
    carrier_start.record()
    if scheduler is not None:
        scheduler.begin_step(
            update_parameter=carrier_optimizer.step_manual,
            deferred_parameter_ids=carrier_optimizer.deferred_parameter_ids,
        )
    carrier_loss = forward_loss(carrier)
    carrier_loss.backward()
    scheduler_audit = None
    if scheduler is not None:
        scheduler_audit = scheduler.finish_step(
            update_parameter=carrier_optimizer.step_manual,
            deferred_parameter_ids=carrier_optimizer.deferred_parameter_ids,
        )
    carrier_optimizer.step_deferred()
    carrier_end.record()
    carrier_end.synchronize()

    ordinary_parameters = dict(ordinary.named_parameters())
    carrier_parameters = dict(carrier.named_parameters())
    if ordinary_parameters.keys() != carrier_parameters.keys():
        raise RuntimeError("ordinary and carrier parameter names differ")
    transposed_parameter_ids = transposed_carrier_parameter_ids(carrier)

    def logical_carrier_view(parameter, value):
        return value.transpose(0, 1) if id(parameter) in transposed_parameter_ids else value

    gradient_rows = []
    for name in ordinary_parameters:
        ordinary_gradient = ordinary_parameters[name].grad
        carrier_gradient = carrier_parameters[name].grad
        if ordinary_gradient is None or carrier_gradient is None:
            raise RuntimeError(f"missing gradient for {name}")
        gradient_rows.append(
            (
                name,
                logical_carrier_view(carrier_parameters[name], carrier_gradient),
                ordinary_gradient,
            )
        )
    gradient_comparison = _aggregate_comparisons(gradient_rows)

    with torch.no_grad():
        for parameter in ordinary_parameters.values():
            parameter.add_(parameter.grad, alpha=-args.learning_rate)
    update_comparison = _aggregate_comparisons(
        [
            (
                name,
                logical_carrier_view(carrier_parameters[name], carrier_parameters[name]),
                ordinary_parameters[name],
            )
            for name in ordinary_parameters
        ]
    )

    plans = carrier_execution_plans(carrier)
    result = {
        "model": args.model,
        "training_batch_size": args.training_batch_size,
        "sequence_length": args.sequence_length,
        "carrier": config.metadata(),
        "weight_gradient_schedule": args.weight_gradient_schedule,
        "weight_gradient_layout": args.weight_gradient_layout,
        "streaming_dw_tasks_per_record": (
            args.streaming_dw_tasks_per_record if scheduler is not None else None
        ),
        "parameters": sum(parameter.numel() for parameter in ordinary_parameters.values()),
        "parameter_tensors": len(ordinary_parameters),
        "carrier_linear_modules": len(carrier_names),
        "transposed_storage_parameter_tensors": len(transposed_parameter_ids),
        "useful_loss_targets": flattened_rows,
        "externally_shifted_next_token_targets": True,
        "ordinary_loss": float(ordinary_loss.detach()),
        "carrier_loss": float(carrier_loss.detach()),
        "loss_absolute_difference": abs(float(ordinary_loss.detach()) - float(carrier_loss.detach())),
        "ordinary_cuda_ms": ordinary_start.elapsed_time(ordinary_end),
        "carrier_cuda_ms": carrier_start.elapsed_time(carrier_end),
        "gradient_comparison": gradient_comparison,
        "updated_parameter_comparison": update_comparison,
        "optimizer_audit": vars(carrier_optimizer.audit()),
        "scheduler_audit": None if scheduler_audit is None else vars(scheduler_audit),
        "shape_audit": {
            "modules": len(plans),
            "forward_launches": sum(plan.forward_launches for plan in plans.values()),
            "input_gradient_launches": sum(plan.input_gradient_launches for plan in plans.values()),
            "weight_gradient_launches": sum(plan.weight_gradient_launches for plan in plans.values()),
            "layout_transform_values": sum(
                plan.layout_transform_values for plan in plans.values()
            ),
            "useful_flops": sum(plan.executed_flops for plan in plans.values()),
            "executed_flops": sum(plan.executed_flops for plan in plans.values()),
            "redundant_flops": sum(plan.redundant_flops for plan in plans.values()),
        },
        "strict_invariants": {
            "inference_cover_tokens": 0,
            "secondary_model_instances_per_process": 0,
            "filler_kernels": 0,
            "all_carrier_gemms_are_useful_training_arithmetic": True,
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "validated_at_unix_seconds": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    carrier_optimizer.close()


if __name__ == "__main__":
    main()
