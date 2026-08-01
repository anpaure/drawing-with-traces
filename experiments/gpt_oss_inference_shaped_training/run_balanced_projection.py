#!/usr/bin/env python3
"""Execute a 1:2 forward/backward stream using real GPT-OSS causal gradients.

For layer-2 ``o_proj`` the input width is 4096.  Accumulating exactly 4096
token rows makes all three useful matrix-multiplication streams equal length:

* forward: 4096 calls with M=1, K=4096, N=2880;
* backward-data: 4096 calls with M=1, K=2880, N=4096;
* backward-weight: 4096 calls with M=1, K=4096, N=2880.

The weight-gradient reduction contains no zero padding and no redundant FLOPs.
Activations and upstream gradients come from the real GPT-OSS causal-LM loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from continuous_similarity import continuous_kernel_similarity
from inference_shaped_linear import InferenceShapeConfig, InferenceShapedLinear
from run_experiment import DEFAULT_MODEL, gradient_error, profile_call


DEFAULT_CORPUS = """
Side-channel measurements reveal how hardware executes machine-learning workloads.
Inference predicts the next token while training propagates a causal loss backward.
This experiment schedules mathematically exact gradients using inference-shaped operations.
Mixture-of-experts transformers route each token through a small subset of local experts.
"""


class CausalGradientBoundary(nn.Module):
    """Detach one real linear boundary while retaining its causal output gradient."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear
        self.input: Tensor | None = None
        self.output: Tensor | None = None

    def forward(self, value: Tensor) -> Tensor:
        self.input = value.detach().reshape(-1, value.shape[-1]).clone()
        with torch.no_grad():
            projected = F.linear(value, self.linear.weight, self.linear.bias)
        self.output = projected.detach().requires_grad_(True)
        return self.output

    def take(self) -> tuple[Tensor, Tensor]:
        if self.input is None or self.output is None or self.output.grad is None:
            raise RuntimeError("The causal boundary did not receive an output gradient")
        return self.input, self.output.grad.detach().reshape(-1, self.output.shape[-1]).clone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--total-rows", type=int, default=4096)
    parser.add_argument("--microbatch-rows", type=int, default=256)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-chrome-traces", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def token_pairs(tokenizer, corpus: str, count: int) -> tuple[Tensor, Tensor]:
    token_ids = tokenizer(corpus, add_special_tokens=False)["input_ids"]
    if len(token_ids) < 2:
        raise ValueError("The corpus must tokenize to at least two tokens")
    inputs = [token_ids[index % (len(token_ids) - 1)] for index in range(count)]
    targets = [token_ids[index % (len(token_ids) - 1) + 1] for index in range(count)]
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def main() -> None:
    args = parse_args()
    if args.total_rows < 1 or args.microbatch_rows < 1:
        raise ValueError("total-rows and microbatch-rows must both be positive")
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
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parent = model.model.layers[args.layer].self_attn
    projection = parent.o_proj
    if not isinstance(projection, nn.Linear):
        raise TypeError(f"Expected o_proj to be nn.Linear, got {type(projection).__name__}")
    if args.total_rows != projection.in_features:
        raise ValueError(
            "The balanced no-padding experiment requires total-rows == o_proj.in_features; "
            f"got total_rows={args.total_rows}, in_features={projection.in_features}"
        )

    source_ids, target_ids = token_pairs(tokenizer, args.corpus, args.total_rows)
    boundary = CausalGradientBoundary(projection)
    parent.o_proj = boundary
    activation_chunks = []
    gradient_chunks = []
    loss_sum = 0.0

    print(
        f"Accumulating {args.total_rows} real causal rows in microbatches of {args.microbatch_rows}",
        flush=True,
    )
    for start in range(0, args.total_rows, args.microbatch_rows):
        end = min(start + args.microbatch_rows, args.total_rows)
        input_ids = source_ids[start:end, None].to("cuda")
        targets = target_ids[start:end].to("cuda")
        output = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        unscaled_loss = F.cross_entropy(output.logits[:, -1, :].float(), targets, reduction="sum")
        (unscaled_loss / args.total_rows).backward()
        activations, upstream_gradient = boundary.take()
        activation_chunks.append(activations)
        gradient_chunks.append(upstream_gradient)
        loss_sum += float(unscaled_loss.detach().cpu())
        print(f"  rows {end}/{args.total_rows}", flush=True)

    activations = torch.cat(activation_chunks, dim=0)
    upstream_gradient = torch.cat(gradient_chunks, dim=0)
    causal_loss_before = loss_sum / args.total_rows
    parent.o_proj = projection
    projection.weight.requires_grad_(True)
    if projection.bias is not None:
        projection.bias.requires_grad_(False)

    print(
        f"Captured x={tuple(activations.shape)}, dY={tuple(upstream_gradient.shape)}, "
        f"causal_loss={causal_loss_before:.6f}",
        flush=True,
    )

    # Ordinary local projection training computes the same vector-Jacobian
    # product as one large batch GEMM.
    ordinary_x = activations.detach().clone().requires_grad_(True)
    projection.weight.grad = None

    def ordinary_projection_training() -> Tensor:
        projection.weight.grad = None
        ordinary_x.grad = None
        output = F.linear(ordinary_x, projection.weight, projection.bias)
        torch.autograd.backward(output, upstream_gradient)
        return output

    ordinary_output, ordinary_profile = profile_call(
        args.output_dir,
        "ordinary_projection_training",
        ordinary_projection_training,
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )
    ordinary_weight_gradient = projection.weight.grad.detach().clone()
    ordinary_input_gradient = ordinary_x.grad.detach().clone()

    # The exact 1:2 decomposition.  Since token rows equal reduction_k, the dW
    # path performs no padding and exactly the useful FLOPs of the ordinary GEMM.
    shaped = InferenceShapedLinear(
        projection,
        InferenceShapeConfig(
            forward_rows=1,
            input_gradient_rows=1,
            weight_gradient_rows=1,
            weight_gradient_inference_rows=projection.in_features,
            reduction_k=projection.in_features,
            bias_epilogue=True,
        ),
    )
    shaped_x = activations.detach().clone().requires_grad_(True)
    projection.weight.grad = None

    def shaped_projection_training() -> Tensor:
        projection.weight.grad = None
        shaped_x.grad = None
        output = shaped(shaped_x)
        torch.autograd.backward(output, upstream_gradient)
        return output

    shaped_output, shaped_profile = profile_call(
        args.output_dir,
        "balanced_inference_shaped_training",
        shaped_projection_training,
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )
    shaped_weight_gradient = projection.weight.grad.detach().clone()
    shaped_input_gradient = shaped_x.grad.detach().clone()

    # One indefinitely repeated inference projection stream.  Its iteration
    # count need not align with training; continuous metrics normalize counts.
    def inference_projection_stream() -> Tensor:
        return torch.cat(
            [
                F.linear(activations[row : row + 1], projection.weight, projection.bias)
                for row in range(args.total_rows)
            ],
            dim=0,
        )

    inference_output, inference_profile = profile_call(
        args.output_dir,
        "inference_projection_stream",
        inference_projection_stream,
        enabled=args.profile,
        save_chrome_trace=args.save_chrome_traces,
    )

    # Apply the real accumulated causal gradient using an FP32 master weight.
    original_weight = projection.weight.detach().clone()
    master_weight = original_weight.float()
    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    def evaluate_causal_loss() -> float:
        total = 0.0
        with torch.no_grad():
            for start in range(0, args.total_rows, args.microbatch_rows):
                end = min(start + args.microbatch_rows, args.total_rows)
                logits = model(
                    input_ids=source_ids[start:end, None].to("cuda"),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                total += float(
                    F.cross_entropy(
                        logits.float(), target_ids[start:end].to("cuda"), reduction="sum"
                    ).cpu()
                )
        return total / args.total_rows

    update_trials = []
    with torch.no_grad():
        for learning_rate in learning_rates:
            candidate = master_weight - learning_rate * shaped_weight_gradient.float()
            projection.weight.copy_(candidate.to(dtype=projection.weight.dtype))
            loss = evaluate_causal_loss()
            changed = projection.weight != original_weight
            update_trials.append(
                {
                    "learning_rate": learning_rate,
                    "causal_loss": loss,
                    "changed_elements": int(changed.count_nonzero().cpu()),
                    "max_parameter_delta": float(
                        (projection.weight.float() - original_weight.float()).abs().max().cpu()
                    ),
                }
            )
        selected_update = min(update_trials, key=lambda row: row["causal_loss"])
        best_weight = master_weight - selected_update["learning_rate"] * shaped_weight_gradient.float()
        projection.weight.copy_(best_weight.to(dtype=projection.weight.dtype))
    causal_loss_after = evaluate_causal_loss()

    comparisons = {}
    if args.profile:
        comparisons = {
            "inference_vs_ordinary": continuous_kernel_similarity(
                inference_profile["cuda_kernel_sequence"],
                ordinary_profile["cuda_kernel_sequence"],
            ),
            "inference_vs_balanced_shaped": continuous_kernel_similarity(
                inference_profile["cuda_kernel_sequence"],
                shaped_profile["cuda_kernel_sequence"],
            ),
        }

    useful_flops_per_stream = 2 * args.total_rows * projection.in_features * projection.out_features
    result = {
        "model": args.model,
        "target_parameter": f"model.layers.{args.layer}.self_attn.o_proj.weight",
        "target_shape": list(projection.weight.shape),
        "total_rows": args.total_rows,
        "microbatch_rows": args.microbatch_rows,
        "stream_gemm_shapes": {
            "forward": [args.total_rows, 1, projection.in_features, projection.out_features],
            "backward_data": [args.total_rows, 1, projection.out_features, projection.in_features],
            "backward_weight": [projection.in_features, 1, args.total_rows, projection.out_features],
        },
        "useful_flops": {
            "forward": useful_flops_per_stream,
            "backward_data": useful_flops_per_stream,
            "backward_weight": useful_flops_per_stream,
            "backward_to_forward_ratio": 2.0,
            "redundant_weight_gradient_flops": 0,
        },
        "causal_loss_before": causal_loss_before,
        "causal_loss_after": causal_loss_after,
        "selected_update": selected_update,
        "update_trials": update_trials,
        "forward_equivalence": gradient_error(shaped_output, ordinary_output),
        "inference_forward_equivalence": gradient_error(inference_output, ordinary_output),
        "weight_gradient_equivalence": gradient_error(
            shaped_weight_gradient, ordinary_weight_gradient
        ),
        "input_gradient_equivalence": gradient_error(shaped_input_gradient, ordinary_input_gradient),
        "comparisons": comparisons,
        "profiles": {
            profile["phase"]: {
                key: profile[key]
                for key in (
                    "wall_ms",
                    "actual_cuda_kernel_count",
                    "actual_cuda_busy_ms",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                )
            }
            for profile in (ordinary_profile, shaped_profile, inference_profile)
            if args.profile
        },
        "checksums": {
            "activations": float(activations.float().sum().cpu()),
            "upstream_gradient": float(upstream_gradient.float().sum().cpu()),
            "ordinary_output": float(ordinary_output.float().sum().cpu()),
            "shaped_output": float(shaped_output.float().sum().cpu()),
            "ordinary_weight_gradient": float(ordinary_weight_gradient.float().sum().cpu()),
            "shaped_weight_gradient": float(shaped_weight_gradient.float().sum().cpu()),
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
