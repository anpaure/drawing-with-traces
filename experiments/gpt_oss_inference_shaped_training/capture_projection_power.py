#!/usr/bin/env python3
"""Capture equal-duration power windows from inference and shaped training.

This is the physical counterpart of ``run_balanced_projection.py``.  It uses
SideCapture and the validated Husky burst path.  Training captures rotate the
first visible phase so triggering at a Python iteration boundary cannot bias
all windows toward the forward phase.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import sidecapture as sc

from inference_shaped_linear import inference_shaped_weight_gradient
from run_balanced_projection import (
    DEFAULT_CORPUS,
    CausalGradientBoundary,
    token_pairs,
)
from run_experiment import DEFAULT_MODEL


PHASES = ("forward", "backward_data", "backward_weight")


def row_linear(value: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    return torch.cat(
        [F.linear(value[row : row + 1], weight, bias) for row in range(value.shape[0])],
        dim=0,
    )


class ProjectionInferenceWorkload(sc.Workload):
    replay_safe = True

    def __init__(self, activations: Tensor, projection: nn.Linear) -> None:
        self.activations = activations
        self.projection = projection

    def run(self, context: sc.CaptureContext):
        context.labels.update(
            process="inference",
            model=DEFAULT_MODEL,
            parameter="model.layers.2.self_attn.o_proj.weight",
            token_rows=int(self.activations.shape[0]),
            gemm_m=1,
            gemm_k=int(self.projection.in_features),
            gemm_n=int(self.projection.out_features),
        )
        # A capture is intentionally a partial window into a longer continuous
        # process, so an end-bounded annotation would correctly fall outside the
        # trace and be rejected.  Keep only the in-window start marker.
        context.cuda_mark("inference.window_start", sync="none")
        output = row_linear(
            self.activations,
            self.projection.weight,
            self.projection.bias,
        )
        return {"output_checksum": float(output.float().sum().detach().cpu())}

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "real_gpt_oss_projection_inference",
            "model": DEFAULT_MODEL,
            "target": "model.layers.2.self_attn.o_proj",
            "rows": int(self.activations.shape[0]),
            "note": "Actual cached GPT-OSS weight and real layer-2 activations, executed one token row at a time.",
        }


class BalancedProjectionTrainingWorkload(sc.Workload):
    replay_safe = False

    def __init__(
        self,
        activations: Tensor,
        upstream_gradient: Tensor,
        projection: nn.Linear,
        *,
        learning_rate: float,
        row_interleaved: bool = False,
        phase_block_rows: int | None = None,
    ) -> None:
        self.activations = activations
        self.upstream_gradient = upstream_gradient
        self.projection = projection
        self.learning_rate = float(learning_rate)
        self.phase_block_rows = 1 if row_interleaved else phase_block_rows
        if self.phase_block_rows is not None and not 1 <= self.phase_block_rows <= activations.shape[0]:
            raise ValueError(
                f"phase_block_rows must be between 1 and {activations.shape[0]}, got {self.phase_block_rows}"
            )
        self.master_weight = projection.weight.detach().float().clone()
        self.accepted_steps = 0

    def snapshot(self):
        snapshot = {
            "master_weight": self.master_weight.clone(),
            "model_weight": self.projection.weight.detach().clone(),
            "accepted_steps": self.accepted_steps,
        }
        torch.cuda.synchronize()
        return snapshot

    def restore(self, snapshot) -> None:
        self.master_weight.copy_(snapshot["master_weight"])
        self.projection.weight.data.copy_(snapshot["model_weight"])
        self.accepted_steps = int(snapshot["accepted_steps"])
        torch.cuda.synchronize()

    def run(self, context: sc.CaptureContext):
        if self.phase_block_rows is not None:
            return self.run_block_interleaved(context)
        return self.run_with_phase(context, context.index % len(PHASES))

    def run_with_phase(self, context: sc.CaptureContext, phase_offset: int):
        if self.phase_block_rows is not None:
            return self.run_block_interleaved(context)
        phase_order = PHASES[phase_offset:] + PHASES[:phase_offset]
        context.labels.update(
            process="training",
            model=DEFAULT_MODEL,
            parameter="model.layers.2.self_attn.o_proj.weight",
            phase_first=phase_order[0],
            phase_order=list(phase_order),
            token_rows=int(self.activations.shape[0]),
            backward_to_forward_flops=2.0,
            redundant_weight_gradient_flops=0,
            learning_rate=self.learning_rate,
            accepted_steps_before=self.accepted_steps,
        )

        results: dict[str, Tensor] = {}
        context.cuda_mark(f"training.window_start.{phase_order[0]}", sync="none")
        for phase in phase_order:
            if phase == "forward":
                results[phase] = row_linear(
                    self.activations,
                    self.projection.weight,
                    self.projection.bias,
                )
            elif phase == "backward_data":
                transposed_weight = self.projection.weight.transpose(0, 1).contiguous()
                zero_bias = self.activations.new_zeros(self.projection.in_features)
                results[phase] = row_linear(
                    self.upstream_gradient,
                    transposed_weight,
                    zero_bias,
                )
            elif phase == "backward_weight":
                results[phase] = inference_shaped_weight_gradient(
                    self.activations,
                    self.upstream_gradient,
                    reduction_k=self.projection.in_features,
                    rows=1,
                    inference_rows=self.projection.in_features,
                    bias_epilogue=True,
                )
            else:  # pragma: no cover - PHASES is fixed above.
                raise RuntimeError(f"Unknown phase {phase}")

        weight_gradient = results["backward_weight"]
        self.master_weight.add_(weight_gradient.float(), alpha=-self.learning_rate)
        self.projection.weight.data.copy_(self.master_weight.to(self.projection.weight.dtype))
        return {
            "forward_checksum": float(results["forward"].float().sum().detach().cpu()),
            "input_gradient_checksum": float(
                results["backward_data"].float().sum().detach().cpu()
            ),
            "weight_gradient_checksum": float(weight_gradient.float().sum().detach().cpu()),
        }

    def run_block_interleaved(self, context: sc.CaptureContext):
        """Pipeline O-forward, Q-shaped dX, and O-shaped dW in coarse row blocks."""

        block_rows = int(self.phase_block_rows or self.activations.shape[0])

        context.labels.update(
            process="training",
            model=DEFAULT_MODEL,
            parameter="model.layers.2.self_attn.o_proj.weight",
            phase_first=f"block_interleaved_{block_rows}",
            phase_order=["forward", "backward_data", "backward_weight"],
            schedule="block_interleaved_O_Q_O",
            phase_block_rows=block_rows,
            token_rows=int(self.activations.shape[0]),
            backward_to_forward_flops=2.0,
            redundant_weight_gradient_flops=0,
            learning_rate=self.learning_rate,
            accepted_steps_before=self.accepted_steps,
        )
        context.cuda_mark(f"training.window_start.block_interleaved_{block_rows}", sync="none")

        transposed_weight = self.projection.weight.transpose(0, 1).contiguous()
        activation_columns = self.activations.transpose(0, 1).contiguous()
        gradient_as_weight = self.upstream_gradient.transpose(0, 1).contiguous()
        zero_input_bias = self.activations.new_zeros(self.projection.in_features)
        zero_output_bias = self.activations.new_zeros(self.projection.out_features)
        forward_rows = []
        input_gradient_rows = []
        weight_gradient_rows = []
        for block_start in range(0, self.activations.shape[0], block_rows):
            block_end = min(block_start + block_rows, self.activations.shape[0])
            for row in range(block_start, block_end):
                forward_rows.append(
                    F.linear(
                        self.activations[row : row + 1],
                        self.projection.weight,
                        self.projection.bias,
                    )
                )
            for row in range(block_start, block_end):
                input_gradient_rows.append(
                    F.linear(
                        self.upstream_gradient[row : row + 1],
                        transposed_weight,
                        zero_input_bias,
                    )
                )
            for row in range(block_start, block_end):
                weight_gradient_rows.append(
                    F.linear(
                        activation_columns[row : row + 1],
                        gradient_as_weight,
                        zero_output_bias,
                    )
                )

        forward = torch.cat(forward_rows, dim=0)
        input_gradient = torch.cat(input_gradient_rows, dim=0)
        weight_gradient = torch.cat(weight_gradient_rows, dim=0).transpose(0, 1).contiguous()
        self.master_weight.add_(weight_gradient.float(), alpha=-self.learning_rate)
        self.projection.weight.data.copy_(self.master_weight.to(self.projection.weight.dtype))
        return {
            "forward_checksum": float(forward.float().sum().detach().cpu()),
            "input_gradient_checksum": float(input_gradient.float().sum().detach().cpu()),
            "weight_gradient_checksum": float(weight_gradient.float().sum().detach().cpu()),
        }

    def on_accept(self, _result: Any) -> None:
        self.accepted_steps += 1

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "real_gpt_oss_balanced_projection_training",
            "model": DEFAULT_MODEL,
            "target": "model.layers.2.self_attn.o_proj",
            "rows": int(self.activations.shape[0]),
            "learning_rate": self.learning_rate,
            "forward_backward_stream_ratio": "1:2",
            "redundant_weight_gradient_flops": 0,
            "phase_sampling": "capture index rotates the first phase across forward, dX, and dW",
            "phase_block_rows": self.phase_block_rows,
            "note": (
                "Actual GPT-OSS weight, activations, and upstream causal-loss gradient. "
                "Every accepted capture executes forward, dX, dW, and a BF16/FP32-master SGD update."
            ),
        }


class InterleavedProjectionWorkload(sc.Workload):
    """Alternate inference and training under one identical capture lifecycle."""

    replay_safe = False

    def __init__(
        self,
        inference: ProjectionInferenceWorkload,
        training: BalancedProjectionTrainingWorkload,
    ) -> None:
        self.inference = inference
        self.training = training

    def snapshot(self):
        # This path runs before every capture, including inference, removing the
        # GPU-memory-copy warm-state confound from the first experiment.
        return self.training.snapshot()

    def restore(self, snapshot) -> None:
        self.training.restore(snapshot)

    def run(self, context: sc.CaptureContext):
        context.labels["capture_schedule"] = "strictly_interleaved"
        if context.index % 2 == 0:
            return {"process": "inference", "result": self.inference.run(context)}
        training_ordinal = context.index // 2
        return {
            "process": "training",
            "result": self.training.run_with_phase(context, training_ordinal % len(PHASES)),
        }

    def on_accept(self, result: dict[str, Any]) -> None:
        if result["process"] == "training":
            self.training.on_accept(result["result"])

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "interleaved_inference_and_balanced_training",
            "schedule": "inference, train-forward, inference, train-dX, inference, train-dW, repeat",
            "inference": self.inference.metadata(),
            "training": self.training.metadata(),
            "common_pre_arm_snapshot": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--total-rows", type=int, default=4096)
    parser.add_argument("--microbatch-rows", type=int, default=256)
    parser.add_argument("--traces-per-process", type=int, default=9)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--row-interleaved-training", action="store_true")
    parser.add_argument("--phase-block-rows", type=int)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def collect_real_causal_inputs(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parent = model.model.layers[args.layer].self_attn
    projection = parent.o_proj
    if not isinstance(projection, nn.Linear):
        raise TypeError(f"Expected o_proj to be nn.Linear, got {type(projection).__name__}")
    if args.total_rows != projection.in_features:
        raise ValueError(
            f"total-rows must equal o_proj.in_features={projection.in_features}, got {args.total_rows}"
        )

    source_ids, target_ids = token_pairs(tokenizer, args.corpus, args.total_rows)
    boundary = CausalGradientBoundary(projection)
    parent.o_proj = boundary
    activation_chunks = []
    gradient_chunks = []
    loss_sum = 0.0
    for start in range(0, args.total_rows, args.microbatch_rows):
        end = min(start + args.microbatch_rows, args.total_rows)
        output = model(
            input_ids=source_ids[start:end, None].to("cuda"),
            use_cache=False,
            logits_to_keep=1,
        )
        unscaled_loss = F.cross_entropy(
            output.logits[:, -1, :].float(),
            target_ids[start:end].to("cuda"),
            reduction="sum",
        )
        (unscaled_loss / args.total_rows).backward()
        activations, upstream_gradient = boundary.take()
        activation_chunks.append(activations)
        gradient_chunks.append(upstream_gradient)
        loss_sum += float(unscaled_loss.detach().cpu())
        print(f"[causal] rows={end}/{args.total_rows}", flush=True)

    parent.o_proj = projection
    activations = torch.cat(activation_chunks, dim=0)
    upstream_gradient = torch.cat(gradient_chunks, dim=0)
    metadata = {
        "causal_loss": loss_sum / args.total_rows,
        "activation_shape": list(activations.shape),
        "upstream_gradient_shape": list(upstream_gradient.shape),
        "activation_checksum": float(activations.float().sum().cpu()),
        "upstream_gradient_checksum": float(upstream_gradient.float().sum().cpu()),
    }

    # Retain only the actual projection and captured tensors during scope I/O.
    del boundary, parent, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return activations, upstream_gradient, projection, metadata


def sampler(args: argparse.Namespace) -> sc.ChipWhispererSampler:
    return sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.duration,
            sample_rate=args.sample_rate,
            mode="burst",
            bits_per_sample=12,
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )


def warmup_operators(activations: Tensor, upstream_gradient: Tensor, projection: nn.Linear) -> None:
    F.linear(activations[:1], projection.weight, projection.bias)
    F.linear(
        upstream_gradient[:1],
        projection.weight.transpose(0, 1).contiguous(),
        activations.new_zeros(projection.in_features),
    )
    inference_shaped_weight_gradient(
        activations[:1],
        upstream_gradient[:1],
        reduction_k=projection.in_features,
        rows=projection.in_features,
        inference_rows=projection.in_features,
    )
    torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.input_cache or (args.output_dir / "causal_projection_inputs.pt")
    if cache_path.exists():
        print(f"[setup] loading cached causal inputs <- {cache_path}", flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        weight = payload["weight"]
        bias = payload["bias"]
        projection = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None).to(
            device="cuda", dtype=weight.dtype
        )
        with torch.no_grad():
            projection.weight.copy_(weight.to("cuda"))
            if bias is not None:
                projection.bias.copy_(bias.to("cuda"))
        activations = payload["activations"].to("cuda")
        upstream_gradient = payload["upstream_gradient"].to("cuda")
        causal_metadata = payload["metadata"]
    else:
        print("[setup] collecting real GPT-OSS causal boundary data", flush=True)
        activations, upstream_gradient, projection, causal_metadata = collect_real_causal_inputs(args)
        torch.save(
            {
                "activations": activations.detach().cpu(),
                "upstream_gradient": upstream_gradient.detach().cpu(),
                "weight": projection.weight.detach().cpu(),
                "bias": None if projection.bias is None else projection.bias.detach().cpu(),
                "metadata": causal_metadata,
            },
            cache_path,
        )
        print(f"[setup] cached causal inputs -> {cache_path}", flush=True)
    warmup_operators(activations, upstream_gradient, projection)

    inference_workload = ProjectionInferenceWorkload(activations, projection)
    training_workload = BalancedProjectionTrainingWorkload(
        activations,
        upstream_gradient,
        projection,
        learning_rate=args.learning_rate,
        row_interleaved=args.row_interleaved_training,
        phase_block_rows=args.phase_block_rows,
    )
    if args.interleaved:
        mixed_root = args.output_dir / "mixed"
        mixed_workload = InterleavedProjectionWorkload(inference_workload, training_workload)
        print(f"[capture] strictly interleaved -> {mixed_root}", flush=True)
        with sc.Experiment(
            sampler=sampler(args),
            workload=mixed_workload,
            store=sc.DirectoryStore(mixed_root, trace_dtype="float32"),
            retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
            workload_sync="cuda",
        ) as experiment:
            mixed_records = experiment.run(2 * args.traces_per_process)
        inference_records = [
            record for record in mixed_records if record["labels"]["process"] == "inference"
        ]
        training_records = [
            record for record in mixed_records if record["labels"]["process"] == "training"
        ]
        record_roots = {"inference": mixed_root, "training": mixed_root}
    else:
        inference_root = args.output_dir / "inference"
        training_root = args.output_dir / "training"
        print(f"[capture] inference -> {inference_root}", flush=True)
        with sc.Experiment(
            sampler=sampler(args),
            workload=inference_workload,
            store=sc.DirectoryStore(inference_root, trace_dtype="float32"),
            retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
            workload_sync="cuda",
        ) as experiment:
            inference_records = experiment.run(args.traces_per_process)

        print(f"[capture] training -> {training_root}", flush=True)
        with sc.Experiment(
            sampler=sampler(args),
            workload=training_workload,
            store=sc.DirectoryStore(training_root, trace_dtype="float32"),
            retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
            workload_sync="cuda",
        ) as experiment:
            training_records = experiment.run(args.traces_per_process)
        record_roots = {"inference": inference_root, "training": training_root}

    image_root = args.output_dir / "plots"
    image_root.mkdir(exist_ok=True)
    plot_paths = []
    for process, records in (("inference", inference_records), ("training", training_records)):
        root = record_roots[process]
        for record in records:
            output = image_root / f"{process}_{record['index']:03d}.png"
            sc.plot_capture(root, index=record["index"], output=output, annotations=True)
            plot_paths.append(str(output))

    metadata = {
        "model": args.model,
        "target": f"model.layers.{args.layer}.self_attn.o_proj",
        "capture": {
            "duration": args.duration,
            "requested_sample_rate": args.sample_rate,
            "gain_db": args.gain_db,
            "traces_per_process": args.traces_per_process,
            "interleaved": args.interleaved,
        },
        "causal_boundary": causal_metadata,
        "inference_records": len(inference_records),
        "training_records": len(training_records),
        "accepted_training_updates": training_workload.accepted_steps,
        "plot_paths": plot_paths,
    }
    (args.output_dir / "capture_summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
