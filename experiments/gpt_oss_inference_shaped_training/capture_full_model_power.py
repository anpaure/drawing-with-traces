#!/usr/bin/env python3
"""SideCapture real GPT-OSS decode vs ordinary/inference-shaped fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import sidecapture as sc

try:  # Support both ``python file.py`` and ``python -m experiments...``.
    from .inference_shaped_linear import InferenceShapeConfig, InferenceShapedLinear
    from .run_experiment import DEFAULT_MODEL
except ImportError:  # pragma: no cover - exercised by hardware CLI invocation.
    from inference_shaped_linear import InferenceShapeConfig, InferenceShapedLinear
    from run_experiment import DEFAULT_MODEL


class InterleavedFullModelWorkload(sc.Workload):
    """Alternate continuous cached decode with one real causal training step."""

    replay_safe = False

    def __init__(
        self,
        *,
        model: nn.Module,
        target_parameters: list[nn.Parameter],
        target_names: list[str],
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        decode_id: Tensor,
        decode_mask: Tensor,
        past_key_values: Any,
        variant: str,
        learning_rate: float,
        decode_repeats: int,
        training_repeats: int,
        cover_decodes_per_training_step: int,
    ) -> None:
        self.model = model
        self.target_parameters = target_parameters
        self.target_names = target_names
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.decode_id = decode_id
        self.decode_mask = decode_mask
        self.past_key_values = past_key_values
        self.variant = variant
        self.learning_rate = float(learning_rate)
        self.decode_repeats = int(decode_repeats)
        self.training_repeats = int(training_repeats)
        self.cover_decodes_per_training_step = int(cover_decodes_per_training_step)
        self.master_weights = [parameter.detach().float().clone() for parameter in target_parameters]
        self.accepted_training_steps = 0
        self.accepted_losses: list[float] = []

    def snapshot(self):
        # Identical pre-arm path for inference and training controls.
        snapshot = {
            "master_weights": [weight.clone() for weight in self.master_weights],
            "model_weights": [parameter.detach().clone() for parameter in self.target_parameters],
            "accepted_training_steps": self.accepted_training_steps,
            "accepted_losses": list(self.accepted_losses),
        }
        torch.cuda.synchronize()
        return snapshot

    def restore(self, snapshot) -> None:
        for master, saved in zip(self.master_weights, snapshot["master_weights"]):
            master.copy_(saved)
        for parameter, saved in zip(self.target_parameters, snapshot["model_weights"]):
            parameter.data.copy_(saved)
        self.accepted_training_steps = int(snapshot["accepted_training_steps"])
        self.accepted_losses = list(snapshot["accepted_losses"])
        for parameter in self.target_parameters:
            parameter.grad = None
        torch.cuda.synchronize()

    def run(self, context: sc.CaptureContext):
        context.labels.update(
            capture_schedule="strictly_interleaved",
            model=DEFAULT_MODEL,
            variant=self.variant,
            parameter_scope=self.target_names,
        )
        if context.index % 2 == 0:
            return self.run_inference(context)
        return self.run_training(context)

    def run_inference(self, context: sc.CaptureContext):
        self.model.eval()
        context.labels.update(
            process="inference",
            task="cached_decode",
            decode_repeats=self.decode_repeats,
            accepted_training_steps_before=self.accepted_training_steps,
        )
        context.cuda_mark(f"{self.variant}.inference.window_start", sync="none")
        output = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(self.decode_repeats):
                output = self.model(
                    input_ids=self.decode_id,
                    attention_mask=self.decode_mask,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
        end.record()
        end.synchronize()
        if output is None:  # pragma: no cover - CLI validates repeats.
            raise RuntimeError("decode_repeats must be positive")
        return {
            "process": "inference",
            "logits_checksum": float(output.logits.float().sum().cpu()),
            "cuda_ms": float(start.elapsed_time(end)),
        }

    def run_training(self, context: sc.CaptureContext):
        self.model.train()
        context.labels.update(
            process="training",
            task="causal_lm_finetuning",
            accepted_training_steps_before=self.accepted_training_steps,
            optimizer="FP32-master SGD",
            learning_rate=self.learning_rate,
            trainable_parameter_tensors=len(self.target_parameters),
            trainable_parameters=sum(parameter.numel() for parameter in self.target_parameters),
            training_repeats=self.training_repeats,
            cover_decodes_per_training_step=self.cover_decodes_per_training_step,
        )
        context.cuda_mark(f"{self.variant}.training.window_start", sync="none")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        losses = []
        gradient_checksum = 0.0
        training_ordinal = context.index // 2
        cycle = ["training", *("decode" for _ in range(self.cover_decodes_per_training_step))]
        phase_offset = training_ordinal % len(cycle)
        cycle = cycle[phase_offset:] + cycle[:phase_offset]
        context.labels["cycle_first"] = cycle[0]
        context.labels["cycle_phase_offset"] = phase_offset

        def training_step() -> None:
            nonlocal gradient_checksum
            for parameter in self.target_parameters:
                parameter.grad = None
            output = self.model(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                labels=self.labels,
                use_cache=False,
            )
            output.loss.backward()
            gradients = [parameter.grad for parameter in self.target_parameters]
            missing = [
                name for name, gradient in zip(self.target_names, gradients) if gradient is None
            ]
            if missing:
                raise RuntimeError(f"GPT-OSS projections did not receive gradients: {missing}")
            for parameter, master, gradient in zip(
                self.target_parameters, self.master_weights, gradients
            ):
                assert gradient is not None
                master.add_(gradient.float(), alpha=-self.learning_rate)
                parameter.data.copy_(master.to(parameter.dtype))
                gradient_checksum += float(gradient.detach().float().sum().cpu())
                parameter.grad = None
            losses.append(float(output.loss.detach().float().cpu()))

        def cover_decode() -> None:
            self.model.eval()
            with torch.inference_mode():
                self.model(
                    input_ids=self.decode_id,
                    attention_mask=self.decode_mask,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )

        start.record()
        for _ in range(self.training_repeats):
            for task in cycle:
                if task == "training":
                    self.model.train()
                    training_step()
                else:
                    cover_decode()
        end.record()
        end.synchronize()
        return {
            "process": "training",
            "losses": losses,
            "gradient_checksum": gradient_checksum,
            "cuda_ms": float(start.elapsed_time(end)),
        }

    def on_accept(self, result: dict[str, Any]) -> None:
        if result["process"] == "training":
            self.accepted_training_steps += len(result["losses"])
            self.accepted_losses.extend(float(loss) for loss in result["losses"])

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "real_gpt_oss_interleaved_decode_finetuning",
            "model": DEFAULT_MODEL,
            "variant": self.variant,
            "targets": self.target_names,
            "trainable_parameter_tensors": len(self.target_parameters),
            "trainable_parameters": sum(parameter.numel() for parameter in self.target_parameters),
            "decode_repeats": self.decode_repeats,
            "training_repeats": self.training_repeats,
            "cover_decodes_per_training_step": self.cover_decodes_per_training_step,
            "optimizer": "FP32-master SGD",
            "learning_rate": self.learning_rate,
            "schedule": "inference, training, repeat",
            "common_pre_arm_snapshot": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--prompt", default="Training can imitate inference")
    parser.add_argument("--traces-per-process", type=int, default=9)
    parser.add_argument("--decode-repeats", type=int, default=3)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--shaped-inference-rows", type=int, default=128)
    parser.add_argument("--shaped-forward-rows", type=int, default=1)
    parser.add_argument("--shaped-input-gradient-rows", type=int, default=1)
    parser.add_argument("--shaped-cover-decodes", type=int, default=0)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-o-projections", action="store_true")
    scope.add_argument("--all-attention-projections", action="store_true")
    scope.add_argument("--layer-attention-projections", action="store_true")
    parser.add_argument("--skip-ordinary", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_sampler(args: argparse.Namespace) -> sc.ChipWhispererSampler:
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


def prepare_decode(model, input_ids: Tensor, attention_mask: Tensor):
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
        # Warm the exact cached-decode path before opening the scope.
        model(
            input_ids=decode_id,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    return decode_id, decode_mask, past_key_values


def capture_variant(
    *,
    args: argparse.Namespace,
    root: Path,
    variant: str,
    model: nn.Module,
    target_parameters: list[nn.Parameter],
    target_names: list[str],
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    cover_decodes_per_training_step: int = 0,
) -> tuple[list[dict[str, Any]], InterleavedFullModelWorkload]:
    decode_id, decode_mask, past_key_values = prepare_decode(model, input_ids, attention_mask)
    workload = InterleavedFullModelWorkload(
        model=model,
        target_parameters=target_parameters,
        target_names=target_names,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decode_id=decode_id,
        decode_mask=decode_mask,
        past_key_values=past_key_values,
        variant=variant,
        learning_rate=args.learning_rate,
        decode_repeats=args.decode_repeats,
        training_repeats=args.training_repeats,
        cover_decodes_per_training_step=cover_decodes_per_training_step,
    )
    run_root = root / "mixed"
    print(f"[capture] {variant} -> {run_root}", flush=True)
    with sc.Experiment(
        sampler=make_sampler(args),
        workload=workload,
        store=sc.DirectoryStore(run_root, trace_dtype="float32"),
        retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
        workload_sync="cuda",
    ) as experiment:
        records = experiment.run(2 * args.traces_per_process)
    return records, workload


def main() -> None:
    args = parse_args()
    if args.traces_per_process < 1 or args.decode_repeats < 1 or args.training_repeats < 1:
        raise ValueError("traces-per-process, decode-repeats, and training-repeats must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[model] loading cached {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][:, :2].to("cuda")
    if input_ids.shape[1] != 2:
        raise ValueError(f"Prompt must produce at least two tokens, got {input_ids.shape[1]}")
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.all_attention_projections:
        slots = [
            (model.model.layers[layer].self_attn, attribute, layer)
            for layer in range(len(model.model.layers))
            for attribute in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        parameter_scope_mode = "all_attention_projections"
    elif args.layer_attention_projections:
        slots = [
            (model.model.layers[args.layer].self_attn, attribute, args.layer)
            for attribute in ("q_proj", "k_proj", "v_proj", "o_proj")
        ]
        parameter_scope_mode = "layer_attention_projections"
    else:
        selected_layers = range(len(model.model.layers)) if args.all_o_projections else [args.layer]
        slots = [
            (model.model.layers[layer].self_attn, "o_proj", layer) for layer in selected_layers
        ]
        parameter_scope_mode = "all_o_projections" if args.all_o_projections else "layer_o_projection"
    ordinary_projections = [getattr(parent, attribute) for parent, attribute, _layer in slots]
    invalid = [type(module).__name__ for module in ordinary_projections if not isinstance(module, nn.Linear)]
    if invalid:
        raise TypeError(f"Expected every selected projection to be nn.Linear, got {invalid}")
    target_parameters = [projection.weight for projection in ordinary_projections]
    target_names = [
        f"model.layers.{layer}.self_attn.{attribute}.weight"
        for _parent, attribute, layer in slots
    ]
    for parameter in target_parameters:
        parameter.requires_grad_(True)
    original_weights = [parameter.detach().clone() for parameter in target_parameters]

    ordinary_records = []
    ordinary_workload = None
    if not args.skip_ordinary:
        ordinary_root = args.output_dir / "ordinary"
        for (parent, attribute, _layer), projection in zip(slots, ordinary_projections):
            setattr(parent, attribute, projection)
        ordinary_records, ordinary_workload = capture_variant(
            args=args,
            root=ordinary_root,
            variant="ordinary",
            model=model,
            target_parameters=target_parameters,
            target_names=target_names,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cover_decodes_per_training_step=0,
        )

    # Identical starting weight and data for the shaped comparison.
    for parameter, original_weight in zip(target_parameters, original_weights):
        parameter.data.copy_(original_weight)
        parameter.grad = None
    shaped_projections = [
        InferenceShapedLinear(
            projection,
            InferenceShapeConfig(
                forward_rows=args.shaped_forward_rows,
                input_gradient_rows=args.shaped_input_gradient_rows,
                weight_gradient_rows=1,
                weight_gradient_inference_rows=min(
                    args.shaped_inference_rows, projection.in_features
                ),
                reduction_k=projection.in_features,
                bias_epilogue=True,
            ),
        )
        for projection in ordinary_projections
    ]
    for (parent, attribute, _layer), projection in zip(slots, shaped_projections):
        setattr(parent, attribute, projection)
    shaped_root = args.output_dir / "shaped"
    shaped_records, shaped_workload = capture_variant(
        args=args,
        root=shaped_root,
        variant="inference_shaped",
        model=model,
        target_parameters=target_parameters,
        target_names=target_names,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        cover_decodes_per_training_step=args.shaped_cover_decodes,
    )

    summary = {
        "model": args.model,
        "parameter_scope_mode": parameter_scope_mode,
        "targets": target_names,
        "target_shapes": [list(parameter.shape) for parameter in target_parameters],
        "trainable_parameters": sum(parameter.numel() for parameter in target_parameters),
        "capture": {
            "duration": args.duration,
            "requested_sample_rate": args.sample_rate,
            "gain_db": args.gain_db,
            "traces_per_process_per_variant": args.traces_per_process,
            "decode_repeats": args.decode_repeats,
            "training_repeats": args.training_repeats,
            "shaped_cover_decodes_per_training_step": args.shaped_cover_decodes,
        },
        "ordinary": {
            "records": len(ordinary_records),
            "accepted_training_steps": 0
            if ordinary_workload is None
            else ordinary_workload.accepted_training_steps,
            "accepted_losses": [] if ordinary_workload is None else ordinary_workload.accepted_losses,
        },
        "inference_shaped": {
            "inference_rows": args.shaped_inference_rows,
            "forward_rows": args.shaped_forward_rows,
            "input_gradient_rows": args.shaped_input_gradient_rows,
            "cover_decodes_per_training_step": args.shaped_cover_decodes,
            "records": len(shaped_records),
            "accepted_training_steps": shaped_workload.accepted_training_steps,
            "accepted_losses": shaped_workload.accepted_losses,
        },
    }
    (args.output_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
