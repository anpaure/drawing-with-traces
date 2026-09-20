"""Separate candidate: contextual LM training with FP32 masters and BF16 AMP.

This does not change the frozen single-token capture program. It reuses that
program's dependency-safe stream schedule, but loads FP32 parameters and runs
ordinary mixed-precision causal forwards on nontrivial token sequences.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

from .benchmark import parse_candidates
from .scheduler import BalancedServiceController, summarize_durations
from .workload import BalancedGigaPassConfig, BalancedGigaPassEngine, load_calibration


def save_result(path: Path, result: dict) -> None:
    """Preserve explicit non-finite diagnostics instead of losing failed runs."""
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(result), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ContextualAMPConfig(BalancedGigaPassConfig):
    training_batch_size: int = 8
    sequence_length: int = 128
    inference_batch_candidates: tuple[int, ...] = tuple(range(16, 73, 4))
    attention_backend: str = "sdpa"
    parameter_dtype: str = "float32"
    autocast_dtype: str = "bfloat16"
    autocast_cache_enabled: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sequence_length < 2:
            raise ValueError("contextual training requires sequence_length >= 2")
        if self.parameter_dtype != "float32" or self.autocast_dtype != "bfloat16":
            raise ValueError("this candidate requires FP32 masters and BF16 autocast")


class ContextualAMPEngine(BalancedGigaPassEngine):
    """Same shared-weight stream schedule, ordinary full-model AMP gradients."""

    def __init__(self, config: ContextualAMPConfig) -> None:
        super().__init__(config)
        self.device_type = "cuda"

    def autocast(self):
        import torch
        return torch.autocast(self.device_type, dtype=torch.bfloat16,
                              cache_enabled=self.config.autocast_cache_enabled)

    def setup(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        random.seed(self.config.compute_seed)
        torch.manual_seed(self.config.compute_seed)
        torch.cuda.manual_seed_all(self.config.compute_seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Do not run the BF16-storage parent's setup and then silently convert.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model, local_files_only=self.config.local_files_only,
            dtype=torch.float32, attn_implementation=self.config.attention_backend,
        ).cuda()
        self.model.train()
        self.model.config.use_cache = False
        if not all(p.requires_grad and p.dtype == torch.float32 for p in self.model.parameters()):
            raise ValueError("every model parameter must be trainable and stored in FP32")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, foreach=True)
        self.inference_stream = torch.cuda.Stream()
        self.training_stream = torch.cuda.current_stream()
        generator = torch.Generator(device="cuda").manual_seed(self.config.compute_seed + 101)
        vocabulary = min(int(self.model.config.vocab_size), 32_000)
        self.inference_ids = torch.randint(
            0, vocabulary, (self.config.data_ring_size, max(self.config.effective_inference_candidates),
                            self.config.sequence_length), generator=generator, device="cuda",
        )
        tokens = torch.randint(
            0, vocabulary, (self.config.data_ring_size, self.config.training_batch_size,
                            self.config.sequence_length + 1), generator=generator, device="cuda",
        )
        self.training_ids = tokens[:, :, :-1].contiguous()
        self.training_targets = tokens[:, :, 1:].contiguous()
        self.parameter_probe_name, parameter = next(
            (name, p) for name, p in self.model.named_parameters() if name.endswith("mlp.down_proj.weight")
        )
        self.parameter_probe = parameter.reshape(-1)[:4096]
        self.initial_parameter_probe = self.parameter_probe.detach().clone()
        torch.cuda.synchronize()
        warmup_batch = self.config.effective_inference_candidates[len(self.config.effective_inference_candidates) // 2]
        for _ in range(self.config.warmup_cycles):
            self.run_mixed_batches([warmup_batch], account_service=False)
        self.calibration = self.config.calibration_override
        if self.calibration is None:
            self.calibration = self._refine_calibration(self._measure_calibration())
        self.controller = BalancedServiceController(self.calibration)
        self._started = time.monotonic()

    def _enqueue_inference(self, batch_size: int) -> None:
        with self.autocast():
            super()._enqueue_inference(batch_size)

    def _enqueue_training_backward(self) -> None:
        import torch
        self.optimizer.zero_grad(set_to_none=not self._gradient_buffers_ready)
        index = self._training_ring_index % self.config.data_ring_size
        self._training_ring_index += 1
        with self.autocast():
            logits = self.model(input_ids=self.training_ids[index], use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), self.training_targets[index].reshape(-1)
            )
        # Autograd follows the precision selected by forward, outside autocast.
        loss.backward()
        self._gradient_buffers_ready = True
        self._last_loss = loss.detach()

    def metadata(self) -> dict:
        return {**super().metadata(), "variant": "contextual-fp32-master-bf16-autocast",
                "autocast_dtype": self.config.autocast_dtype,
                "autocast_cache_enabled": self.config.autocast_cache_enabled,
                "inference_mode": "uncached causal prefill; not autoregressive KV-cache serving",
                "data": "deterministic synthetic token sequences; no language-quality claim"}

    def close(self) -> None:
        super().close()
        self.parameter_probe = self.initial_parameter_probe = None


def validate_amp(engine: ContextualAMPEngine) -> dict:
    """Independent sequential AMP reference; full tensors, not just a checksum."""
    import torch

    torch.cuda.synchronize()
    initial = {name: p.detach().clone() for name, p in engine.model.state_dict().items()}
    retained_logits = []

    def retain_logits(module, inputs, output):
        if not torch.is_grad_enabled():
            retained_logits.append(output.detach().clone())

    handle = engine.model.lm_head.register_forward_hook(retain_logits)
    schedule = [engine.calibration.lower_batch_size, engine.calibration.upper_batch_size]

    def reset():
        torch.cuda.synchronize()
        engine.model.load_state_dict(initial)
        engine.optimizer.zero_grad(set_to_none=False)
        engine._inference_ring_index = engine._training_ring_index = 0
        torch.cuda.synchronize()

    try:
        reset()
        losses = []
        for index, batch in enumerate(schedule):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits = engine.model(input_ids=engine.inference_ids[index, :batch], use_cache=False).logits
                logits[:, -1, :].argmax(-1).sum()
            engine.optimizer.zero_grad(set_to_none=False)
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                logits = engine.model(input_ids=engine.training_ids[index], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), engine.training_targets[index].reshape(-1)
                )
            loss.backward()
            engine.optimizer.step()
            losses.append(float(loss.detach()))
            del logits, loss
        torch.cuda.synchronize()
        reference_logits = retained_logits[:]
        retained_logits.clear()
        reference_parameters = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
        missing = [name for name, p in engine.model.named_parameters() if p.grad is None]
        if missing:
            raise RuntimeError(f"sequential AMP reference did not produce gradients for: {missing}")
        reference_gradients = {n: p.grad.detach().clone() for n, p in engine.model.named_parameters()}
        reset()
        engine.run_mixed_batches(schedule, account_service=False)
        torch.cuda.synchronize()
        rows = []
        for name, parameter in engine.model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"mixed AMP execution did not produce a gradient for: {name}")
            rows.append({
                "name": name, "elements": parameter.numel(),
                "parameter_dtype": str(parameter.dtype), "gradient_dtype": str(gradient.dtype),
                "parameters_bit_equal": torch.equal(parameter, reference_parameters[name]),
                "gradients_bit_equal": torch.equal(gradient, reference_gradients[name]),
                "finite": bool(torch.isfinite(parameter).all() and torch.isfinite(gradient).all()),
                "changed_elements": int(torch.count_nonzero(parameter.detach() != initial[name])),
                "nonzero_gradient_elements": int(torch.count_nonzero(gradient)),
            })
        attention_rows = [r for r in rows if r["name"].endswith(("self_attn.q_proj.weight", "self_attn.k_proj.weight"))]
        checks = {
            "all_parameters_bit_equal": all(r["parameters_bit_equal"] for r in rows),
            "all_gradients_bit_equal": all(r["gradients_bit_equal"] for r in rows),
            "all_parameters_and_gradients_finite": all(r["finite"] for r in rows),
            "all_parameters_and_gradients_fp32": all(r["parameter_dtype"] == r["gradient_dtype"] == "torch.float32" for r in rows),
            "all_inference_logits_bit_equal": len(reference_logits) == len(retained_logits) == 2
                and all(torch.equal(a, b) for a, b in zip(reference_logits, retained_logits)),
            "all_inference_head_outputs_bf16": all(t.dtype == torch.bfloat16 for t in retained_logits),
            "all_inference_head_outputs_finite": all(bool(torch.isfinite(t).all()) for t in retained_logits),
            "last_loss_bit_equal": float(engine._last_loss) == losses[-1],
            "every_attention_query_key_has_nonzero_gradient": len(attention_rows) == 2 * len(engine.model.model.layers)
                and all(r["nonzero_gradient_elements"] > 0 for r in attention_rows),
            "every_attention_query_key_has_changed_weights": bool(attention_rows)
                and all(r["changed_elements"] > 0 for r in attention_rows),
        }
        return {"passed": all(checks.values()), "checks": checks, "cycles": 2,
                "sequence_length": engine.config.sequence_length, "inference_schedule": schedule,
                "reference_losses": losses, "parameters_checked": sum(r["elements"] for r in rows),
                "changed_parameter_elements": sum(r["changed_elements"] for r in rows), "tensors": rows,
                "reference": "ordinary sequential PyTorch FP32 parameters, BF16 autocast forward/loss, backward and SGD outside autocast"}
    finally:
        handle.remove()


def benchmark_amp(engine: ContextualAMPEngine, rounds: int, cycles: int) -> dict:
    planner = BalancedServiceController(engine.calibration)
    gpu = {mode: [] for mode in ("inference", "training", "mixed")}
    wall = {mode: [] for mode in gpu}
    inference_tokens = 0
    for index in range(rounds):
        schedule = planner.plan(cycles)
        inference_tokens += sum(schedule) * engine.config.sequence_length
        order = (("inference", "training", "mixed"), ("mixed", "inference", "training"),
                 ("training", "mixed", "inference"))[index % 3]
        for mode in order:
            if mode == "inference":
                timing = engine.run_dedicated_inference_batches(schedule)
            elif mode == "training":
                timing = engine.run_dedicated_training_cycles(cycles)
            else:
                timing = engine.run_mixed_batches(schedule)
            wall[mode].append(timing.wall_seconds)
            gpu[mode].append(timing.gpu_seconds)
    wall_summary = summarize_durations(wall["inference"], wall["training"], wall["mixed"])
    gpu_summary = summarize_durations(gpu["inference"], gpu["training"], gpu["mixed"])
    training_tokens = rounds * cycles * engine.config.training_batch_size * engine.config.sequence_length
    elapsed = wall_summary["total_seconds"]["mixed"]
    snapshot = engine.snapshot()
    checks = {"finite_loss": math.isfinite(snapshot["last_loss"]),
              "changed_parameters": snapshot["parameter_probe_delta_linf"] > 0,
              "wall_service_imbalance_at_most_2_percent": wall_summary["native_service_relative_imbalance"] <= 0.02,
              "gpu_service_imbalance_at_most_2_percent": gpu_summary["native_service_relative_imbalance"] <= 0.02,
              "wall_slowdown_within_1_percent_of_2x": wall_summary["max_slowdown"] <= 2.02,
              "gpu_slowdown_within_1_percent_of_2x": gpu_summary["max_slowdown"] <= 2.02}
    return {"passed": all(checks.values()), "checks": checks, "wall_timing": wall_summary,
            "gpu_timing": gpu_summary, "snapshot": snapshot,
            "work": {"inference_processed_tokens": inference_tokens, "training_tokens": training_tokens,
                     "mixed_training_updates": rounds * cycles,
                     "inference_processed_tokens_per_second": inference_tokens / elapsed,
                     "training_tokens_per_second": training_tokens / elapsed,
                     "model_flops_per_second_estimate": engine.parameter_count * (2 * inference_tokens + 6 * training_tokens) / elapsed,
                     "flop_estimate_note": "2P/6P token estimate; excludes quadratic attention and non-matmul work"},
            "limits": {"target_slowdown": 2, "measurement_tolerance_fraction": 0.01,
                       "native_service_imbalance": 0.02},
            "physical_detector_evaluation": "not yet performed for this candidate"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--inference-candidates", type=parse_candidates, default=tuple(range(16, 73, 4)))
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--benchmark-rounds", type=int, default=12)
    parser.add_argument("--cycles-per-block", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing contextual AMP result")
    if args.benchmark_rounds < 1 or args.cycles_per_block < 1:
        raise ValueError("benchmark-rounds and cycles-per-block must be positive")
    config = ContextualAMPConfig(
        training_batch_size=args.training_batch_size, sequence_length=args.sequence_length,
        inference_batch_candidates=args.inference_candidates, attention_backend=args.attention_backend,
        calibration_override=None if args.calibration_input is None else load_calibration(args.calibration_input),
    )
    engine = ContextualAMPEngine(config)
    result = {"status": "starting", "config": asdict(config),
              "physical_detector_evaluation": "not yet performed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        save_result(args.output, result)

    try:
        engine.setup()
        result["engine"] = engine.metadata()
        result["calibration"] = engine.calibration.to_dict()
        result["validation"] = validate_amp(engine)
        save()
        if not result["validation"]["passed"]:
            raise RuntimeError("contextual AMP failed full-model sequential-equivalence/attention checks")
        result["benchmark"] = benchmark_amp(engine, args.benchmark_rounds, args.cycles_per_block)
        result["status"] = "complete"
        result["passed"] = result["validation"]["passed"] and result["benchmark"]["passed"]
        save()
        print(json.dumps({"validation_passed": result["validation"]["passed"],
                          "benchmark": result["benchmark"]}), flush=True)
    except Exception as exc:
        result.update(status="failed", passed=False, error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    finally:
        engine.close()


if __name__ == "__main__":
    main()
