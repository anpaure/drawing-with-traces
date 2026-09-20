"""Compare mixed execution with sequential SGD across every parameter and gradient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workload import BalancedGigaPassConfig, BalancedGigaPassEngine, load_calibration


def validate(engine, cycles: int) -> dict:
    import torch

    torch.cuda.synchronize()
    initial = {name: value.detach().clone() for name, value in engine.model.state_dict().items()}
    inference_logits = []

    def retain_logits(module, inputs, output):
        if not torch.is_grad_enabled():
            inference_logits.append(output.detach().clone())

    handle = engine.model.lm_head.register_forward_hook(retain_logits)
    schedule = [engine.calibration.lower_batch_size, engine.calibration.upper_batch_size]
    schedule = [schedule[index % 2] for index in range(cycles)]

    def reset():
        torch.cuda.synchronize()
        engine.model.load_state_dict(initial)
        engine._inference_ring_index = 0
        engine._training_ring_index = 0
        engine.optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize()

    try:
        reset()
        reference_losses = []
        for index, batch in enumerate(schedule):
            # Ordinary PyTorch on one stream, independent of the mixed scheduler.
            with torch.inference_mode():
                logits = engine.model(
                    input_ids=engine.inference_ids[index % engine.config.data_ring_size, :batch],
                    use_cache=False,
                ).logits
                logits[:, -1, :].argmax(-1).sum()
            engine.optimizer.zero_grad(set_to_none=False)
            index %= engine.config.data_ring_size
            logits = engine.model(input_ids=engine.training_ids[index], use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), engine.training_targets[index].reshape(-1)
            )
            loss.backward()
            engine.optimizer.step()
            reference_losses.append(float(loss.detach()))
            del loss, logits
        torch.cuda.synchronize()
        reference_logits = inference_logits[:]
        inference_logits.clear()
        reference_parameters = {n: p.detach().clone() for n, p in engine.model.named_parameters()}
        reference_gradients = {
            n: p.grad.detach().clone() for n, p in engine.model.named_parameters() if p.grad is not None
        }
        reset()
        engine.run_mixed_batches(schedule, account_service=False)
        torch.cuda.synchronize()
        parameters_equal, gradients_equal, all_finite = True, True, True
        mismatches = []
        for name, parameter in engine.model.named_parameters():
            pe = torch.equal(parameter, reference_parameters[name])
            ge = parameter.grad is not None and torch.equal(parameter.grad, reference_gradients[name])
            finite = bool(torch.isfinite(parameter).all() and torch.isfinite(parameter.grad).all())
            parameters_equal &= pe
            gradients_equal &= ge
            all_finite &= finite
            if not pe or not ge or not finite:
                mismatches.append(name)
        logits_equal = len(reference_logits) == len(inference_logits) == cycles and all(
            torch.equal(left, right) for left, right in zip(reference_logits, inference_logits)
        )
        changed = sum(
            int(torch.count_nonzero(parameter.detach() != initial[name]))
            for name, parameter in engine.model.named_parameters()
        )
        last_loss_equal = float(engine._last_loss) == reference_losses[-1]
        return {
            "passed": parameters_equal and gradients_equal and logits_equal and all_finite
            and changed > 0 and last_loss_equal,
            "cycles": cycles,
            "inference_schedule": schedule,
            "all_parameters_bit_equal": parameters_equal,
            "all_gradients_bit_equal": gradients_equal,
            "all_inference_logits_bit_equal": logits_equal,
            "all_parameters_and_gradients_finite": all_finite,
            "last_loss_bit_equal": last_loss_equal,
            "reference_losses": reference_losses,
            "changed_parameter_elements": changed,
            "parameter_elements_checked": engine.parameter_count,
            "mismatched_parameter_names": mismatches,
            "reference": "ordinary sequential PyTorch forward, cross-entropy, backward, foreach SGD",
        }
    finally:
        handle.remove()


def profile(engine, output: Path) -> dict:
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        engine.run_mixed_cycles(2)
    rows = sorted(prof.key_averages(), key=lambda event: event.self_device_time_total, reverse=True)
    payload = {
        "note": "Profiling overhead is excluded from throughput results; CUDA self times may overlap.",
        "operations": [{"name": row.key, "calls": row.count,
                        "cuda_self_us": row.self_device_time_total,
                        "cpu_self_us": row.self_cpu_time_total} for row in rows[:35]],
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--attention-backend", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args()
    if args.cycles < 1:
        raise ValueError("cycles must be positive")
    config = BalancedGigaPassConfig(
        calibration_override=load_calibration(args.calibration),
        attention_backend=args.attention_backend,
    )
    engine = BalancedGigaPassEngine(config)
    try:
        engine.setup()
        result = validate(engine, args.cycles)
        result["engine"] = engine.metadata()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "engine"}, indent=2), flush=True)
        if not result["passed"]:
            raise RuntimeError("mixed execution failed full-model sequential-equivalence validation")
        if args.profile_output:
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            print(json.dumps(profile(engine, args.profile_output), indent=2), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
