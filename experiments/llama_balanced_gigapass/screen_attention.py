"""Same-state numerical and ABBA timing screen of standard attention backends."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .workload import BalancedGigaPassConfig, BalancedGigaPassEngine, load_calibration


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = BalancedGigaPassEngine(BalancedGigaPassConfig(
        calibration_override=load_calibration(args.calibration)
    ))
    try:
        engine.setup()
        original = {n: p.detach().clone() for n, p in engine.model.state_dict().items()}
        references = {}
        numerical = {}
        for backend in ("sdpa", "eager"):
            torch.cuda.synchronize()
            engine.model.load_state_dict(original)
            engine.model.set_attn_implementation(backend)
            engine._inference_ring_index = engine._training_ring_index = 0
            engine.run_mixed_batches([8192], account_service=False)
            gradients = {n: p.grad.detach().clone() for n, p in engine.model.named_parameters()}
            if backend == "sdpa":
                references = gradients
                reference_loss = float(engine._last_loss)
                reference_checksum = int(engine._last_inference_checksum)
            else:
                numerator, denominator, max_abs = 0.0, 0.0, 0.0
                exact, finite = True, True
                for name, gradient in gradients.items():
                    delta = gradient.float() - references[name].float()
                    numerator += float(delta.double().square().sum())
                    denominator += float(references[name].double().square().sum())
                    max_abs = max(max_abs, float(delta.abs().max()))
                    exact &= torch.equal(gradient, references[name])
                    finite &= bool(torch.isfinite(gradient).all())
                numerical = {
                    "reference_loss": reference_loss, "eager_loss": float(engine._last_loss),
                    "loss_equal": reference_loss == float(engine._last_loss),
                    "inference_checksum_equal": reference_checksum == int(engine._last_inference_checksum),
                    "gradients_bit_equal": exact, "gradients_finite": finite,
                    "gradient_relative_l2": (numerator / max(denominator, 1e-30)) ** 0.5,
                    "gradient_max_abs": max_abs,
                }
        del original, references, gradients
        torch.cuda.empty_cache()
        timings = {backend: {mode: [] for mode in ("inference", "training", "mixed")}
                   for backend in ("sdpa", "eager")}
        for backend in ("sdpa", "eager", "eager", "sdpa"):
            engine.model.set_attn_implementation(backend)
            engine.run_mixed_batches([8192] * 2, account_service=False)
            for _ in range(4):
                for mode, run in (
                    ("inference", lambda: engine.run_dedicated_inference_batches([8192] * 4)),
                    ("training", lambda: engine.run_dedicated_training_cycles(4)),
                    ("mixed", lambda: engine.run_mixed_batches([8192] * 4, account_service=False)),
                ):
                    timings[backend][mode].append(run().wall_seconds / 4)
        medians = {backend: {mode: statistics.median(values) for mode, values in rows.items()}
                   for backend, rows in timings.items()}
        payload = {
            "inference_batch": 8192, "training_batch": 1024, "sequence_length": 1,
            "numerical": numerical, "seconds_per_cycle": timings, "median_seconds": medians,
            "eager_speedup": {mode: medians["sdpa"][mode] / medians["eager"][mode]
                              for mode in medians["sdpa"]},
            "balanced_service_certified": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
