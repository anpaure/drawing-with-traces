#!/usr/bin/env python3
"""Benchmark one persistent training schedule without opening the scope."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

try:
    from .continuous_workloads import DEFAULT_MODEL, PersistentGpuWorkload, WorkerConfig
except ImportError:  # pragma: no cover - direct hardware invocation.
    from continuous_workloads import DEFAULT_MODEL, PersistentGpuWorkload, WorkerConfig


class _Context:
    def __init__(self) -> None:
        self.labels: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--session-id", default="benchmark")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--linear-shaping", choices=("none", "token-row", "hybrid"), default="none")
    parser.add_argument("--cover-decode-tokens-per-microbatch", type=int, default=0)
    parser.add_argument("--cover-decode-token-jitter", type=int, default=0)
    parser.add_argument("--cover-backward-layer-interval", type=int, default=0)
    parser.add_argument(
        "--optimizer",
        choices=("adamw8bit", "adamw", "adamw_fused", "sgd"),
        default="adamw8bit",
    )
    parser.add_argument("--observe-seconds", type=float, default=5.0)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--bitsandbytes-path", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def configure_bitsandbytes(path: Path | None, *, required: bool) -> None:
    if path is not None:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"bitsandbytes path does not exist: {path}")
        sys.path.insert(0, str(path))
        previous = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(path) + (os.pathsep + previous if previous else "")
    if required and importlib.util.find_spec("bitsandbytes") is None:
        raise ModuleNotFoundError(
            "bitsandbytes is required for the quantized optimizer; pass --bitsandbytes-path"
        )


def main() -> None:
    args = parse_args()
    if args.observe_seconds <= 0:
        raise ValueError("observe-seconds must be positive")
    configure_bitsandbytes(
        args.bitsandbytes_path,
        required=args.optimizer == "adamw8bit"
        or args.cover_decode_tokens_per_microbatch > 0
        or args.cover_backward_layer_interval > 0,
    )
    config = WorkerConfig(
        mode="training",
        session_id=args.session_id,
        model=args.model,
        training_sequence_length=args.sequence_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        linear_shaping=args.linear_shaping,
        cover_decode_tokens_per_microbatch=args.cover_decode_tokens_per_microbatch,
        cover_decode_token_jitter=args.cover_decode_token_jitter,
        cover_backward_layer_interval=args.cover_backward_layer_interval,
        optimizer=args.optimizer,
    )
    workload = PersistentGpuWorkload(
        config,
        startup_timeout_s=args.startup_timeout_s,
        shutdown_timeout_s=30.0,
    )
    context = _Context()
    workload.setup()
    try:
        first = workload.run(context)  # First complete update is the warmup.
        start_ns = time.monotonic_ns()
        time.sleep(args.observe_seconds)
        final = workload.run(context)
        end_ns = time.monotonic_ns()
    finally:
        workload.teardown()

    elapsed_s = (end_ns - start_ns) / 1e9
    step_delta = int(final["steps"]) - int(first["steps"])
    input_delta = int(final["training_input_tokens"]) - int(first["training_input_tokens"])
    loss_token_delta = int(final["training_loss_tokens"]) - int(first["training_loss_tokens"])
    cover_delta = int(final["cover_decode_tokens"]) - int(first["cover_decode_tokens"])
    result = {
        "config": config.metadata(),
        "warmup": first,
        "final": final,
        "observation_seconds": elapsed_s,
        "completed_updates": step_delta,
        "training_input_tokens_per_second": input_delta / elapsed_s,
        "useful_loss_tokens_per_second": loss_token_delta / elapsed_s,
        "optimizer_updates_per_second": step_delta / elapsed_s,
        "cover_decode_tokens_per_second": cover_delta / elapsed_s,
        "all_parameter_tensors_have_gradients": (
            int(final["gradient_tensors"]) == int(final["parameter_tensors"])
        ),
        "all_parameter_tensors_in_optimizer": (
            int(final["optimizer_parameter_tensors"]) == int(final["parameter_tensors"])
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
