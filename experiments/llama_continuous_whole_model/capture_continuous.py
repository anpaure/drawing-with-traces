#!/usr/bin/env python3
"""Capture arbitrary windows from an already-running whole-model process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import sidecapture as sc

try:  # Support direct and module execution.
    from .continuous_workloads import DEFAULT_MODEL, PersistentGpuWorkload, WorkerConfig
except ImportError:  # pragma: no cover - hardware CLI invokes this file directly.
    from continuous_workloads import DEFAULT_MODEL, PersistentGpuWorkload, WorkerConfig


class RandomPretriggerDelaySampler:
    """Delegate to a sampler after a random delay while it is armed.

    The GPU worker is already active. Sleeping between scope arm and trigger
    destroys deterministic phase coupling without changing captured samples.
    """

    def __init__(
        self,
        sampler,
        *,
        minimum_delay_s: float,
        maximum_delay_s: float,
        seed: int,
    ) -> None:
        if minimum_delay_s < 0 or maximum_delay_s < minimum_delay_s:
            raise ValueError(
                "pretrigger delay bounds must satisfy 0 <= minimum <= maximum; "
                f"got {minimum_delay_s} and {maximum_delay_s}"
            )
        self.sampler = sampler
        self.request = sampler.request
        self.resolved = None
        self.minimum_delay_s = float(minimum_delay_s)
        self.maximum_delay_s = float(maximum_delay_s)
        self.seed = int(seed)
        self._random = random.Random(self.seed)
        self._last_delay_s: float | None = None

    def open(self) -> None:
        self.sampler.open()

    def plan(self):
        self.resolved = self.sampler.plan()
        return self.resolved

    def arm(self) -> None:
        self.sampler.arm()
        self._last_delay_s = self._random.uniform(
            self.minimum_delay_s,
            self.maximum_delay_s,
        )
        if self._last_delay_s:
            time.sleep(self._last_delay_s)

    def trigger(self):
        return self.sampler.trigger()

    def observe_trigger(self, anchor) -> None:
        self.sampler.observe_trigger(anchor)

    def finish(self):
        return self.sampler.finish()

    def abort(self) -> None:
        self.sampler.abort()

    def recover(self, error: BaseException) -> bool:
        return bool(self.sampler.recover(error))

    def close(self) -> None:
        self.sampler.close()

    def metadata(self) -> dict[str, Any]:
        return {
            **self.sampler.metadata(),
            "random_pretrigger_delay": {
                "minimum_s": self.minimum_delay_s,
                "maximum_s": self.maximum_delay_s,
                "seed": self.seed,
                "last_delay_s": self._last_delay_s,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inference", "training"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--captures", type=int, default=24)
    parser.add_argument("--duration", default="100ms")
    parser.add_argument("--sample-rate", default="1.5MHz")
    parser.add_argument("--gain-db", type=float, default=10.0)
    parser.add_argument("--minimum-pretrigger-delay-ms", type=float, default=0.0)
    parser.add_argument("--maximum-pretrigger-delay-ms", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--training-sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--inference-quantization", choices=("nf4", "none"), default="nf4")
    parser.add_argument(
        "--optimizer",
        choices=("adamw8bit", "adamw", "adamw_fused", "sgd"),
        default="adamw8bit",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--linear-shaping",
        choices=("none", "token-row", "hybrid"),
        default="none",
        help="Exact linear schedule used during training; hybrid also shapes one dW column.",
    )
    parser.add_argument("--cover-decode-tokens-per-microbatch", type=int, default=0)
    parser.add_argument("--cover-decode-token-jitter", type=int, default=0)
    parser.add_argument(
        "--cover-backward-layer-interval",
        type=int,
        default=0,
        help="Insert one real decode after every Nth transformer layer gradient; 0 disables it.",
    )
    parser.add_argument("--cover-prompt-tokens", type=int, default=32)
    parser.add_argument("--cover-reset-tokens", type=int, default=32)
    parser.add_argument("--bitsandbytes-path", type=Path)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_bitsandbytes(args: argparse.Namespace) -> None:
    needs_bitsandbytes = (
        args.inference_quantization == "nf4"
        if args.mode == "inference"
        else args.optimizer == "adamw8bit"
        or args.cover_decode_tokens_per_microbatch > 0
        or args.cover_backward_layer_interval > 0
    )
    dependency_path = args.bitsandbytes_path or (
        Path(os.environ["BITSANDBYTES_PATH"]) if os.environ.get("BITSANDBYTES_PATH") else None
    )
    if dependency_path is not None:
        dependency_path = dependency_path.expanduser().resolve()
        if not dependency_path.exists():
            raise FileNotFoundError(f"bitsandbytes dependency path does not exist: {dependency_path}")
        sys.path.insert(0, str(dependency_path))
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(dependency_path) + (os.pathsep + existing if existing else "")
    if needs_bitsandbytes and importlib.util.find_spec("bitsandbytes") is None:
        raise ModuleNotFoundError(
            "This workload requires bitsandbytes. Install it or pass --bitsandbytes-path to an "
            "experiment-local installation. Neither the shared Python environment nor PYTHONPATH "
            "currently provides it."
        )


def main() -> None:
    args = parse_args()
    if args.captures < 1:
        raise ValueError(f"captures must be positive, got {args.captures}")
    configure_bitsandbytes(args)
    worker_config = WorkerConfig(
        mode=args.mode,
        session_id=args.session_id,
        model=args.model,
        seed=args.seed,
        local_files_only=args.local_files_only,
        prompt_tokens=args.prompt_tokens,
        decode_tokens=args.decode_tokens,
        training_sequence_length=args.training_sequence_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        inference_quantization=args.inference_quantization,
        optimizer=args.optimizer,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        linear_shaping=args.linear_shaping,
        cover_decode_tokens_per_microbatch=args.cover_decode_tokens_per_microbatch,
        cover_decode_token_jitter=args.cover_decode_token_jitter,
        cover_backward_layer_interval=args.cover_backward_layer_interval,
        cover_prompt_tokens=args.cover_prompt_tokens,
        cover_reset_tokens=args.cover_reset_tokens,
    )
    base_sampler = sc.ChipWhispererSampler(
        sc.CaptureRequest.create(
            duration=args.duration,
            sample_rate=args.sample_rate,
            mode="burst",
            bits_per_sample=12,
            gain_db=args.gain_db,
        ),
        usb_read_mode="auto",
    )
    sampler = RandomPretriggerDelaySampler(
        base_sampler,
        minimum_delay_s=args.minimum_pretrigger_delay_ms / 1e3,
        maximum_delay_s=args.maximum_pretrigger_delay_ms / 1e3,
        seed=args.seed,
    )
    workload = PersistentGpuWorkload(
        worker_config,
        startup_timeout_s=args.startup_timeout_s,
        shutdown_timeout_s=args.shutdown_timeout_s,
    )
    session_root = args.output_dir / args.mode / args.session_id
    with sc.Experiment(
        sampler=sampler,
        workload=workload,
        store=sc.DirectoryStore(session_root, trace_dtype="float32"),
        retry=sc.RetryPolicy(max_attempts=5, backoff_s=0.5),
        workload_sync="none",
    ) as experiment:
        records = experiment.run(args.captures)
    summary = {
        "session_root": str(session_root),
        "accepted_records": len(records),
        "worker": worker_config.metadata(),
        "capture": {
            "duration": args.duration,
            "sample_rate": args.sample_rate,
            "gain_db": args.gain_db,
            "minimum_pretrigger_delay_ms": args.minimum_pretrigger_delay_ms,
            "maximum_pretrigger_delay_ms": args.maximum_pretrigger_delay_ms,
        },
    }
    (session_root / "session_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
