#!/usr/bin/env python3
"""Build an executable unrotated SCS manifest for identical Triton GEMMs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .profile_events import load_profiled_period
from .sequence_alignment import PaddingCostModel, minimum_cost_supersequence


def build_manifest(
    inference_profile: Path,
    training_profile: Path,
    *,
    inference_repeats: int,
    training_repeats: int,
    duration_weight_per_us: float = 1.0,
    launch_weight: float = 1e-6,
    inference_operand_statistics: dict | None = None,
    training_operand_statistics: dict | None = None,
) -> dict:
    if inference_repeats < 1 or training_repeats < 1:
        raise ValueError("repeat counts must be positive")
    inference_period = tuple(
        block
        for block in load_profiled_period(inference_profile)
        if "identical_gemm_kernel" in block.signature.kernel_id
    )
    training_period = tuple(
        block
        for block in load_profiled_period(training_profile)
        if "identical_gemm_kernel" in block.signature.kernel_id
    )
    if not inference_period or not training_period:
        raise ValueError("both profiles must contain identical_gemm_kernel events")
    cost_model = PaddingCostModel(
        launch_weight=launch_weight,
        duration_weight_per_us=duration_weight_per_us,
    )
    plan = minimum_cost_supersequence(
        inference_period * inference_repeats,
        training_period * training_repeats,
        cost_model=cost_model,
    )
    slots = []
    for index, slot in enumerate(plan.slots):
        slots.append(
            {
                "index": index,
                "signature": asdict(slot.signature),
                "signature_key": slot.signature.key,
                "inference_semantic_id": (
                    None if slot.inference_block is None else slot.inference_block.semantic_id
                ),
                "training_semantic_id": (
                    None if slot.training_block is None else slot.training_block.semantic_id
                ),
                "template_semantic_id": slot.template.semantic_id,
                "estimated_duration_us": slot.estimated_duration_us,
                "estimated_flops": slot.estimated_flops,
                "estimated_memory_bytes": slot.estimated_memory_bytes,
            }
        )
    manifest = {
        "format": "drawing-with-traces-identical-gemm-scs-v1",
        "scope": "identical Triton GEMMs only; non-GEMM kernels are not yet scheduled",
        "alignment": "exact minimum-cost linear SCS; no dependency-breaking cycle rotation",
        "inference_profile": {
            "path": str(inference_profile),
            "sha256": _sha256(inference_profile),
            "gemms_per_period": len(inference_period),
            "repeats": inference_repeats,
        },
        "training_profile": {
            "path": str(training_profile),
            "sha256": _sha256(training_profile),
            "gemms_per_period": len(training_period),
            "repeats": training_repeats,
        },
        "cost_model": asdict(cost_model),
        "plan": plan.to_dict(),
        "slots": slots,
    }
    if inference_operand_statistics is not None or training_operand_statistics is not None:
        if inference_operand_statistics is None or training_operand_statistics is None:
            raise ValueError("both inference and training operand statistics are required")
        manifest["padding_operand_statistics"] = {
            "inference": {
                "source": "measured real training operands",
                "by_signature": training_operand_statistics,
            },
            "training": {
                "source": "measured real inference operands",
                "by_signature": inference_operand_statistics,
            },
        }
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-profile", type=Path, required=True)
    parser.add_argument("--training-profile", type=Path, required=True)
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--duration-weight-per-us", type=float, default=1.0)
    parser.add_argument("--launch-weight", type=float, default=1e-6)
    parser.add_argument("--inference-stats-benchmark", type=Path)
    parser.add_argument("--training-stats-benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_manifest(
        args.inference_profile,
        args.training_profile,
        inference_repeats=args.inference_repeats,
        training_repeats=args.training_repeats,
        duration_weight_per_us=args.duration_weight_per_us,
        launch_weight=args.launch_weight,
        inference_operand_statistics=_load_operand_statistics(
            args.inference_stats_benchmark
        ),
        training_operand_statistics=_load_operand_statistics(
            args.training_stats_benchmark
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "slots"}, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_operand_statistics(path: Path | None) -> dict | None:
    if path is None:
        return None
    document = json.loads(path.read_text())
    try:
        return document["ready"]["common_gemm_real_operand_statistics"]
    except KeyError as exc:
        raise ValueError(f"{path} has no ready common-GEMM operand statistics") from exc


if __name__ == "__main__":
    main()
