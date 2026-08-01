#!/usr/bin/env python3
"""Compact and compare CUDA kernel profiles against cached NF4 inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .kernel_similarity import continuous_kernel_similarity
except ImportError:  # pragma: no cover - direct invocation from copied experiment.
    from kernel_similarity import continuous_kernel_similarity


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("candidate label and path cannot be empty")
    return label, Path(raw_path)


def family_fractions(profile: dict[str, Any]) -> dict[str, float]:
    durations = {
        row["family"]: float(row["total_us"])
        for row in profile["top_kernel_families"]
    }
    total = sum(durations.values())
    if total <= 0:
        return {}
    return {name: duration / total for name, duration in durations.items()}


def gemm_geometry(profile: dict[str, Any]) -> dict[str, Any]:
    total = 0
    one_row = 0
    by_rows: dict[str, int] = {}
    for row in profile["aten_gemm_shapes"]:
        shapes = row.get("input_shapes") or []
        if not shapes or not isinstance(shapes[0], list) or len(shapes[0]) < 2:
            continue
        count = int(row["count"])
        matrix_rows = int(shapes[0][-2])
        total += count
        one_row += count if matrix_rows == 1 else 0
        by_rows[str(matrix_rows)] = by_rows.get(str(matrix_rows), 0) + count
    return {
        "aten_gemm_calls": total,
        "m_equals_1_calls": one_row,
        "m_equals_1_fraction": one_row / total if total else 0.0,
        "calls_by_first_operand_rows": dict(
            sorted(by_rows.items(), key=lambda item: int(item[0]))
        ),
    }


def compact_profile(payload: dict[str, Any]) -> dict[str, Any]:
    profile = payload["profile"]
    return {
        "config": payload["config"],
        "wall_ms_under_profiler": profile["wall_ms"],
        "cuda_kernel_count": profile["cuda_kernel_count"],
        "cuda_busy_ms": profile["cuda_busy_ms"],
        "peak_allocated_bytes": profile["peak_allocated_bytes"],
        "family_duration_fractions": family_fractions(profile),
        "gemm_geometry": gemm_geometry(profile),
        "top_kernel_families": profile["top_kernel_families"][:20],
        "top_cuda_kernels": profile["top_cuda_kernels"][:20],
    }


def compare(reference: dict[str, Any], candidates: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    reference_profile = reference["profile"]
    results = {}
    for label, payload in candidates:
        profile = payload["profile"]
        results[label] = {
            **compact_profile(payload),
            "similarity_to_quantized_cached_decode": continuous_kernel_similarity(
                reference_profile["cuda_kernel_sequence"],
                profile["cuda_kernel_sequence"],
            ),
        }
    return {
        "reference": compact_profile(reference),
        "candidates": results,
        "similarity_definition": (
            "Boundary-free normalized distributions of kernel family/exact duration, cyclic "
            "family n-grams, launch gaps, and kernel durations. This is a scheduling proxy, "
            "not a substitute for physical capture."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate:
        parser.error("at least one --candidate LABEL=PATH is required")
    reference = json.loads(args.reference.read_text())
    candidates = [(label, json.loads(path.read_text())) for label, path in args.candidate]
    result = compare(reference, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
