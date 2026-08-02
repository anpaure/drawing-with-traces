"""Capture and parse one exact steady-state CUDA period into SCS blocks."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from ..llama_identical_microkernel_carrier.backend import IdenticalKernelCall
from .sequence_alignment import BlockSignature, ComputationBlock
from .sequence_alignment import minimum_cost_supersequence


T = TypeVar("T")
_GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def capture_cuda_period(
    label: str,
    function: Callable[[], T],
    output_prefix: Path,
    *,
    logical_gemm_calls: Sequence[IdenticalKernelCall] | None = None,
    record_logical_gemm_calls: bool = False,
) -> tuple[T, dict[str, Any]]:
    """Profile one real call and save both raw Chrome and parsed block traces."""

    if not label:
        raise ValueError("profile label cannot be empty")
    if logical_gemm_calls is not None and record_logical_gemm_calls:
        raise ValueError("pass recorded calls or record this invocation, not both")
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    chrome_path = Path(f"{output_prefix}.chrome.json")
    blocks_path = Path(f"{output_prefix}.blocks.json")

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from ..llama_identical_microkernel_carrier.backend import (
        begin_identical_kernel_sequence,
        cancel_identical_kernel_sequence,
        end_identical_kernel_sequence,
    )

    torch.cuda.synchronize()
    if record_logical_gemm_calls:
        begin_identical_kernel_sequence()
    started = time.perf_counter()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as profiler:
            with record_function(label):
                result = function()
            torch.cuda.synchronize()
        if record_logical_gemm_calls:
            logical_gemm_calls = end_identical_kernel_sequence()
    except BaseException:
        if record_logical_gemm_calls:
            cancel_identical_kernel_sequence()
        raise
    wall_ms = (time.perf_counter() - started) * 1e3
    profiler.export_chrome_trace(str(chrome_path))
    parsed = parse_chrome_trace(chrome_path, logical_gemm_calls=logical_gemm_calls or ())
    summary = {
        "label": label,
        "wall_ms_including_profiler": wall_ms,
        "chrome_trace": str(chrome_path),
        "blocks_trace": str(blocks_path),
        **parsed,
    }
    blocks_path.write_text(json.dumps(summary, indent=2) + "\n")
    return result, summary


def compact_profile_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Return queue/log-safe profile provenance without the full event sequence."""

    keys = (
        "label",
        "wall_ms_including_profiler",
        "chrome_trace",
        "blocks_trace",
        "cuda_blocks",
        "logical_gemm_calls",
        "cuda_busy_us",
        "period_span_us",
        "maximum_preceding_gap_us",
    )
    compact = {key: summary[key] for key in keys}
    compact["unique_kernel_signatures"] = len(summary["kernel_counts"])
    return compact


def parse_chrome_trace(
    path: Path,
    *,
    logical_gemm_calls: Sequence[IdenticalKernelCall] = (),
) -> dict[str, Any]:
    return parse_chrome_document(
        json.loads(Path(path).read_text()),
        logical_gemm_calls=logical_gemm_calls,
    )


def parse_chrome_document(
    document: dict[str, Any],
    *,
    logical_gemm_calls: Sequence[IdenticalKernelCall] = (),
) -> dict[str, Any]:
    """Parse Kineto events, enriching Triton graph launches by recorded order."""

    trace_events = document.get("traceEvents")
    if not isinstance(trace_events, list):
        raise ValueError("Chrome trace has no traceEvents list")
    parents: dict[int, dict[str, Any]] = {}
    for event in trace_events:
        if event.get("cat") != "cpu_op":
            continue
        arguments = event.get("args") or {}
        external_id = _integer(arguments.get("External id"), default=-1)
        if external_id >= 0:
            parents[external_id] = event

    gpu_events = [
        event
        for event in trace_events
        if event.get("cat") in _GPU_CATEGORIES and event.get("ph") == "X"
    ]
    gpu_events.sort(key=lambda event: (_number(event.get("ts")), str(event.get("name"))))
    identical_events = [
        event for event in gpu_events if "identical_gemm_kernel" in str(event.get("name", ""))
    ]
    if logical_gemm_calls and len(identical_events) != len(logical_gemm_calls):
        raise RuntimeError(
            "logical/GPU identical-GEMM count mismatch: "
            f"logical={len(logical_gemm_calls)}, GPU={len(identical_events)}"
        )

    logical_iterator = iter(logical_gemm_calls)
    occurrence_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    previous_end_us: float | None = None
    for index, event in enumerate(gpu_events):
        arguments = event.get("args") or {}
        name = str(event.get("name") or "unnamed-gpu-event")
        category = str(event.get("cat"))
        start_us = _number(event.get("ts"))
        duration_us = max(0.0, _number(event.get("dur")))
        preceding_gap_us = (
            0.0 if previous_end_us is None else max(0.0, start_us - previous_end_us)
        )
        previous_end_us = max(previous_end_us or start_us, start_us + duration_us)
        external_id = _integer(arguments.get("External id"), default=-1)
        parent = parents.get(external_id)
        parent_name = None if parent is None else str(parent.get("name"))
        parent_arguments = {} if parent is None else (parent.get("args") or {})
        logical_call = (
            next(logical_iterator)
            if "identical_gemm_kernel" in name and logical_gemm_calls
            else None
        )

        grid = _triplet(arguments.get("grid"))
        thread_block = _triplet(arguments.get("block"))
        shared_memory = _integer(arguments.get("shared memory"), default=0)
        if logical_call is not None:
            if grid != logical_call.grid:
                raise RuntimeError(
                    f"logical/GPU grid mismatch at GEMM {index}: "
                    f"logical={logical_call.grid}, GPU={grid}"
                )
            logical_shape = logical_call.logical_shape
            layout_class = logical_call.layout_class
            operand_class = logical_call.operand_class
            semantic_id = logical_call.semantic_id
            executed_flops = logical_call.executed_flops
            memory_bytes = logical_call.estimated_memory_bytes
        else:
            logical_shape = _compact_json(parent_arguments.get("Input Dims"))
            layout_class = _compact_json(parent_arguments.get("Input Strides"))
            operand_class = _parent_operand_class(parent_name, parent_arguments)
            occurrence_key = parent_name or name
            occurrence = occurrence_counts[occurrence_key]
            occurrence_counts[occurrence_key] += 1
            semantic_id = f"{occurrence_key}:occurrence{occurrence}"
            executed_flops = 0
            memory_bytes = _integer(arguments.get("bytes"), default=0)

        signature = BlockSignature(
            kernel_id=f"{category}:{name}",
            grid=grid,
            thread_block=thread_block,
            dynamic_shared_bytes=shared_memory,
            logical_shape=logical_shape,
            layout_class=layout_class,
            operand_class=operand_class,
        )
        block = ComputationBlock(
            signature=signature,
            semantic_id=semantic_id,
            duration_us=duration_us,
            executed_flops=executed_flops,
            memory_bytes=memory_bytes,
        )
        rows.append(
            {
                "index": index,
                "start_us": start_us,
                "preceding_gap_us": preceding_gap_us,
                "external_id": external_id,
                "parent_op": parent_name,
                "block": {
                    **asdict(block),
                    "signature_key": block.signature.key,
                },
            }
        )

    first_start = 0.0 if not rows else rows[0]["start_us"]
    last_end = (
        first_start
        if not rows
        else rows[-1]["start_us"] + rows[-1]["block"]["duration_us"]
    )
    return {
        "cuda_blocks": len(rows),
        "logical_gemm_calls": len(logical_gemm_calls),
        "cuda_busy_us": sum(row["block"]["duration_us"] for row in rows),
        "period_span_us": max(0.0, last_end - first_start),
        "maximum_preceding_gap_us": max(
            (row["preceding_gap_us"] for row in rows),
            default=0.0,
        ),
        "kernel_counts": dict(Counter(row["block"]["signature"]["kernel_id"] for row in rows)),
        "blocks": rows,
    }


def load_profiled_period(path: Path) -> tuple[ComputationBlock, ...]:
    document = json.loads(Path(path).read_text())
    return tuple(_block_from_row(row) for row in document["blocks"])


def enrich_graph_profile(
    graph_path: Path,
    eager_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    enriched = enrich_graph_document(
        json.loads(Path(graph_path).read_text()),
        json.loads(Path(eager_path).read_text()),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2) + "\n")
    return enriched


def enrich_graph_document(
    graph: dict[str, Any],
    eager: dict[str, Any],
) -> dict[str, Any]:
    """Transfer eager parent metadata through a proven physical LCS mapping."""

    graph_rows = graph.get("blocks")
    eager_rows = eager.get("blocks")
    if not isinstance(graph_rows, list) or not isinstance(eager_rows, list):
        raise ValueError("both profiles must contain block lists")
    graph_physical = tuple(
        _physical_only_block(_block_from_row(row), semantic_id=f"graph:{index}")
        for index, row in enumerate(graph_rows)
    )
    eager_physical = tuple(
        _physical_only_block(_block_from_row(row), semantic_id=f"eager:{index}")
        for index, row in enumerate(eager_rows)
    )
    mapping_plan = minimum_cost_supersequence(graph_physical, eager_physical)
    mapping = []
    for slot in mapping_plan.slots:
        if slot.inference_block is None or slot.training_block is None:
            continue
        graph_index = int(slot.inference_block.semantic_id.split(":", 1)[1])
        eager_index = int(slot.training_block.semantic_id.split(":", 1)[1])
        mapping.append((graph_index, eager_index))

    enriched = deepcopy(graph)
    for graph_index, eager_index in mapping:
        graph_row = enriched["blocks"][graph_index]
        eager_row = eager_rows[eager_index]
        graph_payload = graph_row["block"]
        eager_payload = eager_row["block"]
        kernel_id = graph_payload["signature"]["kernel_id"]
        if "identical_gemm_kernel" not in kernel_id:
            for field in ("logical_shape", "layout_class", "operand_class"):
                graph_payload["signature"][field] = eager_payload["signature"][field]
            graph_payload["semantic_id"] = eager_payload["semantic_id"]
            graph_payload["executed_flops"] = eager_payload["executed_flops"]
            graph_payload["memory_bytes"] = eager_payload["memory_bytes"]
            graph_payload["signature_key"] = _signature_from_payload(
                graph_payload["signature"]
            ).key
            graph_row["parent_op"] = eager_row.get("parent_op")
        graph_row["enriched_from_eager_index"] = eager_index

    enriched["enrichment"] = {
        "method": "maximum-cardinality common subsequence over exact kernel/grid/block/shared-memory",
        "graph_blocks": len(graph_rows),
        "eager_blocks": len(eager_rows),
        "physically_mapped_blocks": len(mapping),
        "graph_mapping_fraction": 0.0 if not graph_rows else len(mapping) / len(graph_rows),
        "eager_mapping_fraction": 0.0 if not eager_rows else len(mapping) / len(eager_rows),
        "unmapped_graph_blocks": len(graph_rows) - len(mapping),
        "unmapped_eager_blocks": len(eager_rows) - len(mapping),
    }
    return enriched


def _block_from_row(row: dict[str, Any]) -> ComputationBlock:
    payload = row["block"]
    return ComputationBlock(
        signature=_signature_from_payload(payload["signature"]),
        semantic_id=payload["semantic_id"],
        duration_us=payload["duration_us"],
        executed_flops=payload["executed_flops"],
        memory_bytes=payload["memory_bytes"],
    )


def _signature_from_payload(payload: dict[str, Any]) -> BlockSignature:
    return BlockSignature(
        kernel_id=payload["kernel_id"],
        grid=tuple(payload["grid"]),
        thread_block=tuple(payload["thread_block"]),
        dynamic_shared_bytes=payload["dynamic_shared_bytes"],
        logical_shape=payload["logical_shape"],
        layout_class=payload["layout_class"],
        operand_class=payload["operand_class"],
    )


def _physical_only_block(block: ComputationBlock, *, semantic_id: str) -> ComputationBlock:
    signature = block.signature
    return ComputationBlock(
        signature=BlockSignature(
            kernel_id=signature.kernel_id,
            grid=signature.grid,
            thread_block=signature.thread_block,
            dynamic_shared_bytes=signature.dynamic_shared_bytes,
            logical_shape="",
            layout_class="",
            operand_class="physical-launch-only",
        ),
        semantic_id=semantic_id,
        duration_us=block.duration_us,
        executed_flops=block.executed_flops,
        memory_bytes=block.memory_bytes,
    )


def _parent_operand_class(parent_name: str | None, arguments: dict[str, Any]) -> str:
    input_types = arguments.get("Input type") or []
    normalized_types = sorted({str(value).replace("c10::", "") for value in input_types})
    parent_family = "unattributed" if parent_name is None else parent_name
    return f"unmeasured:{parent_family}:{','.join(normalized_types) or 'unknown'}"


def _triplet(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)):
        return (1, 1, 1)
    values = [_integer(item, default=1) for item in value[:3]]
    values.extend([1] * (3 - len(values)))
    return tuple(max(1, item) for item in values)  # type: ignore[return-value]


def _integer(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _compact_json(value: Any) -> str:
    return "" if value is None else json.dumps(value, separators=(",", ":"), sort_keys=True)
