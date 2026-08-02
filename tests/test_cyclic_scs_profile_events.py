from __future__ import annotations

import pytest

from experiments.llama_cyclic_scs_carrier.profile_events import (
    compact_profile_metadata,
    enrich_graph_document,
    parse_chrome_document,
)
from experiments.llama_identical_microkernel_carrier.backend import (
    IdenticalKernelCall,
    begin_identical_kernel_sequence,
    cancel_identical_kernel_sequence,
    end_identical_kernel_sequence,
)


def chrome_document() -> dict:
    return {
        "traceEvents": [
            {
                "cat": "cpu_op",
                "name": "aten::mm",
                "ph": "X",
                "ts": 10,
                "dur": 5,
                "args": {
                    "External id": 7,
                    "Input Dims": [[64, 32], [32, 16]],
                    "Input Strides": [[32, 1], [16, 1]],
                    "Input type": ["c10::BFloat16", "c10::BFloat16"],
                },
            },
            {
                "cat": "kernel",
                "name": "identical_gemm_kernel",
                "ph": "X",
                "ts": 30,
                "dur": 4,
                "args": {
                    "External id": 7,
                    "grid": [2, 1, 1],
                    "block": [128, 1, 1],
                    "shared memory": 4096,
                },
            },
            {
                "cat": "kernel",
                "name": "vectorized_elementwise_kernel",
                "ph": "X",
                "ts": 40,
                "dur": 2,
                "args": {
                    "External id": 9,
                    "grid": [1, 1, 1],
                    "block": [64, 1, 1],
                },
            },
        ]
    }


def logical_call() -> IdenticalKernelCall:
    return IdenticalKernelCall(
        semantic_id="model.layers.0.q_proj:forward:tile0",
        m=64,
        n=16,
        k=32,
        lhs_strides=(32, 1),
        rhs_strides=(16, 1),
        output_strides=(16, 1),
        grid=(2, 1, 1),
        operand_class="bf16-activation-times-weight",
    )


def test_chrome_parser_preserves_order_geometry_parent_and_logical_gemm() -> None:
    result = parse_chrome_document(chrome_document(), logical_gemm_calls=(logical_call(),))

    assert result["cuda_blocks"] == 2
    assert result["logical_gemm_calls"] == 1
    assert result["cuda_busy_us"] == 6
    assert result["period_span_us"] == 12
    first, second = result["blocks"]
    assert first["block"]["semantic_id"] == "model.layers.0.q_proj:forward:tile0"
    assert first["block"]["signature"]["grid"] == (2, 1, 1)
    assert first["block"]["signature"]["logical_shape"] == "m64-n16-k32"
    assert first["block"]["executed_flops"] == 2 * 64 * 16 * 32
    assert second["preceding_gap_us"] == 6


def test_logical_and_gpu_gemm_count_must_match() -> None:
    with pytest.raises(RuntimeError, match="count mismatch"):
        parse_chrome_document(chrome_document(), logical_gemm_calls=(logical_call(), logical_call()))


def test_logical_and_gpu_grid_must_match() -> None:
    bad = IdenticalKernelCall(
        **{**logical_call().__dict__, "grid": (99, 1, 1)}
    )
    with pytest.raises(RuntimeError, match="grid mismatch"):
        parse_chrome_document(chrome_document(), logical_gemm_calls=(bad,))


def test_sequence_recorder_has_explicit_lifecycle() -> None:
    cancel_identical_kernel_sequence()
    begin_identical_kernel_sequence()
    with pytest.raises(RuntimeError, match="already active"):
        begin_identical_kernel_sequence()
    assert end_identical_kernel_sequence() == ()
    with pytest.raises(RuntimeError, match="not active"):
        end_identical_kernel_sequence()


def test_compact_metadata_excludes_event_payloads() -> None:
    summary = {
        "label": "period",
        "wall_ms_including_profiler": 10.0,
        "chrome_trace": "period.chrome.json",
        "blocks_trace": "period.blocks.json",
        "cuda_blocks": 2,
        "logical_gemm_calls": 1,
        "cuda_busy_us": 6.0,
        "period_span_us": 12.0,
        "maximum_preceding_gap_us": 6.0,
        "kernel_counts": {"a": 1, "b": 1},
        "blocks": [{"large": "payload"}],
    }
    compact = compact_profile_metadata(summary)

    assert compact["unique_kernel_signatures"] == 2
    assert "blocks" not in compact
    assert "kernel_counts" not in compact


def test_graph_enrichment_maps_only_exact_physical_subsequence() -> None:
    eager = parse_chrome_document(chrome_document(), logical_gemm_calls=(logical_call(),))
    graph = parse_chrome_document(chrome_document(), logical_gemm_calls=(logical_call(),))
    graph["blocks"].insert(
        1,
        {
            "index": 1,
            "start_us": 35.0,
            "preceding_gap_us": 1.0,
            "external_id": -1,
            "parent_op": None,
            "block": {
                "signature": {
                    "kernel_id": "kernel:graph_only",
                    "grid": (1, 1, 1),
                    "thread_block": (32, 1, 1),
                    "dynamic_shared_bytes": 0,
                    "logical_shape": "",
                    "layout_class": "",
                    "operand_class": "unmeasured:unattributed:unknown",
                },
                "semantic_id": "graph-only",
                "duration_us": 1.0,
                "executed_flops": 0,
                "memory_bytes": 0,
                "signature_key": "old",
            },
        },
    )
    # Simulate parent metadata being unavailable during graph replay.
    graph["blocks"][2]["block"]["signature"]["logical_shape"] = ""
    graph["blocks"][2]["block"]["signature"]["layout_class"] = ""
    graph["blocks"][2]["block"]["signature"]["operand_class"] = (
        "unmeasured:unattributed:unknown"
    )

    enriched = enrich_graph_document(graph, eager)

    assert enriched["enrichment"]["physically_mapped_blocks"] == 2
    assert enriched["enrichment"]["unmapped_graph_blocks"] == 1
    assert enriched["blocks"][1]["block"]["semantic_id"] == "graph-only"
    assert enriched["blocks"][2]["block"]["signature"]["logical_shape"] == (
        eager["blocks"][1]["block"]["signature"]["logical_shape"]
    )
