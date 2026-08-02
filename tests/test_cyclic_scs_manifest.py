from __future__ import annotations

import json
from pathlib import Path

from experiments.llama_cyclic_scs_carrier.build_gemm_manifest import build_manifest
from experiments.llama_cyclic_scs_carrier.sequence_alignment import (
    BlockSignature,
    ComputationBlock,
)


def write_profile(path: Path, blocks: list[ComputationBlock]) -> None:
    rows = []
    for index, block in enumerate(blocks):
        rows.append(
            {
                "index": index,
                "block": {
                    "signature": {
                        "kernel_id": block.signature.kernel_id,
                        "grid": block.signature.grid,
                        "thread_block": block.signature.thread_block,
                        "dynamic_shared_bytes": block.signature.dynamic_shared_bytes,
                        "logical_shape": block.signature.logical_shape,
                        "layout_class": block.signature.layout_class,
                        "operand_class": block.signature.operand_class,
                    },
                    "semantic_id": block.semantic_id,
                    "duration_us": block.duration_us,
                    "executed_flops": block.executed_flops,
                    "memory_bytes": block.memory_bytes,
                },
            }
        )
    path.write_text(json.dumps({"blocks": rows}))


def gemm(name: str, semantic_id: str) -> ComputationBlock:
    return ComputationBlock(
        signature=BlockSignature(
            kernel_id="kernel:identical_gemm_kernel",
            grid=(1 if name == "A" else 2, 1, 1),
            thread_block=(128, 1, 1),
            dynamic_shared_bytes=8192,
            logical_shape=f"shape-{name}",
            layout_class="contiguous",
            operand_class="bf16-activation-times-weight",
        ),
        semantic_id=semantic_id,
        duration_us=2.0,
        executed_flops=100,
        memory_bytes=64,
    )


def test_manifest_contains_executable_slot_bindings(tmp_path: Path) -> None:
    inference = tmp_path / "inference.json"
    training = tmp_path / "training.json"
    write_profile(inference, [gemm("A", "i-a"), gemm("B", "i-b")])
    write_profile(training, [gemm("A", "t-a")])

    manifest = build_manifest(
        inference,
        training,
        inference_repeats=1,
        training_repeats=1,
    )

    assert manifest["format"] == "drawing-with-traces-identical-gemm-scs-v1"
    assert manifest["plan"]["common_schedule_blocks"] == 2
    assert manifest["plan"]["matched_blocks"] == 1
    assert manifest["slots"][0]["inference_semantic_id"] == "i-a"
    assert manifest["slots"][0]["training_semantic_id"] == "t-a"
    assert manifest["slots"][1]["training_semantic_id"] is None
