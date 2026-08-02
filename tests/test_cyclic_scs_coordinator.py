from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from experiments.llama_cyclic_scs_carrier.build_gemm_manifest import build_manifest
from experiments.llama_cyclic_scs_carrier.gemm_coordinator import (
    CommonGemmCoordinator,
    _OperandAccumulator,
    _accumulator_statistics,
    _decode_bf16_reservoir,
    _fill_from_bf16_reservoir,
    _validate_bf16_reservoir,
)
from experiments.llama_cyclic_scs_carrier.sequence_alignment import (
    BlockSignature,
    ComputationBlock,
)


def gemm(name: str, semantic_id: str) -> ComputationBlock:
    dimension = {"A": 16, "B": 32, "C": 48}[name]
    return ComputationBlock(
        signature=BlockSignature(
            kernel_id="kernel:identical_gemm_kernel",
            grid=(1, 1, 1),
            thread_block=(128, 1, 1),
            dynamic_shared_bytes=8192,
            logical_shape=f"m64-n{dimension}-k32",
            layout_class=f"a32x1-b{dimension}x1-c{dimension}x1",
            operand_class="bf16-activation-times-weight",
        ),
        semantic_id=semantic_id,
        duration_us=2.0,
        executed_flops=100,
        memory_bytes=64,
    )


def write_profile(path: Path, blocks: list[ComputationBlock]) -> None:
    rows = []
    for block in blocks:
        rows.append(
            {
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
                }
            }
        )
    path.write_text(json.dumps({"blocks": rows}))


def make_manifest(tmp_path: Path) -> Path:
    inference = tmp_path / "inference.json"
    training = tmp_path / "training.json"
    write_profile(inference, [gemm("A", "i-a:forward"), gemm("B", "i-b:forward")])
    write_profile(training, [gemm("A", "t-a:forward"), gemm("C", "t-c:forward")])
    document = build_manifest(
        inference,
        training,
        inference_repeats=1,
        training_repeats=1,
    )
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(document))
    return output


def operands(signature: BlockSignature) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n, k = map(int, re.fullmatch(r"m(\d+)-n(\d+)-k(\d+)", signature.logical_shape).groups())
    values = tuple(
        map(
            int,
            re.fullmatch(
                r"a(\d+)x(\d+)-b(\d+)x(\d+)-c(\d+)x(\d+)",
                signature.layout_class,
            ).groups(),
        )
    )
    tensors = (
        torch.empty_strided((m, k), values[:2], dtype=torch.bfloat16),
        torch.empty_strided((k, n), values[2:4], dtype=torch.bfloat16),
        torch.empty_strided((m, n), values[4:6], dtype=torch.bfloat16),
    )
    for index, tensor in enumerate(tensors):
        tensor.fill_(index + 1)
    return tensors


def exercise_mode(manifest: Path, mode: str) -> tuple[list[str], dict]:
    coordinator = CommonGemmCoordinator(manifest, mode=mode)
    launches: list[str] = []

    def launcher(lhs, rhs, output, *, semantic_id: str) -> None:
        del output, semantic_id
        launches.append(f"{lhs.shape}x{rhs.shape}")

    coordinator.begin_template_collection()
    for slot in coordinator._real_slots:
        coordinator.execute_real(
            *operands(slot.signature),
            semantic_id=slot.binding(mode),
            launcher=launcher,
        )
    coordinator.finish_template_collection()
    launches.clear()
    coordinator.prepare_padding_templates(
        allocator=lambda signature, lhs, rhs: (
            lhs or operands(signature)[0],
            rhs or operands(signature)[1],
            operands(signature)[2],
        )
    )
    coordinator.begin_period()
    for slot in coordinator._real_slots:
        coordinator.execute_real(
            *operands(slot.signature),
            semantic_id=slot.binding(mode),
            launcher=launcher,
        )
    coordinator.finish_period()
    return launches, coordinator.metadata()


def test_both_modes_launch_the_same_common_sequence(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    inference_launches, inference_audit = exercise_mode(manifest, "inference")
    training_launches, training_audit = exercise_mode(manifest, "training")

    assert len(inference_launches) == len(training_launches) == 3
    assert inference_launches == training_launches
    assert inference_audit["common_slots_per_superperiod"] == 3
    assert training_audit["common_slots_per_superperiod"] == 3
    assert inference_audit["padding_calls"] == 1
    assert training_audit["padding_calls"] == 1
    assert inference_audit["periods_completed"] == 1


def test_semantic_order_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    coordinator = CommonGemmCoordinator(manifest, mode="inference")
    coordinator.begin_template_collection()
    first = coordinator._real_slots[0]
    with pytest.raises(RuntimeError, match="expected"):
        coordinator.execute_real(
            *operands(first.signature),
            semantic_id="wrong:forward",
            launcher=lambda *args, **kwargs: None,
        )


def test_empirical_bf16_reservoir_round_trip_is_bit_exact() -> None:
    source = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 2**-20, 3.140625, 0.5, -0.5],
        dtype=torch.bfloat16,
    )
    accumulator = _OperandAccumulator(
        shape=(2, 4),
        strides=(4, 1),
        samples=[source.clone()],
        values=source.numel(),
    )

    statistics = _accumulator_statistics(accumulator, maximum_values=source.numel())
    decoded = _decode_bf16_reservoir(statistics["reservoir"])

    assert torch.equal(decoded.view(torch.uint16), source.view(torch.uint16))
    assert statistics["bf16_bit_one_fractions_lsb_to_msb"]
    assert len(statistics["bf16_exponent_histogram"]) == 256


def test_empirical_replay_fills_noncontiguous_matrix_in_logical_order() -> None:
    reservoir = torch.tensor([1.0, -2.0, 4.0], dtype=torch.bfloat16)
    target = torch.empty_strided((3, 5), (1, 3), dtype=torch.bfloat16)

    _fill_from_bf16_reservoir(target, reservoir, maximum_chunk_values=7)

    expected = reservoir.repeat(5).reshape(3, 5)
    assert torch.equal(target, expected)
    assert target.stride() == (1, 3)


def test_empirical_reservoir_rejects_checksum_corruption() -> None:
    source = torch.arange(8, dtype=torch.float32).to(torch.bfloat16)
    accumulator = _OperandAccumulator(
        shape=(2, 4),
        strides=(4, 1),
        samples=[source],
        values=source.numel(),
    )
    reservoir = _accumulator_statistics(accumulator, maximum_values=8)["reservoir"]
    reservoir["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checksum"):
        _validate_bf16_reservoir(reservoir)
