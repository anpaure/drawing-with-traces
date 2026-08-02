from __future__ import annotations

import pytest

from experiments.llama_cyclic_scs_carrier.sequence_alignment import (
    BlockSignature,
    ComputationBlock,
    PaddingCostModel,
    align_cyclic_periods,
    minimum_cost_supersequence,
    search_superperiod,
)


def block(
    name: str,
    *,
    duration_us: float = 1.0,
    operand_class: str = "normal-bf16",
    flops: int = 100,
) -> ComputationBlock:
    return ComputationBlock(
        signature=BlockSignature(
            kernel_id=name,
            grid=(4, 1, 1),
            thread_block=(128, 1, 1),
            logical_shape="m64-n64-k32",
            layout_class="row-column",
            operand_class=operand_class,
        ),
        semantic_id=name,
        duration_us=duration_us,
        executed_flops=flops,
        memory_bytes=256,
    )


def keys(blocks) -> list[str]:
    return [item.signature.key for item in blocks]


def test_linear_scs_preserves_both_real_programs() -> None:
    a, b, c, d = (block(name) for name in "ABCD")
    plan = minimum_cost_supersequence((a, b, c), (a, d, c))

    assert keys(plan.real_blocks("inference")) == keys((a, b, c))
    assert keys(plan.real_blocks("training")) == keys((a, d, c))
    assert len(plan.slots) == 4
    assert plan.matched_blocks == 2
    assert plan.accounting("inference").padding_blocks == 1
    assert plan.accounting("training").padding_blocks == 1
    plan.assert_valid()


def test_weighted_crossing_match_preserves_expensive_block() -> None:
    expensive = block("A", duration_us=100.0)
    cheap = block("B", duration_us=1.0)
    model = PaddingCostModel(launch_weight=0.0, duration_weight_per_us=1.0)

    plan = minimum_cost_supersequence(
        (expensive, cheap),
        (cheap, expensive),
        cost_model=model,
    )

    assert [slot.signature.kernel_id for slot in plan.slots] == ["B", "A", "B"]
    assert plan.padding_cost == pytest.approx(2.0)
    assert plan.matched_blocks == 1


def test_operand_distribution_is_part_of_physical_identity() -> None:
    real = block("gemm", operand_class="activation-bf16")
    zeros = block("gemm", operand_class="all-zero")
    plan = minimum_cost_supersequence((real,), (zeros,))

    assert len(plan.slots) == 2
    assert plan.matched_blocks == 0
    assert plan.accounting("inference").padding_blocks == 1
    assert plan.accounting("training").padding_blocks == 1


def test_cyclic_relative_rotation_eliminates_artificial_boundary_padding() -> None:
    a, b, c = (block(name) for name in "ABC")
    plan = align_cyclic_periods((a, b, c), (c, a, b))

    assert len(plan.slots) == 3
    assert plan.matched_blocks == 3
    assert plan.padding_cost == 0
    assert plan.training_rotation == 1
    assert plan.exhaustive_rotation_search


def test_superperiod_search_finds_two_inference_cycles_per_training_cycle() -> None:
    a, b = (block(name) for name in "AB")
    plan = search_superperiod(
        (a, b),
        (a, b, a, b),
        inference_repeat_candidates=(1, 2, 3),
        training_repeat_candidates=(1,),
    )

    assert plan.inference_repeats == 2
    assert plan.training_repeats == 1
    assert plan.padding_cost == 0
    assert plan.maximum_padding_duration_fraction == 0


def test_long_cycle_records_nonexhaustive_anchor_search() -> None:
    period = tuple(block(chr(65 + index % 4)) for index in range(20))
    plan = align_cyclic_periods(
        period,
        period[7:] + period[:7],
        exhaustive_rotation_limit=4,
        maximum_rotation_candidates=8,
    )

    assert not plan.exhaustive_rotation_search
    assert plan.searched_rotations <= 8
    plan.assert_valid()


def test_invalid_cost_model_and_empty_sequences_fail_clearly() -> None:
    with pytest.raises(ValueError, match="at least one"):
        PaddingCostModel(launch_weight=0)
    with pytest.raises(ValueError, match="non-empty"):
        minimum_cost_supersequence((), (block("A"),))
