"""Performance-aware alignment of continuous GPU computation-block streams.

The two real programs are immutable subsequences.  Alignment may only insert a
physical block whose output is discarded by that mode.  Consequently both
modes can execute one common physical schedule without reordering either real
program's dependencies.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence


@dataclass(frozen=True, order=True)
class BlockSignature:
    """Observable identity required before two block occurrences may be shared.

    `operand_class` is deliberately part of equality.  Kernel names and launch
    geometry alone are insufficient when operand values change switching
    activity and therefore power.
    """

    kernel_id: str
    grid: tuple[int, int, int] = (1, 1, 1)
    thread_block: tuple[int, int, int] = (1, 1, 1)
    dynamic_shared_bytes: int = 0
    logical_shape: str = ""
    layout_class: str = ""
    operand_class: str = "unknown"

    def __post_init__(self) -> None:
        if not self.kernel_id:
            raise ValueError("kernel_id cannot be empty")
        if any(value < 1 for value in (*self.grid, *self.thread_block)):
            raise ValueError("grid and thread-block dimensions must be positive")
        if self.dynamic_shared_bytes < 0:
            raise ValueError("dynamic_shared_bytes cannot be negative")
        if not self.operand_class:
            raise ValueError("operand_class cannot be empty")

    @property
    def key(self) -> str:
        return "|".join(
            (
                self.kernel_id,
                "x".join(map(str, self.grid)),
                "x".join(map(str, self.thread_block)),
                str(self.dynamic_shared_bytes),
                self.logical_shape,
                self.layout_class,
                self.operand_class,
            )
        )


@dataclass(frozen=True)
class ComputationBlock:
    """One occurrence in a real inference or training period."""

    signature: BlockSignature
    semantic_id: str
    duration_us: float
    executed_flops: int = 0
    memory_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.semantic_id:
            raise ValueError("semantic_id cannot be empty")
        if not math.isfinite(self.duration_us) or self.duration_us < 0:
            raise ValueError("duration_us must be finite and non-negative")
        if self.executed_flops < 0:
            raise ValueError("executed_flops cannot be negative")
        if self.memory_bytes < 0:
            raise ValueError("memory_bytes cannot be negative")


@dataclass(frozen=True)
class PaddingCostModel:
    """Scalar cost for inserting a physical block into the other program.

    Set only `launch_weight=1` for a literal minimum-block SCS.  Measured GPU
    scheduling should normally use duration as the dominant term, with launch,
    FLOP, and memory terms as deterministic tie-breakers.
    """

    launch_weight: float = 1.0
    duration_weight_per_us: float = 0.0
    flop_weight_per_tflop: float = 0.0
    memory_weight_per_gib: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.launch_weight,
            self.duration_weight_per_us,
            self.flop_weight_per_tflop,
            self.memory_weight_per_gib,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("padding-cost weights must be finite and non-negative")
        if not any(values):
            raise ValueError("at least one padding-cost weight must be positive")

    def insertion_cost(self, block: ComputationBlock) -> float:
        return (
            self.launch_weight
            + self.duration_weight_per_us * block.duration_us
            + self.flop_weight_per_tflop * block.executed_flops / 1e12
            + self.memory_weight_per_gib * block.memory_bytes / (1 << 30)
        )


@dataclass(frozen=True)
class ScheduleSlot:
    """One physical launch in the common schedule and its semantic bindings."""

    signature: BlockSignature
    inference_block: ComputationBlock | None
    training_block: ComputationBlock | None

    def __post_init__(self) -> None:
        if self.inference_block is None and self.training_block is None:
            raise ValueError("a common-schedule slot needs at least one real source block")
        for block in (self.inference_block, self.training_block):
            if block is not None and block.signature != self.signature:
                raise ValueError("slot block does not match its physical signature")

    @property
    def template(self) -> ComputationBlock:
        return self.inference_block or self.training_block  # type: ignore[return-value]

    @property
    def estimated_duration_us(self) -> float:
        return max(
            block.duration_us
            for block in (self.inference_block, self.training_block)
            if block is not None
        )

    @property
    def estimated_flops(self) -> int:
        return max(
            block.executed_flops
            for block in (self.inference_block, self.training_block)
            if block is not None
        )

    @property
    def estimated_memory_bytes(self) -> int:
        return max(
            block.memory_bytes
            for block in (self.inference_block, self.training_block)
            if block is not None
        )


@dataclass(frozen=True)
class ModeAccounting:
    real_blocks: int
    padding_blocks: int
    useful_duration_us: float
    padding_duration_us: float
    useful_flops: int
    padding_flops: int
    useful_memory_bytes: int
    padding_memory_bytes: int

    @property
    def total_blocks(self) -> int:
        return self.real_blocks + self.padding_blocks

    @property
    def padding_fraction_by_duration(self) -> float:
        total = self.useful_duration_us + self.padding_duration_us
        return 0.0 if total == 0 else self.padding_duration_us / total

    @property
    def useful_flop_fraction(self) -> float:
        total = self.useful_flops + self.padding_flops
        return 1.0 if total == 0 else self.useful_flops / total


@dataclass(frozen=True)
class AlignmentPlan:
    """Auditable common schedule for one inference/training superperiod."""

    slots: tuple[ScheduleSlot, ...]
    inference_period: tuple[ComputationBlock, ...]
    training_period: tuple[ComputationBlock, ...]
    inference_repeats: int = 1
    training_repeats: int = 1
    inference_rotation: int = 0
    training_rotation: int = 0
    padding_cost: float = 0.0
    searched_rotations: int = 1
    exhaustive_rotation_search: bool = True

    def __post_init__(self) -> None:
        if self.inference_repeats < 1 or self.training_repeats < 1:
            raise ValueError("period repeat counts must be positive")
        if not self.inference_period or not self.training_period:
            raise ValueError("both real periods must be non-empty")
        if not math.isfinite(self.padding_cost) or self.padding_cost < 0:
            raise ValueError("padding_cost must be finite and non-negative")
        self.assert_valid()

    @property
    def physical_signatures(self) -> tuple[BlockSignature, ...]:
        return tuple(slot.signature for slot in self.slots)

    @property
    def matched_blocks(self) -> int:
        return sum(
            slot.inference_block is not None and slot.training_block is not None
            for slot in self.slots
        )

    def real_blocks(self, mode: Literal["inference", "training"]) -> tuple[ComputationBlock, ...]:
        attribute = f"{mode}_block"
        return tuple(
            block
            for slot in self.slots
            if (block := getattr(slot, attribute)) is not None
        )

    def accounting(self, mode: Literal["inference", "training"]) -> ModeAccounting:
        attribute = f"{mode}_block"
        real_blocks = 0
        padding_blocks = 0
        useful_duration = 0.0
        padding_duration = 0.0
        useful_flops = 0
        padding_flops = 0
        useful_memory = 0
        padding_memory = 0
        for slot in self.slots:
            block = getattr(slot, attribute)
            if block is None:
                padding_blocks += 1
                padding_duration += slot.estimated_duration_us
                padding_flops += slot.estimated_flops
                padding_memory += slot.estimated_memory_bytes
            else:
                real_blocks += 1
                useful_duration += block.duration_us
                useful_flops += block.executed_flops
                useful_memory += block.memory_bytes
        return ModeAccounting(
            real_blocks=real_blocks,
            padding_blocks=padding_blocks,
            useful_duration_us=useful_duration,
            padding_duration_us=padding_duration,
            useful_flops=useful_flops,
            padding_flops=padding_flops,
            useful_memory_bytes=useful_memory,
            padding_memory_bytes=padding_memory,
        )

    @property
    def maximum_padding_duration_fraction(self) -> float:
        return max(
            self.accounting("inference").padding_fraction_by_duration,
            self.accounting("training").padding_fraction_by_duration,
        )

    def assert_valid(self) -> None:
        expected_inference = _repeat_rotated(
            self.inference_period,
            self.inference_repeats,
            self.inference_rotation,
        )
        expected_training = _repeat_rotated(
            self.training_period,
            self.training_repeats,
            self.training_rotation,
        )
        if self.real_blocks("inference") != expected_inference:
            raise ValueError("inference projection does not reproduce the real period")
        if self.real_blocks("training") != expected_training:
            raise ValueError("training projection does not reproduce the real period")
        for mode in ("inference", "training"):
            if len(self.physical_signatures) != self.accounting(mode).total_blocks:
                raise ValueError(f"{mode} does not execute every common-schedule slot")

    def to_dict(self) -> dict[str, object]:
        inference = self.accounting("inference")
        training = self.accounting("training")
        return {
            "inference_period_blocks": len(self.inference_period),
            "training_period_blocks": len(self.training_period),
            "inference_repeats": self.inference_repeats,
            "training_repeats": self.training_repeats,
            "inference_rotation": self.inference_rotation,
            "training_rotation": self.training_rotation,
            "common_schedule_blocks": len(self.slots),
            "matched_blocks": self.matched_blocks,
            "padding_cost": self.padding_cost,
            "searched_rotations": self.searched_rotations,
            "exhaustive_rotation_search": self.exhaustive_rotation_search,
            "maximum_padding_duration_fraction": self.maximum_padding_duration_fraction,
            "inference": asdict(inference),
            "training": asdict(training),
            "physical_signature_keys": [signature.key for signature in self.physical_signatures],
        }


def minimum_cost_supersequence(
    inference: Sequence[ComputationBlock],
    training: Sequence[ComputationBlock],
    *,
    cost_model: PaddingCostModel | None = None,
) -> AlignmentPlan:
    """Return an exact minimum-cost linear SCS and both semantic bindings.

    This is solved as an exact maximum-weight common subsequence with a sparse
    Hunt-Szymanski/Fenwick recurrence.  A match saves the cost of inserting both
    occurrences.  Unlike a dense O(n*m) table, work scales with the number of
    physically compatible pairs and remains practical for multi-thousand-kernel
    periods.
    """

    inference = tuple(inference)
    training = tuple(training)
    if not inference or not training:
        raise ValueError("both sequences must be non-empty")
    cost_model = cost_model or PaddingCostModel()
    inference_costs = tuple(cost_model.insertion_cost(block) for block in inference)
    training_costs = tuple(cost_model.insertion_cost(block) for block in training)
    matched_pairs, saved_cost = _maximum_weight_common_subsequence(
        inference,
        training,
        inference_costs=inference_costs,
        training_costs=training_costs,
    )
    slots: list[ScheduleSlot] = []
    i = 0
    j = 0
    for matched_i, matched_j in matched_pairs:
        _append_interleaved_unmatched(
            slots,
            inference,
            training,
            inference_start=i,
            inference_end=matched_i,
            training_start=j,
            training_end=matched_j,
        )
        slots.append(
            ScheduleSlot(
                signature=inference[matched_i].signature,
                inference_block=inference[matched_i],
                training_block=training[matched_j],
            )
        )
        i = matched_i + 1
        j = matched_j + 1
    _append_interleaved_unmatched(
        slots,
        inference,
        training,
        inference_start=i,
        inference_end=len(inference),
        training_start=j,
        training_end=len(training),
    )

    return AlignmentPlan(
        slots=tuple(slots),
        inference_period=inference,
        training_period=training,
        padding_cost=max(0.0, sum(inference_costs) + sum(training_costs) - saved_cost),
    )


def _maximum_weight_common_subsequence(
    inference: Sequence[ComputationBlock],
    training: Sequence[ComputationBlock],
    *,
    inference_costs: Sequence[float],
    training_costs: Sequence[float],
) -> tuple[tuple[tuple[int, int], ...], float]:
    """Return exact weighted-LCS occurrence pairs in increasing order."""

    training_positions: dict[BlockSignature, list[int]] = {}
    for index, block in enumerate(training):
        training_positions.setdefault(block.signature, []).append(index)

    node_i: list[int] = []
    node_j: list[int] = []
    node_previous: list[int] = []
    node_score: list[float] = []
    node_matches: list[int] = []
    tree = [-1] * (len(training) + 1)

    def better(candidate: int, incumbent: int) -> bool:
        if candidate < 0:
            return False
        if incumbent < 0:
            return True
        candidate_key = (node_score[candidate], node_matches[candidate], -node_j[candidate])
        incumbent_key = (node_score[incumbent], node_matches[incumbent], -node_j[incumbent])
        return candidate_key > incumbent_key

    def query(exclusive_position: int) -> int:
        best = -1
        cursor = exclusive_position
        while cursor > 0:
            if better(tree[cursor], best):
                best = tree[cursor]
            cursor -= cursor & -cursor
        return best

    def update(position: int, node: int) -> None:
        cursor = position + 1
        while cursor < len(tree):
            if better(node, tree[cursor]):
                tree[cursor] = node
            cursor += cursor & -cursor

    for inference_index, inference_block in enumerate(inference):
        # Descending positions prevent two matches from consuming one inference occurrence.
        for training_index in reversed(training_positions.get(inference_block.signature, ())):
            previous = query(training_index)
            score = inference_costs[inference_index] + training_costs[training_index]
            matches = 1
            if previous >= 0:
                score += node_score[previous]
                matches += node_matches[previous]
            node = len(node_i)
            node_i.append(inference_index)
            node_j.append(training_index)
            node_previous.append(previous)
            node_score.append(score)
            node_matches.append(matches)
            update(training_index, node)

    best = query(len(training))
    if best < 0:
        return (), 0.0
    saved_cost = node_score[best]
    pairs = []
    while best >= 0:
        pairs.append((node_i[best], node_j[best]))
        best = node_previous[best]
    pairs.reverse()
    return tuple(pairs), saved_cost


def _append_interleaved_unmatched(
    slots: list[ScheduleSlot],
    inference: Sequence[ComputationBlock],
    training: Sequence[ComputationBlock],
    *,
    inference_start: int,
    inference_end: int,
    training_start: int,
    training_end: int,
) -> None:
    """Merge two unmatched runs without clustering one mode's real operands."""

    i = inference_start
    j = training_start
    inference_count = inference_end - inference_start
    training_count = training_end - training_start
    while i < inference_end or j < training_end:
        take_inference = j >= training_end or (
            i < inference_end
            and (
                training_count == 0
                or (i - inference_start + 1) * training_count
                <= (j - training_start + 1) * inference_count
            )
        )
        if take_inference:
            slots.append(
                ScheduleSlot(
                    signature=inference[i].signature,
                    inference_block=inference[i],
                    training_block=None,
                )
            )
            i += 1
        else:
            slots.append(
                ScheduleSlot(
                    signature=training[j].signature,
                    inference_block=None,
                    training_block=training[j],
                )
            )
            j += 1


def align_cyclic_periods(
    inference_period: Sequence[ComputationBlock],
    training_period: Sequence[ComputationBlock],
    *,
    inference_repeats: int = 1,
    training_repeats: int = 1,
    cost_model: PaddingCostModel | None = None,
    exhaustive_rotation_limit: int = 256,
    maximum_rotation_candidates: int = 64,
) -> AlignmentPlan:
    """Align periodic streams while searching their relative phase.

    One inference phase is fixed without loss of generality: a circular common
    schedule can always be rotated to an inference boundary.  For long periods,
    candidate phases are generated from rare shared anchors to avoid cubic work;
    the returned plan records whether the phase search was exhaustive.
    """

    if inference_repeats < 1 or training_repeats < 1:
        raise ValueError("period repeat counts must be positive")
    if exhaustive_rotation_limit < 1 or maximum_rotation_candidates < 1:
        raise ValueError("rotation limits must be positive")
    inference_period = tuple(inference_period)
    training_period = tuple(training_period)
    if not inference_period or not training_period:
        raise ValueError("both periods must be non-empty")
    cost_model = cost_model or PaddingCostModel()
    inference = inference_period * inference_repeats
    training = training_period * training_repeats
    candidates, exhaustive = _rotation_candidates(
        inference,
        training,
        cost_model=cost_model,
        exhaustive_rotation_limit=exhaustive_rotation_limit,
        maximum_rotation_candidates=maximum_rotation_candidates,
    )

    best: AlignmentPlan | None = None
    best_score: tuple[float, int, float, int] | None = None
    for rotation in candidates:
        rotated_training = _rotate(training, rotation)
        linear = minimum_cost_supersequence(
            inference,
            rotated_training,
            cost_model=cost_model,
        )
        candidate = AlignmentPlan(
            slots=linear.slots,
            inference_period=inference_period,
            training_period=training_period,
            inference_repeats=inference_repeats,
            training_repeats=training_repeats,
            inference_rotation=0,
            training_rotation=rotation % len(training_period),
            padding_cost=linear.padding_cost,
            searched_rotations=len(candidates),
            exhaustive_rotation_search=exhaustive,
        )
        score = (
            candidate.padding_cost,
            sum(candidate.accounting(mode).padding_blocks for mode in ("inference", "training")),
            candidate.maximum_padding_duration_fraction,
            rotation,
        )
        if best_score is None or score < best_score:
            best = candidate
            best_score = score
    assert best is not None
    return best


def search_superperiod(
    inference_period: Sequence[ComputationBlock],
    training_period: Sequence[ComputationBlock],
    *,
    inference_repeat_candidates: Iterable[int],
    training_repeat_candidates: Iterable[int],
    cost_model: PaddingCostModel | None = None,
    maximum_total_real_blocks: int = 4096,
    exhaustive_rotation_limit: int = 256,
    maximum_rotation_candidates: int = 64,
) -> AlignmentPlan:
    """Select the period ratio minimizing the worst per-mode padding fraction."""

    inference_period = tuple(inference_period)
    training_period = tuple(training_period)
    pairs = [
        (inference_repeats, training_repeats)
        for inference_repeats in sorted(set(inference_repeat_candidates))
        for training_repeats in sorted(set(training_repeat_candidates))
        if inference_repeats > 0
        and training_repeats > 0
        and (
            inference_repeats * len(inference_period)
            + training_repeats * len(training_period)
            <= maximum_total_real_blocks
        )
    ]
    if not pairs:
        raise ValueError("no positive repeat pair fits maximum_total_real_blocks")
    plans = [
        align_cyclic_periods(
            inference_period,
            training_period,
            inference_repeats=inference_repeats,
            training_repeats=training_repeats,
            cost_model=cost_model,
            exhaustive_rotation_limit=exhaustive_rotation_limit,
            maximum_rotation_candidates=maximum_rotation_candidates,
        )
        for inference_repeats, training_repeats in pairs
    ]
    return min(
        plans,
        key=lambda plan: (
            plan.maximum_padding_duration_fraction,
            plan.padding_cost
            / (plan.inference_repeats + plan.training_repeats),
            len(plan.slots),
            plan.inference_repeats + plan.training_repeats,
        ),
    )


def _rotate(sequence: Sequence[ComputationBlock], offset: int) -> tuple[ComputationBlock, ...]:
    sequence = tuple(sequence)
    if not sequence:
        return ()
    offset %= len(sequence)
    return sequence[offset:] + sequence[:offset]


def _repeat_rotated(
    period: Sequence[ComputationBlock],
    repeats: int,
    rotation: int,
) -> tuple[ComputationBlock, ...]:
    repeated = tuple(period) * repeats
    return _rotate(repeated, rotation)


def _rotation_candidates(
    inference: Sequence[ComputationBlock],
    training: Sequence[ComputationBlock],
    *,
    cost_model: PaddingCostModel,
    exhaustive_rotation_limit: int,
    maximum_rotation_candidates: int,
) -> tuple[tuple[int, ...], bool]:
    if len(training) <= exhaustive_rotation_limit:
        return tuple(range(len(training))), True

    inference_positions: dict[BlockSignature, list[int]] = {}
    training_positions: dict[BlockSignature, list[int]] = {}
    for index, block in enumerate(inference):
        inference_positions.setdefault(block.signature, []).append(index)
    for index, block in enumerate(training):
        training_positions.setdefault(block.signature, []).append(index)
    shared = set(inference_positions) & set(training_positions)
    ranked = sorted(
        shared,
        key=lambda signature: (
            len(inference_positions[signature]) * len(training_positions[signature]),
            -max(
                cost_model.insertion_cost(inference[index])
                for index in inference_positions[signature]
            ),
            signature.key,
        ),
    )
    candidates = {0}
    for signature in ranked:
        for inference_index in inference_positions[signature]:
            for training_index in training_positions[signature]:
                candidates.add((training_index - inference_index) % len(training))
                if len(candidates) >= maximum_rotation_candidates:
                    return tuple(sorted(candidates)), False
    if len(candidates) < maximum_rotation_candidates:
        stride = max(1, len(training) // maximum_rotation_candidates)
        candidates.update(range(0, len(training), stride))
    return tuple(sorted(candidates)[:maximum_rotation_candidates]), False
