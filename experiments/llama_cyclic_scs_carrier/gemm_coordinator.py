"""Online executor for one common identical-GEMM shortest supersequence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from ..llama_identical_microkernel_carrier.backend import BLOCK_M, BLOCK_N
from .sequence_alignment import BlockSignature


Mode = Literal["inference", "training"]
Launcher = Callable[..., None]
Allocator = Callable[[BlockSignature, Any | None, Any | None], tuple[Any, Any, Any]]


@dataclass(frozen=True)
class ExecutableGemmSlot:
    index: int
    signature: BlockSignature
    inference_semantic_id: str | None
    training_semantic_id: str | None
    template_semantic_id: str

    def binding(self, mode: Mode) -> str | None:
        return getattr(self, f"{mode}_semantic_id")


@dataclass
class _Template:
    lhs: Any | None = None
    rhs: Any | None = None
    padding_output: Any | None = None


@dataclass
class _OperandAccumulator:
    shape: tuple[int, int]
    strides: tuple[int, int]
    samples: list[Any] = field(default_factory=list)
    values: int = 0


class CommonGemmCoordinator:
    """Insert missing GEMMs while preserving every real call's original order."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        mode: Mode,
        maximum_scratch_bytes: int = 16 << 30,
        seed: int = 0x5C5,
        operand_sample_values_per_call: int = 1024,
        maximum_operand_reservoir_values: int = 262_144,
    ) -> None:
        if mode not in {"inference", "training"}:
            raise ValueError("mode must be inference or training")
        if maximum_scratch_bytes < 1:
            raise ValueError("maximum_scratch_bytes must be positive")
        if operand_sample_values_per_call < 64:
            raise ValueError("operand_sample_values_per_call must be at least 64")
        if maximum_operand_reservoir_values < operand_sample_values_per_call:
            raise ValueError(
                "maximum_operand_reservoir_values must be at least "
                "operand_sample_values_per_call"
            )
        self.manifest_path = Path(manifest_path)
        document = json.loads(self.manifest_path.read_text())
        if document.get("format") != "drawing-with-traces-identical-gemm-scs-v1":
            raise ValueError("unsupported common-GEMM manifest format")
        self.mode = mode
        self.maximum_scratch_bytes = int(maximum_scratch_bytes)
        self.seed = int(seed)
        self.operand_sample_values_per_call = int(operand_sample_values_per_call)
        self.maximum_operand_reservoir_values = int(maximum_operand_reservoir_values)
        padding_statistics = document.get("padding_operand_statistics", {})
        self._padding_operand_statistics = (
            padding_statistics.get(mode, {}).get("by_signature", {})
        )
        self.slots = tuple(_slot_from_dict(row) for row in document["slots"])
        if not self.slots:
            raise ValueError("common-GEMM manifest has no slots")
        for expected_index, slot in enumerate(self.slots):
            if slot.index != expected_index:
                raise ValueError("common-GEMM slot indices must be contiguous")
            if slot.signature.kernel_id != "kernel:identical_gemm_kernel":
                raise ValueError("common-GEMM manifest contains a non-identical-GEMM slot")
            if slot.inference_semantic_id is None and slot.training_semantic_id is None:
                raise ValueError("common-GEMM slot has no real source binding")
        profile = document[f"{mode}_profile"]
        self.real_period_repeats = int(profile["repeats"])
        self.real_gemms_per_base_period = int(profile["gemms_per_period"])
        self._real_slots = tuple(slot for slot in self.slots if slot.binding(mode) is not None)
        expected_real = self.real_period_repeats * self.real_gemms_per_base_period
        if len(self._real_slots) != expected_real:
            raise ValueError(
                f"{mode} manifest projection has {len(self._real_slots)} real GEMMs, "
                f"expected {expected_real}"
            )
        self._templates: dict[str, _Template] = {}
        self._operand_accumulators: dict[
            str, dict[Literal["lhs", "rhs"], _OperandAccumulator]
        ] = {}
        self._real_operand_statistics: dict[str, Any] | None = None
        self._phase: Literal["idle", "collecting", "active"] = "idle"
        self._real_cursor = 0
        self._slot_cursor = 0
        self._last_launcher: Launcher | None = None
        self._prepared = False
        self._periods_completed = 0
        self._scheduled_real_calls = 0
        self._padding_calls = 0
        self._collection_calls = 0
        self._scratch_bytes = 0
        self._manifest_sha256 = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()

    @property
    def expected_real_calls_per_superperiod(self) -> int:
        return len(self._real_slots)

    @property
    def expected_padding_calls_per_superperiod(self) -> int:
        return len(self.slots) - len(self._real_slots)

    def begin_template_collection(self) -> None:
        self._require_phase("idle")
        self._phase = "collecting"
        self._real_cursor = 0
        self._operand_accumulators.clear()
        self._real_operand_statistics = None

    def finish_template_collection(self) -> None:
        self._require_phase("collecting")
        if self._real_cursor != len(self._real_slots):
            raise RuntimeError(
                f"template collection observed {self._real_cursor}/{len(self._real_slots)} "
                f"expected {self.mode} GEMMs"
            )
        self._phase = "idle"

    def prepare_padding_templates(self, *, allocator: Allocator | None = None) -> None:
        self._require_phase("idle")
        allocator = allocator or self._allocate_template
        needed = {
            slot.signature.key: slot.signature
            for slot in self.slots
            if slot.binding(self.mode) is None
        }
        for key, signature in needed.items():
            template = self._templates.setdefault(key, _Template())
            force_measured_synthetic = key in self._padding_operand_statistics
            lhs, rhs, output = allocator(
                signature,
                None if force_measured_synthetic else template.lhs,
                None if force_measured_synthetic else template.rhs,
            )
            self._verify_operands(signature, lhs, rhs, output, semantic_id=slot_operand_semantic(signature))
            template.lhs = lhs
            template.rhs = rhs
            template.padding_output = output
        missing = [
            key
            for key in needed
            if self._templates[key].lhs is None
            or self._templates[key].rhs is None
            or self._templates[key].padding_output is None
        ]
        if missing:
            raise RuntimeError(f"padding templates remain incomplete: {missing[:5]}")
        self._prepared = True

    def template_statistics(self, *, maximum_values: int = 262_144) -> dict[str, Any]:
        """Return bounded empirical BF16 reservoirs from the first real base period.

        Samples are cloned while each real call is live.  This avoids both retaining
        full activation tensors and accidentally measuring a reused/mutated buffer
        after the collection period has completed.
        """

        self._require_phase("idle")
        if maximum_values < 1024:
            raise ValueError("maximum_values must be at least 1024")
        if self._real_operand_statistics is not None:
            return self._real_operand_statistics
        statistics: dict[str, Any] = {}
        bounded_values = min(maximum_values, self.maximum_operand_reservoir_values)
        for key, operands in self._operand_accumulators.items():
            statistics[key] = {
                side: _accumulator_statistics(accumulator, maximum_values=bounded_values)
                for side, accumulator in operands.items()
            }
        self._real_operand_statistics = statistics
        self._operand_accumulators.clear()
        return statistics

    def begin_period(self) -> None:
        self._require_phase("idle")
        if not self._prepared:
            raise RuntimeError("padding templates have not been prepared")
        self._phase = "active"
        self._slot_cursor = 0
        self._last_launcher = None

    def finish_period(self) -> None:
        self._require_phase("active")
        if self._last_launcher is None:
            raise RuntimeError("common schedule observed no real GEMM launcher")
        while self._slot_cursor < len(self.slots):
            slot = self.slots[self._slot_cursor]
            if slot.binding(self.mode) is not None:
                raise RuntimeError(
                    f"period ended before real slot {slot.index} ({slot.binding(self.mode)})"
                )
            self._launch_padding(slot, self._last_launcher)
            self._slot_cursor += 1
        self._phase = "idle"
        self._periods_completed += 1

    def execute_real(
        self,
        lhs: Any,
        rhs: Any,
        output: Any,
        *,
        semantic_id: str,
        launcher: Launcher,
    ) -> None:
        if self._phase == "collecting":
            slot = self._real_slots[self._real_cursor]
            self._verify_real(slot, lhs, rhs, output, semantic_id)
            if self._real_cursor < self.real_gemms_per_base_period:
                self._accumulate_operand_samples(slot.signature, lhs=lhs, rhs=rhs)
            template = self._templates.setdefault(slot.signature.key, _Template())
            template.lhs = lhs
            template.rhs = rhs
            launcher(lhs, rhs, output, semantic_id=semantic_id)
            self._real_cursor += 1
            self._collection_calls += 1
            return
        if self._phase != "active":
            raise RuntimeError(
                f"common-GEMM coordinator received a real call while {self._phase}"
            )
        self._last_launcher = launcher
        while self._slot_cursor < len(self.slots):
            slot = self.slots[self._slot_cursor]
            if slot.binding(self.mode) is not None:
                break
            self._launch_padding(slot, launcher)
            self._slot_cursor += 1
        if self._slot_cursor >= len(self.slots):
            raise RuntimeError(f"unexpected extra {self.mode} GEMM {semantic_id!r}")
        slot = self.slots[self._slot_cursor]
        self._verify_real(slot, lhs, rhs, output, semantic_id)
        template = self._templates.setdefault(slot.signature.key, _Template())
        template.lhs = lhs
        template.rhs = rhs
        launcher(lhs, rhs, output, semantic_id=semantic_id)
        self._slot_cursor += 1
        self._scheduled_real_calls += 1

    def metadata(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": self._manifest_sha256,
            "mode": self.mode,
            "common_slots_per_superperiod": len(self.slots),
            "real_base_period_repeats": self.real_period_repeats,
            "real_gemms_per_base_period": self.real_gemms_per_base_period,
            "real_gemms_per_superperiod": len(self._real_slots),
            "padding_gemms_per_superperiod": self.expected_padding_calls_per_superperiod,
            "templates": len(self._templates),
            "scratch_bytes": self._scratch_bytes,
            "measured_padding_distributions": len(self._padding_operand_statistics),
            "empirical_bf16_padding_distributions": sum(
                "reservoir" in operand
                for statistics in self._padding_operand_statistics.values()
                for operand in (statistics.get("lhs", {}), statistics.get("rhs", {}))
            ),
            "prepared": self._prepared,
            "periods_completed": self._periods_completed,
            "collection_calls": self._collection_calls,
            "scheduled_real_calls": self._scheduled_real_calls,
            "padding_calls": self._padding_calls,
        }

    def _launch_padding(self, slot: ExecutableGemmSlot, launcher: Launcher) -> None:
        template = self._templates.get(slot.signature.key)
        if (
            template is None
            or template.lhs is None
            or template.rhs is None
            or template.padding_output is None
        ):
            raise RuntimeError(f"missing padding template for slot {slot.index}")
        launcher(
            template.lhs,
            template.rhs,
            template.padding_output,
            semantic_id=f"padding:{slot.template_semantic_id}",
        )
        self._padding_calls += 1

    def _verify_real(
        self,
        slot: ExecutableGemmSlot,
        lhs: Any,
        rhs: Any,
        output: Any,
        semantic_id: str,
    ) -> None:
        expected_semantic = slot.binding(self.mode)
        if semantic_id != expected_semantic:
            raise RuntimeError(
                f"common schedule expected {expected_semantic!r} at slot {slot.index}, "
                f"got {semantic_id!r}"
            )
        self._verify_operands(slot.signature, lhs, rhs, output, semantic_id=semantic_id)

    def _verify_operands(
        self,
        expected: BlockSignature,
        lhs: Any,
        rhs: Any,
        output: Any,
        *,
        semantic_id: str,
    ) -> None:
        actual = _signature_from_operands(
            lhs,
            rhs,
            output,
            semantic_id=semantic_id,
            thread_block=expected.thread_block,
            dynamic_shared_bytes=expected.dynamic_shared_bytes,
        )
        if actual != expected:
            raise RuntimeError(
                "common schedule physical-signature mismatch:\n"
                f"expected {expected.key}\n"
                f"actual   {actual.key}"
            )

    def _allocate_template(
        self,
        signature: BlockSignature,
        existing_lhs: Any | None,
        existing_rhs: Any | None,
    ) -> tuple[Any, Any, Any]:
        import torch

        m, n, k = _parse_shape(signature.logical_shape)
        lhs_stride, rhs_stride, output_stride = _parse_layout(signature.layout_class)
        generator = torch.Generator(device="cuda").manual_seed(
            self.seed ^ (int(hashlib.sha256(signature.key.encode()).hexdigest()[:16], 16))
        )

        statistics = self._padding_operand_statistics.get(signature.key)

        def allocate(
            shape,
            stride,
            *,
            mean: float,
            std: float,
            zero_fraction: float,
            reservoir: dict[str, Any] | None = None,
        ):
            tensor = torch.empty_strided(
                shape,
                stride,
                device="cuda",
                dtype=torch.bfloat16,
            )
            if reservoir is not None:
                empirical_values = _decode_bf16_reservoir(reservoir).to(device=tensor.device)
                _fill_from_bf16_reservoir(tensor, empirical_values)
                del empirical_values
            elif std == 0:
                tensor.fill_(mean)
            else:
                tensor.normal_(mean=mean, std=std, generator=generator)
            if reservoir is None and zero_fraction >= 0.005:
                mask = torch.rand_like(tensor, memory_format=torch.preserve_format) < zero_fraction
                tensor.masked_fill_(mask, 0)
                del mask
            self._account_scratch(tensor)
            return tensor

        if statistics is None:
            lhs_scale, rhs_scale = _operand_scales(signature.operand_class)
            lhs_distribution = {"mean": 0.0, "std": lhs_scale, "zero_fraction": 0.0}
            rhs_distribution = {"mean": 0.0, "std": rhs_scale, "zero_fraction": 0.0}
        else:
            lhs_distribution = _validated_distribution(
                statistics["lhs"],
                expected_shape=(m, k),
                expected_strides=lhs_stride,
            )
            rhs_distribution = _validated_distribution(
                statistics["rhs"],
                expected_shape=(k, n),
                expected_strides=rhs_stride,
            )
        lhs = (
            existing_lhs
            if existing_lhs is not None
            else allocate((m, k), lhs_stride, **lhs_distribution)
        )
        rhs = (
            existing_rhs
            if existing_rhs is not None
            else allocate((k, n), rhs_stride, **rhs_distribution)
        )
        output = torch.empty_strided(
            (m, n),
            output_stride,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output.zero_()
        self._account_scratch(output)
        return lhs, rhs, output

    def _accumulate_operand_samples(
        self,
        signature: BlockSignature,
        *,
        lhs: Any,
        rhs: Any,
    ) -> None:
        operands = self._operand_accumulators.setdefault(signature.key, {})
        for side, tensor in (("lhs", lhs), ("rhs", rhs)):
            accumulator = operands.get(side)
            shape = tuple(map(int, tensor.shape))
            strides = tuple(map(int, tensor.stride()))
            if accumulator is None:
                accumulator = _OperandAccumulator(shape=shape, strides=strides)
                operands[side] = accumulator
            elif accumulator.shape != shape or accumulator.strides != strides:
                raise RuntimeError(
                    f"real {side} operand layout changed within signature {signature.key}"
                )
            remaining = self.maximum_operand_reservoir_values - accumulator.values
            if remaining <= 0:
                continue
            sample = _sample_bf16_values(
                tensor,
                maximum_values=min(self.operand_sample_values_per_call, remaining),
            )
            accumulator.samples.append(sample)
            accumulator.values += int(sample.numel())

    def _account_scratch(self, tensor: Any) -> None:
        storage_bytes = int(tensor.untyped_storage().nbytes())
        self._scratch_bytes += storage_bytes
        if self._scratch_bytes > self.maximum_scratch_bytes:
            raise RuntimeError(
                f"common-schedule scratch allocation exceeded "
                f"{self.maximum_scratch_bytes / (1 << 30):.2f} GiB; "
                "raise maximum_scratch_bytes only after checking H100 headroom"
            )

    def _require_phase(self, expected: str) -> None:
        if self._phase != expected:
            raise RuntimeError(f"expected coordinator phase {expected}, got {self._phase}")


def _slot_from_dict(row: dict[str, Any]) -> ExecutableGemmSlot:
    payload = row["signature"]
    return ExecutableGemmSlot(
        index=int(row["index"]),
        signature=BlockSignature(
            kernel_id=payload["kernel_id"],
            grid=tuple(payload["grid"]),
            thread_block=tuple(payload["thread_block"]),
            dynamic_shared_bytes=int(payload["dynamic_shared_bytes"]),
            logical_shape=payload["logical_shape"],
            layout_class=payload["layout_class"],
            operand_class=payload["operand_class"],
        ),
        inference_semantic_id=row["inference_semantic_id"],
        training_semantic_id=row["training_semantic_id"],
        template_semantic_id=row["template_semantic_id"],
    )


def _signature_from_operands(
    lhs: Any,
    rhs: Any,
    output: Any,
    *,
    semantic_id: str,
    thread_block: tuple[int, int, int],
    dynamic_shared_bytes: int,
) -> BlockSignature:
    m, k = map(int, lhs.shape)
    rhs_k, n = map(int, rhs.shape)
    if rhs_k != k or tuple(map(int, output.shape)) != (m, n):
        raise RuntimeError("coordinator received incompatible GEMM operands")
    return BlockSignature(
        kernel_id="kernel:identical_gemm_kernel",
        grid=(
            ((m + BLOCK_M - 1) // BLOCK_M) * ((n + BLOCK_N - 1) // BLOCK_N),
            1,
            1,
        ),
        thread_block=thread_block,
        dynamic_shared_bytes=dynamic_shared_bytes,
        logical_shape=f"m{m}-n{n}-k{k}",
        layout_class=(
            f"a{lhs.stride(0)}x{lhs.stride(1)}"
            f"-b{rhs.stride(0)}x{rhs.stride(1)}"
            f"-c{output.stride(0)}x{output.stride(1)}"
        ),
        operand_class=_operand_class(semantic_id),
    )


def _operand_class(semantic_id: str) -> str:
    lowered = semantic_id.lower()
    if ":dx" in lowered:
        return "bf16-gradient-times-weight"
    if ":dw" in lowered:
        return "bf16-activation-times-gradient"
    if ":forward" in lowered:
        return "bf16-activation-times-weight"
    return "bf16-dense-unclassified"


def slot_operand_semantic(signature: BlockSignature) -> str:
    if signature.operand_class == "bf16-gradient-times-weight":
        return "padding:synthetic:dx"
    if signature.operand_class == "bf16-activation-times-gradient":
        return "padding:synthetic:dw"
    if signature.operand_class == "bf16-activation-times-weight":
        return "padding:synthetic:forward"
    return "padding:synthetic"


def _parse_shape(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"m(\d+)-n(\d+)-k(\d+)", value)
    if match is None:
        raise ValueError(f"cannot parse GEMM shape {value!r}")
    return tuple(map(int, match.groups()))  # type: ignore[return-value]


def _parse_layout(value: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    match = re.fullmatch(r"a(\d+)x(\d+)-b(\d+)x(\d+)-c(\d+)x(\d+)", value)
    if match is None:
        raise ValueError(f"cannot parse GEMM layout {value!r}")
    values = tuple(map(int, match.groups()))
    return values[:2], values[2:4], values[4:6]


def _operand_scales(operand_class: str) -> tuple[float, float]:
    if operand_class == "bf16-gradient-times-weight":
        return 0.01, 0.02
    if operand_class == "bf16-activation-times-gradient":
        return 1.0, 0.01
    return 1.0, 0.02


def _validated_distribution(
    statistics: dict[str, Any],
    *,
    expected_shape: tuple[int, int],
    expected_strides: tuple[int, int],
) -> dict[str, Any]:
    if tuple(statistics["shape"]) != expected_shape:
        raise ValueError(
            f"operand-statistics shape {statistics['shape']} does not match {expected_shape}"
        )
    if tuple(statistics["strides"]) != expected_strides:
        raise ValueError(
            f"operand-statistics strides {statistics['strides']} do not match "
            f"{expected_strides}"
        )
    distribution = {
        "mean": float(statistics["mean"]),
        "std": float(statistics["std"]),
        "zero_fraction": float(statistics["zero_fraction"]),
    }
    if not all(math.isfinite(value) for value in distribution.values()):
        raise ValueError("operand statistics must be finite")
    if distribution["std"] < 0 or not 0 <= distribution["zero_fraction"] <= 1:
        raise ValueError("invalid operand standard deviation or zero fraction")
    reservoir = statistics.get("reservoir")
    if reservoir is not None:
        _validate_bf16_reservoir(reservoir)
        distribution["reservoir"] = reservoir
    return distribution


def _sample_bf16_values(tensor: Any, *, maximum_values: int) -> Any:
    """Clone a bounded logical-grid sample without synchronizing it to the CPU."""

    import torch

    if tensor.ndim != 2:
        raise RuntimeError(f"expected a matrix operand, got rank {tensor.ndim}")
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(f"expected a BF16 operand, got {tensor.dtype}")
    if maximum_values < 1:
        raise ValueError("maximum_values must be positive")
    rows, columns = map(int, tensor.shape)
    target_extent = max(1, int(math.sqrt(maximum_values)))
    row_step = max(1, math.ceil(rows / target_extent))
    column_step = max(1, math.ceil(columns / target_extent))
    with torch.no_grad():
        sample = tensor.detach()[::row_step, ::column_step].contiguous().reshape(-1)
        return sample[:maximum_values].clone()


def _accumulator_statistics(
    accumulator: _OperandAccumulator,
    *,
    maximum_values: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if not accumulator.samples:
        raise RuntimeError("real operand accumulator contains no samples")
    with torch.no_grad():
        sample = torch.cat(accumulator.samples)[:maximum_values]
        finite = torch.isfinite(sample)
        finite_sample = sample[finite]
        if finite_sample.numel() == 0:
            raise RuntimeError("real operand sample contains no finite values")
        finite_values = finite_sample.float()
        quantiles = torch.quantile(
            finite_values,
            torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=finite_values.device),
        )
        cpu_sample = finite_sample.contiguous().cpu()

    raw_bits = np.asarray(cpu_sample.view(torch.uint16).numpy(), dtype="<u2")
    raw = raw_bits.tobytes()
    compressed = zlib.compress(raw, level=9)
    xor_bits = np.bitwise_xor(raw_bits[1:], raw_bits[:-1])
    if xor_bits.size:
        adjacent_hamming_fraction = float(
            np.unpackbits(xor_bits.view(np.uint8)).reshape(-1, 16).sum(axis=1).mean() / 16
        )
    else:
        adjacent_hamming_fraction = 0.0
    bit_positions = np.arange(16, dtype=np.uint16)
    bit_one_fractions = ((raw_bits[:, None] >> bit_positions) & 1).mean(axis=0)
    exponent_histogram = np.bincount(
        ((raw_bits >> 7) & 0xFF).astype(np.int64),
        minlength=256,
    )
    reservoir = {
        "encoding": "base64-zlib-little-endian-u16",
        "dtype": "bfloat16",
        "values": int(raw_bits.size),
        "uncompressed_bytes": len(raw),
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(compressed).decode("ascii"),
    }
    return {
        "shape": list(accumulator.shape),
        "strides": list(accumulator.strides),
        "sample_values": int(sample.numel()),
        "finite_fraction": float(finite.float().mean()),
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(unbiased=False)),
        "mean_absolute": float(finite_values.abs().mean()),
        "zero_fraction": float((finite_values == 0).float().mean()),
        "minimum": float(finite_values.min()),
        "maximum": float(finite_values.max()),
        "quantiles": [float(value) for value in quantiles],
        "bf16_sign_one_fraction": float(((raw_bits >> 15) & 1).mean()),
        "bf16_bit_one_fractions_lsb_to_msb": [float(value) for value in bit_one_fractions],
        "bf16_exponent_histogram": [int(value) for value in exponent_histogram],
        "bf16_adjacent_hamming_fraction": adjacent_hamming_fraction,
        "reservoir": reservoir,
    }


def _validate_bf16_reservoir(reservoir: dict[str, Any]) -> bytes:
    if reservoir.get("encoding") != "base64-zlib-little-endian-u16":
        raise ValueError("unsupported empirical BF16 reservoir encoding")
    if reservoir.get("dtype") != "bfloat16":
        raise ValueError("empirical operand reservoir must contain BF16 values")
    values = int(reservoir.get("values", 0))
    if values < 1:
        raise ValueError("empirical BF16 reservoir must contain at least one value")
    try:
        raw = zlib.decompress(base64.b64decode(reservoir["data"], validate=True))
    except (KeyError, ValueError, binascii.Error, zlib.error) as exc:
        raise ValueError("invalid empirical BF16 reservoir payload") from exc
    if len(raw) != values * 2 or len(raw) != int(reservoir.get("uncompressed_bytes", -1)):
        raise ValueError("empirical BF16 reservoir byte count is inconsistent")
    if hashlib.sha256(raw).hexdigest() != reservoir.get("sha256"):
        raise ValueError("empirical BF16 reservoir checksum mismatch")
    return raw


def _decode_bf16_reservoir(reservoir: dict[str, Any]) -> Any:
    import numpy as np
    import torch

    raw = _validate_bf16_reservoir(reservoir)
    bits = np.frombuffer(raw, dtype="<u2").copy()
    return torch.from_numpy(bits).view(torch.bfloat16)


def _fill_from_bf16_reservoir(
    tensor: Any,
    reservoir: Any,
    *,
    maximum_chunk_values: int = 1 << 20,
) -> None:
    """Fill a possibly-strided matrix by replaying empirical BF16 values in order."""

    import torch

    if tensor.ndim != 2 or reservoir.ndim != 1:
        raise ValueError("empirical replay expects a matrix and a one-dimensional reservoir")
    if tensor.dtype != torch.bfloat16 or reservoir.dtype != torch.bfloat16:
        raise ValueError("empirical replay supports BF16 tensors only")
    if tensor.device != reservoir.device:
        raise ValueError("empirical replay tensor and reservoir must be on the same device")
    if reservoir.numel() < 1:
        raise ValueError("empirical replay reservoir cannot be empty")
    if maximum_chunk_values < 1:
        raise ValueError("maximum_chunk_values must be positive")

    rows, columns = map(int, tensor.shape)
    rows_per_chunk = max(1, maximum_chunk_values // max(columns, 1))
    offset = 0
    with torch.no_grad():
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            values = (row_end - row_start) * columns
            indices = torch.arange(values, device=tensor.device, dtype=torch.int64)
            indices.add_(offset).remainder_(int(reservoir.numel()))
            chunk = reservoir.index_select(0, indices).reshape(row_end - row_start, columns)
            tensor[row_start:row_end].copy_(chunk)
            offset = (offset + values) % int(reservoir.numel())
