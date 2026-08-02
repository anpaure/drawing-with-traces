"""Audited launcher for one runtime-shape/stride Triton GEMM binary."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any


BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
NUM_WARPS = 4
NUM_STAGES = 3


@dataclass(frozen=True)
class IdenticalKernelAudit:
    locked: bool
    calls_observed: int
    unique_logical_signatures: int
    logical_signature_counts: dict[str, int]
    compiled_object_ids: list[int]
    triton_kernel_hashes: list[str]
    cubin_sha256: list[str]
    kernel_names: list[str]
    block_m: int = BLOCK_M
    block_n: int = BLOCK_N
    block_k: int = BLOCK_K
    num_warps: int = NUM_WARPS
    num_stages: int = NUM_STAGES
    one_compiled_binary_proven: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_locked = False
_calls = 0
_signatures: Counter[str] = Counter()
_compiled_object_ids: set[int] = set()
_triton_hashes: set[str] = set()
_cubin_hashes: set[str] = set()
_kernel_names: set[str] = set()


def reset_identical_kernel_audit() -> None:
    global _locked, _calls
    _locked = False
    _calls = 0
    _signatures.clear()
    _compiled_object_ids.clear()
    _triton_hashes.clear()
    _cubin_hashes.clear()
    _kernel_names.clear()


def _signature(lhs, rhs, output) -> str:
    return (
        f"m{lhs.shape[0]}-n{rhs.shape[1]}-k{lhs.shape[1]}"
        f"-a{lhs.stride(0)}x{lhs.stride(1)}"
        f"-b{rhs.stride(0)}x{rhs.stride(1)}"
        f"-c{output.stride(0)}x{output.stride(1)}"
    )


def identical_kernel_audit() -> IdenticalKernelAudit:
    one_binary = (
        bool(_calls)
        and len(_compiled_object_ids) == 1
        and len(_triton_hashes) == 1
        and len(_cubin_hashes) == 1
        and len(_kernel_names) == 1
    )
    return IdenticalKernelAudit(
        locked=_locked,
        calls_observed=_calls,
        unique_logical_signatures=len(_signatures),
        logical_signature_counts=dict(_signatures),
        compiled_object_ids=sorted(_compiled_object_ids),
        triton_kernel_hashes=sorted(_triton_hashes),
        cubin_sha256=sorted(_cubin_hashes),
        kernel_names=sorted(_kernel_names),
        one_compiled_binary_proven=one_binary,
    )


def lock_identical_kernel_audit() -> IdenticalKernelAudit:
    global _locked
    audit = identical_kernel_audit()
    if not audit.one_compiled_binary_proven:
        raise RuntimeError(
            "identical Triton carrier observed more than one compiled kernel: "
            f"objects={audit.compiled_object_ids}, Triton hashes={audit.triton_kernel_hashes}, "
            f"cubin hashes={audit.cubin_sha256}, names={audit.kernel_names}"
        )
    _locked = True
    return identical_kernel_audit()


def triton_mm_into(lhs, rhs, output) -> None:
    """Compute ``output = lhs @ rhs`` with the one audited BF16 kernel."""

    global _calls
    import torch

    if not (lhs.is_cuda and rhs.is_cuda and output.is_cuda):
        raise ValueError("identical Triton GEMM requires CUDA tensors")
    if lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16:
        raise ValueError("identical Triton GEMM requires BF16 inputs")
    if output.dtype != torch.bfloat16:
        raise ValueError("identical Triton GEMM requires a BF16 output")
    if lhs.ndim != 2 or rhs.ndim != 2 or output.ndim != 2:
        raise ValueError("identical Triton GEMM requires matrices")
    if lhs.shape[1] != rhs.shape[0] or output.shape != (lhs.shape[0], rhs.shape[1]):
        raise ValueError(
            f"incompatible GEMM shapes: {tuple(lhs.shape)}, {tuple(rhs.shape)}, "
            f"output={tuple(output.shape)}"
        )
    if any(stride <= 0 for tensor in (lhs, rhs, output) for stride in tensor.stride()):
        raise ValueError("identical Triton GEMM requires positive matrix strides")

    from .triton_kernel import identical_gemm_kernel

    rows, reduction = lhs.shape
    columns = rhs.shape[1]
    arguments = (
        lhs,
        rhs,
        output,
        rows,
        columns,
        reduction,
        lhs.stride(0),
        lhs.stride(1),
        rhs.stride(0),
        rhs.stride(1),
        output.stride(0),
        output.stride(1),
    )
    grid = (
        ((rows + BLOCK_M - 1) // BLOCK_M)
        * ((columns + BLOCK_N - 1) // BLOCK_N),
    )
    if not _locked:
        compiled = identical_gemm_kernel.warmup(
            *arguments,
            grid=grid,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        _compiled_object_ids.add(id(compiled))
        _triton_hashes.add(str(compiled.hash))
        _cubin_hashes.add(hashlib.sha256(compiled.asm["cubin"]).hexdigest())
        _kernel_names.add(str(compiled.name))
    _calls += 1
    _signatures[_signature(lhs, rhs, output)] += 1
    identical_gemm_kernel[grid](
        *arguments,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
