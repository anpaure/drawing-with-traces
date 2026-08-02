"""Triton source kept separate so CPU-only hosts never import Triton."""

import triton
import triton.language as tl


_RUNTIME_ARGUMENTS = (
    "M",
    "N",
    "K",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
)


@triton.jit(
    do_not_specialize=_RUNTIME_ARGUMENTS,
    do_not_specialize_on_alignment=_RUNTIME_ARGUMENTS,
)
def identical_gemm_kernel(
    a,
    b,
    c,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One runtime-shape/stride BF16 GEMM kernel used by every carrier path."""

    program = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    program_m = program // grid_n
    program_n = program - program_m * grid_n
    offsets_m = program_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = program_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for reduction_start in tl.range(0, K, BLOCK_K):
        a_pointers = (
            a
            + offsets_m[:, None] * stride_am
            + (reduction_start + offsets_k[None, :]) * stride_ak
        )
        b_pointers = (
            b
            + (reduction_start + offsets_k[:, None]) * stride_bk
            + offsets_n[None, :] * stride_bn
        )
        a_values = tl.load(
            a_pointers,
            mask=(offsets_m[:, None] < M)
            & (reduction_start + offsets_k[None, :] < K),
            other=0.0,
        )
        b_values = tl.load(
            b_pointers,
            mask=(reduction_start + offsets_k[:, None] < K)
            & (offsets_n[None, :] < N),
            other=0.0,
        )
        accumulator = tl.dot(a_values, b_values, accumulator)

    c_pointers = c + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(
        c_pointers,
        accumulator.to(tl.bfloat16),
        mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
    )
