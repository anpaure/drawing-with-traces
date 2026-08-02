"""One-binary Triton GEMM carrier for inference and exact training."""

from .backend import (
    identical_kernel_audit,
    lock_identical_kernel_audit,
    reset_identical_kernel_audit,
    triton_mm_into,
)

__all__ = [
    "identical_kernel_audit",
    "lock_identical_kernel_audit",
    "reset_identical_kernel_audit",
    "triton_mm_into",
]
