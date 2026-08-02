"""One-binary Triton GEMM carrier for inference and exact training."""

from .backend import (
    begin_identical_kernel_sequence,
    cancel_identical_kernel_sequence,
    end_identical_kernel_sequence,
    identical_kernel_audit,
    install_identical_kernel_coordinator,
    lock_identical_kernel_audit,
    reset_identical_kernel_audit,
    triton_mm_into,
    uninstall_identical_kernel_coordinator,
)

__all__ = [
    "begin_identical_kernel_sequence",
    "cancel_identical_kernel_sequence",
    "end_identical_kernel_sequence",
    "identical_kernel_audit",
    "install_identical_kernel_coordinator",
    "lock_identical_kernel_audit",
    "reset_identical_kernel_audit",
    "triton_mm_into",
    "uninstall_identical_kernel_coordinator",
]
