from __future__ import annotations

from pathlib import Path

import pytest

from experiments.llama_identical_microkernel_carrier.backend import (
    identical_kernel_audit,
    lock_identical_kernel_audit,
    reset_identical_kernel_audit,
)
from experiments.llama_shared_kernel_training_carrier.shared_carrier import (
    SharedCarrierConfig,
)
from experiments.llama_strict_inference_shaped_training.strict_workloads import (
    StrictWorkloadConfig,
)


def test_empty_identical_kernel_audit_cannot_be_locked() -> None:
    reset_identical_kernel_audit()
    audit = identical_kernel_audit()
    assert not audit.locked
    assert not audit.one_compiled_binary_proven
    assert audit.calls_observed == 0
    with pytest.raises(RuntimeError, match="more than one compiled kernel"):
        lock_identical_kernel_audit()


def test_shared_carrier_accepts_only_named_gemm_backends() -> None:
    config = SharedCarrierConfig(gemm_backend="identical-triton")
    assert config.metadata()["gemm_backend"] == "identical-triton"
    with pytest.raises(ValueError, match="gemm_backend must be"):
        SharedCarrierConfig(gemm_backend="unknown")


def test_identical_inference_config_does_not_apply_training_row_constraint() -> None:
    config = StrictWorkloadConfig(
        mode="inference",
        session_id="identical-inference",
        training_batch_size=3,
        training_sequence_length=1,
        tile_rows=1024,
        shaping_backend="shared-carrier",
        shared_carrier_weight_gradient_layout="inference-balanced-strided",
        shared_carrier_gemm_backend="identical-triton",
        weight_gradient_schedule="inline",
        inference_batch_size=1024,
    )
    assert config.shared_carrier_gemm_backend == "identical-triton"


def test_identical_backend_requires_shared_carrier() -> None:
    with pytest.raises(ValueError, match="requires --shaping-backend shared-carrier"):
        StrictWorkloadConfig(
            mode="inference",
            session_id="bad-identical-inference",
            shared_carrier_gemm_backend="identical-triton",
        )


def test_triton_source_forbids_runtime_shape_and_stride_specialization() -> None:
    source = (
        Path(__file__).parents[1]
        / "experiments"
        / "llama_identical_microkernel_carrier"
        / "triton_kernel.py"
    ).read_text()
    assert "do_not_specialize=_RUNTIME_ARGUMENTS" in source
    assert "do_not_specialize_on_alignment=_RUNTIME_ARGUMENTS" in source
    for argument in (
        '"M"',
        '"N"',
        '"K"',
        '"stride_am"',
        '"stride_ak"',
        '"stride_bk"',
        '"stride_bn"',
        '"stride_cm"',
        '"stride_cn"',
    ):
        assert argument in source
