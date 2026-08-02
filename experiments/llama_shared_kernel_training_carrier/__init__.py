"""Shared high-MFU GEMM carrier for strict inference-shaped training."""

from .shared_carrier import (
    SharedCarrierConfig,
    SharedCarrierGradientScheduler,
    SharedCarrierLinear,
    carrier_execution_plans,
    replace_linear_modules_with_shared_carrier,
)

__all__ = [
    "SharedCarrierConfig",
    "SharedCarrierGradientScheduler",
    "SharedCarrierLinear",
    "carrier_execution_plans",
    "replace_linear_modules_with_shared_carrier",
]
