"""Role-blind fused Llama inference/training workload."""

from .workload import (
    ComputationConfig,
    DualRoleConfig,
    PersistentDualRoleWorkload,
    Role,
    role_artifact,
    split_dual_logits,
)

__all__ = [
    "ComputationConfig",
    "DualRoleConfig",
    "PersistentDualRoleWorkload",
    "Role",
    "role_artifact",
    "split_dual_logits",
]
