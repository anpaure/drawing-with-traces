"""Interleaved optimizer updates for strict no-cover training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class OptimizerAudit:
    parameter_tensors: int
    parameter_values: int
    interleaved_parameter_tensors: int
    manual_parameter_tensors: int
    deferred_shared_parameter_tensors: int
    interleaved_update_bucket_size: int
    fused_interleaved_update_flushes: int
    fused_interleaved_update_tensors: int
    manual_update_bucket_size: int
    fused_manual_update_flushes: int
    fused_manual_update_tensors: int
    learning_rate: float
    update_rule: str = "parameter -= learning_rate * exact_gradient"


def shared_parameter_ids(model: nn.Module) -> set[int]:
    """Return parameter identities that occur under more than one qualified name."""

    counts = Counter(id(parameter) for _name, parameter in model.named_parameters(remove_duplicate=False))
    return {parameter_id for parameter_id, count in counts.items() if count > 1}


class InterleavedSGD:
    """Apply real SGD updates as gradients become available during backward.

    A post-accumulate hook queues each unshared parameter after its complete
    gradient reaches the leaf. Small ready-parameter buckets are updated with one
    foreach launch while backward is still running. Shared parameters are
    deliberately deferred until backward has returned so every contribution is
    included exactly once.

    The implementation has no momentum or weight decay.  It is mathematically
    equivalent to a conventional SGD step with the same learning rate, while its
    fused update kernels are distributed through the reverse pass instead of
    forming one distinctive optimizer burst.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        learning_rate: float,
        deferred_parameter_ids: set[int] | None = None,
        manual_parameter_ids: set[int] | None = None,
        manual_update_bucket_size: int = 1,
        interleaved_update_bucket_size: int | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        self.learning_rate = float(learning_rate)
        if manual_update_bucket_size < 1:
            raise ValueError("manual_update_bucket_size must be positive")
        self.manual_update_bucket_size = int(manual_update_bucket_size)
        if interleaved_update_bucket_size is None:
            interleaved_update_bucket_size = manual_update_bucket_size
        if interleaved_update_bucket_size < 1:
            raise ValueError("interleaved_update_bucket_size must be positive")
        self.interleaved_update_bucket_size = int(interleaved_update_bucket_size)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.deferred_parameter_ids = (
            shared_parameter_ids(model) if deferred_parameter_ids is None else set(deferred_parameter_ids)
        )
        self.manual_parameter_ids = set(manual_parameter_ids or ())
        known_ids = {id(parameter) for parameter in self.parameters}
        unknown = (self.deferred_parameter_ids | self.manual_parameter_ids) - known_ids
        if unknown:
            raise ValueError("optimizer parameter IDs contain parameters outside the model")
        self.deferred_parameters = [
            parameter for parameter in self.parameters if id(parameter) in self.deferred_parameter_ids
        ]
        self.manual_parameters = [
            parameter
            for parameter in self.parameters
            if id(parameter) in self.manual_parameter_ids and id(parameter) not in self.deferred_parameter_ids
        ]
        self.interleaved_parameters = [
            parameter
            for parameter in self.parameters
            if id(parameter) not in self.deferred_parameter_ids
            and id(parameter) not in self.manual_parameter_ids
        ]
        self._pending_interleaved_parameters: list[Tensor] = []
        self._pending_manual_parameters: list[Tensor] = []
        self._fused_interleaved_update_flushes = 0
        self._fused_interleaved_update_tensors = 0
        self._fused_manual_update_flushes = 0
        self._fused_manual_update_tensors = 0
        self._handles = [
            parameter.register_post_accumulate_grad_hook(self._make_hook())
            for parameter in self.interleaved_parameters
        ]

    def _make_hook(self):
        @torch.no_grad()
        def update(parameter: Tensor) -> None:
            if parameter.grad is None:
                raise RuntimeError("interleaved SGD hook received a parameter without a gradient")
            if any(candidate is parameter for candidate in self._pending_interleaved_parameters):
                raise RuntimeError("interleaved SGD parameter was queued more than once")
            self._pending_interleaved_parameters.append(parameter)
            if len(self._pending_interleaved_parameters) >= self.interleaved_update_bucket_size:
                self.flush_interleaved()

        return update

    @torch.no_grad()
    def flush_interleaved(self) -> None:
        """Fuse ready leaf-gradient updates while backward is still in progress."""

        if not self._pending_interleaved_parameters:
            return
        gradients = []
        for parameter in self._pending_interleaved_parameters:
            if parameter.grad is None:
                raise RuntimeError("queued interleaved SGD parameter has no gradient")
            gradients.append(parameter.grad)
        torch._foreach_add_(
            self._pending_interleaved_parameters,
            gradients,
            alpha=-self.learning_rate,
        )
        self._fused_interleaved_update_flushes += 1
        self._fused_interleaved_update_tensors += len(self._pending_interleaved_parameters)
        self._pending_interleaved_parameters.clear()

    @torch.no_grad()
    def step_manual(self, parameter: Tensor) -> None:
        """Update one manually-computed, unshared gradient at its completion point."""

        if id(parameter) not in self.manual_parameter_ids:
            raise ValueError("step_manual received a parameter not registered for manual gradients")
        if id(parameter) in self.deferred_parameter_ids:
            raise ValueError("shared/deferred parameters must be updated by step_deferred")
        if parameter.grad is None:
            raise RuntimeError("manual SGD parameter has no gradient")
        if any(candidate is parameter for candidate in self._pending_manual_parameters):
            raise RuntimeError("manual SGD parameter was queued more than once")
        self._pending_manual_parameters.append(parameter)
        if len(self._pending_manual_parameters) >= self.manual_update_bucket_size:
            self.flush_manual()

    @torch.no_grad()
    def flush_manual(self) -> None:
        """Fuse the completed manual-gradient updates currently in one bucket."""

        if not self._pending_manual_parameters:
            return
        gradients = []
        for parameter in self._pending_manual_parameters:
            if parameter.grad is None:
                raise RuntimeError("queued manual SGD parameter has no gradient")
            gradients.append(parameter.grad)
        torch._foreach_add_(
            self._pending_manual_parameters,
            gradients,
            alpha=-self.learning_rate,
        )
        self._fused_manual_update_flushes += 1
        self._fused_manual_update_tensors += len(self._pending_manual_parameters)
        self._pending_manual_parameters.clear()

    @torch.no_grad()
    def step_deferred(self) -> None:
        """Update shared parameters once all backward contributions are present."""

        self.flush_manual()
        self.flush_interleaved()
        for parameter in self.deferred_parameters:
            if parameter.grad is None:
                raise RuntimeError("deferred SGD parameter has no gradient")
            parameter.add_(parameter.grad, alpha=-self.learning_rate)

    @torch.no_grad()
    def zero_grad(self, *, set_to_none: bool = False) -> None:
        """Clear gradients while supporting the fixed buffers required by CUDA graphs."""

        if self._pending_manual_parameters or self._pending_interleaved_parameters:
            raise RuntimeError("cannot clear gradients with an unflushed optimizer update bucket")
        for parameter in self.parameters:
            if parameter.grad is None:
                if not set_to_none:
                    parameter.grad = torch.zeros_like(parameter)
            elif set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def audit(self) -> OptimizerAudit:
        return OptimizerAudit(
            parameter_tensors=len(self.parameters),
            parameter_values=sum(parameter.numel() for parameter in self.parameters),
            interleaved_parameter_tensors=len(self.interleaved_parameters),
            manual_parameter_tensors=len(self.manual_parameters),
            deferred_shared_parameter_tensors=len(self.deferred_parameters),
            interleaved_update_bucket_size=self.interleaved_update_bucket_size,
            fused_interleaved_update_flushes=self._fused_interleaved_update_flushes,
            fused_interleaved_update_tensors=self._fused_interleaved_update_tensors,
            manual_update_bucket_size=self.manual_update_bucket_size,
            fused_manual_update_flushes=self._fused_manual_update_flushes,
            fused_manual_update_tensors=self._fused_manual_update_tensors,
            learning_rate=self.learning_rate,
        )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> InterleavedSGD:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
