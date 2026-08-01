"""Exact linear decompositions for inference-shaped transformer training.

The wrappers in this module do not replace, detach, or approximate a gradient.
They change only the schedule used to evaluate a linear map and its gradients.
In particular, ``token-row`` mode evaluates the forward and input-gradient
matrix products one token row at a time, reproducing the ``M=1`` geometry of a
batch-one cached decode.  ``hybrid`` mode additionally constructs a small,
exact part of each weight gradient using decode-shaped matrix products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


LinearShaping = Literal["none", "token-row", "hybrid"]


@dataclass(frozen=True)
class LinearShapeConfig:
    """Geometry controls for one exact shaped linear operation."""

    forward_rows: int = 1
    input_gradient_rows: int = 1
    weight_gradient_inference_rows: int = 0
    bias_epilogue: bool = True

    def validate(self, in_features: int) -> None:
        if self.forward_rows < 1:
            raise ValueError(f"forward_rows must be >= 1, got {self.forward_rows}")
        if self.input_gradient_rows < 1:
            raise ValueError(
                "input_gradient_rows must be >= 1, got "
                f"{self.input_gradient_rows}"
            )
        if not 0 <= self.weight_gradient_inference_rows <= in_features:
            raise ValueError(
                "weight_gradient_inference_rows must be between 0 and in_features; "
                f"got {self.weight_gradient_inference_rows} for {in_features=}"
            )


def _row_tiled_linear(x: Tensor, weight: Tensor, bias: Tensor | None, rows: int) -> Tensor:
    if x.ndim != 2:
        raise ValueError(f"expected a matrix, got shape={tuple(x.shape)}")
    if rows >= x.shape[0]:
        return F.linear(x, weight, bias)
    return torch.cat(
        [F.linear(x[start : start + rows], weight, bias) for start in range(0, x.shape[0], rows)]
    )


def hybrid_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    inference_rows: int,
    bias_epilogue: bool,
) -> Tensor:
    """Compute an exact weight gradient with selected decode-shaped columns.

    For ``y = x @ W.T``, the ordinary gradient is ``grad_output.T @ x``.
    A selected input-feature column can instead be obtained from an ``M=1``
    linear call.  The token reduction is zero-padded to ``in_features`` so its
    geometry is ``[1, in_features] @ [in_features, out_features]``—the same
    dimensions as a one-token forward projection.  Padding contributes zero.
    The unselected columns use the ordinary reduction.
    """

    if x.ndim != 2 or grad_output.ndim != 2:
        raise ValueError(
            f"x and grad_output must be matrices, got {tuple(x.shape)} and "
            f"{tuple(grad_output.shape)}"
        )
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError("x and grad_output must have the same token-row count")
    token_rows, in_features = x.shape
    out_features = grad_output.shape[1]
    if not 0 <= inference_rows <= in_features:
        raise ValueError(
            f"inference_rows must be between 0 and {in_features}, got {inference_rows}"
        )
    if inference_rows == 0:
        return grad_output.transpose(0, 1) @ x

    # The common experiment has token_rows << in_features.  If it does not,
    # accumulate exact chunks so every shaped call still has K=in_features.
    shaped_t: Tensor | None = None
    zero_bias = x.new_zeros(out_features) if bias_epilogue else None
    for token_start in range(0, token_rows, in_features):
        token_end = min(token_start + in_features, token_rows)
        chunk_rows = token_end - token_start
        padded_x_columns = x.new_zeros((inference_rows, in_features))
        padded_grad_weight = grad_output.new_zeros((out_features, in_features))
        padded_x_columns[:, :chunk_rows] = x[
            token_start:token_end, :inference_rows
        ].transpose(0, 1)
        padded_grad_weight[:, :chunk_rows] = grad_output[
            token_start:token_end
        ].transpose(0, 1)
        current = F.linear(padded_x_columns, padded_grad_weight, zero_bias)
        shaped_t = current if shaped_t is None else shaped_t + current

    assert shaped_t is not None
    shaped = shaped_t.transpose(0, 1).contiguous()
    if inference_rows == in_features:
        return shaped
    bulk = grad_output.transpose(0, 1) @ x[:, inference_rows:]
    return torch.cat((shaped, bulk), dim=1)


class _ShapedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        config: LinearShapeConfig,
    ) -> Tensor:
        if x.shape[-1] != weight.shape[1]:
            raise ValueError(
                f"input width {x.shape[-1]} does not match weight width {weight.shape[1]}"
            )
        config.validate(weight.shape[1])
        flat_x = x.reshape(-1, x.shape[-1])
        flat_output = _row_tiled_linear(flat_x, weight, bias, config.forward_rows)
        ctx.save_for_backward(flat_x, weight)
        ctx.input_shape = tuple(x.shape)
        ctx.has_bias = bias is not None
        ctx.config = config
        return flat_output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        flat_x, weight = ctx.saved_tensors
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        config: LinearShapeConfig = ctx.config

        grad_x = None
        if ctx.needs_input_grad[0]:
            transposed_weight = weight.transpose(0, 1).contiguous()
            zero_bias = flat_x.new_zeros(weight.shape[1]) if config.bias_epilogue else None
            flat_grad_x = _row_tiled_linear(
                flat_grad_output,
                transposed_weight,
                zero_bias,
                config.input_gradient_rows,
            )
            grad_x = flat_grad_x.reshape(ctx.input_shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = hybrid_weight_gradient(
                flat_x,
                flat_grad_output,
                inference_rows=config.weight_gradient_inference_rows,
                bias_epilogue=config.bias_epilogue,
            )
        grad_bias = flat_grad_output.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_x, grad_weight, grad_bias, None


class ShapedLinear(nn.Module):
    """Drop-in ``nn.Linear`` using an exact inference-shaped schedule."""

    def __init__(self, source: nn.Linear, config: LinearShapeConfig) -> None:
        super().__init__()
        if not isinstance(source, nn.Linear):
            raise TypeError(f"source must be nn.Linear, got {type(source).__name__}")
        config.validate(source.in_features)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = source.weight
        self.bias = source.bias
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        return _ShapedLinearFunction.apply(x, self.weight, self.bias, self.config)


def shape_linear_modules(model: nn.Module, mode: LinearShaping) -> list[str]:
    """Replace every dense linear recursively and return its qualified name."""

    if mode == "none":
        return []
    if mode not in {"token-row", "hybrid"}:
        raise ValueError(f"unknown linear shaping mode: {mode!r}")
    config = LinearShapeConfig(
        forward_rows=1,
        input_gradient_rows=1,
        weight_gradient_inference_rows=1 if mode == "hybrid" else 0,
    )
    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []

    def collect(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                replacements.append((parent, child_name, qualified, child))
            else:
                collect(child, qualified)

    collect(model, "")
    for parent, child_name, _qualified, child in replacements:
        setattr(parent, child_name, ShapedLinear(child, config))
    return [qualified for _parent, _name, qualified, _child in replacements]
