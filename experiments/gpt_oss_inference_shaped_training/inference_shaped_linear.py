"""Exact linear autograd with decode-shaped CUDA work in backward.

This module is intentionally independent from GPT-OSS and Transformers.  The
experiment runner installs :class:`InferenceShapedLinear` in one real GPT-OSS
projection, but the gradient construction can be tested on small CPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class InferenceShapeConfig:
    """Controls how a linear operation is decomposed.

    ``forward_rows=1`` makes each forward GEMM represent one token.  During the
    weight gradient, ``weight_gradient_rows=1`` emits one output row at a time.
    ``reduction_k`` pads the token reduction with zeros; setting it to the
    projection's input width makes each weight-gradient GEMM have the exact
    ``M=1, K=in_features, N=out_features`` geometry of one-token inference.
    """

    forward_rows: int = 1
    input_gradient_rows: int = 1
    weight_gradient_rows: int = 1
    weight_gradient_inference_rows: int | None = None
    reduction_k: int | None = None
    bias_epilogue: bool = True

    def validate(self, in_features: int) -> int:
        for name in ("forward_rows", "input_gradient_rows", "weight_gradient_rows"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        reduction_k = in_features if self.reduction_k is None else self.reduction_k
        if reduction_k < 1:
            raise ValueError(f"reduction_k must be >= 1, got {reduction_k}")
        if self.weight_gradient_inference_rows is not None and not (
            0 <= self.weight_gradient_inference_rows <= in_features
        ):
            raise ValueError(
                "weight_gradient_inference_rows must be between 0 and in_features; "
                f"got {self.weight_gradient_inference_rows} for in_features={in_features}"
            )
        return reduction_k


def _row_tiled_linear(x: Tensor, weight: Tensor, bias: Tensor | None, rows: int) -> Tensor:
    """Compute ``F.linear`` exactly while limiting each call to ``rows`` rows."""

    if x.ndim != 2:
        raise ValueError(f"_row_tiled_linear expects a matrix, got shape={tuple(x.shape)}")
    if rows >= x.shape[0]:
        return F.linear(x, weight, bias)
    return torch.cat([F.linear(x[i : i + rows], weight, bias) for i in range(0, x.shape[0], rows)])


def inference_shaped_weight_gradient(
    x: Tensor,
    grad_output: Tensor,
    *,
    reduction_k: int,
    rows: int = 1,
    inference_rows: int | None = None,
    bias_epilogue: bool = True,
) -> Tensor:
    """Return the exact linear weight gradient using inference-shaped GEMMs.

    For ``y = x @ weight.T``, the ordinary gradient is
    ``grad_output.T @ x``.  We instead construct its transpose one row at a
    time.  For each token chunk, zeros pad both operands to ``reduction_k``::

        [rows, reduction_k] @ [reduction_k, out_features]

    With ``rows=1`` and ``reduction_k=in_features``, this is the same GEMM
    geometry as one-token inference.  Padding contributes exactly zero, so no
    approximate or surrogate gradient is used.
    """

    if x.ndim != 2 or grad_output.ndim != 2:
        raise ValueError(
            "x and grad_output must both be matrices; "
            f"got x={tuple(x.shape)}, grad_output={tuple(grad_output.shape)}"
        )
    if x.shape[0] != grad_output.shape[0]:
        raise ValueError(
            "x and grad_output must have the same token-row count; "
            f"got {x.shape[0]} and {grad_output.shape[0]}"
        )
    if reduction_k < 1 or rows < 1:
        raise ValueError(f"reduction_k and rows must be >= 1, got {reduction_k=} and {rows=}")

    token_rows, in_features = x.shape
    out_features = grad_output.shape[1]
    shaped_feature_rows = in_features if inference_rows is None else inference_rows
    if not 0 <= shaped_feature_rows <= in_features:
        raise ValueError(
            f"inference_rows must be between 0 and {in_features}, got {shaped_feature_rows}"
        )
    if shaped_feature_rows == 0:
        return grad_output.transpose(0, 1) @ x

    grad_weight_t: Tensor | None = None
    zero_bias = x.new_zeros(out_features) if bias_epilogue else None

    # A token chunk is padded with zeros so K is fixed even for short training
    # sequences.  ``fake_weight`` is contiguous in the same [N, K] layout as a
    # normal nn.Linear weight, causing F.linear to use the normal bias epilogue.
    for token_start in range(0, token_rows, reduction_k):
        token_end = min(token_start + reduction_k, token_rows)
        chunk_rows = token_end - token_start
        if chunk_rows == reduction_k:
            # Balanced batches need no padding.  Materialize only the layouts
            # consumed by F.linear; there are no fill kernels or zero operands.
            padded_x_columns = (
                x[token_start:token_end, :shaped_feature_rows].transpose(0, 1).contiguous()
            )
            padded_grad_weight = grad_output[token_start:token_end].transpose(0, 1).contiguous()
        else:
            padded_x_columns = x.new_zeros((shaped_feature_rows, reduction_k))
            padded_grad_weight = grad_output.new_zeros((out_features, reduction_k))
            padded_x_columns[:, :chunk_rows] = x[
                token_start:token_end, :shaped_feature_rows
            ].transpose(0, 1)
            padded_grad_weight[:, :chunk_rows] = grad_output[token_start:token_end].transpose(0, 1)

        chunk_result = torch.cat(
            [
                F.linear(
                    padded_x_columns[row_start : row_start + rows],
                    padded_grad_weight,
                    zero_bias,
                )
                for row_start in range(0, shaped_feature_rows, rows)
            ],
            dim=0,
        )
        grad_weight_t = chunk_result if grad_weight_t is None else grad_weight_t + chunk_result

    if grad_weight_t is None:  # A zero-row tensor is unusual but mathematically well-defined.
        grad_weight_t = x.new_zeros((shaped_feature_rows, out_features))
    shaped_gradient = grad_weight_t.transpose(0, 1).contiguous()
    if shaped_feature_rows == in_features:
        return shaped_gradient

    # The remaining columns are the ordinary exact reduction.  This hybrid is
    # useful when matching the whole transformer trace: a small number of M=1
    # GEMMs can reproduce decode's kernel mix without letting them dominate it.
    bulk_gradient = grad_output.transpose(0, 1) @ x[:, shaped_feature_rows:]
    return torch.cat((shaped_gradient, bulk_gradient), dim=1)


class _InferenceShapedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        config: InferenceShapeConfig,
    ) -> Tensor:
        if x.shape[-1] != weight.shape[1]:
            raise ValueError(
                f"input width {x.shape[-1]} does not match weight width {weight.shape[1]}"
            )
        reduction_k = config.validate(weight.shape[1])
        flat_x = x.reshape(-1, x.shape[-1])
        flat_output = _row_tiled_linear(flat_x, weight, bias, config.forward_rows)

        ctx.save_for_backward(flat_x, weight)
        ctx.input_shape = tuple(x.shape)
        ctx.has_bias = bias is not None
        ctx.config = config
        ctx.reduction_k = reduction_k
        return flat_output.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        flat_x, weight = ctx.saved_tensors
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        config: InferenceShapeConfig = ctx.config

        grad_x = None
        if ctx.needs_input_grad[0]:
            # A contiguous transposed weight makes these M=1 calls match the
            # geometry of the complementary GPT-OSS q projection.
            input_grad_weight = weight.transpose(0, 1).contiguous()
            zero_bias = flat_x.new_zeros(weight.shape[1]) if config.bias_epilogue else None
            flat_grad_x = _row_tiled_linear(
                flat_grad_output,
                input_grad_weight,
                zero_bias,
                config.input_gradient_rows,
            )
            grad_x = flat_grad_x.reshape(ctx.input_shape)

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = inference_shaped_weight_gradient(
                flat_x,
                flat_grad_output,
                reduction_k=ctx.reduction_k,
                rows=config.weight_gradient_rows,
                inference_rows=config.weight_gradient_inference_rows,
                bias_epilogue=config.bias_epilogue,
            )

        grad_bias = flat_grad_output.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return grad_x, grad_weight, grad_bias, None


class InferenceShapedLinear(nn.Module):
    """Drop-in ``nn.Linear`` wrapper with exact decode-shaped autograd work."""

    def __init__(self, linear: nn.Linear, config: InferenceShapeConfig | None = None) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"linear must be nn.Linear, got {type(linear).__name__}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.config = config or InferenceShapeConfig()
        self.config.validate(self.in_features)

    def forward(self, x: Tensor) -> Tensor:
        return _InferenceShapedLinearFunction.apply(x, self.weight, self.bias, self.config)
