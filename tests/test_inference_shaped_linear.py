from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.gpt_oss_inference_shaped_training.inference_shaped_linear import (
    InferenceShapeConfig,
    InferenceShapedLinear,
    inference_shaped_weight_gradient,
)


@pytest.mark.parametrize("token_rows", [1, 2, 7])
@pytest.mark.parametrize("tile_rows", [1, 2, 4])
def test_padded_weight_gradient_matches_reference(token_rows: int, tile_rows: int) -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(token_rows, 5, generator=generator, dtype=torch.float64)
    grad_output = torch.randn(token_rows, 3, generator=generator, dtype=torch.float64)

    actual = inference_shaped_weight_gradient(
        x,
        grad_output,
        reduction_k=4,
        rows=tile_rows,
        bias_epilogue=True,
    )
    expected = grad_output.transpose(0, 1) @ x

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_module_matches_linear_forward_and_backward() -> None:
    generator = torch.Generator().manual_seed(23)
    reference = nn.Linear(5, 3, bias=True, dtype=torch.float64)
    shaped_source = nn.Linear(5, 3, bias=True, dtype=torch.float64)
    shaped_source.load_state_dict(reference.state_dict())
    shaped = InferenceShapedLinear(
        shaped_source,
        InferenceShapeConfig(
            forward_rows=1,
            input_gradient_rows=1,
            weight_gradient_rows=1,
            reduction_k=5,
        ),
    )

    x_reference = torch.randn(2, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    x_shaped = x_reference.detach().clone().requires_grad_(True)
    upstream = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64)

    output_reference = reference(x_reference)
    output_shaped = shaped(x_shaped)
    output_reference.backward(upstream)
    output_shaped.backward(upstream)

    torch.testing.assert_close(output_shaped, output_reference, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(x_shaped.grad, x_reference.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shaped.weight.grad, reference.weight.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shaped.bias.grad, reference.bias.grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("inference_rows", [0, 1, 3, 5])
def test_hybrid_weight_gradient_matches_reference(inference_rows: int) -> None:
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    grad_output = torch.randn(7, 3, generator=generator, dtype=torch.float64)

    actual = inference_shaped_weight_gradient(
        x,
        grad_output,
        reduction_k=4,
        rows=1,
        inference_rows=inference_rows,
    )
    expected = grad_output.transpose(0, 1) @ x

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_invalid_configuration_has_specific_error() -> None:
    with pytest.raises(ValueError, match="weight_gradient_rows must be >= 1"):
        InferenceShapeConfig(weight_gradient_rows=0).validate(5)
    with pytest.raises(ValueError, match="weight_gradient_inference_rows must be between"):
        InferenceShapeConfig(weight_gradient_inference_rows=6).validate(5)
