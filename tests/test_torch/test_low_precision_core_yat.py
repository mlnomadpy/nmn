"""Large-magnitude low-precision regressions for core YAT arithmetic."""

import numpy as np
import pytest
import torch

from nmn.torch import YatNMN
from nmn.torch.attention import yat_attention


def _dense_value_and_grads(dtype):
    layer = YatNMN(2, 1, bias=False, alpha=False, epsilon=1.0,
                   dtype=dtype, param_dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[-100.0, -99.0]], dtype=dtype))
    x = torch.tensor([[100.0, 100.0]], dtype=dtype, requires_grad=True)
    output = layer(x)
    input_grad, kernel_grad = torch.autograd.grad(
        output.float().sum(), (x, layer.weight)
    )
    return output, input_grad, kernel_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_magnitude_dense_matches_fp32_forward_and_gradients(dtype):
    reference = _dense_value_and_grads(torch.float32)
    lowp = _dense_value_and_grads(dtype)
    for actual, expected in zip(lowp, reference):
        np.testing.assert_allclose(
            actual.float().detach().numpy(), expected.detach().numpy(),
            rtol=5e-3, atol=0.15,
        )
    assert lowp[0].dtype == dtype


def _aggregate_dense_grads(dtype):
    layer = YatNMN(2, 1, bias=True, alpha=True, epsilon=1.0,
                   dtype=dtype, param_dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[-100.0, -99.0]], dtype=dtype))
        layer.bias.fill_(0.5)
        layer.alpha.fill_(1.25)
    x = torch.full((4096, 2), 100.0, dtype=dtype, requires_grad=True)
    output = layer(x)
    gradients = torch.autograd.grad(
        output.float().sum(), (x, layer.weight, layer.bias, layer.alpha)
    )
    return output, gradients


def test_fp16_dense_aggregate_cotangents_match_saturated_fp32_reference():
    reference_output, reference_grads = _aggregate_dense_grads(torch.float32)
    output, grads = _aggregate_dense_grads(torch.float16)
    limit = torch.finfo(torch.float16)
    assert torch.isfinite(output).all()
    np.testing.assert_allclose(
        output.float().detach().numpy(), reference_output.detach().numpy(),
        rtol=5e-3, atol=2.0,
    )
    for actual, expected in zip(grads, reference_grads):
        clipped = expected.clamp(limit.min, limit.max).to(torch.float16)
        assert torch.isfinite(actual).all()
        np.testing.assert_allclose(
            actual.detach().numpy(), clipped.detach().numpy(), rtol=5e-3, atol=8.0
        )


def _attention_value_and_grads(dtype):
    query = torch.full((1, 1, 1, 2), 100.0, dtype=dtype, requires_grad=True)
    key = torch.full((1, 2, 1, 2), 100.0, dtype=dtype, requires_grad=True)
    value = torch.tensor([[[[1.0]], [[2.0]]]], dtype=dtype, requires_grad=True)
    output = yat_attention(
        query, key, value, training=False, epsilon=1.0
    )
    gradients = torch.autograd.grad(
        output.float().sum() * 0.015625, (query, key, value)
    )
    return output, gradients


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_large_magnitude_attention_matches_fp32_forward_and_gradients(dtype):
    ref_output, ref_grads = _attention_value_and_grads(torch.float32)
    output, grads = _attention_value_and_grads(dtype)
    np.testing.assert_allclose(
        output.float().detach().numpy(), ref_output.detach().numpy(), atol=2e-3
    )
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(
            actual.float().detach().numpy(), expected.detach().numpy(),
            rtol=7e-3, atol=8.0,
        )


def _aggregate_attention_grads(dtype):
    query = torch.full((1, 1, 1, 2), 100.0, dtype=dtype, requires_grad=True)
    key = torch.full((1, 2, 1, 2), 99.0, dtype=dtype, requires_grad=True)
    value = torch.tensor([[[[0.0]], [[1.0]]]], dtype=dtype, requires_grad=True)
    output = yat_attention(query, key, value, training=False, epsilon=1.0)
    gradients = torch.autograd.grad(output.float().sum(), (query, key, value))
    return output, gradients


def test_fp16_attention_aggregate_cotangents_match_saturated_fp32_reference():
    reference_output, reference_grads = _aggregate_attention_grads(torch.float32)
    output, grads = _aggregate_attention_grads(torch.float16)
    limit = torch.finfo(torch.float16)
    np.testing.assert_allclose(
        output.float().detach().numpy(), reference_output.detach().numpy(), atol=2e-3
    )
    for actual, expected in zip(grads, reference_grads):
        clipped = expected.clamp(limit.min, limit.max).to(torch.float16)
        assert torch.isfinite(actual).all()
        np.testing.assert_allclose(
            actual.detach().numpy(), clipped.detach().numpy(), rtol=7e-3, atol=8.0
        )


def test_low_precision_dense_and_attention_preserve_genuine_nan():
    layer = YatNMN(2, 1, bias=False, alpha=False, epsilon=1.0,
                   dtype=torch.float16, param_dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    assert torch.isnan(layer(torch.tensor([[float("nan"), 1.0]], dtype=torch.float16))).all()

    query = torch.tensor([[[[float("nan"), 100.0]]]], dtype=torch.float16)
    key = torch.full((1, 2, 1, 2), 100.0, dtype=torch.float16)
    weights = yat_attention(
        query, key, torch.ones((1, 2, 1, 1), dtype=torch.float16),
        training=False,
    )
    assert torch.isnan(weights).all()
