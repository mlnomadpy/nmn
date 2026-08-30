"""Stable learnable-epsilon initialization across Torch YAT families."""

import io

import pytest
import torch
import torch.nn.functional as F

from nmn.torch import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatNMN,
)


EPSILONS = (1e-20, 1e-5, 1000.0)
FAMILIES = (
    (YatNMN, (2, 1), (1, 2)),
    (YatConv1D, (1, 1, 1), (1, 1, 1)),
    (YatConv2D, (1, 1, 1, 1), (1, 1, 1)),
    (YatConv3D, (1, 1, 1), (1, 1, 1, 1, 1)),
    (YatConvTranspose1D, (1, 1, 1), (1, 1, 1)),
    (YatConvTranspose2D, (1, 1, 1, 1), (1, 1, 1)),
    (YatConvTranspose3D, (1, 1, 1), (1, 1, 1, 1, 1)),
)


def _make(layer_cls, args, epsilon):
    kwargs = dict(epsilon=epsilon, learnable_epsilon=True)
    if layer_cls is YatNMN:
        kwargs["bias"] = False
        kwargs["alpha"] = False
    else:
        kwargs["bias"] = False
        kwargs["use_alpha"] = False
    return layer_cls(*args, **kwargs)


@pytest.mark.parametrize("layer_cls,args,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
def test_learnable_epsilon_constructs_runs_and_differentiates(
    layer_cls, args, input_shape, epsilon
):
    layer = _make(layer_cls, args, epsilon)
    with torch.no_grad():
        layer.weight.fill_(0.3)
    inputs = torch.full(input_shape, 0.2, requires_grad=True)
    output = layer(inputs)
    input_grad, kernel_grad, epsilon_grad = torch.autograd.grad(
        output.sum(), (inputs, layer.weight, layer.epsilon_param)
    )

    effective = F.softplus(layer.epsilon_param.detach()).item()
    assert effective == pytest.approx(epsilon, rel=2e-6, abs=0.0)
    assert output.isfinite().all()
    assert input_grad.isfinite().all()
    assert kernel_grad.isfinite().all()
    assert epsilon_grad.isfinite().all()
    assert epsilon_grad.abs().item() > 0.0


@pytest.mark.parametrize("layer_cls,args,_", FAMILIES)
@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan"), float("inf")])
def test_epsilon_must_be_finite_and_strictly_positive(layer_cls, args, _, epsilon):
    with pytest.raises(ValueError, match="positive and finite"):
        _make(layer_cls, args, epsilon)


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_learnable_epsilon_traces_and_state_dict_roundtrips(epsilon):
    layer = _make(YatNMN, (2, 1), epsilon)
    inputs = torch.full((1, 2), 0.2)
    traced = torch.jit.trace(layer, inputs)
    assert traced(inputs).isfinite().all()

    buffer = io.BytesIO()
    torch.save(layer.state_dict(), buffer)
    buffer.seek(0)
    restored = _make(YatNMN, (2, 1), epsilon)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored.epsilon_param, layer.epsilon_param)


def test_default_epsilon_remains_backward_compatible():
    layer = YatNMN(1, 2, learnable_epsilon=True)
    assert layer.epsilon == 1e-5
    assert F.softplus(layer.epsilon_param).item() == pytest.approx(1e-5, rel=2e-6)
