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


def _make(layer_cls, args, epsilon, dtype=None):
    kwargs = dict(epsilon=epsilon, learnable_epsilon=True)
    if dtype is not None:
        kwargs.update(dtype=dtype, param_dtype=dtype)
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


@pytest.mark.parametrize("layer_cls,args,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", [1e-8, 1e-20, 1e5])
def test_float16_uses_fp32_epsilon_storage(layer_cls, args, input_shape, epsilon):
    layer = _make(layer_cls, args, epsilon, torch.float16)
    with torch.no_grad():
        layer.weight.fill_(0.3)
    inputs = torch.full(input_shape, 0.2, dtype=torch.float16, requires_grad=True)
    output = layer(inputs)
    (epsilon_grad,) = torch.autograd.grad(output.float().sum(), (layer.epsilon_param,))
    assert layer.epsilon_param.dtype == torch.float32
    assert F.softplus(layer.epsilon_param).item() == pytest.approx(epsilon, rel=2e-6)
    assert output.dtype == torch.float16 and output.isfinite().all()
    assert epsilon_grad.isfinite().all() and epsilon_grad.abs().max() > 0


@pytest.mark.parametrize("layer_cls,args,_", [FAMILIES[0], FAMILIES[1]])
@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(layer_cls, args, _, epsilon):
    with pytest.raises(ValueError, match="not representable"):
        _make(layer_cls, args, epsilon, torch.float32)


@pytest.mark.parametrize("layer_cls,args,input_shape", [FAMILIES[0], FAMILIES[1]])
@pytest.mark.parametrize("epsilon", [2.0**-1022, 1e150])
def test_float64_extreme_epsilon_is_effective_and_differentiable(
    layer_cls, args, input_shape, epsilon
):
    layer = _make(layer_cls, args, epsilon, torch.float64)
    with torch.no_grad():
        layer.weight.fill_(0.3)
    inputs = torch.full(input_shape, 0.2, dtype=torch.float64, requires_grad=True)
    output = layer(inputs)
    (epsilon_grad,) = torch.autograd.grad(output.sum(), (layer.epsilon_param,))
    assert F.softplus(layer.epsilon_param).item() == pytest.approx(epsilon, rel=2e-14)
    assert output.isfinite().all() and epsilon_grad.isfinite().all()
    assert epsilon_grad.abs().max() > 0


@pytest.mark.parametrize("layer_cls,args,_", [FAMILIES[0], FAMILIES[1]])
def test_float64_rejects_softplus_underflow(layer_cls, args, _):
    with pytest.raises(ValueError, match="not representable"):
        _make(layer_cls, args, 5e-324, torch.float64)


@pytest.mark.parametrize("layer_cls,args,input_shape", FAMILIES[:2])
@pytest.mark.parametrize("epsilon", [1e-20, 1e5])
@pytest.mark.parametrize(
    "migration,target_dtype",
    [
        ("half", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("to", torch.float16),
    ],
)
def test_module_dtype_migration_preserves_epsilon_storage_identity_and_gradient(
    layer_cls, args, input_shape, epsilon, migration, target_dtype
):
    layer = _make(layer_cls, args, epsilon)
    epsilon_param = layer.epsilon_param
    if migration == "to":
        layer.to(dtype=target_dtype)
    else:
        getattr(layer, migration)()

    assert layer.epsilon_param is epsilon_param
    assert layer.epsilon_param.dtype == torch.float32
    assert layer.weight.dtype == target_dtype
    assert layer.state_dict()["epsilon_param"].dtype == torch.float32

    with torch.no_grad():
        layer.weight.fill_(0.3)
    inputs = torch.full(input_shape, 0.2, dtype=target_dtype)
    output = layer(inputs)
    (epsilon_grad,) = torch.autograd.grad(output.float().sum(), (layer.epsilon_param,))
    assert output.isfinite().all()
    assert epsilon_grad.isfinite().all() and epsilon_grad.abs().max() > 0

    restored = _make(layer_cls, args, epsilon)
    if migration == "to":
        restored.to(dtype=target_dtype)
    else:
        getattr(restored, migration)()
    restored.load_state_dict(layer.state_dict())
    assert restored.epsilon_param.dtype == torch.float32
    torch.testing.assert_close(restored.epsilon_param, layer.epsilon_param)
