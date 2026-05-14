"""Basic shape / dtype / parameter tests for the MLX YatNMN layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import YatNMN, YatDense  # noqa: E402


def test_module_imports():
    """``YatDense`` is the back-compat alias for ``YatNMN``."""
    assert YatDense is YatNMN


def test_yat_nmn_forward_shape_2d():
    layer = YatNMN(features=10)
    x = mx.random.normal(shape=(4, 8))
    y = layer(x)
    assert y.shape == (4, 10)
    assert y.dtype == mx.float32


def test_yat_nmn_forward_shape_3d():
    """Multiple batch dims should be preserved (broadcast over the last axis)."""
    layer = YatNMN(features=5)
    x = mx.random.normal(shape=(2, 3, 4))
    y = layer(x)
    assert y.shape == (2, 3, 5)


def test_yat_nmn_parameters_discovered():
    layer = YatNMN(features=6)
    _ = layer(mx.zeros((1, 4)))
    params = layer.parameters()
    assert set(params.keys()) == {"kernel", "bias", "alpha"}
    assert params["kernel"].shape == (6, 4)
    assert params["bias"].shape == (6,)
    assert params["alpha"].shape == (1,)


def test_constant_alpha_drops_learnable_alpha():
    layer = YatNMN(features=6, constant_alpha=True)
    _ = layer(mx.zeros((1, 4)))
    assert layer._constant_alpha_value == math.sqrt(2.0)
    assert "alpha" not in layer.parameters()


def test_constant_alpha_float_value():
    layer = YatNMN(features=6, constant_alpha=1.25)
    _ = layer(mx.zeros((1, 4)))
    assert layer._constant_alpha_value == 1.25
    assert "alpha" not in layer.parameters()


def test_constant_bias_drops_learnable_bias():
    layer = YatNMN(features=6, constant_bias=0.5)
    _ = layer(mx.zeros((1, 4)))
    assert layer._constant_bias_value == 0.5
    assert "bias" not in layer.parameters()


def test_use_alpha_false():
    layer = YatNMN(features=6, use_alpha=False)
    _ = layer(mx.zeros((1, 4)))
    assert "alpha" not in layer.parameters()


def test_use_bias_false():
    layer = YatNMN(features=6, use_bias=False)
    _ = layer(mx.zeros((1, 4)))
    assert "bias" not in layer.parameters()


def test_learnable_epsilon_creates_param():
    layer = YatNMN(features=4, learnable_epsilon=True, epsilon=1e-3)
    _ = layer(mx.zeros((1, 3)))
    assert "epsilon_param" in layer.parameters()
    # softplus(raw) should ≈ epsilon at initialization.
    eps = float(mlx_nn.softplus(layer.epsilon_param)[0])
    assert abs(eps - 1e-3) < 1e-5


def test_invalid_epsilon_rejected():
    with pytest.raises(ValueError):
        YatNMN(features=4, epsilon=0.0)
    with pytest.raises(ValueError):
        YatNMN(features=4, epsilon=-1.0)


def test_input_shape_change_rejected():
    layer = YatNMN(features=4)
    _ = layer(mx.zeros((2, 3)))
    with pytest.raises(ValueError):
        layer(mx.zeros((2, 5)))


def test_positive_init():
    layer = YatNMN(features=4, positive_init=True)
    _ = layer(mx.zeros((1, 3)))
    # Kernel entries are abs() of the Xavier draw — strictly non-negative.
    assert float(mx.min(layer.kernel)) >= 0.0


def test_get_set_weights_round_trip():
    layer = YatNMN(features=4)
    _ = layer(mx.zeros((1, 3)))
    weights = layer.get_weights()
    # Perturb and round-trip.
    perturbed = [w + 0.1 for w in weights]
    layer.set_weights(perturbed)
    restored = layer.get_weights()
    for got, want in zip(restored, perturbed):
        assert np.max(np.abs(np.array(got) - np.array(want))) < 1e-6


def test_return_weights_tuple():
    layer = YatNMN(features=4, return_weights=True)
    y, w = layer(mx.zeros((1, 3)))
    assert y.shape == (1, 4)
    assert w.shape == (4, 3)


def test_gradient_flows_and_reduces_loss():
    """One AdamW step should reduce a squared-error loss."""

    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    layer = YatNMN(features=3)
    x = mx.random.normal(shape=(8, 5))
    y = mx.random.normal(shape=(8, 3))
    _ = layer(x)  # build

    grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
    loss, grads = grad_fn(layer, x, y)
    # All three learnable params should receive gradients.
    assert set(grads.keys()) == {"kernel", "bias", "alpha"}

    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(layer, grads)
    mx.eval(layer.parameters())

    loss_after = float(loss_fn(layer, x, y))
    assert loss_after < float(loss)
