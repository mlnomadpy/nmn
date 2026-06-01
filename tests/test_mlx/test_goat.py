"""Tests for the MLX GOAT (value-only / projection-free YAT) attention."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

from nmn.mlx import (  # noqa: E402
    GoatYatAttention,
    goat_yat_attention_weights,
)


def _pos_scalars(h):
    return mlx_nn.softplus(mx.zeros((h,))), mlx_nn.softplus(mx.zeros((h,)))


@pytest.mark.parametrize("variant", ["v", "nov"])
def test_output_shape_and_finite(variant):
    attn = GoatYatAttention(embed_dim=8, num_heads=2, variant=variant)
    y = attn(mx.random.normal(shape=(2, 5, 8)))
    assert y.shape == (2, 5, 8)
    assert bool(mx.all(mx.isfinite(y)))


def test_weights_are_l1_row_stochastic():
    B, L, H, D = 2, 6, 2, 4
    q = mx.random.normal(shape=(B, L, H, D))
    b, eps = _pos_scalars(H)
    w = goat_yat_attention_weights(q, q, b, eps, self_mask=True)
    assert w.shape == (B, H, L, L)
    rows = np.array(mx.sum(w, axis=-1))
    np.testing.assert_allclose(rows, 1.0, atol=1e-4)


def test_self_mask_zeros_diagonal():
    B, L, H, D = 1, 5, 1, 4
    q = mx.random.normal(shape=(B, L, H, D))
    b, eps = _pos_scalars(H)
    w = np.array(goat_yat_attention_weights(q, q, b, eps, self_mask=True))[0, 0]
    np.testing.assert_allclose(np.diagonal(w), 0.0, atol=1e-7)


def test_unmasked_collapses_to_diagonal():
    # k(u,u) is the row maximum, so without the self-mask the diagonal dominates
    # every row -- the documented collapse the self-mask exists to prevent.
    B, L, H, D = 1, 5, 1, 4
    q = mx.random.normal(shape=(B, L, H, D))
    b, eps = _pos_scalars(H)
    w = np.array(goat_yat_attention_weights(q, q, b, eps, self_mask=False))[0, 0]
    assert np.all(np.argmax(w, axis=-1) == np.arange(L))


def test_mask_zeroes_disallowed_positions():
    B, L, H, D = 1, 4, 1, 3
    q = mx.random.normal(shape=(B, L, H, D))
    b, eps = _pos_scalars(H)
    # forbid the last key for every query (True = attend)
    mask = mx.concatenate([mx.ones((B, H, L, L - 1)), mx.zeros((B, H, L, 1))], axis=-1)
    w = np.array(goat_yat_attention_weights(q, q, b, eps, mask=mask, self_mask=True))[0, 0]
    np.testing.assert_allclose(w[:, -1], 0.0, atol=1e-7)


@pytest.mark.parametrize("variant", ["v", "nov"])
def test_gradients_flow(variant):
    attn = GoatYatAttention(embed_dim=8, num_heads=2, variant=variant)
    x = mx.random.normal(shape=(2, 5, 8))

    def loss_fn(model):
        return mx.mean(model(x) ** 2)

    loss, grads = mlx_nn.value_and_grad(attn, loss_fn)(attn)
    assert bool(mx.isfinite(loss))
    # per-head kernel scalars are trainable and receive finite gradients
    assert "b_raw" in grads and "eps_raw" in grads
    assert bool(mx.all(mx.isfinite(grads["b_raw"])))
    assert bool(mx.all(mx.isfinite(grads["eps_raw"])))
