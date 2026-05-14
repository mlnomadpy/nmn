"""Tests for the MLX YatEmbed layer."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import YatEmbed  # noqa: E402


def test_embed_lookup_shape_1d():
    layer = YatEmbed(num_embeddings=20, features=8)
    ids = mx.array([1, 2, 3, 4], dtype=mx.int32)
    out = layer(ids)
    assert out.shape == (4, 8)


def test_embed_lookup_shape_2d():
    layer = YatEmbed(num_embeddings=20, features=8)
    ids = mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32)
    out = layer(ids)
    assert out.shape == (2, 3, 8)


def test_embed_rejects_non_integer_input():
    layer = YatEmbed(num_embeddings=10, features=4)
    with pytest.raises(ValueError):
        layer(mx.array([0.0, 1.0]))


def test_embed_attend_shape():
    layer = YatEmbed(num_embeddings=50, features=16)
    query = mx.random.normal(shape=(3, 16))
    scores = layer.attend(query)
    assert scores.shape == (3, 50)


def test_embed_attend_math_parity():
    """``attend`` output matches a numpy YAT reference."""
    mx.random.seed(0)
    layer = YatEmbed(num_embeddings=8, features=4, use_alpha=True)
    query = mx.random.normal(shape=(2, 4))
    scores = np.array(layer.attend(query))

    E = np.array(layer.embedding)
    a = float(np.array(layer.alpha)[0])
    qn = np.array(query)
    dot = qn @ E.T
    q_sq = (qn ** 2).sum(axis=-1, keepdims=True)
    e_sq = (E ** 2).sum(axis=-1)[None, :]
    dist = np.maximum(q_sq + e_sq - 2 * dot, 0.0)
    ref = a * (dot ** 2) / (dist + 1e-5)
    assert np.max(np.abs(scores - ref)) < 1e-5


def test_embed_constant_alpha():
    layer = YatEmbed(num_embeddings=10, features=4, constant_alpha=True)
    assert layer._constant_alpha_value == math.sqrt(2.0)
    assert "alpha" not in layer.parameters()


def test_embed_spherical_attend_shape():
    layer = YatEmbed(num_embeddings=12, features=6, spherical=True)
    out = layer.attend(mx.random.normal(shape=(2, 6)))
    assert out.shape == (2, 12)


def test_embed_weight_normalized_init():
    layer = YatEmbed(num_embeddings=8, features=4, weight_normalized=True)
    E = np.array(layer.embedding)
    row_norms = np.linalg.norm(E, axis=1)
    assert np.max(np.abs(row_norms - 1.0)) < 1e-5


def test_embed_gradient_reduces_loss():
    def loss_fn(model, ids, target):
        return mx.mean((model.attend(model(ids)) - target) ** 2)

    layer = YatEmbed(num_embeddings=8, features=4)
    ids = mx.array([0, 1, 2, 3], dtype=mx.int32)
    target = mx.random.normal(shape=(4, 8))

    grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
    loss, grads = grad_fn(layer, ids, target)
    assert "embedding" in grads
    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(layer, grads)
    mx.eval(layer.parameters())
    loss_after = float(loss_fn(layer, ids, target))
    assert loss_after < float(loss)
