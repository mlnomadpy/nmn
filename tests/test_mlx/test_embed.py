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
    q_sq = (qn**2).sum(axis=-1, keepdims=True)
    e_sq = (E**2).sum(axis=-1)[None, :]
    dist = np.maximum(q_sq + e_sq - 2 * dot, 0.0)
    ref = a * (dot**2) / (dist + 1e-5)
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


def _mlx_attend_value_and_grads(dtype, spherical, compiled, loss_scale=1.0):
    layer = YatEmbed(
        num_embeddings=2, features=2, epsilon=1.0, spherical=spherical, dtype=dtype
    )
    layer.embedding = mx.array([[-100.0, -99.0], [100.0, -99.0]], dtype=dtype)
    layer.alpha = mx.array([1.25], dtype=dtype)
    query = mx.array([[100.0, 100.0]], dtype=dtype)

    query_grad_fn = mx.value_and_grad(
        lambda value: mx.sum(layer.attend(value).astype(mx.float32)) * loss_scale
    )
    raw_parameter_grad_fn = mlx_nn.value_and_grad(
        layer,
        lambda model, value: (
            mx.sum(model.attend(value).astype(mx.float32)) * loss_scale
        ),
    )
    attend_fn = layer.attend
    if compiled:
        state = layer.state
        query_grad_fn = mx.compile(query_grad_fn, inputs=state)
        parameter_grad_fn = mx.compile(
            lambda value: raw_parameter_grad_fn(layer, value),
            inputs=state,
        )
        attend_fn = mx.compile(attend_fn, inputs=state)

    output = attend_fn(query)
    _, query_grad = query_grad_fn(query)
    _, parameter_grads = (
        parameter_grad_fn(query) if compiled else raw_parameter_grad_fn(layer, query)
    )
    mx.eval(output, query_grad, parameter_grads)
    return output, (query_grad, parameter_grads["embedding"], parameter_grads["alpha"])


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_low_precision_attend_matches_fp32_on_native_metal(
    mlx_gpu, dtype, spherical, compiled
):
    del mlx_gpu
    expected_output, expected_grads = _mlx_attend_value_and_grads(
        mx.float32, spherical, compiled
    )
    output, grads = _mlx_attend_value_and_grads(dtype, spherical, compiled)

    assert output.dtype == dtype
    assert np.allclose(
        np.array(output.astype(mx.float32)),
        np.array(expected_output),
        rtol=2e-2,
        atol=2e-2,
    )
    for actual, expected in zip(grads, expected_grads):
        assert np.isfinite(np.array(actual.astype(mx.float32))).all()
        assert np.allclose(
            np.array(actual.astype(mx.float32)),
            np.array(expected),
            rtol=3e-2,
            atol=2e-2,
        )


@pytest.mark.parametrize("spherical", [False, True])
def test_fp16_attend_saturates_output_and_gradients_and_preserves_nan(
    mlx_gpu, spherical
):
    del mlx_gpu
    layer = YatEmbed(
        num_embeddings=2, features=2, spherical=spherical, dtype=mx.float16
    )
    layer.embedding = mx.array([[300.0, 300.0], [300.0, -300.0]], dtype=mx.float16)
    query = mx.array([[300.0, 300.0]], dtype=mx.float16)

    # Materialize the eager NaN probe before compiling against ``layer.state``.
    # MLX transfers ownership of captured lazy state arrays to the compiled
    # graph, so constructing a new eager graph from that state afterwards can
    # leave arrays without a primitive.
    nan_score = layer.attend(mx.array([[float("nan"), 300.0]], mx.float16))
    mx.eval(nan_score)
    nan_score_np = np.array(nan_score.astype(mx.float32))

    query_grad_fn = mx.grad(
        lambda value: mx.sum(layer.attend(value).astype(mx.float32))
    )
    parameter_grad_fn = mlx_nn.value_and_grad(
        layer,
        lambda model, value: mx.sum(model.attend(value).astype(mx.float32)),
    )

    def compiled_bundle(value):
        score = layer.attend(value)
        query_grad = query_grad_fn(value)
        _, parameter_grads = parameter_grad_fn(layer, value)
        return score, query_grad, parameter_grads

    score, query_grad, parameter_grads = mx.compile(
        compiled_bundle, inputs=layer.state
    )(query)
    mx.eval(score, query_grad, parameter_grads)

    assert float(score[0, 0]) == np.finfo(np.float16).max
    assert np.isfinite(np.array(query_grad)).all()
    assert all(np.isfinite(np.array(value)).all() for value in parameter_grads.values())
    assert np.isnan(nan_score_np).all()


def test_fp16_attend_returning_gradients_saturate_against_fp32(mlx_gpu):
    del mlx_gpu
    spherical, loss_scale = False, 1e4
    _, expected_grads = _mlx_attend_value_and_grads(
        mx.float32, spherical, True, loss_scale
    )
    _, grads = _mlx_attend_value_and_grads(mx.float16, spherical, True, loss_scale)
    limits = np.finfo(np.float16)
    for actual, expected in zip(grads, expected_grads):
        clipped = np.clip(np.array(expected), limits.min, limits.max).astype(np.float16)
        assert np.isfinite(np.array(actual)).all()
        assert np.allclose(np.array(actual), clipped, rtol=2e-2, atol=32.0)
