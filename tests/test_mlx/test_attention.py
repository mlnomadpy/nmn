"""Tests for the MLX YAT attention module + functional helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import (  # noqa: E402
    MultiHeadYatAttention,
    normalize_qk,
    yat_attention,
    yat_attention_weights,
)

# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def test_normalize_qk_unit_norm():
    q = mx.random.normal(shape=(2, 4, 3, 5))
    k = mx.random.normal(shape=(2, 6, 3, 5))
    qn, kn = normalize_qk(q, k)
    norms_q = np.linalg.norm(np.array(qn), axis=-1)
    norms_k = np.linalg.norm(np.array(kn), axis=-1)
    assert np.max(np.abs(norms_q - 1.0)) < 1e-4
    assert np.max(np.abs(norms_k - 1.0)) < 1e-4


def test_yat_attention_shape():
    B, Q, K, H, D = 1, 4, 6, 2, 3
    q = mx.random.normal(shape=(B, Q, H, D))
    k = mx.random.normal(shape=(B, K, H, D))
    v = mx.random.normal(shape=(B, K, H, D))
    out = yat_attention(q, k, v)
    assert out.shape == (B, Q, H, D)


def test_yat_attention_weights_softmax_sum():
    q = mx.random.normal(shape=(1, 3, 2, 4))
    k = mx.random.normal(shape=(1, 5, 2, 4))
    w = np.array(yat_attention_weights(q, k))
    # Last axis should sum to 1.
    sums = w.sum(axis=-1)
    assert np.max(np.abs(sums - 1.0)) < 1e-4


def test_yat_attention_weights_math_parity():
    mx.random.seed(0)
    B, Q, K, H, D = 1, 3, 4, 2, 5
    q = mx.random.normal(shape=(B, Q, H, D))
    k = mx.random.normal(shape=(B, K, H, D))
    w = np.array(yat_attention_weights(q, k, epsilon=1e-5))

    qn = np.array(q)
    kn = np.array(k)
    dim_scale = math.sqrt(D)
    ref = np.zeros((B, H, Q, K))
    for b in range(B):
        for h in range(H):
            for i in range(Q):
                row = np.zeros(K)
                for j in range(K):
                    dot = float((qn[b, i, h] * kn[b, j, h]).sum())
                    dist = float(((qn[b, i, h] - kn[b, j, h]) ** 2).sum())
                    row[j] = dot * dot / ((dist + 1e-5) * dim_scale)
                row = row - row.max()
                ex = np.exp(row)
                ref[b, h, i] = ex / ex.sum()
    assert np.max(np.abs(w - ref)) < 1e-5


def test_yat_attention_causal_mask():
    """With a strict causal mask the upper-triangle of weights should be ~0."""
    B, L, H, D = 1, 5, 2, 4
    q = mx.random.normal(shape=(B, L, H, D))
    k = mx.random.normal(shape=(B, L, H, D))
    # mask[b, h, i, j] = (i >= j)  -> True = attend.
    causal = np.tril(np.ones((L, L), dtype=bool))[None, None, :, :]
    mask = mx.array(np.broadcast_to(causal, (B, H, L, L)))
    w = np.array(yat_attention_weights(q, k, mask=mask))
    upper = np.triu(np.ones_like(w[0, 0]), k=1).astype(bool)
    masked_vals = w[..., upper]
    assert float(np.max(np.abs(masked_vals))) < 1e-5


def test_yat_attention_spherical_shape():
    B, Q, K, H, D = 1, 3, 4, 2, 5
    q = mx.random.normal(shape=(B, Q, H, D))
    k = mx.random.normal(shape=(B, K, H, D))
    v = mx.random.normal(shape=(B, K, H, D))
    out = yat_attention(q, k, v, spherical=True)
    assert out.shape == (B, Q, H, D)


def test_negative_scale_cannot_make_masked_key_win_softmax():
    q = mx.ones((1, 1, 1, 4))
    k = mx.ones((1, 2, 1, 4))
    weights = yat_attention_weights(q, k, mask=mx.array([True, False]), scale=-1.0)
    mx.eval(weights)
    np.testing.assert_array_equal(np.asarray(weights), [[[[1.0, 0.0]]]])


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
@pytest.mark.parametrize("spherical", [False, True])
def test_fully_masked_rows_are_zero_with_finite_gradients(spherical, dtype):
    q = mx.random.normal(shape=(1, 2, 2, 4)).astype(dtype)
    k = mx.random.normal(shape=(1, 3, 2, 4)).astype(dtype)
    v = mx.random.normal(shape=(1, 3, 2, 5)).astype(dtype)
    mask = mx.array([[[[False, False, False], [True, False, True]]]])

    def loss(q, k, v):
        return mx.sum(yat_attention(q, k, v, mask=mask, spherical=spherical))

    output = yat_attention(q, k, v, mask=mask, spherical=spherical)
    weights = yat_attention_weights(q, k, mask=mask, spherical=spherical)
    grads = mx.value_and_grad(loss, argnums=(0, 1, 2))(q, k, v)[1]
    mx.eval(output, weights, *grads)
    np.testing.assert_array_equal(np.asarray(output[:, 0]), 0.0)
    np.testing.assert_array_equal(np.asarray(weights[..., 0, :]), 0.0)
    assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)


# ---------------------------------------------------------------------------
# MultiHeadYatAttention
# ---------------------------------------------------------------------------


def test_mha_self_attn_shape():
    mha = MultiHeadYatAttention(embed_dim=16, num_heads=4)
    x = mx.random.normal(shape=(2, 7, 16))
    assert mha(x).shape == (2, 7, 16)


def test_mha_cross_attn_shape():
    mha = MultiHeadYatAttention(embed_dim=16, num_heads=4)
    q = mx.random.normal(shape=(2, 5, 16))
    ctx = mx.random.normal(shape=(2, 11, 16))
    assert mha(q, ctx, ctx).shape == (2, 5, 16)


@pytest.mark.parametrize("mask_rank", [2, 4])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_mha_fully_masked_rows_stay_zero_after_biased_projection(
    cross_attention, mask_rank
):
    mha = MultiHeadYatAttention(embed_dim=8, num_heads=2)
    query = mx.random.normal(shape=(1, 2, 8))
    context = mx.random.normal(shape=(1, 3, 8))
    _ = mha(query)
    mha.out_bias = mx.full(mha.out_bias.shape, 3.0)
    kv_length = 3 if cross_attention else 2
    shape = (2, kv_length) if mask_rank == 2 else (1, 1, 2, kv_length)
    mask_np = np.ones(shape, dtype=bool)
    mask_np[..., 0, :] = False
    mask = mx.array(mask_np)
    output = (
        mha(query, context, context, mask=mask)
        if cross_attention
        else mha(query, mask=mask)
    )
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output[:, 0]), 0.0)
    assert np.all(np.isfinite(np.asarray(output)))
    if cross_attention:
        grads = mx.grad(lambda q, c: mx.sum(mha(q, c, c, mask=mask)), argnums=(0, 1))(
            query, context
        )
    else:
        grads = (mx.grad(lambda q: mx.sum(mha(q, mask=mask)))(query),)
    mx.eval(*grads)
    assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)


def test_mha_invalid_dim_rejected():
    with pytest.raises(ValueError):
        MultiHeadYatAttention(embed_dim=10, num_heads=4)


def test_mha_constant_alpha():
    mha = MultiHeadYatAttention(embed_dim=8, num_heads=2, constant_alpha=True)
    _ = mha(mx.random.normal(shape=(1, 3, 8)))
    assert "alpha" not in mha.parameters()


def test_mha_no_out_proj():
    mha = MultiHeadYatAttention(embed_dim=8, num_heads=2, use_out_proj=False)
    _ = mha(mx.random.normal(shape=(1, 3, 8)))
    params = mha.parameters()
    assert "out_kernel" not in params
    assert "out_bias" not in params


def test_mha_gradient_reduces_loss():
    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    mha = MultiHeadYatAttention(embed_dim=8, num_heads=2)
    x = mx.random.normal(shape=(2, 5, 8))
    y = mx.random.normal(shape=(2, 5, 8))
    _ = mha(x)  # build lazy params before value_and_grad captures the tree

    grad_fn = mlx_nn.value_and_grad(mha, loss_fn)
    loss, grads = grad_fn(mha, x, y)
    # Should have grads for at least the four projection kernels and alpha.
    assert {"q_kernel", "k_kernel", "v_kernel", "out_kernel", "alpha"} <= set(
        grads.keys()
    )

    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(mha, grads)
    mx.eval(mha.parameters())
    loss_after = float(loss_fn(mha, x, y))
    assert loss_after < float(loss)
