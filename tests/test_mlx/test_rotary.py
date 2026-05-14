"""Tests for MLX Rotary YAT Attention."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import (  # noqa: E402
    RotaryYatAttention,
    precompute_freqs_cis,
    apply_rotary_emb,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    yat_attention_weights,
)


# ---------------------------------------------------------------------------
# precompute_freqs_cis
# ---------------------------------------------------------------------------


def test_precompute_freqs_cis_shape():
    cos, sin = precompute_freqs_cis(64, 128)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)


def test_precompute_freqs_cis_rejects_odd_dim():
    with pytest.raises(ValueError):
        precompute_freqs_cis(63, 16)


def test_precompute_freqs_cis_first_row_is_zero_angle():
    """t=0 -> all angles are 0, cos=1, sin=0."""
    cos, sin = precompute_freqs_cis(32, 4)
    np.testing.assert_allclose(np.array(cos[0]), np.ones(16), atol=1e-6)
    np.testing.assert_allclose(np.array(sin[0]), np.zeros(16), atol=1e-6)


# ---------------------------------------------------------------------------
# apply_rotary_emb
# ---------------------------------------------------------------------------


def test_apply_rotary_emb_shape_preserved():
    cos, sin = precompute_freqs_cis(16, 32)
    x = mx.random.normal(shape=(2, 8, 3, 16))
    out = apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape


def test_apply_rotary_emb_preserves_per_pair_norm():
    """RoPE is a rotation of each (even, odd) pair → preserves vector norms."""
    cos, sin = precompute_freqs_cis(64, 32)
    x = mx.random.normal(shape=(2, 8, 4, 64))
    x_rot = apply_rotary_emb(x, cos, sin)
    in_norms = np.linalg.norm(np.array(x), axis=-1)
    out_norms = np.linalg.norm(np.array(x_rot), axis=-1)
    assert np.max(np.abs(in_norms - out_norms)) < 1e-4


def test_apply_rotary_emb_offset_changes_output():
    """Same input, different ``position_offset`` → different rotated output."""
    mx.random.seed(0)
    cos, sin = precompute_freqs_cis(32, 64)
    x = mx.random.normal(shape=(1, 10, 2, 32))
    out_0 = np.array(apply_rotary_emb(x, cos, sin, position_offset=0))
    out_5 = np.array(apply_rotary_emb(x, cos, sin, position_offset=5))
    assert np.max(np.abs(out_0 - out_5)) > 1e-2


def test_apply_rotary_emb_rejects_overflow():
    cos, sin = precompute_freqs_cis(16, 8)
    x = mx.zeros((1, 8, 1, 16))
    with pytest.raises(ValueError):
        apply_rotary_emb(x, cos, sin, position_offset=4)


def test_apply_rotary_emb_matches_numpy_reference():
    """Compare against a hand-rolled numpy implementation."""
    mx.random.seed(3)
    head_dim = 16
    cos, sin = precompute_freqs_cis(head_dim, 8)
    x = mx.random.normal(shape=(1, 5, 2, head_dim))
    y = np.array(apply_rotary_emb(x, cos, sin))

    xn = np.array(x)
    cn = np.array(cos)[:5]
    sn = np.array(sin)[:5]
    x_even = xn[..., 0::2]
    x_odd = xn[..., 1::2]
    cn_b = cn.reshape(1, 5, 1, -1)
    sn_b = sn.reshape(1, 5, 1, -1)
    rot_even = x_even * cn_b - x_odd * sn_b
    rot_odd = x_even * sn_b + x_odd * cn_b
    ref = np.stack([rot_even, rot_odd], axis=-1).reshape(xn.shape)
    assert np.max(np.abs(y - ref)) < 1e-5


# ---------------------------------------------------------------------------
# Functional rotary YAT attention
# ---------------------------------------------------------------------------


def test_rotary_yat_attention_weights_relative_position_invariance():
    """Shifting Q and K by the same offset must leave attention weights
    unchanged (only relative position matters in RoPE+YAT)."""
    mx.random.seed(11)
    cos, sin = precompute_freqs_cis(16, 32)
    q = mx.random.normal(shape=(1, 6, 2, 16))
    k = mx.random.normal(shape=(1, 6, 2, 16))
    w0 = np.array(rotary_yat_attention_weights(q, k, cos, sin, position_offset=0))
    w3 = np.array(rotary_yat_attention_weights(q, k, cos, sin, position_offset=3))
    # Should be equal up to fp32 noise.
    assert np.max(np.abs(w0 - w3)) < 1e-4


def test_rotary_yat_attention_softmax_sums_to_one():
    cos, sin = precompute_freqs_cis(16, 32)
    q = mx.random.normal(shape=(1, 4, 2, 16))
    k = mx.random.normal(shape=(1, 5, 2, 16))
    w = np.array(rotary_yat_attention_weights(q, k, cos, sin))
    assert np.max(np.abs(w.sum(axis=-1) - 1.0)) < 1e-4


def test_rotary_yat_attention_v_output_shape():
    cos, sin = precompute_freqs_cis(16, 32)
    q = mx.random.normal(shape=(1, 4, 2, 16))
    k = mx.random.normal(shape=(1, 5, 2, 16))
    v = mx.random.normal(shape=(1, 5, 2, 16))
    out = rotary_yat_attention(q, k, v, cos, sin)
    assert out.shape == (1, 4, 2, 16)


def test_rotary_yat_attention_diverges_from_no_rope():
    """With RoPE the weights should differ from plain YAT (otherwise we'd
    be losing the positional signal)."""
    mx.random.seed(0)
    cos, sin = precompute_freqs_cis(16, 16)
    q = mx.random.normal(shape=(1, 8, 1, 16))
    k = mx.random.normal(shape=(1, 8, 1, 16))
    w_no_rope = np.array(yat_attention_weights(q, k))
    w_rope = np.array(rotary_yat_attention_weights(q, k, cos, sin))
    assert np.max(np.abs(w_no_rope - w_rope)) > 1e-2


# ---------------------------------------------------------------------------
# RotaryYatAttention module
# ---------------------------------------------------------------------------


def test_module_self_attn_shape():
    mha = RotaryYatAttention(embed_dim=64, num_heads=8, max_seq_len=128)
    x = mx.random.normal(shape=(2, 10, 64))
    out = mha(x)
    assert out.shape == (2, 10, 64)


def test_module_rejects_bad_embed_dim():
    with pytest.raises(ValueError):
        RotaryYatAttention(embed_dim=10, num_heads=4, max_seq_len=32)


def test_module_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryYatAttention(embed_dim=15, num_heads=3, max_seq_len=32)


def test_module_rejects_too_long_input():
    mha = RotaryYatAttention(embed_dim=32, num_heads=2, max_seq_len=8)
    with pytest.raises(ValueError):
        mha(mx.random.normal(shape=(1, 10, 32)))


def test_module_constant_alpha_drops_learnable():
    mha = RotaryYatAttention(
        embed_dim=32, num_heads=2, max_seq_len=16, constant_alpha=True
    )
    _ = mha(mx.random.normal(shape=(1, 4, 32)))
    assert mha._constant_alpha_value == math.sqrt(2.0)
    assert "alpha" not in mha.parameters()


def test_module_no_out_proj():
    mha = RotaryYatAttention(
        embed_dim=32, num_heads=2, max_seq_len=16, use_out_proj=False
    )
    _ = mha(mx.random.normal(shape=(1, 4, 32)))
    assert "out_kernel" not in mha.parameters()


def test_module_gradient_reduces_loss():
    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    mha = RotaryYatAttention(embed_dim=32, num_heads=2, max_seq_len=16)
    x = mx.random.normal(shape=(2, 5, 32))
    y = mx.random.normal(shape=(2, 5, 32))
    _ = mha(x)  # build lazy params

    grad_fn = mlx_nn.value_and_grad(mha, loss_fn)
    loss, grads = grad_fn(mha, x, y)
    assert {"q_kernel", "k_kernel", "v_kernel", "out_kernel"} <= set(grads.keys())

    opt = mlx_optim.AdamW(learning_rate=1e-2)
    opt.update(mha, grads)
    mx.eval(mha.parameters())
    loss_after = float(loss_fn(mha, x, y))
    assert loss_after < float(loss)


def test_module_causal_mask_zeroes_upper_triangle():
    """Attention with a causal mask should respect token ordering."""
    mha = RotaryYatAttention(embed_dim=32, num_heads=2, max_seq_len=16)
    x = mx.random.normal(shape=(1, 6, 32))
    causal = np.tril(np.ones((6, 6), dtype=bool))[None, None, :, :]
    mask = mx.array(np.broadcast_to(causal, (1, 2, 6, 6)))
    # Use the functional helpers directly to inspect attention weights.
    q = mx.reshape(mha._linear(x, mha.q_kernel, getattr(mha, "q_bias", None)),
                   (1, 6, 2, 16)) if mha.is_built else None
    if q is None:
        _ = mha(x)
        q = mx.reshape(mha._linear(x, mha.q_kernel, getattr(mha, "q_bias", None)),
                       (1, 6, 2, 16))
    k = mx.reshape(mha._linear(x, mha.k_kernel, getattr(mha, "k_bias", None)),
                   (1, 6, 2, 16))
    w = np.array(rotary_yat_attention_weights(q, k, mha.freqs_cos, mha.freqs_sin, mask=mask))
    upper = np.triu(np.ones_like(w[0, 0]), k=1).astype(bool)
    assert float(np.max(np.abs(w[..., upper]))) < 1e-5
