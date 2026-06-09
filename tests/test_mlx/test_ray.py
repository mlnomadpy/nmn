"""Tests for MLX RAY (radial) bias-aware spherical-YAT features."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import (  # noqa: E402
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)


def test_projection_shapes():
    d, m, R, D = 16, 8, 4, 8
    p = create_radial_projection(
        head_dim=d, sketch_m=m, num_radial=R, radial_dim=D,
        bias=1.0, epsilon=0.5, seed=0,
    )
    assert p["W1"].shape == (m, d + 1)
    assert p["W2"].shape == (m, d + 1)
    assert p["omegas"].shape == (R, D, d + 1)
    assert p["phases"].shape == (R, D)
    assert p["coef"].shape == (R,)


def test_seeded_projection_is_deterministic():
    p1 = create_radial_projection(head_dim=8, seed=3)
    p2 = create_radial_projection(head_dim=8, seed=3)
    assert np.array_equal(np.array(p1["W1"]), np.array(p2["W1"]))
    assert np.array_equal(np.array(p1["omegas"]), np.array(p2["omegas"]))


def test_feature_shape_is_product():
    d, m, R, D = 16, 8, 4, 8
    p = create_radial_projection(head_dim=d, sketch_m=m, num_radial=R,
                                 radial_dim=D, seed=0)
    x = mx.random.normal(shape=(2, 5, 3, d))
    feat = radial_features(x, p)
    assert feat.shape == (2, 5, 3, m * R * D)


def test_features_no_nan_inf():
    p = create_radial_projection(head_dim=16, bias=1.0, epsilon=0.5, seed=0)
    x = mx.random.normal(shape=(2, 6, 2, 16)) * 4.0
    feat = np.array(radial_features(x, p))
    assert np.isfinite(feat).all()


def test_features_deterministic():
    p = create_radial_projection(head_dim=8, seed=0)
    x = mx.random.normal(shape=(1, 4, 2, 8))
    f1 = np.array(radial_features(x, p))
    f2 = np.array(radial_features(x, p))
    assert np.array_equal(f1, f2)


def test_attention_output_shape_non_causal():
    p = create_radial_projection(head_dim=16, seed=0)
    q = mx.random.normal(shape=(2, 4, 3, 16))
    k = mx.random.normal(shape=(2, 5, 3, 16))
    v = mx.random.normal(shape=(2, 5, 3, 16))
    out = radial_yat_attention(q, k, v, p, causal=False)
    assert out.shape == (2, 4, 3, 16)
    assert np.isfinite(np.array(out)).all()


def test_attention_output_shape_causal():
    p = create_radial_projection(head_dim=16, seed=0)
    x = mx.random.normal(shape=(1, 6, 2, 16))
    out = radial_yat_attention(x, x, x, p, causal=True)
    assert out.shape == (1, 6, 2, 16)
    assert np.isfinite(np.array(out)).all()


def test_causal_position_0_unaffected_by_future():
    mx.random.seed(0)
    p = create_radial_projection(head_dim=8, seed=0)
    x = mx.random.normal(shape=(1, 5, 1, 8))
    v = mx.random.normal(shape=(1, 5, 1, 8))
    out_a = radial_yat_attention(x, x, v, p, causal=True)
    v_pert = mx.concatenate([v[:, :4], v[:, 4:5] + 100.0], axis=1)
    out_b = radial_yat_attention(x, x, v_pert, p, causal=True)
    assert float(mx.max(mx.abs(out_a[:, 0] - out_b[:, 0]))) < 1e-2


def test_attention_dtype_promotion():
    p = create_radial_projection(head_dim=8, seed=0)
    q = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    k = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    v = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float16)
    out = radial_yat_attention(q, k, v, p)
    assert out.dtype == mx.float32
    assert out.shape == (1, 3, 1, 8)
