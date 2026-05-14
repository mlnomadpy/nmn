"""Tests for the MLX Spherical-YAT-Performer."""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import (  # noqa: E402
    create_yat_tp_projection,
    yat_tp_features,
    yat_tp_attention,
)


# ---------------------------------------------------------------------------
# create_yat_tp_projection
# ---------------------------------------------------------------------------


def test_projection_shapes():
    params = create_yat_tp_projection(
        head_dim=16, num_prf_features=8, num_quad_nodes=2,
        num_anchor_features=16, seed=0,
    )
    assert params["projections"].shape == (2, 8, 16)
    assert params["anchors"].shape == (16, 16)
    assert params["quad_nodes"].shape == (2,)
    assert params["quad_weights"].shape == (2,)
    assert params["num_prf_features"] == 8
    assert params["num_anchor_features"] == 16
    assert params["num_scales"] == 2


def test_anchors_unit_norm():
    params = create_yat_tp_projection(
        head_dim=8, num_anchor_features=10, num_prf_features=4, seed=0,
    )
    anchors = np.array(params["anchors"])
    norms = np.linalg.norm(anchors, axis=-1)
    assert np.max(np.abs(norms - 1.0)) < 1e-5


def test_seeded_projection_is_deterministic():
    p1 = create_yat_tp_projection(head_dim=8, num_anchor_features=4,
                                  num_prf_features=4, seed=42)
    p2 = create_yat_tp_projection(head_dim=8, num_anchor_features=4,
                                  num_prf_features=4, seed=42)
    assert np.array_equal(np.array(p1["anchors"]), np.array(p2["anchors"]))
    assert np.array_equal(np.array(p1["projections"]), np.array(p2["projections"]))


# ---------------------------------------------------------------------------
# yat_tp_features
# ---------------------------------------------------------------------------


def test_feature_shape_matches_rpm():
    """Feature dim should equal R·P·M."""
    R, P, M, d = 2, 16, 8, 16
    params = create_yat_tp_projection(
        head_dim=d, num_quad_nodes=R, num_anchor_features=P,
        num_prf_features=M, seed=0,
    )
    x = mx.random.normal(shape=(1, 5, 2, d))
    feat = yat_tp_features(x, params)
    assert feat.shape == (1, 5, 2, R * P * M)


def test_feature_non_negative():
    """φ(x) = √w · (a·x)² · exp(...) ≥ 0 always."""
    params = create_yat_tp_projection(
        head_dim=8, num_anchor_features=8, num_prf_features=4, seed=0,
    )
    x = mx.random.normal(shape=(1, 4, 2, 8)) * 3.0
    feat = np.array(yat_tp_features(x, params))
    # Tiny negatives only from fp32 rounding inside the normalization
    # denominator — allow ~1e-7 slack.
    assert float(feat.min()) >= -1e-6


def test_feature_normalization_is_optional():
    """``normalize=False`` should produce different features than the
    default (which L2-normalizes)."""
    params = create_yat_tp_projection(
        head_dim=8, num_anchor_features=4, num_prf_features=4, seed=0,
    )
    x = mx.random.normal(shape=(1, 4, 2, 8)) * 5.0
    feat_norm = np.array(yat_tp_features(x, params, normalize=True))
    feat_raw = np.array(yat_tp_features(x, params, normalize=False))
    assert np.max(np.abs(feat_norm - feat_raw)) > 1e-2


# ---------------------------------------------------------------------------
# yat_tp_attention
# ---------------------------------------------------------------------------


def test_attention_output_shape_non_causal():
    params = create_yat_tp_projection(head_dim=16, seed=0)
    q = mx.random.normal(shape=(2, 4, 3, 16))
    k = mx.random.normal(shape=(2, 5, 3, 16))
    v = mx.random.normal(shape=(2, 5, 3, 16))
    out = yat_tp_attention(q, k, v, params, causal=False)
    assert out.shape == (2, 4, 3, 16)


def test_attention_output_shape_causal():
    params = create_yat_tp_projection(head_dim=16, seed=0)
    q = mx.random.normal(shape=(1, 6, 2, 16))
    out = yat_tp_attention(q, q, q, params, causal=True)
    assert out.shape == (1, 6, 2, 16)


def test_causal_attention_position_0_unaffected_by_future_values():
    """Mutating value[t > 0] must not change output[t = 0] under causal mask."""
    mx.random.seed(0)
    params = create_yat_tp_projection(head_dim=8, seed=0)
    x = mx.random.normal(shape=(1, 5, 1, 8))
    v = mx.random.normal(shape=(1, 5, 1, 8))
    out_a = yat_tp_attention(x, x, v, params, causal=True)
    # Perturb v at the last timestep.
    v_perturbed = mx.concatenate([v[:, :4], v[:, 4:5] + 100.0], axis=1)
    out_b = yat_tp_attention(x, x, v_perturbed, params, causal=True)
    assert float(mx.max(mx.abs(out_a[:, 0] - out_b[:, 0]))) < 1e-3
    # And the later position should change.
    assert float(mx.max(mx.abs(out_a[:, 4] - out_b[:, 4]))) > 1e-2


def test_causal_and_non_causal_differ():
    mx.random.seed(0)
    params = create_yat_tp_projection(head_dim=16, seed=0)
    x = mx.random.normal(shape=(1, 8, 2, 16))
    v = mx.random.normal(shape=(1, 8, 2, 16))
    out_nc = yat_tp_attention(x, x, v, params, causal=False)
    out_c = yat_tp_attention(x, x, v, params, causal=True)
    assert float(mx.max(mx.abs(out_nc - out_c))) > 1e-3


def test_kernel_approximation_quality_at_dot_zero():
    """When q·k = 0, the exact spherical YAT score is 0; the linearized
    feature inner product should be small too."""
    params = create_yat_tp_projection(
        head_dim=16, num_anchor_features=32, num_prf_features=8,
        num_quad_nodes=2, seed=0,
    )
    # Build q and k orthogonal unit vectors.
    rng = np.random.default_rng(0)
    q_np = rng.standard_normal(16).astype(np.float32)
    q_np /= np.linalg.norm(q_np)
    k_np = rng.standard_normal(16).astype(np.float32)
    k_np -= np.dot(k_np, q_np) * q_np
    k_np /= np.linalg.norm(k_np)
    assert abs(float(np.dot(q_np, k_np))) < 1e-5

    q = mx.array(q_np).reshape((1, 1, 1, 16))
    k = mx.array(k_np).reshape((1, 1, 1, 16))
    qf = yat_tp_features(q, params)
    kf = yat_tp_features(k, params)
    score = float(mx.sum(qf * kf))
    # Exact: (q·k)² / (C − 2·q·k) = 0 / (2+ε − 0) = 0.
    # Approx should be very small (PRF tails decay → magnitude ~ 1/√(P·M·R)).
    assert score < 0.05


def test_yat_tp_attention_dtype_promotion():
    """Mixed dtypes should not crash — query dtype wins."""
    params = create_yat_tp_projection(head_dim=8, seed=0, dtype=mx.float32)
    q = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    k = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    v = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float16)
    out = yat_tp_attention(q, k, v, params)
    assert out.dtype == mx.float32
    assert out.shape == (1, 3, 1, 8)
