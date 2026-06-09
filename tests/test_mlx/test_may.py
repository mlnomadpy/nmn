"""Tests for MLX MAY (Random Maclaurin) bias-aware spherical-YAT features."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import (  # noqa: E402
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_yat_tp_projection,
    yat_tp_attention,
)


# ---------------------------------------------------------------------------
# Helpers (NumPy reference mirrors).
# ---------------------------------------------------------------------------


def _normalize_rows(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _kappa(s, b, eps):
    C = 2.0 + eps
    return (s + b) ** 2 / (C - 2.0 * s)


def _median_sq(qn, kn):
    diff = qn[:, None, :] - kn[None, :, :]
    return float(np.median((diff * diff).sum(-1)))


def _exact_attention(q, k, v, b, eps):
    qn, kn = _normalize_rows(q), _normalize_rows(k)
    s = qn @ kn.T
    w = _kappa(s, b, eps)
    w = w / (w.sum(-1, keepdims=True) + 1e-9)
    return w @ v


def _cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# coefficients + projection
# ---------------------------------------------------------------------------


def test_maclaurin_coeffs_non_negative():
    a = maclaurin_coeffs(b=1.0, epsilon=0.5, nmax=40)
    assert a.shape == (41,)
    assert float(a.min()) >= 0.0
    assert float(a.sum()) > 0.0


def test_projection_shapes():
    p = create_maclaurin_projection(
        head_dim=16, num_features=64, bias=1.0, epsilon=0.5, nmax=40, seed=0,
    )
    assert p["omegas"].shape == (64, 40, 16)
    assert p["degrees"].shape == (64,)
    assert p["num_features"] == 64
    assert p["nmax"] == 40
    assert p["Z"] > 0.0


def test_projection_rademacher_entries():
    p = create_maclaurin_projection(head_dim=8, num_features=16, seed=0)
    om = np.array(p["omegas"])
    assert set(np.unique(om).tolist()).issubset({-1.0, 1.0})


def test_seeded_projection_is_deterministic():
    p1 = create_maclaurin_projection(head_dim=8, num_features=16, seed=42)
    p2 = create_maclaurin_projection(head_dim=8, num_features=16, seed=42)
    assert np.array_equal(np.array(p1["omegas"]), np.array(p2["omegas"]))
    assert np.array_equal(np.array(p1["degrees"]), np.array(p2["degrees"]))


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def test_feature_shape_is_M():
    M, d = 32, 16
    p = create_maclaurin_projection(head_dim=d, num_features=M, seed=0)
    x = mx.random.normal(shape=(2, 5, 3, d))
    feat = maclaurin_features(x, p)
    assert feat.shape == (2, 5, 3, M)


def test_features_no_nan_inf():
    p = create_maclaurin_projection(head_dim=16, num_features=64, bias=1.0, seed=0)
    x = mx.random.normal(shape=(2, 6, 2, 16)) * 4.0
    feat = np.array(maclaurin_features(x, p))
    assert np.isfinite(feat).all()


def test_features_deterministic():
    p = create_maclaurin_projection(head_dim=8, num_features=32, seed=0)
    x = mx.random.normal(shape=(1, 4, 2, 8))
    f1 = np.array(maclaurin_features(x, p))
    f2 = np.array(maclaurin_features(x, p))
    assert np.array_equal(f1, f2)


def test_inner_product_tracks_kappa():
    """E[phi(q)·phi(k)] = kappa(q·k): pointwise IP should track kappa and rise
    monotonically with the cosine similarity s."""
    d, M, b, eps = 32, 8000, 1.0, 0.5
    # MAY is an unbiased estimator; average several independent projections to
    # tame the Monte-Carlo variance of the high-degree terms.
    projs = [create_maclaurin_projection(head_dim=d, num_features=M, bias=b,
                                         epsilon=eps, seed=s) for s in range(8)]
    rng = np.random.default_rng(1)
    q = rng.standard_normal(d).astype(np.float32)
    q /= np.linalg.norm(q)
    svals = [-0.5, 0.0, 0.5, 0.8]
    ips, ks = [], []
    for s in svals:
        r = rng.standard_normal(d).astype(np.float32)
        r -= r.dot(q) * q
        r /= np.linalg.norm(r)
        k = (s * q + np.sqrt(1.0 - s * s) * r).astype(np.float32)
        est = []
        for p in projs:
            qf = np.array(maclaurin_features(mx.array(q).reshape(1, 1, 1, d), p)).ravel()
            kf = np.array(maclaurin_features(mx.array(k).reshape(1, 1, 1, d), p)).ravel()
            est.append(float(qf.dot(kf)))
        ips.append(float(np.mean(est)))
        ks.append(float(_kappa(s, b, eps)))
    # Monotone increasing in s (the kernel rises with similarity).
    assert all(ips[i] < ips[i + 1] for i in range(len(ips) - 1))
    # Tracks kappa to a loose Monte-Carlo tolerance.
    ips, ks = np.array(ips), np.array(ks)
    assert np.max(np.abs(ips - ks)) < 0.25


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------


def test_attention_output_shape_non_causal():
    p = create_maclaurin_projection(head_dim=16, num_features=128, seed=0)
    q = mx.random.normal(shape=(2, 4, 3, 16))
    k = mx.random.normal(shape=(2, 5, 3, 16))
    v = mx.random.normal(shape=(2, 5, 3, 16))
    out = maclaurin_yat_attention(q, k, v, p, causal=False)
    assert out.shape == (2, 4, 3, 16)
    assert np.isfinite(np.array(out)).all()


def test_attention_output_shape_causal():
    p = create_maclaurin_projection(head_dim=16, num_features=128, seed=0)
    x = mx.random.normal(shape=(1, 6, 2, 16))
    out = maclaurin_yat_attention(x, x, x, p, causal=True)
    assert out.shape == (1, 6, 2, 16)
    assert np.isfinite(np.array(out)).all()


def test_causal_position_0_unaffected_by_future():
    mx.random.seed(0)
    p = create_maclaurin_projection(head_dim=8, num_features=64, seed=0)
    x = mx.random.normal(shape=(1, 5, 1, 8))
    v = mx.random.normal(shape=(1, 5, 1, 8))
    out_a = maclaurin_yat_attention(x, x, v, p, causal=True)
    v_pert = mx.concatenate([v[:, :4], v[:, 4:5] + 100.0], axis=1)
    out_b = maclaurin_yat_attention(x, x, v_pert, p, causal=True)
    assert float(mx.max(mx.abs(out_a[:, 0] - out_b[:, 0]))) < 1e-2
    assert float(mx.max(mx.abs(out_a[:, 4] - out_b[:, 4]))) > 1e-2


def test_attention_dtype_promotion():
    p = create_maclaurin_projection(head_dim=8, num_features=32, seed=0)
    q = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    k = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float32)
    v = mx.random.normal(shape=(1, 3, 1, 8)).astype(mx.float16)
    out = maclaurin_yat_attention(q, k, v, p)
    assert out.dtype == mx.float32
    assert out.shape == (1, 3, 1, 8)


def test_may_beats_slay_at_bias_ge_1():
    """MAY is bias-aware; SLAY only approximates b=0. At b>=1 MAY should have
    clearly higher cosine similarity to exact attention."""
    d, N = 64, 256
    rng = np.random.default_rng(100)
    q = rng.standard_normal((N, d))
    k = rng.standard_normal((N, d))
    v = rng.standard_normal((N, d))
    eps = _median_sq(_normalize_rows(q), _normalize_rows(k))

    def to(x):
        return mx.array(x.astype(np.float32)).reshape(1, N, 1, d)

    qm, km, vm = to(q), to(k), to(v)
    for b in [1.0, 2.0]:
        exact = _exact_attention(q, k, v, b, eps)
        mp = create_maclaurin_projection(d, num_features=256, bias=b,
                                         epsilon=eps, seed=0)
        may = np.array(maclaurin_yat_attention(qm, km, vm, mp)).reshape(N, d)
        sp = create_yat_tp_projection(d, num_prf_features=8, num_quad_nodes=1,
                                      num_anchor_features=32, epsilon=eps, seed=0)
        slay = np.array(yat_tp_attention(qm, km, vm, sp)).reshape(N, d)
        assert _cos(may, exact) > _cos(slay, exact)


def test_more_features_higher_fidelity():
    """More MAY features -> closer to exact attention (cosine similarity up)."""
    d, N, b = 48, 192, 1.0
    rng = np.random.default_rng(7)
    q = rng.standard_normal((N, d))
    k = rng.standard_normal((N, d))
    v = rng.standard_normal((N, d))
    eps = _median_sq(_normalize_rows(q), _normalize_rows(k))

    def to(x):
        return mx.array(x.astype(np.float32)).reshape(1, N, 1, d)

    qm, km, vm = to(q), to(k), to(v)
    exact = _exact_attention(q, k, v, b, eps)
    cos_small, cos_large = [], []
    for seed in range(3):
        ps = create_maclaurin_projection(d, num_features=64, bias=b, epsilon=eps, seed=seed)
        pl = create_maclaurin_projection(d, num_features=1024, bias=b, epsilon=eps, seed=seed)
        a_s = np.array(maclaurin_yat_attention(qm, km, vm, ps)).reshape(N, d)
        a_l = np.array(maclaurin_yat_attention(qm, km, vm, pl)).reshape(N, d)
        cos_small.append(_cos(a_s, exact))
        cos_large.append(_cos(a_l, exact))
    assert np.mean(cos_large) > np.mean(cos_small)
