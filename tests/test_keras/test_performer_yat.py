"""Tests for Keras MAY (Random Maclaurin) and RAY (radial) feature maps.

The Keras/TensorFlow backend is not installed in the local dev environment, so
these tests are guarded by ``pytest.importorskip("tensorflow")`` and run in CI.

The math mirrors the verified NumPy reference at ``.context/may-ray/reference.py``
exactly, so the qualitative guarantees (inner product tracks kappa, MAY beats
the bias-free SLAY baseline at ``b >= 1``) are reproduced here.
"""

import math

import numpy as np
import pytest

pytest.importorskip("tensorflow")
keras = pytest.importorskip("keras")

from nmn.keras.performer_yat import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)


# --------------------------------------------------------------------------
# Pure-NumPy oracles (copied from the verified reference) for cross-checking.
# --------------------------------------------------------------------------
def _normalize_rows(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _exact_kernel(s, b, eps):
    C = 2.0 + eps
    return (s + b) ** 2 / (C - 2.0 * s)


def _exact_attention(q, k, v, b, eps):
    qn, kn = _normalize_rows(q), _normalize_rows(k)
    s = qn @ kn.T
    w = _exact_kernel(s, b, eps)
    w = w / (w.sum(-1, keepdims=True) + 1e-9)
    return w @ v


def _linear_attention_np(phi_q, phi_k, v, eps=1e-9):
    kv = phi_k.T @ v
    num = phi_q @ kv
    den = phi_q @ phi_k.sum(0)
    return num / (den[:, None] + eps)


def _slay_create(d, P=32, M=8, R=1, eps=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    C = 2.0 + eps
    nodes, weights = np.polynomial.laguerre.laggauss(R)
    nodes, weights = nodes / C, weights / C
    anchors = _normalize_rows(rng.standard_normal((P, d)))
    proj = rng.standard_normal((R, M, d))
    return dict(anchors=anchors, proj=proj, nodes=nodes, weights=weights, P=P, M=M, R=R)


def _slay_features(x, params):
    x = _normalize_rows(x)
    P, M, R = params["P"], params["M"], params["R"]
    poly = (x @ params["anchors"].T) ** 2 / np.sqrt(P)
    proj_all = np.einsum("nd,rmd->nrm", x, params["proj"])
    out = []
    for r in range(R):
        s = params["nodes"][r]
        arg = np.clip(proj_all[:, r] * np.sqrt(2 * max(s, 0)) - s, -20, 20)
        prf = np.exp(arg) / np.sqrt(M)
        fused = np.einsum("np,nm->npm", poly, prf).reshape(x.shape[0], -1)
        out.append(np.sqrt(max(params["weights"][r], 0)) * fused)
    return np.concatenate(out, axis=1)


def _median_sq_distance(qn, kn):
    diff = qn[:, None, :] - kn[None, :, :]
    return float(np.median((diff * diff).sum(-1)))


def _per_token_cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


# --------------------------------------------------------------------------
# Maclaurin coefficient sanity
# --------------------------------------------------------------------------
def test_maclaurin_coeffs_nonnegative():
    a = maclaurin_coeffs(b=1.0, eps=0.05, nmax=40)
    assert a.shape == (41,)
    assert np.all(a >= 0.0)
    assert a.sum() > 0.0


# --------------------------------------------------------------------------
# MAY: shapes, finiteness, determinism
# --------------------------------------------------------------------------
def test_may_feature_shape():
    d, M = 16, 64
    params = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=0)
    x = np.random.randn(2, 5, 4, d).astype(np.float32)
    feat = np.array(maclaurin_features(x, params))
    assert feat.shape == (2, 5, 4, M)


def test_may_no_nan_inf():
    d, M = 16, 64
    params = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=0)
    x = np.random.randn(2, 5, 4, d).astype(np.float32)
    feat = np.array(maclaurin_features(x, params))
    assert np.all(np.isfinite(feat))


def test_may_determinism():
    d, M = 16, 64
    x = np.random.randn(3, 7, 2, d).astype(np.float32)
    p1 = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=42)
    p2 = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=42)
    f1 = np.array(maclaurin_features(x, p1))
    f2 = np.array(maclaurin_features(x, p2))
    np.testing.assert_allclose(f1, f2, atol=1e-6)


def test_may_attention_shape_and_finite():
    d, M = 16, 128
    params = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=0)
    q = np.random.randn(2, 6, 3, d).astype(np.float32)
    k = np.random.randn(2, 8, 3, d).astype(np.float32)
    v = np.random.randn(2, 8, 3, 10).astype(np.float32)
    out = np.array(maclaurin_yat_attention(q, k, v, params))
    assert out.shape == (2, 6, 3, 10)
    assert np.all(np.isfinite(out))


def test_may_causal_shape():
    d, M = 16, 64
    params = create_maclaurin_projection(d, num_features=M, bias=1.0, epsilon=0.05, seed=0)
    q = np.random.randn(2, 7, 2, d).astype(np.float32)
    v = np.random.randn(2, 7, 2, d).astype(np.float32)
    out = np.array(maclaurin_yat_attention(q, q, v, params, causal=True))
    assert out.shape == (2, 7, 2, d)
    assert np.all(np.isfinite(out))


# --------------------------------------------------------------------------
# MAY: inner product tracks kappa (unbiased estimator), averaged over seeds.
# --------------------------------------------------------------------------
def test_may_inner_product_tracks_kappa():
    d, M, b, eps = 16, 4096, 1.0, 0.05
    rng = np.random.default_rng(0)
    q = rng.standard_normal((200, d)).astype(np.float32)
    k = rng.standard_normal((200, d)).astype(np.float32)
    qhat, khat = _normalize_rows(q), _normalize_rows(k)
    s = (qhat * khat).sum(-1)
    kap = _exact_kernel(s, b, eps)

    # Average the (high-variance, unbiased) inner product over several seeds.
    ips = []
    for seed in range(40):
        params = create_maclaurin_projection(d, num_features=M, bias=b, epsilon=eps, seed=seed)
        qf = np.array(maclaurin_features(q, params))
        kf = np.array(maclaurin_features(k, params))
        ips.append((qf * kf).sum(-1))
    ip = np.mean(ips, axis=0)

    # Tracks kappa (rises with s) — strong positive correlation.
    assert np.corrcoef(ip, kap)[0, 1] > 0.9


def test_may_more_features_better():
    """More features -> higher cosine similarity to exact attention."""
    d, N, b = 32, 128, 1.0
    rng = np.random.default_rng(7)
    q = rng.standard_normal((N, d)).astype(np.float32)
    k = rng.standard_normal((N, d)).astype(np.float32)
    v = rng.standard_normal((N, d)).astype(np.float32)
    eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
    exact = _exact_attention(q, k, v, b, eps)

    def may_cos(M):
        # average over seeds to reduce variance of the estimate
        cs = []
        for seed in range(5):
            p = create_maclaurin_projection(d, num_features=M, bias=b, epsilon=eps, seed=seed)
            qf = np.array(maclaurin_features(q, p))
            kf = np.array(maclaurin_features(k, p))
            out = _linear_attention_np(qf, kf, v)
            cs.append(_per_token_cos(out, exact))
        return float(np.mean(cs))

    assert may_cos(512) > may_cos(64)


def test_may_beats_slay_at_b1():
    """At b >= 1 MAY (bias-aware) must clearly beat SLAY (bias-free)."""
    d, N, b = 64, 256, 1.0
    F = 256  # matched feature budget
    seeds = [0, 1, 2]
    may_cos, slay_cos = [], []
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        q = rng.standard_normal((N, d)).astype(np.float32)
        k = rng.standard_normal((N, d)).astype(np.float32)
        v = rng.standard_normal((N, d)).astype(np.float32)
        eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
        exact = _exact_attention(q, k, v, b, eps)

        mp = create_maclaurin_projection(d, num_features=F, bias=b, epsilon=eps, seed=seed)
        qf = np.array(maclaurin_features(q, mp))
        kf = np.array(maclaurin_features(k, mp))
        may_out = _linear_attention_np(qf, kf, v)
        may_cos.append(_per_token_cos(may_out, exact))

        sp = _slay_create(d, P=32, M=8, R=1, eps=eps, seed=seed)
        sa = _linear_attention_np(_slay_features(q, sp), _slay_features(k, sp), v)
        slay_cos.append(_per_token_cos(sa, exact))

    assert np.mean(may_cos) > np.mean(slay_cos)


# --------------------------------------------------------------------------
# RAY: shapes, finiteness, determinism
# --------------------------------------------------------------------------
def test_ray_feature_shape():
    d = 16
    params = create_radial_projection(
        d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.05, seed=0
    )
    x = np.random.randn(2, 5, 3, d).astype(np.float32)
    feat = np.array(radial_features(x, params))
    assert feat.shape == (2, 5, 3, 8 * 4 * 8)
    assert np.all(np.isfinite(feat))


def test_ray_determinism():
    d = 16
    x = np.random.randn(2, 5, 3, d).astype(np.float32)
    p1 = create_radial_projection(d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.05, seed=1)
    p2 = create_radial_projection(d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.05, seed=1)
    np.testing.assert_allclose(
        np.array(radial_features(x, p1)), np.array(radial_features(x, p2)), atol=1e-6
    )


def test_ray_attention_shape_and_finite():
    d = 16
    params = create_radial_projection(
        d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.05, seed=0
    )
    q = np.random.randn(2, 6, 2, d).astype(np.float32)
    k = np.random.randn(2, 8, 2, d).astype(np.float32)
    v = np.random.randn(2, 8, 2, 10).astype(np.float32)
    out = np.array(radial_yat_attention(q, k, v, params))
    assert out.shape == (2, 6, 2, 10)
    assert np.all(np.isfinite(out))
