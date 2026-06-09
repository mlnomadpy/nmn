"""Tests for TensorFlow MAY (Random Maclaurin) and RAY (radial) feature maps.

These mirror the verified NumPy reference at ``.context/may-ray/reference.py``.
TensorFlow is not installed in the local dev environment, so the whole module is
guarded by ``pytest.importorskip`` and exercised in CI.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf.performer_yat import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
)


# --------------------------------------------------------------------------
# NumPy reference helpers (verified math) for the kernel / baselines
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


def _median_sq_distance(qn, kn):
    diff = qn[:, None, :] - kn[None, :, :]
    return float(np.median((diff * diff).sum(-1)))


def _per_token_cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


# NumPy SLAY baseline (bias-free b=0 design point), straight from the reference.
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


def _linear_attention_np(phi_q, phi_k, v, eps=1e-9):
    kv = phi_k.T @ v
    num = phi_q @ kv
    den = phi_q @ phi_k.sum(0)
    return num / (den[:, None] + eps)


def _bhd(x):
    """[N, d] -> [1, N, 1, d] (batch=1, single head) for the tf attention API."""
    return tf.constant(x[None, :, None, :], dtype=tf.float32)


# --------------------------------------------------------------------------
# Maclaurin coefficients
# --------------------------------------------------------------------------
def test_maclaurin_coeffs_nonnegative():
    a = maclaurin_coeffs(b=1.0, eps=0.5, nmax=40)
    assert a.shape == (41,)
    assert np.all(a >= 0.0)
    assert a.sum() > 0.0


# --------------------------------------------------------------------------
# MAY -- shapes, finiteness, determinism
# --------------------------------------------------------------------------
def test_may_features_shape():
    params = create_maclaurin_projection(head_dim=16, num_features=64, bias=1.0, epsilon=0.5, seed=0)
    x = tf.random.normal((2, 10, 4, 16))
    feat = maclaurin_features(x, params)
    assert feat.shape == (2, 10, 4, 64)


def test_may_features_finite():
    params = create_maclaurin_projection(head_dim=16, num_features=128, bias=1.0, epsilon=0.5, seed=0)
    x = tf.random.normal((2, 10, 4, 16))
    feat = maclaurin_features(x, params).numpy()
    assert np.all(np.isfinite(feat))


def test_may_features_deterministic():
    p0 = create_maclaurin_projection(head_dim=16, num_features=64, bias=1.0, epsilon=0.5, seed=7)
    p1 = create_maclaurin_projection(head_dim=16, num_features=64, bias=1.0, epsilon=0.5, seed=7)
    np.testing.assert_array_equal(p0["omegas"].numpy(), p1["omegas"].numpy())
    np.testing.assert_array_equal(p0["degrees"].numpy(), p1["degrees"].numpy())
    x = tf.random.normal((3, 5, 2, 16))
    f0 = maclaurin_features(x, p0).numpy()
    f1 = maclaurin_features(x, p1).numpy()
    np.testing.assert_allclose(f0, f1, rtol=1e-6, atol=1e-6)


def test_may_attention_shape_and_finite():
    params = create_maclaurin_projection(head_dim=16, num_features=128, bias=1.0, epsilon=0.5, seed=0)
    q = tf.random.normal((2, 6, 4, 16))
    k = tf.random.normal((2, 8, 4, 16))
    v = tf.random.normal((2, 8, 4, 16))
    out = maclaurin_yat_attention(q, k, v, params)
    assert out.shape == (2, 6, 4, 16)
    assert np.all(np.isfinite(out.numpy()))


def test_may_attention_causal_shape():
    params = create_maclaurin_projection(head_dim=16, num_features=64, bias=1.0, epsilon=0.5, seed=0)
    q = tf.random.normal((2, 7, 4, 16))
    k = tf.random.normal((2, 7, 4, 16))
    v = tf.random.normal((2, 7, 4, 16))
    out = maclaurin_yat_attention(q, k, v, params, causal=True)
    assert out.shape == (2, 7, 4, 16)
    assert np.all(np.isfinite(out.numpy()))


# --------------------------------------------------------------------------
# MAY -- unbiasedness: inner product tracks kappa
# --------------------------------------------------------------------------
def test_may_inner_product_tracks_kappa():
    """E[phi(q)·phi(k)] = kappa(q·k). The empirical Gram entries must correlate
    strongly with the exact kernel and rise with s."""
    d, b, eps = 32, 1.0, 0.5
    M = 20000  # many features -> tight Monte-Carlo estimate
    params = create_maclaurin_projection(head_dim=d, num_features=M, bias=b, epsilon=eps, seed=1)

    rng = np.random.default_rng(0)
    x = _normalize_rows(rng.standard_normal((8, d)))
    xt = tf.constant(x[None, :, None, :], dtype=tf.float32)  # [1, N, 1, d]
    feat = maclaurin_features(xt, params).numpy()[0, :, 0, :]  # [N, M]

    gram = feat @ feat.T
    s = x @ x.T
    iu = np.triu_indices(8, k=1)
    approx = gram[iu]
    exact = _exact_kernel(s[iu], b, eps)

    # Strong linear correlation between approximation and exact kernel.
    corr = np.corrcoef(approx, exact)[0, 1]
    assert corr > 0.95
    # Reasonable absolute agreement with a large feature budget.
    np.testing.assert_allclose(approx, exact, atol=0.12)


def test_may_more_features_better():
    """More features -> closer to exact attention (higher cosine)."""
    d, N, b = 48, 96, 1.0
    rng = np.random.default_rng(3)
    q = rng.standard_normal((N, d))
    k = rng.standard_normal((N, d))
    v = rng.standard_normal((N, d))
    eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
    exact = _exact_attention(q, k, v, b, eps)

    def cos_for(M):
        params = create_maclaurin_projection(head_dim=d, num_features=M, bias=b, epsilon=eps, seed=0)
        out = maclaurin_yat_attention(_bhd(q), _bhd(k), _bhd(v), params).numpy()[0, :, 0, :]
        return _per_token_cos(out, exact)

    assert cos_for(512) > cos_for(32)


def test_may_beats_slay_at_high_bias():
    """At b >= 1 MAY (bias-aware) must clearly beat SLAY (bias-free anchor)."""
    d, N, F = 64, 256, 256
    seeds = [0, 1, 2]
    for b in (1.0, 2.0):
        may_cos, slay_cos = [], []
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            q = rng.standard_normal((N, d))
            k = rng.standard_normal((N, d))
            v = rng.standard_normal((N, d))
            eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
            exact = _exact_attention(q, k, v, b, eps)

            mp = create_maclaurin_projection(head_dim=d, num_features=F, bias=b, epsilon=eps, seed=seed)
            ma = maclaurin_yat_attention(_bhd(q), _bhd(k), _bhd(v), mp).numpy()[0, :, 0, :]
            may_cos.append(_per_token_cos(ma, exact))

            sp = _slay_create(d, P=32, M=8, R=1, eps=eps, seed=seed)
            sa = _linear_attention_np(_slay_features(q, sp), _slay_features(k, sp), v)
            slay_cos.append(_per_token_cos(sa, exact))

        assert np.mean(may_cos) > np.mean(slay_cos), (
            f"b={b}: MAY {np.mean(may_cos):.3f} <= SLAY {np.mean(slay_cos):.3f}"
        )


# --------------------------------------------------------------------------
# RAY -- shapes, finiteness, determinism
# --------------------------------------------------------------------------
def test_ray_features_shape():
    params = create_radial_projection(
        head_dim=16, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.5, seed=0
    )
    x = tf.random.normal((2, 10, 4, 16))
    feat = radial_features(x, params)
    assert feat.shape == (2, 10, 4, 8 * 4 * 8)


def test_ray_features_finite_and_deterministic():
    p0 = create_radial_projection(
        head_dim=16, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.5, seed=5
    )
    p1 = create_radial_projection(
        head_dim=16, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.5, seed=5
    )
    x = tf.random.normal((2, 6, 2, 16))
    f0 = radial_features(x, p0).numpy()
    f1 = radial_features(x, p1).numpy()
    assert np.all(np.isfinite(f0))
    np.testing.assert_allclose(f0, f1, rtol=1e-6, atol=1e-6)


def test_ray_attention_shape_and_finite():
    params = create_radial_projection(
        head_dim=16, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.5, seed=0
    )
    q = tf.random.normal((2, 6, 4, 16))
    k = tf.random.normal((2, 8, 4, 16))
    v = tf.random.normal((2, 8, 4, 16))
    out = radial_yat_attention(q, k, v, params)
    assert out.shape == (2, 6, 4, 16)
    assert np.all(np.isfinite(out.numpy()))
