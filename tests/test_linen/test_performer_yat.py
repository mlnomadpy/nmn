"""Tests for the Linen MAY (Random Maclaurin) and RAY (radial) feature maps
and the generic linear-attention readout (issue #36).

These mirror the verified NumPy reference in ``.context/may-ray/reference.py``:
the MAY inner product must track the spherical kernel ``kappa``, and MAY must
clearly beat the bias-free SLAY baseline at ``b >= 1``.
"""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from nmn.linen import (  # noqa: E402
    spherical_kappa,
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
    linear_attention,
)


# --------------------------------------------------------------------------
# Helpers (NumPy SLAY baseline + exact attention, from the reference)
# --------------------------------------------------------------------------
def _normalize(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def _exact_attention(q, k, v, b, eps):
    qn, kn = _normalize(q), _normalize(k)
    s = qn @ kn.T
    w = (s + b) ** 2 / ((2.0 + eps) - 2.0 * s)
    w = w / (w.sum(-1, keepdims=True) + 1e-9)
    return w @ v


def _median_sq_distance(qn, kn):
    diff = qn[:, None, :] - kn[None, :, :]
    return float(np.median((diff * diff).sum(-1)))


def _slay_create(d, P, M, R, eps, seed):
    rng = np.random.default_rng(seed)
    C = 2.0 + eps
    nodes, weights = np.polynomial.laguerre.laggauss(R)
    nodes, weights = nodes / C, weights / C
    anchors = _normalize(rng.standard_normal((P, d)))
    proj = rng.standard_normal((R, M, d))
    return dict(anchors=anchors, proj=proj, nodes=nodes, weights=weights, P=P, M=M, R=R)


def _slay_features(x, params):
    x = _normalize(x)
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


def _np_linear_attention(phi_q, phi_k, v, eps=1e-9):
    kv = phi_k.T @ v
    num = phi_q @ kv
    den = phi_q @ phi_k.sum(0)
    return num / (den[:, None] + eps)


def _per_token_cos(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return float(np.mean(num / den))


# --------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------
def test_maclaurin_coeffs_nonnegative():
    a = np.asarray(maclaurin_coeffs(b=1.0, eps=0.5, nmax=40))
    assert (a >= 0).all()
    assert a.shape == (41,)
    assert a.sum() > 0


# --------------------------------------------------------------------------
# Shapes / finiteness / determinism
# --------------------------------------------------------------------------
def test_may_feature_shape_and_finite():
    d, M = 16, 64
    key = jax.random.PRNGKey(0)
    p = create_maclaurin_projection(key, d, num_features=M, bias=1.0, epsilon=0.5)
    x = jax.random.normal(jax.random.PRNGKey(3), (2, 5, 3, d))  # (B, L, H, d)
    feat = maclaurin_features(x, p)
    assert feat.shape == (2, 5, 3, M)
    assert bool(jnp.isfinite(feat).all())


def test_ray_feature_shape_and_finite():
    d = 16
    key = jax.random.PRNGKey(0)
    p = create_radial_projection(
        key, d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.5
    )
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 5, 3, d))
    feat = radial_features(x, p)
    assert feat.shape == (2, 5, 3, 8 * 4 * 8)
    assert bool(jnp.isfinite(feat).all())


def test_may_determinism():
    d, M = 16, 64
    key = jax.random.PRNGKey(7)
    p1 = create_maclaurin_projection(key, d, num_features=M, bias=1.0, epsilon=0.5)
    p2 = create_maclaurin_projection(key, d, num_features=M, bias=1.0, epsilon=0.5)
    x = jax.random.normal(jax.random.PRNGKey(9), (4, d))
    f1 = maclaurin_features(x, p1)
    f2 = maclaurin_features(x, p2)
    assert np.allclose(np.asarray(f1), np.asarray(f2))


def test_attention_shapes():
    d, dv, M = 16, 12, 64
    key = jax.random.PRNGKey(0)
    p = create_maclaurin_projection(key, d, num_features=M, bias=1.0, epsilon=0.5)
    q = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 3, d))
    k = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 3, d))
    v = jax.random.normal(jax.random.PRNGKey(3), (2, 7, 3, dv))
    out = maclaurin_yat_attention(q, k, v, p)
    assert out.shape == (2, 7, 3, dv)
    assert bool(jnp.isfinite(out).all())
    out_causal = maclaurin_yat_attention(q, k, v, p, causal=True)
    assert out_causal.shape == (2, 7, 3, dv)
    assert bool(jnp.isfinite(out_causal).all())


def test_ray_attention_shapes():
    d, dv = 16, 12
    key = jax.random.PRNGKey(0)
    p = create_radial_projection(
        key, d, sketch_m=4, num_radial=2, radial_dim=4, bias=1.0, epsilon=0.5
    )
    q = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 3, d))
    k = jax.random.normal(jax.random.PRNGKey(2), (2, 7, 3, d))
    v = jax.random.normal(jax.random.PRNGKey(3), (2, 7, 3, dv))
    out = radial_yat_attention(q, k, v, p)
    assert out.shape == (2, 7, 3, dv)
    assert bool(jnp.isfinite(out).all())


# --------------------------------------------------------------------------
# Kernel-approximation correctness: MAY inner product tracks kappa
# --------------------------------------------------------------------------
def test_may_inner_product_tracks_kappa():
    d, M, b, eps = 64, 30000, 1.0, 0.5
    key = jax.random.PRNGKey(0)
    p = create_maclaurin_projection(key, d, num_features=M, bias=b, epsilon=eps)

    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((6, d)))
    k = jnp.asarray(rng.standard_normal((6, d)))
    qn = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    kn = k / jnp.linalg.norm(k, axis=-1, keepdims=True)
    s = qn @ kn.T

    pq = maclaurin_features(q, p)
    pk = maclaurin_features(k, p)
    approx = np.asarray(pq @ pk.T)
    exact = np.asarray(spherical_kappa(s, b, eps))

    # Unbiased Monte-Carlo estimator: high correlation, modest relative error.
    rel = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    corr = np.corrcoef(approx.ravel(), exact.ravel())[0, 1]
    assert rel < 0.3
    assert corr > 0.95


def test_may_inner_product_rises_with_s():
    """kappa is increasing in s; the MAY inner product must track that trend.

    Averaged over several independent projections to reduce Monte-Carlo
    variance of the unbiased estimator.
    """
    d, M, b, eps = 32, 40000, 1.0, 0.5

    # Build pairs with an EXACT, increasing cosine s via Gram-Schmidt:
    # base and perp are orthonormal, so mix = s*base + sqrt(1-s^2)*perp has
    # cosine exactly s with base.
    rng = np.random.default_rng(0)
    base = _normalize(rng.standard_normal((1, d)))
    raw = rng.standard_normal((1, d))
    perp = raw - (raw @ base.T) * base
    perp = _normalize(perp)
    svals = np.linspace(-0.6, 0.9, 6)
    mixes = [s * base + np.sqrt(max(1 - s * s, 0.0)) * perp for s in svals]

    ips = np.zeros(len(svals))
    n_proj = 5
    for j in range(n_proj):
        p = create_maclaurin_projection(
            jax.random.PRNGKey(j), d, num_features=M, bias=b, epsilon=eps
        )
        pq = maclaurin_features(jnp.asarray(base), p)
        for i, mix in enumerate(mixes):
            pk = maclaurin_features(jnp.asarray(mix), p)
            ips[i] += float((pq * pk).sum())
    ips /= n_proj

    exact = np.asarray(spherical_kappa(svals, b, eps))
    # kappa is increasing (convex, not linear) in s; the unbiased estimator must
    # track that trend. We assert the robust trend statistics rather than strict
    # pointwise monotonicity, which is fragile to Monte-Carlo variance at very
    # negative s where kappa is tiny:
    #   (a) strong rank correlation with the exact kappa values,
    #   (b) a clear net increase across the swept range,
    #   (c) no dip larger than a small fraction of the value scale.
    assert np.corrcoef(ips, exact)[0, 1] > 0.99, f"weak correlation: {ips} vs {exact}"
    assert ips[-1] > ips[0], f"no net increase: {ips}"
    tol = 0.02 * (ips.max() - ips.min())
    assert np.all(np.diff(ips) > -tol), f"large non-monotone dip: {ips}"


def test_may_more_features_better():
    """More MAY features -> higher cosine similarity to exact attention."""
    d, N, b = 64, 128, 1.0
    rng = np.random.default_rng(5)
    q = rng.standard_normal((N, d))
    k = rng.standard_normal((N, d))
    v = rng.standard_normal((N, d))
    eps = _median_sq_distance(_normalize(q), _normalize(k))
    exact = _exact_attention(q, k, v, b, eps)

    def may_cos(M):
        key = jax.random.PRNGKey(0)
        p = create_maclaurin_projection(key, d, num_features=M, bias=b, epsilon=eps)
        pq = np.asarray(maclaurin_features(jnp.asarray(q), p))
        pk = np.asarray(maclaurin_features(jnp.asarray(k), p))
        out = _np_linear_attention(pq, pk, v)
        return _per_token_cos(out, exact)

    assert may_cos(1024) > may_cos(64)


# --------------------------------------------------------------------------
# MAY beats SLAY at b >= 1 (the headline qualitative claim)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("b", [1.0, 2.0])
def test_may_beats_slay_at_high_bias(b):
    d, N, F = 64, 256, 256
    cos_may, cos_slay = [], []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        q = rng.standard_normal((N, d))
        k = rng.standard_normal((N, d))
        v = rng.standard_normal((N, d))
        eps = _median_sq_distance(_normalize(q), _normalize(k))
        exact = _exact_attention(q, k, v, b, eps)

        key = jax.random.PRNGKey(seed)
        mp = create_maclaurin_projection(key, d, num_features=F, bias=b, epsilon=eps)
        pq = np.asarray(maclaurin_features(jnp.asarray(q), mp))
        pk = np.asarray(maclaurin_features(jnp.asarray(k), mp))
        may = _np_linear_attention(pq, pk, v)
        cos_may.append(_per_token_cos(may, exact))

        sp = _slay_create(d, P=32, M=8, R=1, eps=eps, seed=seed)
        sa = _np_linear_attention(_slay_features(q, sp), _slay_features(k, sp), v)
        cos_slay.append(_per_token_cos(sa, exact))

    assert np.mean(cos_may) > np.mean(cos_slay)
