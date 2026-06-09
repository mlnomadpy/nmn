"""Tests for MAY (Random Maclaurin) and RAY (radial) linear-attention feature
maps for spherical YAT (issue #36, PyTorch backend).

Verifies: shapes, no NaN/Inf, determinism (seeded), the MAY inner-product
tracks the kernel ``kappa(s) = (s + b)²/(C − 2s)``, and that MAY beats a
SLAY-style (b=0) baseline at ``b ≥ 1`` (the bias-aware regime).
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nmn.torch.attention.performer_yat import (
    maclaurin_coeffs,
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
    linear_attention_readout,
)
from nmn.torch.attention.yat_attention import yat_attention_normalized


# --------------------------------------------------------------------------
# Reference helpers (NumPy, mirror .context/may-ray/reference.py)
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


# SLAY-style (b=0) NumPy baseline (anchor performer) from the reference.
def _slay_create(d, P, M, R, eps, seed):
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


# --------------------------------------------------------------------------
# Maclaurin coefficient sanity
# --------------------------------------------------------------------------
def test_maclaurin_coeffs_nonneg_and_reconstruct():
    b, eps, nmax = 1.5, 0.3, 60
    a = maclaurin_coeffs(b, eps, nmax)
    assert (a >= 0).all()
    # Series reconstructs kappa(s) for |s| < C/2.
    C = 2.0 + eps
    for s in [-0.5, 0.0, 0.3, 0.7]:
        approx = sum(a[n] * s ** n for n in range(nmax + 1))
        exact = (s + b) ** 2 / (C - 2.0 * s)
        assert abs(approx - exact) < 1e-6


# --------------------------------------------------------------------------
# Shapes / no NaN-Inf / determinism — MAY
# --------------------------------------------------------------------------
def test_may_feature_shape():
    d, M = 16, 64
    params = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=0)
    x = torch.randn(2, 5, 4, d)
    feat = maclaurin_features(x, params)
    assert feat.shape == (2, 5, 4, M)


def test_may_attention_shape_and_finite():
    d, M = 16, 64
    params = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=0)
    q = torch.randn(2, 5, 4, d)
    k = torch.randn(2, 7, 4, d)
    v = torch.randn(2, 7, 4, 8)
    out = maclaurin_yat_attention(q, k, v, params)
    assert out.shape == (2, 5, 4, 8)
    assert torch.isfinite(out).all()


def test_may_causal_shape_and_finite():
    d, M = 16, 64
    params = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=1)
    q = torch.randn(2, 6, 3, d)
    v = torch.randn(2, 6, 3, 8)
    out = maclaurin_yat_attention(q, q, v, params, causal=True)
    assert out.shape == (2, 6, 3, 8)
    assert torch.isfinite(out).all()


def test_may_determinism():
    d, M = 16, 64
    x = torch.randn(3, 4, 2, d)
    p1 = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=42)
    p2 = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=42)
    f1 = maclaurin_features(x, p1)
    f2 = maclaurin_features(x, p2)
    assert torch.allclose(f1, f2)


def test_may_causal_matches_full_at_last_query():
    d, M = 16, 64
    params = create_maclaurin_projection(head_dim=d, num_features=M, bias=1.0, epsilon=0.2, seed=3)
    x = torch.randn(1, 5, 2, d)
    v = torch.randn(1, 5, 2, 6)
    causal = maclaurin_yat_attention(x, x, v, params, causal=True)
    full = maclaurin_yat_attention(x, x, v, params, causal=False)
    # The last query attends to all keys in both modes.
    assert torch.allclose(causal[:, -1], full[:, -1], atol=1e-4)


# --------------------------------------------------------------------------
# MAY inner product tracks kappa, and more features -> better fit
# --------------------------------------------------------------------------
def test_may_inner_product_tracks_kappa():
    d, b, eps = 32, 1.0, 0.3
    C = 2.0 + eps
    rng = np.random.default_rng(0)
    # Build pairs at controlled cosine s by interpolating.
    base = _normalize_rows(rng.standard_normal((1, d)))
    other = _normalize_rows(rng.standard_normal((1, d)))
    s_targets = np.linspace(-0.6, 0.8, 8)

    params = create_maclaurin_projection(
        head_dim=d, num_features=20000, bias=b, epsilon=eps, seed=0
    )

    est, exact = [], []
    for s in s_targets:
        # Construct k̂ with q̂·k̂ ≈ s via Gram-Schmidt mixing.
        perp = other - (other @ base.T) * base
        perp = _normalize_rows(perp)
        kv = s * base + math.sqrt(max(1 - s * s, 0.0)) * perp
        kv = _normalize_rows(kv)
        qf = maclaurin_features(torch.as_tensor(base, dtype=torch.float32), params)
        kf = maclaurin_features(torch.as_tensor(kv, dtype=torch.float32), params)
        est.append(float((qf * kf).sum()))
        exact.append(float(_exact_kernel(s, b, eps)))

    est = np.array(est)
    exact = np.array(exact)
    # Monotone-rising and close to kappa (rises with s, tracks the kernel).
    assert np.corrcoef(est, exact)[0, 1] > 0.99
    assert np.all(np.diff(est) > 0)  # strictly rising in s, like kappa
    # Unbiased Monte-Carlo estimator: tight on absolute error, looser on
    # relative error at small-kappa points (sampling variance dominates there).
    assert np.median(np.abs(est - exact) / (np.abs(exact) + 1e-6)) < 0.25


def test_may_more_features_better_attention_fit():
    d, N, b = 48, 96, 1.0
    rng = np.random.default_rng(7)
    q = rng.standard_normal((N, d)).astype(np.float32)
    k = rng.standard_normal((N, d)).astype(np.float32)
    v = rng.standard_normal((N, d)).astype(np.float32)
    eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
    exact = _exact_attention(q, k, v, b, eps)

    def fit(M):
        p = create_maclaurin_projection(head_dim=d, num_features=M, bias=b, epsilon=eps, seed=0)
        qf = maclaurin_features(torch.as_tensor(q), p).numpy()
        kf = maclaurin_features(torch.as_tensor(k), p).numpy()
        out = _linear_attention_np(qf, kf, v)
        return _per_token_cos(out, exact)

    low = fit(64)
    high = fit(1024)
    assert high > low
    assert high > 0.9


# --------------------------------------------------------------------------
# MAY beats SLAY (b=0 design) at b >= 1
# --------------------------------------------------------------------------
@pytest.mark.parametrize("b", [1.0, 2.0])
def test_may_beats_slay_at_high_bias(b):
    d, N, F = 64, 256, 256
    cos_may, cos_slay = [], []
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(100 + seed)
        q = rng.standard_normal((N, d)).astype(np.float32)
        k = rng.standard_normal((N, d)).astype(np.float32)
        v = rng.standard_normal((N, d)).astype(np.float32)
        eps = _median_sq_distance(_normalize_rows(q), _normalize_rows(k))
        exact = _exact_attention(q, k, v, b, eps)

        mp = create_maclaurin_projection(head_dim=d, num_features=F, bias=b, epsilon=eps, seed=seed)
        qf = maclaurin_features(torch.as_tensor(q), mp).numpy()
        kf = maclaurin_features(torch.as_tensor(k), mp).numpy()
        cos_may.append(_per_token_cos(_linear_attention_np(qf, kf, v), exact))

        sp = _slay_create(d, P=32, M=8, R=1, eps=eps, seed=seed)
        sqf = _slay_features(q, sp)
        skf = _slay_features(k, sp)
        cos_slay.append(_per_token_cos(_linear_attention_np(sqf, skf, v), exact))

    assert np.mean(cos_may) > np.mean(cos_slay)


# --------------------------------------------------------------------------
# RAY — shapes, finite, determinism (weaker method; no win expected)
# --------------------------------------------------------------------------
def test_ray_feature_shape_and_finite():
    d = 16
    params = create_radial_projection(
        head_dim=d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.3, seed=0
    )
    x = torch.randn(2, 5, 3, d)
    feat = radial_features(x, params)
    assert feat.shape[:-1] == (2, 5, 3)
    assert feat.shape[-1] == 8 * 4 * 8
    assert torch.isfinite(feat).all()


def test_ray_attention_shape_and_finite():
    d = 16
    params = create_radial_projection(
        head_dim=d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.3, seed=0
    )
    q = torch.randn(2, 5, 3, d)
    k = torch.randn(2, 7, 3, d)
    v = torch.randn(2, 7, 3, 8)
    out = radial_yat_attention(q, k, v, params)
    assert out.shape == (2, 5, 3, 8)
    assert torch.isfinite(out).all()


def test_ray_determinism():
    d = 16
    x = torch.randn(2, 4, 2, d)
    p1 = create_radial_projection(head_dim=d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.3, seed=5)
    p2 = create_radial_projection(head_dim=d, sketch_m=8, num_radial=4, radial_dim=8, bias=1.0, epsilon=0.3, seed=5)
    assert torch.allclose(radial_features(x, p1), radial_features(x, p2))


# --------------------------------------------------------------------------
# Readout sanity: matches a hand NumPy readout
# --------------------------------------------------------------------------
def test_linear_attention_readout_matches_numpy():
    torch.manual_seed(0)
    B, Q, K, H, F, D = 1, 3, 4, 2, 5, 6
    qf = torch.randn(B, Q, H, F)
    kf = torch.randn(B, K, H, F)
    v = torch.randn(B, K, H, D)
    out = linear_attention_readout(qf, kf, v, causal=False, epsilon=1e-5)
    # NumPy per (b,h)
    for h in range(H):
        num_np = qf[0, :, h].numpy() @ (kf[0, :, h].numpy().T @ v[0, :, h].numpy())
        den_np = qf[0, :, h].numpy() @ kf[0, :, h].numpy().sum(0)
        ref = num_np / (den_np[:, None] + 1e-5)
        assert np.allclose(out[0, :, h].numpy(), ref, atol=1e-4)
