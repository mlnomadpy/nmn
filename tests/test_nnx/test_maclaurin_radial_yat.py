"""Tests for MAY (Random Maclaurin) and RAY (radial) bias-aware Yat features.

Covers issue #36 for the nnx backend:
  - shapes & determinism
  - no NaN/Inf
  - MAY inner product tracks the kernel kappa(s) = (s+b)^2/(C-2s)
  - MAY beats SLAY at b >= 1
  - RAY runs and produces finite output
  - RotaryYatAttention performer_kind selector ('slay'/'maclaurin'/'radial')
"""

import pytest
import numpy as np

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from jax import random

from nmn.nnx.layers.attention import (
    create_maclaurin_projection,
    maclaurin_features,
    maclaurin_yat_attention,
    maclaurin_coeffs,
    create_radial_projection,
    radial_features,
    radial_yat_attention,
    create_yat_tp_projection,
    yat_tp_attention,
)


def _normalize(x):
    return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)


def _exact_attention(q, k, v, b, eps):
    qn, kn = _normalize(q), _normalize(k)
    s = qn @ kn.T
    w = (s + b) ** 2 / ((2.0 + eps) - 2.0 * s)
    w = w / (w.sum(-1, keepdims=True) + 1e-9)
    return w @ v


def _cos(a, b):
    num = (a * b).sum(-1)
    den = jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-12
    return float(jnp.mean(num / den))


# --------------------------------------------------------------------------
# Maclaurin coefficients
# --------------------------------------------------------------------------
class TestMaclaurinCoeffs:
    def test_nonnegative_for_positive_bias(self):
        for b in [0.0, 0.5, 1.0, 2.0]:
            a = maclaurin_coeffs(b, epsilon=0.3, nmax=40)
            assert np.all(a >= 0.0)
            assert a.shape == (41,)

    def test_recurrence_matches_definition(self):
        b, eps, nmax = 1.5, 0.2, 10
        C = 2.0 + eps
        n = np.arange(nmax + 1)
        beta = (1.0 / C) * (2.0 / C) ** n
        a = maclaurin_coeffs(b, eps, nmax)
        expect = b * b * beta.copy()
        expect[1:] += 2.0 * b * beta[:-1]
        expect[2:] += beta[:-2]
        np.testing.assert_allclose(a, expect, rtol=1e-6)


# --------------------------------------------------------------------------
# MAY features
# --------------------------------------------------------------------------
class TestMaclaurinFeatures:
    def test_shape(self):
        key = random.PRNGKey(0)
        p = create_maclaurin_projection(key, head_dim=16, num_features=64, bias=1.0, epsilon=0.1)
        x = random.normal(random.PRNGKey(1), (2, 8, 4, 16))
        f = maclaurin_features(x, p)
        assert f.shape == (2, 8, 4, 64)

    def test_no_nan_inf(self):
        key = random.PRNGKey(0)
        p = create_maclaurin_projection(key, head_dim=16, num_features=128, bias=1.0, epsilon=0.1)
        x = random.normal(random.PRNGKey(2), (3, 5, 2, 16)) * 5.0
        f = maclaurin_features(x, p)
        assert bool(jnp.all(jnp.isfinite(f)))

    def test_determinism(self):
        key = random.PRNGKey(7)
        p = create_maclaurin_projection(key, head_dim=16, num_features=64, bias=1.0, epsilon=0.1)
        x = random.normal(random.PRNGKey(3), (2, 4, 16))
        f1 = maclaurin_features(x, p)
        f2 = maclaurin_features(x, p)
        np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))

    def test_same_key_same_projection(self):
        p1 = create_maclaurin_projection(random.PRNGKey(11), 16, num_features=32, bias=1.0, epsilon=0.1)
        p2 = create_maclaurin_projection(random.PRNGKey(11), 16, num_features=32, bias=1.0, epsilon=0.1)
        np.testing.assert_array_equal(np.asarray(p1['omegas']), np.asarray(p2['omegas']))
        np.testing.assert_array_equal(np.asarray(p1['degrees']), np.asarray(p2['degrees']))

    def test_inner_product_tracks_kappa(self):
        """E[phi(q)·phi(k)] = kappa(s); the inner product must rise with s."""
        d, b, eps = 32, 1.0, 0.3
        C = 2.0 + eps
        # Large feature budget to reduce estimator variance.
        p = create_maclaurin_projection(random.PRNGKey(0), d, num_features=40000, bias=b, epsilon=eps, nmax=40)
        q = random.normal(random.PRNGKey(5), (d,))
        q = q / jnp.linalg.norm(q)
        ko = random.normal(random.PRNGKey(7), (d,))
        ko = ko - jnp.dot(ko, q) * q
        ko = ko / jnp.linalg.norm(ko)

        targets = [-0.5, 0.0, 0.5, 0.9]
        approx_vals, exact_vals = [], []
        for s in targets:
            k = s * q + np.sqrt(max(0.0, 1 - s ** 2)) * ko
            k = k / jnp.linalg.norm(k)
            fq = maclaurin_features(q[None], p)[0]
            fk = maclaurin_features(k[None], p)[0]
            approx_vals.append(float(jnp.sum(fq * fk)))
            ss = float(jnp.dot(q, k))
            exact_vals.append((ss + b) ** 2 / (C - 2 * ss))

        # Monotonic rise with s (the kernel is increasing on this range).
        assert approx_vals[0] < approx_vals[1] < approx_vals[2] < approx_vals[3]
        assert exact_vals[0] < exact_vals[1] < exact_vals[2] < exact_vals[3]
        # Reasonable tracking (high-variance estimator, so generous tolerance).
        for ap, ex in zip(approx_vals, exact_vals):
            assert ap == pytest.approx(ex, rel=0.5, abs=0.3)


# --------------------------------------------------------------------------
# MAY attention readout
# --------------------------------------------------------------------------
class TestMaclaurinAttention:
    def test_shape_and_finite(self):
        d = 16
        p = create_maclaurin_projection(random.PRNGKey(0), d, num_features=128, bias=1.0, epsilon=0.2)
        q = random.normal(random.PRNGKey(1), (2, 10, 4, d))
        k = random.normal(random.PRNGKey(2), (2, 10, 4, d))
        v = random.normal(random.PRNGKey(3), (2, 10, 4, d))
        out = maclaurin_yat_attention(q, k, v, p)
        assert out.shape == (2, 10, 4, d)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_causal_runs(self):
        d = 16
        p = create_maclaurin_projection(random.PRNGKey(0), d, num_features=64, bias=1.0, epsilon=0.2)
        q = random.normal(random.PRNGKey(1), (1, 8, 2, d))
        out = maclaurin_yat_attention(q, q, q, p, causal=True)
        assert out.shape == (1, 8, 2, d)
        assert bool(jnp.all(jnp.isfinite(out)))

    @pytest.mark.parametrize("b", [1.0, 2.0, 4.0])
    def test_may_beats_slay_at_high_bias(self, b):
        d, N = 64, 256
        q = random.normal(random.PRNGKey(1), (N, d))
        k = random.normal(random.PRNGKey(2), (N, d))
        v = random.normal(random.PRNGKey(3), (N, d))
        qn, kn = _normalize(q), _normalize(k)
        diff = qn[:, None, :] - kn[None, :, :]
        eps = float(jnp.median((diff * diff).sum(-1)))
        exact = _exact_attention(q, k, v, b, eps)

        mp = create_maclaurin_projection(random.PRNGKey(0), d, num_features=256, bias=b, epsilon=eps)
        ma = maclaurin_yat_attention(q[:, None, :], k[:, None, :], v[:, None, :], mp)[:, 0, :]

        sp = create_yat_tp_projection(random.PRNGKey(0), d, num_prf_features=8,
                                      num_quad_nodes=1, num_anchor_features=32)
        sa = yat_tp_attention(q[:, None, :], k[:, None, :], v[:, None, :], sp)[:, 0, :]

        assert _cos(ma, exact) > _cos(sa, exact)


# --------------------------------------------------------------------------
# RAY features & attention
# --------------------------------------------------------------------------
class TestRadialFeatures:
    def test_shape(self):
        d = 16
        p = create_radial_projection(random.PRNGKey(0), d, sketch_m=8, num_radial=4,
                                     radial_dim=8, bias=1.0, epsilon=0.2)
        x = random.normal(random.PRNGKey(1), (2, 6, 3, d))
        f = radial_features(x, p)
        assert f.shape == (2, 6, 3, 8 * 4 * 8)
        assert bool(jnp.all(jnp.isfinite(f)))

    def test_determinism(self):
        d = 16
        p = create_radial_projection(random.PRNGKey(0), d, sketch_m=8, num_radial=4,
                                     radial_dim=8, bias=1.0, epsilon=0.2)
        x = random.normal(random.PRNGKey(1), (2, 3, d))
        np.testing.assert_array_equal(
            np.asarray(radial_features(x, p)), np.asarray(radial_features(x, p))
        )

    def test_attention_shape_and_finite(self):
        d = 16
        p = create_radial_projection(random.PRNGKey(0), d, sketch_m=8, num_radial=4,
                                     radial_dim=8, bias=1.0, epsilon=0.2)
        q = random.normal(random.PRNGKey(1), (2, 7, 2, d))
        k = random.normal(random.PRNGKey(2), (2, 7, 2, d))
        v = random.normal(random.PRNGKey(3), (2, 7, 2, d))
        out = radial_yat_attention(q, k, v, p)
        assert out.shape == (2, 7, 2, d)
        assert bool(jnp.all(jnp.isfinite(out)))


# --------------------------------------------------------------------------
# RotaryYatAttention performer_kind selector
# --------------------------------------------------------------------------
class TestPerformerSelector:
    @pytest.mark.parametrize("kind", ["slay", "maclaurin", "radial"])
    def test_selector_runs(self, kind):
        from flax import nnx
        from nmn.nnx.layers.attention import RotaryYatAttention

        attn = RotaryYatAttention(
            embed_dim=32,
            num_heads=4,
            max_seq_len=64,
            use_performer=True,
            performer_kind=kind,
            performer_num_features=64,
            performer_sketch_m=4,
            performer_num_radial=2,
            performer_radial_dim=4,
            rngs=nnx.Rngs(0),
        )
        x = random.normal(random.PRNGKey(1), (2, 16, 32))
        out = attn(x, deterministic=True)
        assert out.shape == (2, 16, 32)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_default_is_slay(self):
        from flax import nnx
        from nmn.nnx.layers.attention import RotaryYatAttention

        attn = RotaryYatAttention(
            embed_dim=32, num_heads=4, max_seq_len=64,
            use_performer=True, rngs=nnx.Rngs(0),
        )
        assert attn.performer_kind == "slay"

    def test_invalid_kind_raises(self):
        from flax import nnx
        from nmn.nnx.layers.attention import RotaryYatAttention

        with pytest.raises(ValueError):
            RotaryYatAttention(
                embed_dim=32, num_heads=4, max_seq_len=64,
                use_performer=True, performer_kind="bogus", rngs=nnx.Rngs(0),
            )
