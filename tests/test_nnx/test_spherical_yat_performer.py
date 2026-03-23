"""Tests for Spherical YAT Performer (Anchor Approximation).

Tests the anchor-based tensor product decomposition of the YAT kernel:
    K(q,k) = (q·k)^2 / (C - 2·q·k)
           ≈ Σ_r w_r · φ_poly(q)·φ_poly(k) · φ_PRF(q;s_r)·φ_PRF(k;s_r)
"""

import pytest
import numpy as np

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from jax import random


class TestCreateProjection:
    """Tests for create_yat_tp_projection."""

    def test_default_params(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import create_yat_tp_projection

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64)

        assert params['num_anchor_features'] == 16
        assert params['num_prf_features'] == 8
        assert params['num_scales'] == 1
        assert params['head_dim'] == 64
        assert params['projections'].shape == (1, 8, 64)
        assert params['anchors'].shape == (16, 64)
        assert params['quad_nodes'].shape == (1,)
        assert params['quad_weights'].shape == (1,)

    def test_custom_params(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import create_yat_tp_projection

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(
            key, head_dim=32, num_prf_features=4, num_quad_nodes=3, num_anchor_features=8,
        )

        assert params['projections'].shape == (3, 4, 32)
        assert params['anchors'].shape == (8, 32)
        assert params['quad_nodes'].shape == (3,)

    def test_anchors_are_unit_norm(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import create_yat_tp_projection

        key = random.PRNGKey(42)
        params = create_yat_tp_projection(key, head_dim=64)
        norms = jnp.sqrt(jnp.sum(params['anchors'] ** 2, axis=-1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_quadrature_weights_positive(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import create_yat_tp_projection

        key = random.PRNGKey(0)
        for R in [1, 2, 4]:
            params = create_yat_tp_projection(key, head_dim=64, num_quad_nodes=R)
            assert jnp.all(params['quad_weights'] > 0)


class TestAnchorPolyFeatures:
    """Tests for anchor polynomial features."""

    def test_output_shape(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            _anchor_poly_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64, num_anchor_features=16)
        x = random.normal(key, (2, 10, 4, 64))
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))

        poly = _anchor_poly_features(x, params['anchors'], 16)
        assert poly.shape == (2, 10, 4, 16)

    def test_non_negative(self):
        """Anchor features (a·x)^2 must be non-negative."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            _anchor_poly_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        params = create_yat_tp_projection(key, head_dim=64)
        x = random.normal(key, (4, 32, 8, 64))
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))

        poly = _anchor_poly_features(x, params['anchors'], params['num_anchor_features'])
        assert jnp.all(poly >= 0)


class TestPRFFeatures:
    """Tests for positive random features."""

    def test_output_shape(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            _prf_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64, num_prf_features=8)
        x = random.normal(key, (2, 10, 4, 64))
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))

        prf = _prf_features(x, params['projections'][0], params['quad_nodes'][0], 8)
        assert prf.shape == (2, 10, 4, 8)

    def test_positive(self):
        """PRF features (exp-based) must be positive."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            _prf_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        params = create_yat_tp_projection(key, head_dim=64)
        x = random.normal(key, (4, 32, 8, 64))
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True))

        prf = _prf_features(x, params['projections'][0], params['quad_nodes'][0], params['num_prf_features'])
        assert jnp.all(prf > 0)


class TestYatTPFeatures:
    """Tests for full tensor-product feature computation."""

    def test_output_shape(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        P, M, R = 16, 8, 1
        params = create_yat_tp_projection(
            key, head_dim=64, num_prf_features=M, num_quad_nodes=R, num_anchor_features=P,
        )

        x = random.normal(key, (2, 10, 4, 64))
        feat = yat_tp_features(x, params, normalize=True)
        assert feat.shape == (2, 10, 4, R * P * M)

    def test_no_nan_inf(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        params = create_yat_tp_projection(key, head_dim=64)
        x = random.normal(key, (2, 32, 4, 64))

        feat = yat_tp_features(x, params, normalize=True)
        assert not jnp.any(jnp.isnan(feat))
        assert not jnp.any(jnp.isinf(feat))

    def test_different_quad_nodes(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        x = random.normal(key, (2, 16, 4, 64))

        for R in [1, 2, 4]:
            params = create_yat_tp_projection(
                key, head_dim=64, num_prf_features=4, num_quad_nodes=R, num_anchor_features=8,
            )
            feat = yat_tp_features(x, params)
            assert feat.shape[-1] == R * 8 * 4


class TestYatTPAttention:
    """Tests for the full attention computation."""

    def test_output_shape(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64)
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (2, 16, 4, 64))
        k = random.normal(k2, (2, 16, 4, 64))
        v = random.normal(k3, (2, 16, 4, 64))

        out = yat_tp_attention(q, k, v, params)
        assert out.shape == (2, 16, 4, 64)

    def test_causal_output_shape(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64)
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (2, 16, 4, 64))
        k = random.normal(k2, (2, 16, 4, 64))
        v = random.normal(k3, (2, 16, 4, 64))

        out = yat_tp_attention(q, k, v, params, causal=True)
        assert out.shape == (2, 16, 4, 64)

    def test_no_nan_inf(self):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        params = create_yat_tp_projection(key, head_dim=64)
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (2, 32, 4, 64))
        k = random.normal(k2, (2, 32, 4, 64))
        v = random.normal(k3, (2, 32, 4, 64))

        for causal in [False, True]:
            out = yat_tp_attention(q, k, v, params, causal=causal)
            assert not jnp.any(jnp.isnan(out)), f"NaN in output (causal={causal})"
            assert not jnp.any(jnp.isinf(out)), f"Inf in output (causal={causal})"

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        params = create_yat_tp_projection(key, head_dim=64)
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (2, 16, 4, 64))
        k = random.normal(k2, (2, 16, 4, 64))
        v = random.normal(k3, (2, 16, 4, 64))

        out1 = yat_tp_attention(q, k, v, params)
        out2 = yat_tp_attention(q, k, v, params)
        np.testing.assert_allclose(out1, out2, atol=1e-6)


class TestKernelApproximation:
    """Tests for kernel approximation quality."""

    def test_kernel_shape_monotonic(self):
        """Approximation should increase with dot product for positive dots."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        head_dim = 64
        params = create_yat_tp_projection(key, head_dim)

        q = random.normal(key, (head_dim,))
        q = q / jnp.linalg.norm(q)

        approx_values = []
        for target_dot in [0.5, 0.7, 0.9, 0.95]:
            k_orth = random.normal(random.split(key)[0], (head_dim,))
            k_orth = k_orth - jnp.dot(k_orth, q) * q
            k_orth = k_orth / jnp.linalg.norm(k_orth)
            k = target_dot * q + jnp.sqrt(1 - target_dot ** 2) * k_orth
            k = k / jnp.linalg.norm(k)

            q_r = q.reshape(1, 1, 1, head_dim)
            k_r = k.reshape(1, 1, 1, head_dim)
            q_feat = yat_tp_features(q_r, params, normalize=True)
            k_feat = yat_tp_features(k_r, params, normalize=True)
            approx_values.append(float(jnp.sum(q_feat * k_feat)))

        # Should be monotonically increasing
        for i in range(len(approx_values) - 1):
            assert approx_values[i] < approx_values[i + 1], (
                f"Not monotonic: approx[{i}]={approx_values[i]} >= approx[{i+1}]={approx_values[i+1]}"
            )

    def test_kernel_near_zero_at_orthogonal(self):
        """Kernel should be near zero when q and k are orthogonal."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_features, create_yat_tp_projection,
        )

        key = random.PRNGKey(42)
        head_dim = 64
        params = create_yat_tp_projection(key, head_dim)

        q = random.normal(key, (head_dim,))
        q = q / jnp.linalg.norm(q)

        # Construct orthogonal k
        k = random.normal(random.split(key)[0], (head_dim,))
        k = k - jnp.dot(k, q) * q
        k = k / jnp.linalg.norm(k)

        q_r = q.reshape(1, 1, 1, head_dim)
        k_r = k.reshape(1, 1, 1, head_dim)
        q_feat = yat_tp_features(q_r, params, normalize=True)
        k_feat = yat_tp_features(k_r, params, normalize=True)
        approx = float(jnp.sum(q_feat * k_feat))

        # High similarity value for reference
        k_sim = 0.9 * q + jnp.sqrt(1 - 0.9 ** 2) * (k / jnp.linalg.norm(k))
        k_sim = k_sim / jnp.linalg.norm(k_sim)
        k_sim_r = k_sim.reshape(1, 1, 1, head_dim)
        k_sim_feat = yat_tp_features(k_sim_r, params, normalize=True)
        approx_high = float(jnp.sum(q_feat * k_sim_feat))

        # Orthogonal should be much smaller than high-similarity
        assert approx < approx_high * 0.5

    def test_approximation_improves_with_features(self):
        """More features should give better cosine similarity to exact."""
        from nmn.nnx.layers.attention.yat_attention import yat_attention_normalized
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        batch, seq_len, heads, head_dim = 2, 64, 4, 64
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (batch, seq_len, heads, head_dim))
        k = random.normal(k2, (batch, seq_len, heads, head_dim))
        v = random.normal(k3, (batch, seq_len, heads, head_dim))

        exact = yat_attention_normalized(q, k, v)

        cos_sims = []
        for M in [2, 8, 32]:
            params = create_yat_tp_projection(
                key, head_dim, num_prf_features=M, num_quad_nodes=1, num_anchor_features=16,
            )
            approx = yat_tp_attention(q, k, v, params)
            cos_sim = float(
                jnp.sum(exact * approx)
                / (jnp.linalg.norm(exact) * jnp.linalg.norm(approx) + 1e-8)
            )
            cos_sims.append(cos_sim)

        # More features should generally improve quality
        assert cos_sims[-1] > cos_sims[0], (
            f"More features didn't improve: M=2 cos={cos_sims[0]:.4f}, M=32 cos={cos_sims[-1]:.4f}"
        )

    def test_mean_error_decreases_with_seq_length(self):
        """Mean absolute error should decrease with longer sequences (averaging effect)."""
        from nmn.nnx.layers.attention.yat_attention import yat_attention_normalized
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        key = random.PRNGKey(0)
        head_dim = 64
        params = create_yat_tp_projection(key, head_dim)

        mean_errors = []
        for seq_len in [32, 128, 512]:
            k1, k2, k3 = random.split(random.PRNGKey(seq_len), 3)
            q = random.normal(k1, (2, seq_len, 4, head_dim))
            k = random.normal(k2, (2, seq_len, 4, head_dim))
            v = random.normal(k3, (2, seq_len, 4, head_dim))

            exact = yat_attention_normalized(q, k, v)
            approx = yat_tp_attention(q, k, v, params)
            mean_errors.append(float(jnp.mean(jnp.abs(exact - approx))))

        # Mean error should decrease as sequence length grows
        assert mean_errors[-1] < mean_errors[0], (
            f"Error didn't decrease: seq=32 err={mean_errors[0]:.4f}, seq=512 err={mean_errors[-1]:.4f}"
        )


class TestParameterSweep:
    """Parametric sweep testing all P x M x R combinations for quality and speed."""

    P_VALUES = [8, 16, 32]
    M_VALUES = [2, 4, 8]
    R_VALUES = [1, 2, 4]

    @pytest.fixture(scope="class")
    def reference_data(self):
        from nmn.nnx.layers.attention.yat_attention import yat_attention_normalized

        key = random.PRNGKey(0)
        batch, seq_len, heads, head_dim = 2, 64, 4, 64
        k1, k2, k3 = random.split(key, 3)
        q = random.normal(k1, (batch, seq_len, heads, head_dim))
        k = random.normal(k2, (batch, seq_len, heads, head_dim))
        v = random.normal(k3, (batch, seq_len, heads, head_dim))
        exact = yat_attention_normalized(q, k, v)
        return q, k, v, exact, head_dim

    @pytest.mark.parametrize("P", P_VALUES)
    @pytest.mark.parametrize("M", M_VALUES)
    @pytest.mark.parametrize("R", R_VALUES)
    def test_no_nan_inf(self, P, M, R, reference_data):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        q, k, v, _, head_dim = reference_data
        params = create_yat_tp_projection(
            random.PRNGKey(0), head_dim,
            num_prf_features=M, num_quad_nodes=R, num_anchor_features=P,
        )
        out = yat_tp_attention(q, k, v, params)
        assert not jnp.any(jnp.isnan(out)), f"NaN with P={P}, M={M}, R={R}"
        assert not jnp.any(jnp.isinf(out)), f"Inf with P={P}, M={M}, R={R}"

    @pytest.mark.parametrize("P", P_VALUES)
    @pytest.mark.parametrize("M", M_VALUES)
    @pytest.mark.parametrize("R", R_VALUES)
    def test_correct_output_shape(self, P, M, R, reference_data):
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        q, k, v, exact, head_dim = reference_data
        params = create_yat_tp_projection(
            random.PRNGKey(0), head_dim,
            num_prf_features=M, num_quad_nodes=R, num_anchor_features=P,
        )
        out = yat_tp_attention(q, k, v, params)
        assert out.shape == exact.shape, f"Shape mismatch with P={P}, M={M}, R={R}"

    @pytest.mark.parametrize("P", P_VALUES)
    @pytest.mark.parametrize("M", M_VALUES)
    @pytest.mark.parametrize("R", R_VALUES)
    def test_positive_cosine_similarity(self, P, M, R, reference_data):
        """All configs should have positive cosine similarity with exact attention."""
        from nmn.nnx.layers.attention.spherical_yat_performer import (
            yat_tp_attention, create_yat_tp_projection,
        )

        q, k, v, exact, head_dim = reference_data
        params = create_yat_tp_projection(
            random.PRNGKey(0), head_dim,
            num_prf_features=M, num_quad_nodes=R, num_anchor_features=P,
        )
        approx = yat_tp_attention(q, k, v, params)
        cos_sim = float(
            jnp.sum(exact * approx)
            / (jnp.linalg.norm(exact) * jnp.linalg.norm(approx) + 1e-8)
        )
        assert cos_sim > 0.3, (
            f"Cosine similarity too low ({cos_sim:.4f}) with P={P}, M={M}, R={R}"
        )


class TestRotaryYatPerformer:
    """Tests for RotaryYatAttention with performer mode using new anchor features."""

    def test_performer_output_shape(self):
        from nmn.nnx.layers.attention import RotaryYatAttention
        from flax import nnx

        rngs = nnx.Rngs(0)
        attn = RotaryYatAttention(
            embed_dim=64, num_heads=4, max_seq_len=128,
            use_performer=True, rngs=rngs,
        )
        x = jnp.ones((2, 16, 64))
        out = attn(x, deterministic=True)
        assert out.shape == (2, 16, 64)

    def test_performer_no_nan(self):
        from nmn.nnx.layers.attention import RotaryYatAttention
        from flax import nnx

        rngs = nnx.Rngs(42)
        attn = RotaryYatAttention(
            embed_dim=64, num_heads=4, max_seq_len=128,
            use_performer=True, rngs=rngs,
        )
        x = random.normal(random.PRNGKey(0), (2, 32, 64))
        out = attn(x, deterministic=True)
        assert not jnp.any(jnp.isnan(out))
        assert not jnp.any(jnp.isinf(out))

    def test_performer_vs_standard_same_shape(self):
        from nmn.nnx.layers.attention import RotaryYatAttention
        from flax import nnx

        rngs = nnx.Rngs(0)
        x = jnp.ones((2, 16, 64))

        attn_std = RotaryYatAttention(
            embed_dim=64, num_heads=4, max_seq_len=128,
            use_performer=False, rngs=rngs,
        )
        attn_perf = RotaryYatAttention(
            embed_dim=64, num_heads=4, max_seq_len=128,
            use_performer=True, rngs=rngs,
        )

        out_std = attn_std(x, deterministic=True)
        out_perf = attn_perf(x, deterministic=True)
        assert out_std.shape == out_perf.shape
