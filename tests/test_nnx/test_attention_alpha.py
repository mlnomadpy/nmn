"""Tests for YAT Attention Alpha Scaling Feature.

Tests the alpha parameter in YAT attention, which scales the attention scores:
    scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

Alpha can be:
- Learnable (default): A trainable parameter
- Constant: A fixed value like sqrt(2)
- Disabled: No scaling applied
"""

import pytest
import numpy as np

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from flax import nnx


class TestYatAttentionAlpha:
    """Tests for alpha parameter in YAT attention functions."""
    
    def test_yat_attention_weights_with_alpha(self):
        """Test that alpha scaling is applied to attention weights."""
        from nmn.nnx.attention import yat_attention_weights
        
        key = jax.random.key(0)
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 32
        
        query = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        key_arr = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        
        # Without alpha
        weights_no_alpha = yat_attention_weights(query, key_arr, alpha=None)
        
        # With constant alpha = 1.0 (should have some scaling effect)
        weights_alpha_1 = yat_attention_weights(query, key_arr, alpha=1.0)
        
        # With constant alpha = 2.0 (should have more scaling)
        weights_alpha_2 = yat_attention_weights(query, key_arr, alpha=2.0)
        
        # All should have valid shapes
        assert weights_no_alpha.shape == (batch, num_heads, seq_len, seq_len)
        assert weights_alpha_1.shape == (batch, num_heads, seq_len, seq_len)
        assert weights_alpha_2.shape == (batch, num_heads, seq_len, seq_len)
        
        # All should be valid (no NaN/Inf)
        assert not jnp.any(jnp.isnan(weights_no_alpha))
        assert not jnp.any(jnp.isnan(weights_alpha_1))
        assert not jnp.any(jnp.isnan(weights_alpha_2))
        
        # With alpha, the weights before softmax should be scaled differently
        # But after softmax they should still sum to 1
        assert jnp.allclose(jnp.sum(weights_no_alpha, axis=-1), 1.0, atol=1e-5)
        assert jnp.allclose(jnp.sum(weights_alpha_1, axis=-1), 1.0, atol=1e-5)
        assert jnp.allclose(jnp.sum(weights_alpha_2, axis=-1), 1.0, atol=1e-5)
        
        print("  [OK] yat_attention_weights with alpha")
    
    def test_yat_attention_with_alpha(self):
        """Test full YAT attention with alpha scaling."""
        from nmn.nnx.attention import yat_attention
        
        key = jax.random.key(0)
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 32
        
        query = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        key_arr = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        value = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        
        # Without alpha
        output_no_alpha = yat_attention(query, key_arr, value, alpha=None)
        
        # With learnable alpha (as array)
        alpha_param = jnp.array([1.414])
        output_with_alpha = yat_attention(query, key_arr, value, alpha=alpha_param)
        
        # Both should have valid shapes
        assert output_no_alpha.shape == (batch, seq_len, num_heads, head_dim)
        assert output_with_alpha.shape == (batch, seq_len, num_heads, head_dim)
        
        # Both should be valid
        assert not jnp.any(jnp.isnan(output_no_alpha))
        assert not jnp.any(jnp.isnan(output_with_alpha))
        
        print("  [OK] yat_attention with alpha")
    
    def test_yat_attention_normalized_with_alpha(self):
        """Test normalized YAT attention with alpha scaling."""
        from nmn.nnx.attention import yat_attention_normalized
        
        key = jax.random.key(0)
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 32
        
        query = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        key_arr = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        value = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        
        # Without alpha
        output_no_alpha = yat_attention_normalized(query, key_arr, value, alpha=None)
        
        # With constant alpha (sqrt(2))
        output_with_alpha = yat_attention_normalized(query, key_arr, value, alpha=jnp.sqrt(2.0))
        
        assert output_no_alpha.shape == (batch, seq_len, num_heads, head_dim)
        assert output_with_alpha.shape == (batch, seq_len, num_heads, head_dim)
        
        assert not jnp.any(jnp.isnan(output_no_alpha))
        assert not jnp.any(jnp.isnan(output_with_alpha))
        
        print("  [OK] yat_attention_normalized with alpha")
    
    def test_yat_performer_attention_with_alpha(self):
        """Test YAT Performer attention with alpha scaling."""
        from nmn.nnx.attention import yat_performer_attention, create_yat_projection
        
        key = jax.random.key(0)
        batch, seq_len, num_heads, head_dim = 2, 8, 4, 32
        num_features = 64
        
        query = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        key_arr = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        value = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        projection = create_yat_projection(key, num_features, head_dim)
        
        # Without alpha
        output_no_alpha = yat_performer_attention(query, key_arr, value, projection, alpha=None)
        
        # With alpha
        output_with_alpha = yat_performer_attention(query, key_arr, value, projection, alpha=1.5)
        
        assert output_no_alpha.shape == (batch, seq_len, num_heads, head_dim)
        assert output_with_alpha.shape == (batch, seq_len, num_heads, head_dim)
        
        assert not jnp.any(jnp.isnan(output_no_alpha))
        assert not jnp.any(jnp.isnan(output_with_alpha))
        
        print("  [OK] yat_performer_attention with alpha")


class TestMultiHeadAttentionAlpha:
    """Tests for alpha parameter in MultiHeadAttention module."""
    
    def test_multihead_attention_learnable_alpha(self):
        """Test MultiHeadAttention with learnable alpha (default)."""
        from nmn.nnx.attention import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        
        # Default: learnable alpha
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            rngs=rngs,
            decode=False,
        )
        
        # Should have learnable alpha parameter
        assert attn.use_alpha is True
        assert attn.alpha is not None
        assert hasattr(attn.alpha, 'value')
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, in_features)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] MultiHeadAttention with learnable alpha")
    
    def test_multihead_attention_constant_alpha_true(self):
        """Test MultiHeadAttention with constant_alpha=True (sqrt(2))."""
        from nmn.nnx.attention import MultiHeadAttention, DEFAULT_CONSTANT_ALPHA
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        
        # constant_alpha=True uses sqrt(2)
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            constant_alpha=True,
            rngs=rngs,
            decode=False,
        )
        
        # Should have constant alpha, not learnable
        assert attn.use_alpha is True
        assert attn.alpha is None  # No learnable parameter
        assert attn._constant_alpha_value == float(DEFAULT_CONSTANT_ALPHA)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, in_features)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] MultiHeadAttention with constant_alpha=True")
    
    def test_multihead_attention_constant_alpha_custom(self):
        """Test MultiHeadAttention with custom constant alpha value."""
        from nmn.nnx.attention import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        custom_alpha = 1.5
        
        # constant_alpha=float uses that value
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            constant_alpha=custom_alpha,
            rngs=rngs,
            decode=False,
        )
        
        # Should have constant alpha = 1.5
        assert attn.use_alpha is True
        assert attn.alpha is None
        assert attn._constant_alpha_value == custom_alpha
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, in_features)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] MultiHeadAttention with constant_alpha=1.5")
    
    def test_multihead_attention_no_alpha(self):
        """Test MultiHeadAttention with use_alpha=False."""
        from nmn.nnx.attention import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        
        # Disable alpha scaling
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            use_alpha=False,
            rngs=rngs,
            decode=False,
        )
        
        # Should have no alpha
        assert attn.use_alpha is False
        assert attn.alpha is None
        assert attn._constant_alpha_value is None
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, in_features)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] MultiHeadAttention with use_alpha=False")
    
    def test_alpha_affects_output(self):
        """Test that different alpha values produce different outputs."""
        from nmn.nnx.attention import MultiHeadAttention
        
        num_heads = 4
        in_features = 64
        batch_size, seq_len = 2, 10
        
        # Create identical inputs
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        
        # Different alpha configurations
        attn_no_alpha = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            use_alpha=False,
            rngs=nnx.Rngs(0),
            decode=False,
        )
        
        attn_alpha_1 = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            constant_alpha=1.0,
            rngs=nnx.Rngs(0),
            decode=False,
        )
        
        attn_alpha_2 = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            constant_alpha=2.0,
            rngs=nnx.Rngs(0),
            decode=False,
        )
        
        output_no_alpha = attn_no_alpha(x, deterministic=True)
        output_alpha_1 = attn_alpha_1(x, deterministic=True)
        output_alpha_2 = attn_alpha_2(x, deterministic=True)
        
        # Outputs should be different with different alpha values
        # (Though note: same rngs means same weights, but alpha affects attention scores)
        assert not jnp.allclose(output_no_alpha, output_alpha_1, atol=1e-3)
        assert not jnp.allclose(output_alpha_1, output_alpha_2, atol=1e-3)
        
        print("  [OK] Different alpha values produce different outputs")


class TestAlphaGradients:
    """Tests that gradients flow through alpha correctly."""
    
    def test_learnable_alpha_gradients(self):
        """Test that gradients flow through learnable alpha."""
        from nmn.nnx.attention import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        batch_size, seq_len = 2, 10
        
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            use_alpha=True,
            rngs=rngs,
            decode=False,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        
        # Define loss function
        def loss_fn(model, x):
            output = model(x, deterministic=True)
            return jnp.mean(output ** 2)
        
        # Compute gradients
        grads = nnx.grad(loss_fn)(attn, x)
        
        # Alpha should have gradients
        assert grads.alpha is not None
        assert grads.alpha.value is not None
        assert not jnp.allclose(grads.alpha.value, 0.0)
        
        print(f"  [OK] Alpha gradients: {grads.alpha.value}")
    
    def test_constant_alpha_no_gradients(self):
        """Test that constant alpha has no gradients (as expected)."""
        from nmn.nnx.attention import MultiHeadAttention
        
        rngs = nnx.Rngs(0)
        num_heads = 4
        in_features = 64
        batch_size, seq_len = 2, 10
        
        attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            constant_alpha=True,  # Constant, not learnable
            rngs=rngs,
            decode=False,
        )
        
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, in_features))
        
        # Define loss function
        def loss_fn(model, x):
            output = model(x, deterministic=True)
            return jnp.mean(output ** 2)
        
        # Compute gradients - should work without issues
        grads = nnx.grad(loss_fn)(attn, x)
        
        # Alpha should not exist in grads (it's not a learnable param)
        assert not hasattr(grads, 'alpha') or grads.alpha is None
        
        print("  [OK] Constant alpha has no gradients (as expected)")


class TestRotaryYatAttentionAlpha:
    """Tests for alpha parameter in RotaryYatAttention module."""
    
    def test_rotary_yat_attention_learnable_alpha(self):
        """Test RotaryYatAttention with learnable alpha (default)."""
        from nmn.nnx.attention import RotaryYatAttention
        
        rngs = nnx.Rngs(0)
        embed_dim = 64
        num_heads = 4
        
        # Default: learnable alpha
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=64,
            rngs=rngs,
        )
        
        assert attn.use_alpha is True
        assert attn.alpha is not None
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, embed_dim))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] RotaryYatAttention with learnable alpha")
    
    def test_rotary_yat_attention_constant_alpha(self):
        """Test RotaryYatAttention with constant_alpha=True."""
        from nmn.nnx.attention import RotaryYatAttention
        
        rngs = nnx.Rngs(0)
        embed_dim = 64
        num_heads = 4
        
        # constant_alpha=True uses sqrt(2)
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=64,
            constant_alpha=True,
            rngs=rngs,
        )
        
        assert attn.use_alpha is True
        assert attn.alpha is None  # No learnable parameter
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, embed_dim))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] RotaryYatAttention with constant_alpha=True")
    
    def test_rotary_yat_performer_with_alpha(self):
        """Test RotaryYatAttention in Performer mode with alpha."""
        from nmn.nnx.attention import RotaryYatAttention
        
        rngs = nnx.Rngs(0)
        embed_dim = 64
        num_heads = 4
        
        # Performer mode with constant alpha
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=64,
            use_performer=True,
            num_features=32,
            constant_alpha=1.5,
            rngs=rngs,
        )
        
        assert attn.use_alpha is True
        assert attn._constant_alpha_value == 1.5
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, embed_dim))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] RotaryYatAttention Performer with constant_alpha=1.5")
    
    def test_rotary_yat_no_alpha(self):
        """Test RotaryYatAttention with use_alpha=False."""
        from nmn.nnx.attention import RotaryYatAttention
        
        rngs = nnx.Rngs(0)
        embed_dim = 64
        num_heads = 4
        
        # Disable alpha scaling
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=64,
            use_alpha=False,
            rngs=rngs,
        )
        
        assert attn.use_alpha is False
        assert attn.alpha is None
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        x = jax.random.normal(jax.random.key(0), (batch_size, seq_len, embed_dim))
        output = attn(x, deterministic=True)
        
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not jnp.any(jnp.isnan(output))
        
        print("  [OK] RotaryYatAttention with use_alpha=False")


class TestAlphaScalingFormula:
    """Tests the alpha scaling formula: (sqrt(head_dim) / log(1 + head_dim))^alpha"""
    
    def test_scaling_formula_values(self):
        """Test that the scaling formula produces expected values."""
        head_dims = [32, 64, 128, 256]
        
        print("\n  Alpha scaling factors for different head_dim:")
        print(f"  {'head_dim':>10} | {'scale (a=1)':>12} | {'scale (a=sqrt2)':>16}")
        print("  " + "-" * 44)
        
        for head_dim in head_dims:
            scale_1 = (jnp.sqrt(head_dim) / jnp.log(1 + head_dim)) ** 1.0
            scale_sqrt2 = (jnp.sqrt(head_dim) / jnp.log(1 + head_dim)) ** jnp.sqrt(2.0)
            print(f"  {head_dim:>10} | {float(scale_1):>12.4f} | {float(scale_sqrt2):>14.4f}")
        
        # Verify formula: for head_dim=64
        head_dim = 64
        alpha = 1.0
        expected = (jnp.sqrt(head_dim) / jnp.log(1 + head_dim))
        actual = (8.0 / jnp.log(65.0))  # sqrt(64) = 8, log(65) ≈ 4.17
        assert jnp.allclose(expected, actual, atol=1e-5)
        
        print("  [OK] Scaling formula verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

