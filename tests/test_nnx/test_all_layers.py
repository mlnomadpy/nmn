"""
Comprehensive Tests for All Flax NNX YAT Layers.

Tests all layer variants with consistent test patterns.
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    NNX_AVAILABLE = True
except ImportError:
    NNX_AVAILABLE = False


pytestmark = pytest.mark.skipif(not NNX_AVAILABLE, reason="JAX/Flax NNX not available")


# ============================================================================
# Layer Import Tests
# ============================================================================

class TestLayerImports:
    """Test that all layers can be imported."""
    
    def test_import_yat_nmn(self):
        """Test YatNMN import."""
        from nmn.nnx.nmn import YatNMN
        assert YatNMN is not None
    
    def test_import_yat_conv(self):
        """Test YatConv import."""
        from nmn.nnx.yatconv import YatConv
        assert YatConv is not None
    
    def test_import_yat_conv_transpose(self):
        """Test YatConvTranspose import."""
        from nmn.nnx.yatconv_transpose import YatConvTranspose
        assert YatConvTranspose is not None
    
    def test_import_yat_attention(self):
        """Test attention imports."""
        from nmn.nnx.yatattention import yat_attention_weights, MultiHeadAttention
        assert yat_attention_weights is not None
        assert MultiHeadAttention is not None
    
    def test_import_rnn_cells(self):
        """Test RNN cell imports."""
        from nmn.nnx.rnn import YatSimpleCell, YatLSTMCell, YatGRUCell
        assert YatSimpleCell is not None
        assert YatLSTMCell is not None
        assert YatGRUCell is not None
    
    def test_import_squashers(self):
        """Test squasher imports."""
        from nmn.nnx.squashers import softermax, softer_sigmoid, soft_tanh
        assert softermax is not None
        assert softer_sigmoid is not None
        assert soft_tanh is not None


# ============================================================================
# YatNMN (Dense) Tests
# ============================================================================

class TestYatNMNComprehensive:
    """Comprehensive tests for NNX YatNMN (dense layer)."""
    
    def test_instantiation(self):
        """Test basic instantiation."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        assert layer.out_features == 16
    
    def test_forward_pass(self):
        """Test forward pass."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        x = jnp.ones((2, 8))
        output = layer(x)
        assert output.shape == (2, 16)
    
    def test_with_bias(self):
        """Test with bias enabled."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, use_bias=True, rngs=rngs)
        assert layer.bias is not None
    
    def test_without_bias(self):
        """Test without bias."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, use_bias=False, rngs=rngs)
        assert layer.bias is None
    
    def test_with_alpha(self):
        """Test with alpha scaling."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, use_alpha=True, rngs=rngs)
        assert layer.alpha is not None
    
    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        x = jnp.ones((2, 8))
        
        @nnx.value_and_grad
        def loss_fn(model):
            return jnp.sum(model(x))
        
        loss, grads = loss_fn(layer)
        assert grads.kernel.value is not None
    
    def test_positive_outputs_no_bias(self):
        """Test that YAT produces non-negative outputs without bias."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(42)
        
        layer = YatNMN(in_features=8, out_features=16, use_bias=False, use_alpha=False, rngs=rngs)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (4, 8))
        output = layer(x)
        
        assert jnp.all(output >= 0), "YAT should produce non-negative values"
    
    def test_deterministic_mode(self):
        """Test deterministic mode disables DropConnect."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        
        layer = YatNMN(
            in_features=8, out_features=16,
            use_dropconnect=True, drop_rate=0.5, rngs=rngs
        )
        x = jnp.ones((2, 8))
        
        # Deterministic mode should give consistent outputs
        out1 = layer(x, deterministic=True)
        out2 = layer(x, deterministic=True)
        
        np.testing.assert_allclose(np.array(out1), np.array(out2), rtol=1e-6)


# ============================================================================
# YatConv Tests
# ============================================================================

class TestYatConvComprehensive:
    """Comprehensive tests for NNX YatConv."""
    
    def test_conv1d_forward_valid_padding(self):
        """Test 1D convolution forward pass with valid padding."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3,), padding='VALID', rngs=rngs)
        x = jnp.ones((2, 16, 3))
        output = layer(x)
        assert output.shape == (2, 14, 8)
    
    def test_conv1d_forward_same_padding(self):
        """Test 1D convolution forward pass with same padding."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3,), padding='SAME', rngs=rngs)
        x = jnp.ones((2, 16, 3))
        output = layer(x)
        assert output.shape == (2, 16, 8)
    
    def test_conv2d_forward_valid_padding(self):
        """Test 2D convolution forward pass with valid padding."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3, 3), padding='VALID', rngs=rngs)
        x = jnp.ones((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 14, 14, 8)
    
    def test_conv2d_forward_same_padding(self):
        """Test 2D convolution forward pass with same padding."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        x = jnp.ones((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 16, 16, 8)
    
    def test_conv2d_stride(self):
        """Test 2D convolution with stride."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        x = jnp.ones((2, 16, 16, 3))
        output = layer(x)
        assert output.shape == (2, 8, 8, 8)
    
    def test_gradient_flow(self):
        """Test gradient computation."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(0)
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3, 3), rngs=rngs)
        x = jnp.ones((2, 16, 16, 3))
        
        @nnx.value_and_grad
        def loss_fn(model):
            return jnp.sum(model(x))
        
        loss, grads = loss_fn(layer)
        assert grads.kernel.value is not None
    
    def test_positive_outputs(self):
        """Test that outputs are non-negative without bias."""
        from nmn.nnx.yatconv import YatConv
        rngs = nnx.Rngs(42)
        
        layer = YatConv(in_features=3, out_features=8, kernel_size=(3, 3), use_bias=False, use_alpha=False, rngs=rngs)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 16, 16, 3))
        output = layer(x)
        
        assert jnp.all(output >= 0), "YAT should produce non-negative values"


# ============================================================================
# YatConvTranspose Tests
# ============================================================================

class TestYatConvTransposeComprehensive:
    """Comprehensive tests for NNX YatConvTranspose."""
    
    def test_conv_transpose2d_forward(self):
        """Test 2D transposed convolution forward pass."""
        from nmn.nnx.yatconv_transpose import YatConvTranspose
        rngs = nnx.Rngs(0)
        layer = YatConvTranspose(
            in_features=8, out_features=3, kernel_size=(2, 2), strides=(2, 2), rngs=rngs
        )
        x = jnp.ones((2, 8, 8, 8))
        output = layer(x)
        # Should upsample spatial dimensions
        assert output.shape[0] == 2
        assert output.shape[3] == 3
        assert output.shape[1] > 8
        assert output.shape[2] > 8
    
    def test_gradient_flow(self):
        """Test gradient computation for transposed conv."""
        from nmn.nnx.yatconv_transpose import YatConvTranspose
        rngs = nnx.Rngs(0)
        layer = YatConvTranspose(
            in_features=8, out_features=3, kernel_size=(2, 2), strides=(2, 2), rngs=rngs
        )
        x = jnp.ones((2, 8, 8, 8))
        
        @nnx.value_and_grad
        def loss_fn(model):
            return jnp.sum(model(x))
        
        loss, grads = loss_fn(layer)
        assert grads.kernel.value is not None


# ============================================================================
# Squasher Tests
# ============================================================================

class TestSquashers:
    """Test squasher activation functions."""
    
    def test_softermax_basic(self):
        """Test basic softermax functionality."""
        from nmn.nnx.squashers import softermax
        
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = softermax(x)
        
        # Should sum to 1 along last axis
        sums = jnp.sum(output, axis=-1)
        np.testing.assert_allclose(np.array(sums), np.ones(2), rtol=1e-5)
    
    def test_softermax_no_nan(self):
        """Test that softermax doesn't produce NaN."""
        from nmn.nnx.squashers import softermax
        
        # Large values that might cause overflow
        x = jnp.array([[100.0, 200.0, 300.0]])
        output = softermax(x)
        
        assert not jnp.isnan(output).any()
        assert not jnp.isinf(output).any()
    
    def test_softer_sigmoid_no_nan(self):
        """Test that softer_sigmoid doesn't produce NaN."""
        from nmn.nnx.squashers import softer_sigmoid
        
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
        output = softer_sigmoid(x)
        
        # Check for NaN/Inf
        assert not jnp.isnan(output).any()
        assert not jnp.isinf(output).any()
    
    def test_soft_tanh_no_nan(self):
        """Test that soft_tanh doesn't produce NaN."""
        from nmn.nnx.squashers import soft_tanh
        
        x = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
        output = soft_tanh(x)
        
        # Check for NaN/Inf
        assert not jnp.isnan(output).any()
        assert not jnp.isinf(output).any()


# ============================================================================
# RNN Cell Tests (basic tests only - these have specific API requirements)
# ============================================================================

class TestRNNCells:
    """Test RNN cell implementations."""
    
    def test_simple_cell_instantiation(self):
        """Test YatSimpleCell can be instantiated."""
        from nmn.nnx.rnn import YatSimpleCell
        rngs = nnx.Rngs(0)
        
        cell = YatSimpleCell(in_features=8, hidden_features=16, rngs=rngs)
        assert cell.in_features == 8
        assert cell.hidden_features == 16
    
    def test_lstm_cell_instantiation(self):
        """Test YatLSTMCell can be instantiated."""
        from nmn.nnx.rnn import YatLSTMCell
        rngs = nnx.Rngs(0)
        
        cell = YatLSTMCell(in_features=8, hidden_features=16, rngs=rngs)
        assert cell.in_features == 8
        assert cell.hidden_features == 16
    
    def test_gru_cell_instantiation(self):
        """Test YatGRUCell can be instantiated."""
        from nmn.nnx.rnn import YatGRUCell
        rngs = nnx.Rngs(0)
        
        cell = YatGRUCell(in_features=8, hidden_features=16, rngs=rngs)
        assert cell.in_features == 8
        assert cell.hidden_features == 16


# ============================================================================
# MultiHeadAttention Tests
# ============================================================================

class TestMultiHeadAttention:
    """Test MultiHeadAttention implementation."""
    
    def test_attention_forward(self):
        """Test basic attention forward pass."""
        from nmn.nnx.yatattention import MultiHeadAttention
        rngs = nnx.Rngs(0)
        
        attn = MultiHeadAttention(
            num_heads=4,
            in_features=32,
            rngs=rngs
        )
        
        # Self-attention
        x = jnp.ones((2, 10, 32))  # (batch, seq_len, features)
        output = attn(x, x, decode=False)
        
        assert output.shape == (2, 10, 32)
    
    def test_attention_with_mask(self):
        """Test attention with mask."""
        from nmn.nnx.yatattention import MultiHeadAttention
        rngs = nnx.Rngs(0)
        
        attn = MultiHeadAttention(
            num_heads=4,
            in_features=32,
            rngs=rngs
        )
        
        x = jnp.ones((2, 10, 32))
        mask = jnp.ones((2, 4, 10, 10))  # (batch, heads, q_len, kv_len)
        
        output = attn(x, x, mask=mask, decode=False)
        assert output.shape == (2, 10, 32)
    
    def test_self_attention(self):
        """Test self-attention (q and kv are the same)."""
        from nmn.nnx.yatattention import MultiHeadAttention
        rngs = nnx.Rngs(0)
        
        attn = MultiHeadAttention(
            num_heads=4,
            in_features=32,
            rngs=rngs
        )
        
        x = jnp.ones((2, 8, 32))  # Self-attention
        
        output = attn(x, x, decode=False)
        assert output.shape == (2, 8, 32)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        x = jnp.ones((1, 8))
        output = layer(x)
        assert output.shape == (1, 16)
    
    def test_large_batch(self):
        """Test with large batch."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        x = jnp.ones((64, 8))
        output = layer(x)
        assert output.shape == (64, 16)
    
    def test_jit_compatibility(self):
        """Test that layers work with JAX JIT compilation."""
        from nmn.nnx.nmn import YatNMN
        rngs = nnx.Rngs(0)
        layer = YatNMN(in_features=8, out_features=16, rngs=rngs)
        
        @jax.jit
        def forward(x):
            return layer(x)
        
        x = jnp.ones((2, 8))
        output = forward(x)
        assert output.shape == (2, 16)
