"""Comprehensive tests for NNX implementation."""

import pytest
import numpy as np


def test_nnx_yat_attention():
    """Test MultiHeadAttention layer with Yat operations."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.yatattention import MultiHeadAttention
        
        # Test parameters
        num_heads = 4
        in_features = 64
        qkv_features = 64
        out_features = 64
        key = jax.random.key(0)
        
        # Create attention layer
        attention = MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=out_features,
            rngs=nnx.Rngs(params=key)
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        dummy_input = jax.random.normal(jax.random.key(1), (batch_size, seq_len, in_features))
        output = attention(dummy_input)
        
        assert output.shape == (batch_size, seq_len, out_features)
        assert output.dtype == dummy_input.dtype
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_yat_conv_transpose():
    """Test YatConvTranspose layer."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.yatconv_transpose import YatConvTranspose
        
        # Test parameters
        in_channels, out_channels = 16, 8
        kernel_size = (3, 3)
        key = jax.random.key(0)
        
        # Create layer
        layer = YatConvTranspose(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=(2, 2),
            padding='VALID',
            rngs=nnx.Rngs(params=key)
        )
        
        # Test forward pass
        dummy_input = jax.random.normal(jax.random.key(1), (1, 8, 8, in_channels))
        output = layer(dummy_input)
        
        # For transpose conv with stride=2, output should be roughly 2x input size
        assert output.shape[0] == 1
        assert output.shape[-1] == out_channels
        assert output.shape[1] > dummy_input.shape[1]
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_rnn_simple():
    """Test YatSimpleCell RNN."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.rnn import YatSimpleCell
        
        # Test parameters
        in_features, hidden_features = 32, 64
        key = jax.random.key(0)
        
        # Create RNN cell
        cell = YatSimpleCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=nnx.Rngs(params=key)
        )
        
        # Initialize carry
        batch_size = 4
        carry = cell.initialize_carry(jax.random.key(1), (batch_size,))
        
        # Test forward pass
        dummy_input = jax.random.normal(jax.random.key(2), (batch_size, in_features))
        new_carry, output = cell(carry, dummy_input)
        
        assert output.shape == (batch_size, hidden_features)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_rnn_lstm():
    """Test YatLSTMCell."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.rnn import YatLSTMCell
        
        # Test parameters
        in_features, hidden_features = 32, 64
        key = jax.random.key(0)
        
        # Create LSTM cell
        cell = YatLSTMCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=nnx.Rngs(params=key)
        )
        
        # Initialize carry
        batch_size = 4
        carry = cell.initialize_carry(jax.random.key(1), (batch_size,))
        
        # Test forward pass
        dummy_input = jax.random.normal(jax.random.key(2), (batch_size, in_features))
        new_carry, output = cell(carry, dummy_input)
        
        assert output.shape == (batch_size, hidden_features)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_rnn_gru():
    """Test YatGRUCell."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.rnn import YatGRUCell
        
        # Test parameters
        in_features, hidden_features = 32, 64
        key = jax.random.key(0)
        
        # Create GRU cell
        cell = YatGRUCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=nnx.Rngs(params=key)
        )
        
        # Initialize carry
        batch_size = 4
        carry = cell.initialize_carry(jax.random.key(1), (batch_size,))
        
        # Test forward pass
        dummy_input = jax.random.normal(jax.random.key(2), (batch_size, in_features))
        new_carry, output = cell(carry, dummy_input)
        
        assert output.shape == (batch_size, hidden_features)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_squashers():
    """Test custom activation functions."""
    try:
        import jax
        import jax.numpy as jnp
        from nmn.nnx.squashers import softermax, softer_sigmoid, soft_tanh
        
        # Test data
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        
        # Test softermax
        out1 = softermax(x)
        assert out1.shape == x.shape
        assert jnp.allclose(jnp.sum(out1), 1.0, rtol=1e-5)
        
        # Test softer_sigmoid
        out2 = softer_sigmoid(x)
        assert out2.shape == x.shape
        assert jnp.all(out2 >= 0) and jnp.all(out2 <= 1)
        
        # Test soft_tanh
        out3 = soft_tanh(x)
        assert out3.shape == x.shape
        assert jnp.all(out3 >= -1) and jnp.all(out3 <= 1)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_yat_conv_dropconnect():
    """Test YatConv with DropConnect."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.yatconv import YatConv
        
        # Test parameters
        in_channels, out_channels = 3, 8
        kernel_size = (3, 3)
        key = jax.random.key(0)
        drop_key = jax.random.key(1)
        
        # Create layer with DropConnect
        layer = YatConv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            use_dropconnect=True,
            drop_rate=0.2,
            padding='VALID',
            rngs=nnx.Rngs(params=key, dropout=drop_key)
        )
        
        # Test forward pass in training mode
        dummy_input = jax.random.normal(jax.random.key(2), (1, 28, 28, in_channels))
        output_train = layer(dummy_input, deterministic=False)
        output_eval = layer(dummy_input, deterministic=True)
        
        assert output_train.shape == output_eval.shape
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")


def test_nnx_yat_nmn_dropconnect():
    """Test YatNMN with DropConnect."""
    try:
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from nmn.nnx.nmn import YatNMN
        
        # Test parameters
        in_features, out_features = 64, 32
        key = jax.random.key(0)
        drop_key = jax.random.key(1)
        
        # Create layer with DropConnect
        layer = YatNMN(
            in_features=in_features,
            out_features=out_features,
            use_dropconnect=True,
            drop_rate=0.2,
            rngs=nnx.Rngs(params=key, dropout=drop_key)
        )
        
        # Test forward pass
        dummy_input = jax.random.normal(jax.random.key(2), (4, in_features))
        output_train = layer(dummy_input, deterministic=False)
        output_eval = layer(dummy_input, deterministic=True)
        
        assert output_train.shape == (4, out_features)
        assert output_eval.shape == (4, out_features)
        
    except ImportError:
        pytest.skip("JAX/Flax dependencies not available")

