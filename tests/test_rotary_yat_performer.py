"""Test script to debug Rotary YAT Performer attention for loss explosion.

This script tests various aspects of the attention mechanism to identify
potential sources of numerical instability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

# Import the attention modules
from nmn.nnx.attention import RotaryYatAttention
from nmn.nnx.attention.yat_attention import (
    normalize_qk,
    yat_performer_feature_map,
    yat_performer_attention,
    create_yat_projection,
)
from nmn.nnx.attention.rotary_yat import (
    precompute_freqs_cis,
    apply_rotary_emb,
    rotary_yat_performer_attention,
)


def print_stats(name: str, arr: jnp.ndarray):
    """Print statistics about an array."""
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Min: {float(jnp.min(arr)):.6g}")
    print(f"  Max: {float(jnp.max(arr)):.6g}")
    print(f"  Mean: {float(jnp.mean(arr)):.6g}")
    print(f"  Std: {float(jnp.std(arr)):.6g}")
    print(f"  Has NaN: {bool(jnp.any(jnp.isnan(arr)))}")
    print(f"  Has Inf: {bool(jnp.any(jnp.isinf(arr)))}")


def test_normalize_qk():
    """Test the Q/K normalization function."""
    print("\n" + "="*60)
    print("TEST: normalize_qk")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    
    # Test with normal random inputs
    q = jax.random.normal(key, (2, 16, 8, 64))  # batch, seq, heads, dim
    k = jax.random.normal(jax.random.split(key)[0], (2, 16, 8, 64))
    
    print_stats("Input Q", q)
    print_stats("Input K", k)
    
    q_norm, k_norm = normalize_qk(q, k)
    
    print_stats("Normalized Q", q_norm)
    print_stats("Normalized K", k_norm)
    
    # Check that vectors are unit norm
    q_norms = jnp.sqrt(jnp.sum(q_norm ** 2, axis=-1))
    k_norms = jnp.sqrt(jnp.sum(k_norm ** 2, axis=-1))
    print(f"\n  Q norms after normalization (should be ~1): mean={float(jnp.mean(q_norms)):.6f}, std={float(jnp.std(q_norms)):.6g}")
    print(f"  K norms after normalization (should be ~1): mean={float(jnp.mean(k_norms)):.6f}, std={float(jnp.std(k_norms)):.6g}")
    
    # Test with extreme inputs (potential issue)
    print("\n--- Testing with extreme values ---")
    q_extreme = jax.random.normal(key, (2, 16, 8, 64)) * 100
    k_extreme = jax.random.normal(jax.random.split(key)[0], (2, 16, 8, 64)) * 0.001
    
    print_stats("Extreme Q (large)", q_extreme)
    print_stats("Extreme K (small)", k_extreme)
    
    q_norm_ext, k_norm_ext = normalize_qk(q_extreme, k_extreme)
    print_stats("Normalized Extreme Q", q_norm_ext)
    print_stats("Normalized Extreme K", k_norm_ext)


def test_feature_map():
    """Test the YAT performer feature map."""
    print("\n" + "="*60)
    print("TEST: yat_performer_feature_map")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    head_dim = 64
    num_features = 32
    
    # Create projection matrix
    projection = create_yat_projection(key, num_features, head_dim)
    print_stats("Projection matrix", projection)
    
    # Create normalized input
    x = jax.random.normal(key, (2, 16, 8, head_dim))
    x_normalized, _ = normalize_qk(x, x)
    
    print_stats("Normalized input", x_normalized)
    
    # Test feature map with pre_normalized=True
    features = yat_performer_feature_map(x_normalized, projection, pre_normalized=True)
    print_stats("Features (pre_normalized=True)", features)
    
    # Test feature map with pre_normalized=False  
    features_unnorm = yat_performer_feature_map(x, projection, pre_normalized=False)
    print_stats("Features (pre_normalized=False)", features_unnorm)


def test_rotary_embeddings():
    """Test RoPE embeddings."""
    print("\n" + "="*60)
    print("TEST: Rotary Embeddings")
    print("="*60)
    
    head_dim = 64
    max_seq_len = 1024
    
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len)
    print_stats("freqs_cos", freqs_cos)
    print_stats("freqs_sin", freqs_sin)
    
    # Test applying to a vector
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (2, 16, 8, head_dim))
    
    x_rotated = apply_rotary_emb(x, freqs_cos, freqs_sin)
    print_stats("Input", x)
    print_stats("After RoPE", x_rotated)
    
    # Check that norms are preserved (RoPE should be a rotation)
    input_norms = jnp.sqrt(jnp.sum(x ** 2, axis=-1))
    output_norms = jnp.sqrt(jnp.sum(x_rotated ** 2, axis=-1))
    norm_diff = jnp.abs(input_norms - output_norms)
    print(f"\n  Norm preservation (should be ~0): mean_diff={float(jnp.mean(norm_diff)):.6g}, max_diff={float(jnp.max(norm_diff)):.6g}")


def test_yat_performer_attention():
    """Test the full YAT performer attention."""
    print("\n" + "="*60)
    print("TEST: yat_performer_attention")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64
    num_features = 32
    
    # Create inputs
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim))
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim))
    
    projection = create_yat_projection(k4, num_features, head_dim)
    
    print_stats("Query", q)
    print_stats("Key", k)
    print_stats("Value", v)
    
    # Test with normalize_inputs=True (the default for performer mode)
    output = yat_performer_attention(
        q, k, v, projection,
        normalize_inputs=True,
        epsilon=1e-5,
    )
    print_stats("Output (normalize_inputs=True)", output)
    
    # Test with normalize_inputs=False
    output_unnorm = yat_performer_attention(
        q, k, v, projection,
        normalize_inputs=False,
        epsilon=1e-5,
    )
    print_stats("Output (normalize_inputs=False)", output_unnorm)


def test_rotary_yat_performer():
    """Test the full Rotary YAT Performer attention."""
    print("\n" + "="*60)
    print("TEST: rotary_yat_performer_attention")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64
    num_features = 32
    max_seq_len = 1024
    
    # Create inputs
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim))
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim))
    
    projection = create_yat_projection(k4, num_features, head_dim)
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len)
    
    # Test with normalize_inputs=True
    output = rotary_yat_performer_attention(
        q, k, v,
        freqs_cos, freqs_sin,
        projection,
        normalize_inputs=True,
        epsilon=1e-5,
    )
    print_stats("Output (normalize_inputs=True)", output)
    
    # Test with normalize_inputs=False
    output_unnorm = rotary_yat_performer_attention(
        q, k, v,
        freqs_cos, freqs_sin,
        projection,
        normalize_inputs=False,
        epsilon=1e-5,
    )
    print_stats("Output (normalize_inputs=False)", output_unnorm)


def test_rotary_yat_attention_module():
    """Test the RotaryYatAttention nnx.Module."""
    print("\n" + "="*60)
    print("TEST: RotaryYatAttention Module (Performer mode)")
    print("="*60)
    
    rngs = nnx.Rngs(42)
    embed_dim = 256
    num_heads = 8
    max_seq_len = 1024
    batch_size = 2
    seq_len = 16
    
    # Create module in Performer mode
    attn = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_performer=True,
        num_features=embed_dim // 2,
        performer_normalize=True,
        constant_alpha=True,  # sqrt(2)
        rngs=rngs,
    )
    
    # Create input
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
    
    print_stats("Input", x)
    
    # Forward pass
    output = attn(x, deterministic=True)
    print_stats("Output", output)


def test_gradient_flow():
    """Test gradient flow through the attention."""
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)
    
    rngs = nnx.Rngs(42)
    embed_dim = 256
    num_heads = 8
    max_seq_len = 1024
    batch_size = 2
    seq_len = 16
    
    # Create module
    attn = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_performer=True,
        num_features=embed_dim // 2,
        performer_normalize=True,
        constant_alpha=True,
        rngs=rngs,
    )
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
    target = jax.random.normal(jax.random.split(key)[0], (batch_size, seq_len, embed_dim))
    
    def loss_fn(model, x, target):
        output = model(x, deterministic=True)
        return jnp.mean((output - target) ** 2)
    
    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(attn, x, target)
    
    print(f"\nLoss: {float(loss):.6g}")
    
    # Check gradients
    grad_state = nnx.state(grads)
    for path, value in nnx.iter_graph(grad_state):
        if hasattr(value, 'value') and value.value is not None:
            arr = value.value
            if isinstance(arr, jnp.ndarray):
                has_nan = bool(jnp.any(jnp.isnan(arr)))
                has_inf = bool(jnp.any(jnp.isinf(arr)))
                if has_nan or has_inf:
                    print(f"  ⚠️ {path}: NaN={has_nan}, Inf={has_inf}")
                else:
                    max_grad = float(jnp.max(jnp.abs(arr)))
                    print(f"  ✓ {path}: max_abs_grad={max_grad:.6g}")


def test_long_sequence():
    """Test with longer sequences to check for scaling issues."""
    print("\n" + "="*60)
    print("TEST: Long Sequence (seq_len=512)")
    print("="*60)
    
    rngs = nnx.Rngs(42)
    embed_dim = 256
    num_heads = 8
    max_seq_len = 1024
    batch_size = 2
    seq_len = 512
    
    attn = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_performer=True,
        num_features=embed_dim // 2,
        performer_normalize=True,
        constant_alpha=True,
        rngs=rngs,
    )
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
    
    print_stats("Input", x)
    output = attn(x, deterministic=True)
    print_stats("Output", output)


def test_feature_map_kernel_approximation():
    """Test if the feature map properly approximates the YAT kernel."""
    print("\n" + "="*60)
    print("TEST: Feature Map Kernel Approximation")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    head_dim = 64
    num_features = 256  # More features for better approximation
    
    # Create normalized q and k vectors
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (head_dim,))
    k = jax.random.normal(k2, (head_dim,))
    q = q / jnp.linalg.norm(q)
    k = k / jnp.linalg.norm(k)
    
    # Compute exact YAT kernel: (q·k)² / (2(1 - q·k) + ε)
    dot = jnp.dot(q, k)
    exact_yat = (dot ** 2) / (2 * (1 - dot) + 1e-5)
    print(f"\nDot product: {float(dot):.6f}")
    print(f"Exact YAT kernel: {float(exact_yat):.6f}")
    
    # Compute approximation via feature maps
    projection = create_yat_projection(k3, num_features, head_dim)
    
    # Reshape for feature map function
    q_reshaped = q.reshape(1, 1, 1, head_dim)
    k_reshaped = k.reshape(1, 1, 1, head_dim)
    
    q_features = yat_performer_feature_map(q_reshaped, projection, pre_normalized=True)
    k_features = yat_performer_feature_map(k_reshaped, projection, pre_normalized=True)
    
    # Approximate kernel: φ(q)·φ(k)
    approx_kernel = float(jnp.sum(q_features * k_features))
    print(f"Approximate kernel: {approx_kernel:.6f}")
    print(f"Relative error: {abs(approx_kernel - exact_yat) / abs(exact_yat) * 100:.2f}%")
    
    # The feature map uses softplus, not the exact YAT formula
    # Let's check what the features look like
    print_stats("Q features", q_features)
    print_stats("K features", k_features)


def main():
    print("="*60)
    print("DEBUGGING ROTARY YAT PERFORMER ATTENTION")
    print("="*60)
    
    test_normalize_qk()
    test_feature_map()
    test_rotary_embeddings()
    test_yat_performer_attention()
    test_rotary_yat_performer()
    test_rotary_yat_attention_module()
    test_gradient_flow()
    test_long_sequence()
    test_feature_map_kernel_approximation()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
