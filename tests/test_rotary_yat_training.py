"""Test training stability of Rotary YAT Performer attention.

This script tests the attention mechanism in isolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from nmn.nnx.attention import RotaryYatAttention
from nmn.nnx.nmn import YatNMN
from nmn.nnx.attention.yat_attention import (
    normalize_qk,
    yat_performer_feature_map,
    create_yat_projection,
)
from nmn.nnx.attention.rotary_yat import (
    precompute_freqs_cis,
    apply_rotary_emb,
    rotary_yat_performer_attention,
)


def print_stats(name: str, arr: jnp.ndarray):
    """Print statistics about an array."""
    if arr is None:
        print(f"{name}: None")
        return
    print(f"{name}: min={float(jnp.min(arr)):.4g}, max={float(jnp.max(arr)):.4g}, "
          f"mean={float(jnp.mean(arr)):.4g}, std={float(jnp.std(arr)):.4g}, "
          f"nan={bool(jnp.any(jnp.isnan(arr)))}, inf={bool(jnp.any(jnp.isinf(arr)))}")


def test_intermediate_values():
    """Test intermediate values during forward pass."""
    print("\n" + "="*70)
    print("TEST: Intermediate Values During Forward Pass")
    print("="*70)
    
    rngs = nnx.Rngs(42)
    embed_dim = 128
    num_heads = 4
    maxlen = 64
    batch_size = 2
    
    # Create just the attention layer
    attn = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=maxlen,
        use_performer=True,
        num_features=embed_dim // 2,
        performer_normalize=True,
        constant_alpha=True,
        rngs=rngs,
    )
    
    # Create input with different scales to test robustness
    key = jax.random.PRNGKey(42)
    
    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        print(f"\n--- Input scale: {scale} ---")
        x = jax.random.normal(key, (batch_size, maxlen, embed_dim)) * scale
        
        # Forward pass
        output = attn(x, deterministic=True)
        
        print_stats("Input", x)
        print_stats("Output", output)
        
        # Check ratio
        output_scale = float(jnp.std(output))
        input_scale = float(jnp.std(x))
        print(f"Output/Input std ratio: {output_scale/input_scale:.4f}")


def test_alpha_scaling_impact():
    """Test the impact of alpha scaling on stability."""
    print("\n" + "="*70)
    print("TEST: Alpha Scaling Impact")
    print("="*70)
    
    key = jax.random.PRNGKey(42)
    embed_dim = 128
    num_heads = 4
    maxlen = 64
    batch_size = 2
    
    x = jax.random.normal(key, (batch_size, maxlen, embed_dim))
    
    # Test with different alpha configurations
    configs = [
        ("No alpha", {"use_alpha": False, "constant_alpha": None}),
        ("Constant alpha=sqrt(2)", {"use_alpha": True, "constant_alpha": True}),
        ("Constant alpha=1.0", {"use_alpha": True, "constant_alpha": 1.0}),
        ("Constant alpha=2.0", {"use_alpha": True, "constant_alpha": 2.0}),
    ]
    
    for name, kwargs in configs:
        rngs = nnx.Rngs(42)
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=maxlen,
            use_performer=True,
            num_features=embed_dim // 2,
            performer_normalize=True,
            rngs=rngs,
            **kwargs,
        )
        
        output = attn(x, deterministic=True)
        print(f"\n{name}:")
        print_stats("  Output", output)


def test_gradient_magnitudes():
    """Test gradient magnitudes through the attention layer."""
    print("\n" + "="*70)
    print("TEST: Gradient Magnitudes")
    print("="*70)
    
    rngs = nnx.Rngs(42)
    embed_dim = 256
    num_heads = 8
    maxlen = 128
    batch_size = 4
    
    attn = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=maxlen,
        use_performer=True,
        num_features=embed_dim // 2,
        performer_normalize=True,
        constant_alpha=True,
        rngs=rngs,
    )
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, maxlen, embed_dim))
    target = jax.random.normal(jax.random.split(key)[0], (batch_size, maxlen, embed_dim))
    
    def loss_fn(model, x, target):
        output = model(x, deterministic=True)
        return jnp.mean((output - target) ** 2)
    
    # Compute gradients
    loss, grads = nnx.value_and_grad(loss_fn)(attn, x, target)
    
    print(f"\nLoss: {float(loss):.6g}")
    
    # Check gradients on the projections
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if hasattr(grads, name):
            proj = getattr(grads, name)
            if hasattr(proj, 'kernel') and hasattr(proj.kernel, 'value'):
                g = proj.kernel.value
                if g is not None:
                    has_nan = bool(jnp.any(jnp.isnan(g)))
                    has_inf = bool(jnp.any(jnp.isinf(g)))
                    max_grad = float(jnp.max(jnp.abs(g)))
                    mean_grad = float(jnp.mean(jnp.abs(g)))
                    status = "✓" if not (has_nan or has_inf) else "⚠️"
                    print(f"  {status} {name}: max_abs={max_grad:.6g}, mean_abs={mean_grad:.6g}, nan={has_nan}, inf={has_inf}")


def test_feature_map_behavior():
    """Deep dive into the feature map behavior."""
    print("\n" + "="*70)
    print("TEST: Feature Map Behavior Analysis")
    print("="*70)
    
    key = jax.random.PRNGKey(42)
    head_dim = 64
    num_features = 32
    
    # Create projection
    projection = create_yat_projection(key, num_features, head_dim)
    
    # Test with normalized vectors of varying dot products
    print("\n--- Dot product vs Feature map output ---")
    
    # Create a fixed query vector
    q = jax.random.normal(key, (head_dim,))
    q = q / jnp.linalg.norm(q)  # Unit norm
    
    # Create keys with varying angles to query
    for angle_factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
        # Create a key that has controlled dot product with q
        k_random = jax.random.normal(jax.random.split(key)[0], (head_dim,))
        k_random = k_random - jnp.dot(k_random, q) * q  # Orthogonal component
        k_random = k_random / jnp.linalg.norm(k_random)
        
        # Interpolate: k = cos(angle) * q + sin(angle) * k_orth
        angle = angle_factor * jnp.pi  # 0 to pi
        k = jnp.cos(angle) * q + jnp.sin(angle) * k_random
        k = k / jnp.linalg.norm(k)
        
        dot = float(jnp.dot(q, k))
        
        # Compute exact YAT kernel
        exact_yat = (dot ** 2) / (2 * (1 - dot) + 1e-5)
        
        # Compute feature map approximation
        q_r = q.reshape(1, 1, 1, head_dim)
        k_r = k.reshape(1, 1, 1, head_dim)
        q_feat = yat_performer_feature_map(q_r, projection, pre_normalized=True)
        k_feat = yat_performer_feature_map(k_r, projection, pre_normalized=True)
        approx = float(jnp.sum(q_feat * k_feat))
        
        print(f"  angle_factor={angle_factor:.2f}, dot={dot:+.4f}, exact_yat={exact_yat:.4f}, approx={approx:.4f}")


def test_normalizer_stability():
    """Test the normalizer computation in performer attention."""
    print("\n" + "="*70)
    print("TEST: Normalizer Stability in Performer Attention")
    print("="*70)
    
    key = jax.random.PRNGKey(42)
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    num_features = 32
    
    # Setup
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim))
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim))
    
    projection = create_yat_projection(k4, num_features, head_dim)
    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, 1024)
    
    # Manually trace through the computation to find the normalizer
    
    # Apply RoPE
    q_rot = apply_rotary_emb(q, freqs_cos, freqs_sin)
    k_rot = apply_rotary_emb(k, freqs_cos, freqs_sin)
    
    # Normalize
    q_norm, k_norm = normalize_qk(q_rot, k_rot, epsilon=1e-6)
    
    # Feature maps
    q_features = yat_performer_feature_map(q_norm, projection, pre_normalized=True)
    k_features = yat_performer_feature_map(k_norm, projection, pre_normalized=True)
    
    print_stats("Q features", q_features)
    print_stats("K features", k_features)
    
    # Compute normalizer
    k_sum = jnp.sum(k_features, axis=-3)  # Sum over sequence
    print_stats("K sum (over seq)", k_sum)
    
    normalizer = jnp.einsum("...qhm,...hm->...qh", q_features, k_sum)
    print_stats("Normalizer (before epsilon)", normalizer)
    
    normalizer_with_eps = normalizer + 1e-5
    print_stats("Normalizer (after epsilon)", normalizer_with_eps)
    
    # Check if normalizer can be very small
    min_normalizer = float(jnp.min(normalizer))
    print(f"\nMin normalizer value: {min_normalizer:.6g}")
    print(f"This could cause issues if close to or below epsilon ({1e-5:.0e})")


def test_long_sequence_scaling():
    """Test how output scales with sequence length."""
    print("\n" + "="*70)
    print("TEST: Long Sequence Scaling")
    print("="*70)
    
    key = jax.random.PRNGKey(42)
    embed_dim = 128
    num_heads = 4
    batch_size = 1
    
    for seq_len in [16, 64, 256, 512, 1024]:
        rngs = nnx.Rngs(42)
        attn = RotaryYatAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=1024,
            use_performer=True,
            num_features=embed_dim // 2,
            performer_normalize=True,
            constant_alpha=True,
            rngs=rngs,
        )
        
        x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
        output = attn(x, deterministic=True)
        
        out_std = float(jnp.std(output))
        in_std = float(jnp.std(x))
        print(f"  seq_len={seq_len:4d}: output_std={out_std:.4f}, ratio={out_std/in_std:.4f}")


def main():
    print("="*70)
    print("TESTING ROTARY YAT PERFORMER STABILITY")
    print("="*70)
    
    test_intermediate_values()
    test_alpha_scaling_impact()
    test_gradient_magnitudes()
    test_feature_map_behavior()
    test_normalizer_stability()
    test_long_sequence_scaling()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
