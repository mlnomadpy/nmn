"""Spherical Yat-Performer Attention.

Implements a Multi-Scale FAVOR+ approximation for the YAT kernel.

The YAT kernel K(x,y) = (x·y)² / (C - 2x·y) has a singularity at x·y = 1.
To approximate this "sharp" kernel using random features, we use a
weighted sum of exponentials at different scales:

    K(x,y) ≈ Σ_r w_r exp(s_r · (x·y - 1))

This is implemented using FAVOR+ features for each scale s_r, effectively
providing the model with a "multi-resolution" view of vector similarity,
capturing both long-range diffuse attention and short-range sharp attention.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import random
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array


def create_orthogonal_features(key, num_features, dim, dtype=jnp.float32):
    """Create orthogonal random features scaled by sqrt(dim)."""
    num_blocks = (num_features + dim - 1) // dim
    blocks = []
    for i in range(num_blocks):
        key, subkey = random.split(key)
        random_matrix = random.normal(subkey, (dim, dim), dtype=dtype)
        q, _ = jnp.linalg.qr(random_matrix)
        blocks.append(q)
    projection = jnp.concatenate(blocks, axis=0)[:num_features]
    return projection * jnp.sqrt(dim).astype(dtype)


def create_yat_tp_projection(
    key: Array,
    head_dim: int,
    num_prf_features: int = 64,  # features per scale
    num_quad_nodes: int = 8,     # number of scales
    epsilon: float = 1e-5,
    dtype: Dtype = jnp.float32,
) -> dict:
    """Create parameters for Multi-Scale YAT attention."""
    
    # Use geometrically spaced scales to capture the singularity
    # Scales correspond to 'sharpness' of attention
    # Range 0.1 to 100 covers diffusion to very sharp
    scales = jnp.logspace(-1, 2, num_quad_nodes, dtype=dtype)
    
    # Create one projection matrix per scale
    projections = []
    for r in range(num_quad_nodes):
        key, subkey = random.split(key)
        proj = create_orthogonal_features(subkey, num_prf_features, head_dim, dtype)
        projections.append(proj)
        
    projections = jnp.stack(projections)  # [R, m, d]
    
    return {
        'projections': projections,  # [R, m, d]
        'scales': scales,           # [R]
        'head_dim': head_dim,
        'num_prf_features': num_prf_features,
        'num_scales': num_quad_nodes,
    }


def yat_tp_features(
    x: Array,
    params: dict,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Compute Multi-Scale FAVOR+ features.
    
    For each scale s_r:
        φ_r(x) = exp(sqrt(s_r) W_r x - s_r/2) / sqrt(m)
        
    The dot product will approximate exp(s_r x·y).
    Summing across r approximates sum(exp(s_r x·y)).
    """
    if normalize:
        x_norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + epsilon)
        x = x / x_norm
    
    projections = params['projections']  # [R, m, d]
    scales = params['scales']           # [R]
    d = params['head_dim']
    m = params['num_prf_features']
    R = params['num_scales']
    
    all_features = []
    
    for r in range(R):
        s_r = scales[r]
        W_r = projections[r]  # [m, d]
        
        # Project: wx = Wx / sqrt(d)
        # Result has variance d/m * 1/d = 1/m (if W has rows of norm sqrt(d))
        # We need unit variance for FAVOR+ approximation
        wx = jnp.einsum("...d,md->...m", x, W_r) / jnp.sqrt(d).astype(x.dtype)
        wx = wx * jnp.sqrt(m).astype(x.dtype)
        
        # FAVOR+ for exp(s_r * x·y)
        # Feature: exp(sqrt(s_r) * wx - s_r/2)
        # E[feat(x) feat(y)] = exp(s_r * x·y)
        
        # We want sharpness `s_r`.
        # Standard FAVOR+ approximates exp(q.k).
        # We want exp(s_r * q.k).
        # So we scale wx by sqrt(s_r).
        # Offset is s_r/2 because E[exp(A z)] = exp(A^2/2) for z~N(0,1).
        # Here z = wx. A = sqrt(s_r). A^2/2 = s_r/2.
        
        arg = jnp.sqrt(s_r).astype(x.dtype) * wx - (s_r / 2.0)
        feat_r = jnp.exp(arg)
        
        # Optional: Add linear features for better low-frequency capture?
        # Mixing linear and exp is generally robust.
        # Let's keep just exp for clean singularity approximation.
        
        all_features.append(feat_r)
    
    # Concatenate all scales
    features = jnp.concatenate(all_features, axis=-1)
    
    # Normalize
    # We sum R scales, each with m features.
    # Total variance per scale is proportional to exp(0) = 1.
    # We want average magnitude ~ 1?
    # Let's normalize by sqrt(m * R) to keep scale reasonable
    features = features / jnp.sqrt(m * R).astype(x.dtype)
    
    return features


def yat_tp_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict,
    causal: bool = False,
    epsilon: float = 1e-5,
    precision: PrecisionLike = None,
    gradient_scaling: bool = True,
) -> Array:
    """Compute YAT attention using Multi-Scale features."""
    query, key, value = promote_dtype((query, key, value), dtype=None)
    seq_len = query.shape[-3]
    
    q_feat = yat_tp_features(query, params, normalize=True, epsilon=epsilon)
    k_feat = yat_tp_features(key, params, normalize=True, epsilon=epsilon)
    
    # Features are exp() so alreayd positive.
    # Add epsilon for safety.
    q_feat = q_feat + epsilon
    k_feat = k_feat + epsilon
    
    if causal:
        kv = jnp.einsum("...khm,...khd->...khmd", k_feat, value, precision=precision)
        kv_cum = jnp.cumsum(kv, axis=-4)
        k_cum = jnp.cumsum(k_feat, axis=-3)
        
        num = jnp.einsum("...qhm,...qhmd->...qhd", q_feat, kv_cum, precision=precision)
        den = jnp.einsum("...qhm,...qhm->...qh", q_feat, k_cum, precision=precision)
        den = den[..., None] + epsilon
        return num / den
    else:
        # Non-causal: O(L) via reordering
        kv = jnp.einsum("...khm,...khd->...hmd", k_feat, value, precision=precision)
        num = jnp.einsum("...qhm,...hmd->...qhd", q_feat, kv, precision=precision)
        
        k_sum = jnp.sum(k_feat, axis=-3)
        den = jnp.einsum("...qhm,...hm->...qh", q_feat, k_sum, precision=precision)
        den = den[..., None] + epsilon
        return num / den


def test_yat_tp_approximation():
    """Test YAT kernel approximation quality."""
    print("="*60)
    print("MULTI-SCALE KERNEL APPROXIMATION")
    print("="*60)
    
    key = random.PRNGKey(42)
    head_dim = 64
    
    params = create_yat_tp_projection(
        key, head_dim, num_prf_features=64, num_quad_nodes=8
    )
    
    R = params['num_scales']
    
    # Create unit vector q
    q = random.normal(key, (head_dim,))
    q = q / jnp.linalg.norm(q)
    
    print(f"\n{'dot':>6} | {'exact_YAT':>10} | {'approx':>10} | {'ratio':>8}")
    print("-" * 42)
    
    for target_dot in [-0.8, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        k_orth = random.normal(random.split(key)[0], (head_dim,))
        k_orth = k_orth - jnp.dot(k_orth, q) * q
        k_orth = k_orth / jnp.linalg.norm(k_orth)
        
        k = target_dot * q + jnp.sqrt(max(0, 1 - target_dot**2)) * k_orth
        k = k / jnp.linalg.norm(k)
        
        dot = float(jnp.dot(q, k))
        # Exact YAT
        C = 2.00001
        exact = (dot ** 2) / (C - 2 * dot)
        
        # Approx
        q_r = q.reshape(1, 1, 1, head_dim)
        k_r = k.reshape(1, 1, 1, head_dim)
        
        q_feat = yat_tp_features(q_r, params, normalize=True)
        k_feat = yat_tp_features(k_r, params, normalize=True)
        
        approx = float(jnp.sum(q_feat * k_feat))
        
        # Scale approx to match exact at 0.9?
        # We just want to see the SHAPE.
        ratio = approx / exact if abs(exact) > 1e-4 else 0
        
        print(f"{dot:+6.2f} | {exact:10.4f} | {approx:10.4f} | {ratio:7.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_yat_tp_approximation()
