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
    poly_sketch_dim: int = 64,   # Output dimension for sketch
    dtype: Dtype = jnp.float32,
) -> dict:
    """Create parameters for Multi-Scale YAT attention."""
    
    # 1. TensorSketch Parameters (Polynomial Kernel)
    key, sketch_key = random.split(key)
    sketch_params = create_tensorsketch_params(
        sketch_key, head_dim, poly_sketch_dim, dtype
    )
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
        'sketch_params': sketch_params,
        'poly_sketch_dim': poly_sketch_dim,
    }

def create_tensorsketch_params(key, input_dim, sketch_dim, dtype=jnp.float32):
    """Create random indices and signs for CountSketch."""
    key_h, key_s = random.split(key)
    # h: indices in [0, sketch_dim)
    h = random.randint(key_h, (input_dim,), 0, sketch_dim)
    # s: signs {-1, 1}
    s = random.choice(key_s, jnp.array([-1.0, 1.0], dtype=dtype), (input_dim,))
    return {'h': h, 's': s}


def apply_tensorsketch(x, params, sketch_dim):
    """Compute TensorSketch for polynomial degree 2.
    
    Approximates <x, y>^2.
    1. CountSketch(x) -> C(x)
    2. Convolution via FFT: IFFT(FFT(C(x))^2)
    """
    # x: [... d]
    h = params['h'] # [d]
    s = params['s'] # [d]
    
    # We need to scatter add x[..., i] * s[i] into index h[i].
    # Use jax.vmap or straightforward scatter if x is single vector.
    # Handling batches:
    # Use einsum-like scatter is tricky. 
    # Use matrix multiplication for scatter?
    # Index matrix P of shape [d, sketch_dim].
    # P[i, j] = s[i] if h[i] == j else 0.
    # C(x) = x @ P.
    # This is efficient for moderate dimensions.
    
    d = h.shape[0]
    # Construct projection matrix P dynamically or store it?
    # Constructing on fly is cheap for small d.
    # Actually P is sparse. But d=64, sketch=64 is small. Dense is fine.
    
    indices = jnp.stack([jnp.arange(d), h], axis=1) # [d, 2]
    # We need P to be [d, sketch_dim]
    # jax.ops.index_update is deprecated. Use .at
    P = jnp.zeros((d, sketch_dim), dtype=x.dtype)
    P = P.at[jnp.arange(d), h].set(s)
    
    Cx = jnp.dot(x, P) # [... sketch_dim]
    
    # Convolution via FFT
    # FFT(C(x))
    fft_cx = jnp.fft.fft(Cx, axis=-1)
    # Square in freq domain (convolution)
    fft_cx_sq = fft_cx ** 2
    # IFFT
    Mx = jnp.fft.ifft(fft_cx_sq, axis=-1)
    
    # Result is real (theoretically), but FFT gives complex.
    return Mx.real


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
    
    # --- 1. Polynomial Features (TensorSketch) ---
    sketch_params = params['sketch_params']
    poly_sketch_dim = params['poly_sketch_dim']
    poly_feat = apply_tensorsketch(x, sketch_params, poly_sketch_dim) # [... D_p]
    
    # --- 2. Exponential Features (FAVOR+) ---
    # ... existing processing ...
    all_exp_features = []
    
    for r in range(R):
        s_r = scales[r]
        W_r = projections[r]  # [m, d]
        
        # Project: wx = Wx / sqrt(d)
        wx = jnp.einsum("...d,md->...m", x, W_r) / jnp.sqrt(d).astype(x.dtype)
        wx = wx * jnp.sqrt(m).astype(x.dtype)
        
        # Feature: exp(sqrt(s_r) * wx - s_r/2)
        arg = jnp.sqrt(s_r).astype(x.dtype) * wx - (s_r / 2.0)
        feat_r = jnp.exp(arg)
        all_exp_features.append(feat_r)
    
    # Concatenate all scales: [... M_total]
    exp_features = jnp.concatenate(all_exp_features, axis=-1)
    
    # Normalize Exponential part
    exp_features = exp_features / jnp.sqrt(m * R).astype(x.dtype)

    # --- 3. Fusion (Tensor Product) ---
    # Psi = Poly \otimes Exp
    # Flattened dimension: D_p * (M * R)
    # Use einsum outer product
    
    # poly: [... D_p]
    # exp:  [... M_total]
    # out:  [... D_p * M_total]
    
    # To save reshaping, calculate explicitly
    # [... D_p, 1] * [... 1, M_total] -> [... D_p, M_total] -> Flatten
    
    features = jnp.einsum('...p,...e->...pe', poly_feat, exp_features)
    features_flat = features.reshape(features.shape[:-2] + (-1,))
    
    return features_flat


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
        # Use scan to avoid O(L*M*D) memory which is massive (e.g. 375GB for L=1024)
        # We process sequence step by step updating the running KV sum.
        
        def scan_body(carry, inputs):
            numerator_state, denominator_state = carry
            q_t, k_t, v_t = inputs
            # q_t, k_t: [... H, M]
            # v_t: [... H, D]
             
            # Update states
            # numerator_state: [... H, M, D]
            # denominator_state: [... H, M]
             
            # k_t[..., None] * v_t[..., None, :] -> [... H, M, D]
            kv_update = jnp.einsum('...hm,...hd->...hmd', k_t, v_t, precision=precision)
            new_num_state = numerator_state + kv_update
            new_den_state = denominator_state + k_t
             
            # Compute output
            # num: [... H, M] @ [... H, M, D] -> [... H, D]
            num = jnp.einsum('...hm,...hmd->...hd', q_t, new_num_state, precision=precision)
            den = jnp.einsum('...hm,...hm->...h', q_t, new_den_state, precision=precision)
             
            out = num / (den[..., None] + epsilon)
             
            return (new_num_state, new_den_state), out

        # Prepare for scan: move Sequence dimension (axis -3) to front (axis 0)
        # q_feat: [Batch, Seq, Heads, M] -> [Seq, Batch, Heads, M]
        # We need to be careful with arbitrary batch dimensions. 
        # jnp.moveaxis is safe.
        q_T = jnp.moveaxis(q_feat, -3, 0)
        k_T = jnp.moveaxis(k_feat, -3, 0)
        v_T = jnp.moveaxis(value, -3, 0)
        
        # Initial state
        # Shape should match the inner dimensions (excluding Seq)
        # [... H, M, D] and [... H, M]
        batch_shape = q_T.shape[1:-2]
        H = q_T.shape[-2]
        M = q_T.shape[-1]
        D = v_T.shape[-1]
        
        init_num = jnp.zeros(batch_shape + (H, M, D), dtype=query.dtype)
        init_den = jnp.zeros(batch_shape + (H, M), dtype=query.dtype)
        
        _, output_T = jax.lax.scan(
            scan_body,
            (init_num, init_den),
            (q_T, k_T, v_T)
        )
        
        # Move Seq back: [Seq, Batch, Heads, D] -> [Batch, Seq, Heads, D]
        return jnp.moveaxis(output_T, 0, -3)
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
