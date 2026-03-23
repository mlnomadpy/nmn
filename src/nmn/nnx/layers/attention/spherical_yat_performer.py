"""Spherical Yat-Performer Attention (Anchor Approximation).

Implements linearized spherical YAT attention using the anchor approach from
the SLAY paper:

1. Bernstein integral representation to decompose the YAT kernel
2. Anchor features for the polynomial kernel (q·k)²
3. Positive Random Features (PRF) for the exponential kernel exp(2s·q·k)
4. Gauss-Laguerre quadrature to discretize the integral
5. Tensor product of polynomial × exponential features

The YAT kernel K(q,k) = (q·k)² / (C - 2·q·k) is decomposed via:

    (q·k)² / (C - 2·q·k) = ∫₀^∞ e^{-sC} · (q·k)² · e^{2s·q·k} ds

Discretized with Gauss-Laguerre quadrature:

    ≈ Σ_r w_r · (q·k)² · exp(2s_r · q·k)

Each term uses a tensor product of two feature maps:
- Anchor features φ_poly(x) = [(a₁·x)², ..., (aₚ·x)²] / √P  for (q·k)²
- PRF features φ_PRF(x; s) = (1/√M) exp(√(2s)·ω^T·x - s)  for exp(2s·q·k)

Final feature per quadrature node:
    φ_r(x) = √w_r · (φ_poly(x) ⊗ φ_PRF(x; s_r))

where ⊗ is the tensor (outer) product, yielding P*M features per node,
and R*P*M total features.
"""

from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from jax import random, lax
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
    num_prf_features: int = 4,
    num_quad_nodes: int = 2,
    num_anchor_features: int = 32,
    epsilon: float = 1e-5,
    dtype: Dtype = jnp.float32,
) -> dict:
    """Create parameters for anchor-based YAT attention.

    Uses Gauss-Laguerre quadrature to discretize the Bernstein integral,
    anchor features for the polynomial kernel, and orthogonal random
    projections for PRF features.

    Args:
        key: JAX random key.
        head_dim: Dimension of each attention head.
        num_prf_features: Number of PRF features M per quadrature node.
        num_quad_nodes: Number of Gauss-Laguerre quadrature nodes R.
        num_anchor_features: Number of anchor features P for polynomial kernel.
        epsilon: Stability parameter (C = 2 + epsilon).
        dtype: Data type.

    Returns:
        Dictionary with projection parameters.
    """
    C = 2.0 + epsilon

    # Gauss-Laguerre quadrature nodes and weights
    # After change of variables t = Cs: s_r = t_r/C, w_r = alpha_r/C
    import numpy as np
    nodes_np, weights_np = np.polynomial.laguerre.laggauss(num_quad_nodes)
    nodes = jnp.array(nodes_np / C, dtype=dtype)     # s_r = t_r / C
    weights = jnp.array(weights_np / C, dtype=dtype)  # w_r = alpha_r / C

    # Anchor vectors on unit sphere for polynomial kernel (q·k)²
    key, subkey = random.split(key)
    anchors = random.normal(subkey, (num_anchor_features, head_dim), dtype=dtype)
    anchors = anchors / jnp.sqrt(jnp.sum(anchors ** 2, axis=-1, keepdims=True) + 1e-8)

    # PRF random projections: one set of M features per quadrature node
    # Standard normal ω ~ N(0, I) matching slay implementation
    projections = []
    for r in range(num_quad_nodes):
        key, subkey = random.split(key)
        proj = random.normal(subkey, (num_prf_features, head_dim), dtype=dtype)
        projections.append(proj)
    projections = jnp.stack(projections)  # [R, M, d]

    return {
        'projections': projections,       # [R, M, d] - PRF projections
        'anchors': anchors,               # [P, d] - anchor vectors on unit sphere
        'quad_nodes': nodes,              # [R] - quadrature nodes (s_r)
        'quad_weights': weights,          # [R] - quadrature weights (w_r)
        'head_dim': head_dim,
        'num_prf_features': num_prf_features,
        'num_anchor_features': num_anchor_features,
        'num_scales': num_quad_nodes,
    }


def _anchor_poly_features(
    x: Array,
    anchors: Array,
    num_anchor_features: int,
) -> Array:
    """Compute anchor features for the polynomial kernel (q·k)².

    φ_anc(x) = [(a₁·x)², ..., (aₚ·x)²] / √P

    Key property: φ_anc(q)·φ_anc(k) ≥ 0 always (preserves non-negativity).

    Args:
        x: Input tensor [..., head_dim], L2-normalized.
        anchors: Anchor vectors [P, head_dim], unit norm.
        num_anchor_features: Number of anchors P.

    Returns:
        Anchor features [..., P].
    """
    dots = jnp.einsum("...d,pd->...p", x, anchors)
    return (dots ** 2) / math.sqrt(num_anchor_features)


def _prf_features(
    x: Array,
    projection: Array,
    s_r: Array,
    num_features: int,
) -> Array:
    """Compute Positive Random Features for exp(2s·q·k).

    φ_PRF(x; s) = (1/√M) exp(√(2s)·ω^T·x - s)

    satisfying E[φ_PRF(q; s)^T φ_PRF(k; s)] = exp(2s·q·k)

    Args:
        x: Input tensor [..., head_dim], L2-normalized.
        projection: Random projection matrix [M, head_dim], standard normal entries.
        s_r: Quadrature node value (scalar).
        num_features: Number of random features M.

    Returns:
        PRF features [..., M].
    """
    # Project: ω^T x  (ω ~ N(0, I), no extra scaling)
    proj = jnp.einsum("...d,md->...m", x, projection)

    # φ_PRF(x; s) = (1/√M) exp(√(2s) · ω^T x - s)
    sqrt_2s = jnp.sqrt(2.0 * jnp.maximum(s_r, 0.0)).astype(x.dtype)
    exp_arg = jnp.clip(proj * sqrt_2s - s_r.astype(x.dtype), -20.0, 20.0)

    return jnp.exp(exp_arg) / math.sqrt(num_features)


def yat_tp_features(
    x: Array,
    params: dict,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Compute anchor-based YAT features using tensor product.

    For each quadrature node r:
        φ_r(x) = √w_r · flatten(φ_poly(x) ⊗ φ_PRF(x; s_r))

    where ⊗ is the outer product (P features × M features = P*M features).
    All R nodes are concatenated: φ(x) has R*P*M total features.

    Uses lax.scan over quadrature nodes for efficient JIT compilation.

    Args:
        x: Input tensor [..., seq_len, num_heads, head_dim].
        params: Parameters from create_yat_tp_projection.
        normalize: If True, L2-normalize inputs first.
        epsilon: Numerical stability constant.

    Returns:
        Feature tensor [..., seq_len, num_heads, R*P*M].
    """
    if normalize:
        x_norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + epsilon)
        x = x / x_norm

    projections = params['projections']       # [R, M, d]
    anchors = params['anchors']               # [P, d]
    quad_nodes = params['quad_nodes']         # [R]
    quad_weights = params['quad_weights']     # [R]
    M = params['num_prf_features']
    P = params['num_anchor_features']

    # Polynomial features (shared across all quadrature nodes)
    poly_feat = _anchor_poly_features(x, anchors, P)  # [..., P]

    # Compute all PRF projections at once: x @ projections^T
    # x: [..., d], projections: [R, M, d] -> proj_all: [..., R, M]
    proj_all = jnp.einsum("...d,rmd->...rm", x, projections)

    # Precompute sqrt(2s) and s for broadcasting: [R, 1]
    sqrt_2s = jnp.sqrt(2.0 * jnp.maximum(quad_nodes, 0.0))[:, None].astype(x.dtype)
    s_vals = quad_nodes[:, None].astype(x.dtype)
    sq_weights = jnp.sqrt(jnp.maximum(quad_weights, 0.0))[:, None].astype(x.dtype)

    # PRF features for all nodes: [..., R, M]
    exp_arg = jnp.clip(proj_all * sqrt_2s - s_vals, -20.0, 20.0)
    prf_all = jnp.exp(exp_arg) / math.sqrt(M)  # [..., R, M]

    # Tensor product + weight via lax.scan over R quadrature nodes
    # scan_inputs: slice along the R axis of prf_all and sq_weights
    # We need to scan over axis -2 of prf_all which is R
    # Reshape for scan: move R to front
    # prf_all: [..., R, M] -> [R, ..., M]
    prf_scan = jnp.moveaxis(prf_all, -2, 0)        # [R, ..., M]
    sq_w_scan = sq_weights[:, 0]                     # [R]

    def scan_fn(carry, r_inputs):
        prf_feat, sq_w = r_inputs  # prf: [..., M], sq_w: scalar
        # Outer product: [..., P] x [..., M] -> [..., P, M]
        outer = poly_feat[..., :, None] * prf_feat[..., None, :]
        # Flatten and weight: [..., P*M]
        fused = outer.reshape(outer.shape[:-2] + (P * M,)) * sq_w
        return carry, fused

    _, fused_stack = lax.scan(scan_fn, None, (prf_scan, sq_w_scan))
    # fused_stack: [R, ..., P*M]

    # Move R to end and flatten: [..., R*P*M]
    R_actual = fused_stack.shape[0]
    fused_moved = jnp.moveaxis(fused_stack, 0, -1)  # [..., P*M, R]
    features = fused_moved.reshape(fused_moved.shape[:-2] + (R_actual * P * M,))

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
    """Compute YAT attention using anchor-based tensor product features.

    Uses the decomposition:
        K(q,k) = (q·k)² / (C - 2·q·k)
               ≈ Σ_r w_r · φ_poly(q)·φ_poly(k) · φ_PRF(q;s_r)·φ_PRF(k;s_r)
               = φ(q)^T φ(k)

    where φ is the concatenated tensor-product feature map, enabling
    O(n) linear attention via associativity reordering.

    Args:
        query: [..., q_length, num_heads, head_dim].
        key: [..., kv_length, num_heads, head_dim].
        value: [..., kv_length, num_heads, v_dim].
        params: Parameters from create_yat_tp_projection.
        causal: If True, use causal attention via prefix sums.
        epsilon: Numerical stability constant.
        precision: JAX precision.
        gradient_scaling: Unused, kept for API compatibility.

    Returns:
        Output [..., q_length, num_heads, v_dim].
    """
    query, key, value = promote_dtype((query, key, value), dtype=None)

    q_feat = yat_tp_features(query, params, normalize=True, epsilon=epsilon)
    k_feat = yat_tp_features(key, params, normalize=True, epsilon=epsilon)

    # Anchor features produce non-negative values (squares × exp),
    # add epsilon for numerical safety
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
        # Non-causal: O(n) via reordering
        kv = jnp.einsum("...khm,...khd->...hmd", k_feat, value, precision=precision)
        num = jnp.einsum("...qhm,...hmd->...qhd", q_feat, kv, precision=precision)

        k_sum = jnp.sum(k_feat, axis=-3)
        den = jnp.einsum("...qhm,...hm->...qh", q_feat, k_sum, precision=precision)
        den = den[..., None] + epsilon
        return num / den


def test_yat_tp_approximation():
    """Test YAT kernel approximation quality with anchor features."""
    print("=" * 60)
    print("ANCHOR TENSOR-PRODUCT KERNEL APPROXIMATION")
    print("=" * 60)

    key = random.PRNGKey(42)
    head_dim = 64
    P = 64   # anchor features
    M = 8    # PRF features per quadrature node
    R = 2    # quadrature nodes

    params = create_yat_tp_projection(
        key, head_dim,
        num_prf_features=M,
        num_quad_nodes=R,
        num_anchor_features=P,
    )

    print(f"Config: P={P} anchors, M={M} PRF, R={R} quad nodes")
    print(f"Total features per token: R*P*M = {R*P*M}")

    # Create unit vector q
    q = random.normal(key, (head_dim,))
    q = q / jnp.linalg.norm(q)

    C = 2.0 + 1e-5

    print(f"\n{'dot':>6} | {'exact_YAT':>10} | {'approx':>10} | {'ratio':>8} | {'scaled':>10}")
    print("-" * 60)

    # Collect all values first to compute scaling factor
    results = []
    for target_dot in [-0.8, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        k_orth = random.normal(random.split(key)[0], (head_dim,))
        k_orth = k_orth - jnp.dot(k_orth, q) * q
        k_orth = k_orth / jnp.linalg.norm(k_orth)

        k = target_dot * q + jnp.sqrt(max(0, 1 - target_dot ** 2)) * k_orth
        k = k / jnp.linalg.norm(k)

        dot = float(jnp.dot(q, k))
        exact = (dot ** 2) / (C - 2 * dot)

        # Approx via anchor tensor-product features
        q_r = q.reshape(1, 1, 1, head_dim)
        k_r = k.reshape(1, 1, 1, head_dim)

        q_feat = yat_tp_features(q_r, params, normalize=True)
        k_feat = yat_tp_features(k_r, params, normalize=True)

        approx = float(jnp.sum(q_feat * k_feat))
        results.append((dot, exact, approx))

    # Compute best-fit scale at dot=0.9 for shape comparison
    ref_exact = [r[1] for r in results if abs(r[0] - 0.9) < 0.05]
    ref_approx = [r[2] for r in results if abs(r[0] - 0.9) < 0.05]
    scale = ref_exact[0] / ref_approx[0] if ref_approx[0] > 1e-10 else 1.0

    for dot, exact, approx in results:
        ratio = approx / exact if abs(exact) > 1e-4 else 0
        scaled = approx * scale
        print(f"{dot:+6.2f} | {exact:10.4f} | {approx:10.4f} | {ratio:7.2f} | {scaled:10.4f}")

    print(f"\n(Scale factor: {scale:.2f} — applied to 'scaled' column to match at dot=0.9)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_yat_tp_approximation()
