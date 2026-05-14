"""Spherical-YAT-Performer attention for MLX.

Linearized spherical YAT attention via the anchor approach:

  K(q, k) = (q·k)² / (C − 2·q·k)
          = ∫₀^∞ e^{−sC} · (q·k)² · e^{2s·q·k} ds
          ≈ Σ_r w_r · (q·k)² · exp(2 s_r · q·k)

Each term factorizes as ``φ_poly(q)·φ_poly(k) · φ_PRF(q;s_r)·φ_PRF(k;s_r)``,
giving a kernel feature map ``φ(x)`` that lets attention run in
``O(seq_len)`` via the standard reordering trick (``Σ_k φ(k) v_kᵀ``
then ``φ(q) · …``).

Mirrors ``nmn.nnx.layers.attention.spherical_yat_performer`` (without
the docstring's experimental approximation-quality table — the
math is unchanged).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import numpy as np


__all__ = [
    "create_yat_tp_projection",
    "yat_tp_features",
    "yat_tp_attention",
]


def create_yat_tp_projection(
    head_dim: int,
    num_prf_features: int = 8,
    num_quad_nodes: int = 1,
    num_anchor_features: int = 16,
    epsilon: float = 1e-5,
    dtype: mx.Dtype = mx.float32,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Precompute the parameters for anchor-based YAT attention.

    Args:
        head_dim: Per-head dimension.
        num_prf_features: ``M`` — PRF features per quadrature node.
        num_quad_nodes: ``R`` — Gauss-Laguerre quadrature nodes.
        num_anchor_features: ``P`` — anchor features for ``(q·k)²``.
        epsilon: Stability constant (``C = 2 + epsilon``).
        dtype: MLX dtype for the precomputed arrays.
        seed: Optional integer seed for deterministic projections.

    Returns:
        Dict with ``projections``, ``anchors``, ``quad_nodes``,
        ``quad_weights``, plus the integers ``head_dim``,
        ``num_prf_features``, ``num_anchor_features``, ``num_scales``.
    """
    rng = np.random.default_rng(seed)

    C = 2.0 + epsilon

    # Gauss-Laguerre nodes/weights — change of variables t = C·s.
    nodes_np, weights_np = np.polynomial.laguerre.laggauss(num_quad_nodes)
    quad_nodes = mx.array((nodes_np / C).astype(np.float32)).astype(dtype)
    quad_weights = mx.array((weights_np / C).astype(np.float32)).astype(dtype)

    # Anchor vectors on the unit sphere.
    anchors_np = rng.standard_normal((num_anchor_features, head_dim)).astype(np.float32)
    anchors_np = anchors_np / (
        np.sqrt((anchors_np ** 2).sum(axis=-1, keepdims=True)) + 1e-8
    )
    anchors = mx.array(anchors_np).astype(dtype)

    # PRF projections: one ``(M, head_dim)`` matrix per quadrature node.
    projections_np = rng.standard_normal(
        (num_quad_nodes, num_prf_features, head_dim)
    ).astype(np.float32)
    projections = mx.array(projections_np).astype(dtype)

    return {
        "projections": projections,        # (R, M, d)
        "anchors": anchors,                # (P, d)
        "quad_nodes": quad_nodes,          # (R,)
        "quad_weights": quad_weights,      # (R,)
        "head_dim": head_dim,
        "num_prf_features": num_prf_features,
        "num_anchor_features": num_anchor_features,
        "num_scales": num_quad_nodes,
    }


def yat_tp_features(
    x: mx.array,
    params: dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> mx.array:
    """Compute ``φ(x) ∈ ℝ^(R·P·M)`` for each ``(seq, head)`` token.

    Args:
        x: ``(..., seq_len, num_heads, head_dim)``.
        params: Output of :func:`create_yat_tp_projection`.
        normalize: L2-normalize inputs first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., seq_len, num_heads, R·P·M)``.
    """
    if normalize:
        x_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + epsilon)
        x = x / x_norm

    projections = params["projections"]       # (R, M, d)
    anchors = params["anchors"]               # (P, d)
    quad_nodes = params["quad_nodes"]         # (R,)
    quad_weights = params["quad_weights"]     # (R,)
    M = params["num_prf_features"]
    P = params["num_anchor_features"]
    R = params["num_scales"]

    # Polynomial features: (a·x)² / √P  →  (..., P)
    dots = mx.einsum("...d,pd->...p", x, anchors)
    poly_feat = (dots * dots) / math.sqrt(P)

    # PRF projections for every node at once: (..., R, M)
    proj_all = mx.einsum("...d,rmd->...rm", x, projections)

    sqrt_2s = mx.sqrt(mx.maximum(quad_nodes * 2.0, 0.0)).astype(x.dtype)
    s_vals = quad_nodes.astype(x.dtype)
    sq_w = mx.sqrt(mx.maximum(quad_weights, 0.0)).astype(x.dtype)

    sqrt_2s_b = sqrt_2s.reshape((R, 1))
    s_b = s_vals.reshape((R, 1))

    exp_arg = mx.clip(proj_all * sqrt_2s_b - s_b, -20.0, 20.0)
    prf_all = mx.exp(exp_arg) / math.sqrt(M)  # (..., R, M)

    # Outer product φ_poly(x) ⊗ φ_PRF(x; s_r) per quadrature node, with
    # √w_r applied. R is small (1-2) so a Python loop is fine.
    leading_shape = poly_feat.shape[:-1]
    fused_per_r = []
    for r in range(R):
        prf_r = prf_all[..., r, :]                       # (..., M)
        outer = poly_feat[..., :, None] * prf_r[..., None, :]   # (..., P, M)
        fused = mx.reshape(outer, leading_shape + (P * M,)) * sq_w[r]
        fused_per_r.append(fused)

    # Stack on a new last axis then flatten the (R, P*M) tail → (..., R*P*M).
    stacked = mx.stack(fused_per_r, axis=-2)  # (..., R, P*M)
    return mx.reshape(stacked, leading_shape + (R * P * M,))


def yat_tp_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    params: dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> mx.array:
    """Linear-complexity YAT attention via tensor-product features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, v_dim)``.
        params: Output of :func:`create_yat_tp_projection`.
        causal: If ``True``, use prefix sums so each query only sees keys
            up to its own position.
        epsilon: Numerical stability constant.

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    if query.dtype != value.dtype:
        value = value.astype(query.dtype)
    if key.dtype != query.dtype:
        key = key.astype(query.dtype)

    q_feat = yat_tp_features(query, params, normalize=True, epsilon=epsilon) + epsilon
    k_feat = yat_tp_features(key, params, normalize=True, epsilon=epsilon) + epsilon

    if causal:
        # kv[b, k, h, m, d] = k_feat[b, k, h, m] * value[b, k, h, d]
        kv = mx.einsum("bkhm,bkhd->bkhmd", k_feat, value)
        kv_cum = mx.cumsum(kv, axis=1)
        k_cum = mx.cumsum(k_feat, axis=1)

        num = mx.einsum("bqhm,bqhmd->bqhd", q_feat, kv_cum)
        den = mx.einsum("bqhm,bqhm->bqh", q_feat, k_cum)
        den = den[..., None] + epsilon
        return num / den

    # Non-causal: O(n) via the associativity trick.
    kv = mx.einsum("bkhm,bkhd->bhmd", k_feat, value)
    num = mx.einsum("bqhm,bhmd->bqhd", q_feat, kv)

    k_sum = mx.sum(k_feat, axis=1)
    den = mx.einsum("bqhm,bhm->bqh", q_feat, k_sum)
    den = den[..., None] + epsilon
    return num / den
