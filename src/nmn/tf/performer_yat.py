"""MAY (Random Maclaurin) and RAY (radial) linear-attention feature maps for
spherical YAT attention in TensorFlow.

Spherical YAT kernel on unit vectors ``s = q̂·k̂`` (``||q̂|| = ||k̂|| = 1``)::

    kappa(s) = (s + b)^2 / (C - 2 s),   C = 2 + eps

where ``b`` is the per-head bias (deployment regime ``b > 0``) and ``eps`` is the
YAT epsilon. The exact reference for the underlying quadratic attention is
:func:`nmn.tf.attention.yat_attention_normalized`.

This module provides two **bias-aware** feature maps that linearize the kernel,
both reusing a generic linear-attention readout
(``num = phi_q · (phi_kᵀ v)``, ``den = phi_q · sum_k phi_k``):

MAY -- Random Maclaurin features (the headline method)
    Expands ``kappa`` in a Maclaurin series on the sphere. With the geometric
    coefficients ``beta_n = (1/C)(2/C)^n`` of ``1/(C - 2 s)``, the kernel
    coefficients are

        a_n = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n   (beta_{-1}=beta_{-2}=0)

    which are all ``>= 0`` for ``b >= 0``. Sampling degrees ``N_r ~ a_n / Z``
    (``Z = sum_n a_n``) and forming products of ``N_r`` Rademacher projections
    yields an **unbiased** estimator: ``E[phi(q̂)·phi(k̂)] = kappa(q̂·k̂)``.
    Degree-0 draws produce a constant feature, which is exactly what lets MAY
    carry the bias and attend diffusely -- something the bias-free SLAY anchor
    method (which only approximates the ``b = 0`` kernel) cannot do.

RAY -- sketched degree-2 modulation × radial RFF (secondary method)
    Augments ``z = [x̂, sqrt(b)]`` so ``z_q·z_k = s + b`` and
    ``||z_q - z_k||^2 = ||q̂ - k̂||^2``. Then
    ``kappa = (z_q·z_k)^2 · 1/(eps + ||z_q - z_k||^2)`` -- a product of two PSD
    kernels approximated by a degree-2 polynomial sketch and a radial random
    Fourier feature (Gauss-Laguerre over the exponential-integral
    representation of ``1/(eps + r^2)``). RAY is the weaker method (the sharp-eps
    regime is hard); expect it to be comparable-ish to SLAY and to underperform
    MAY.

**Caveat (sign-indefinite features).** Both MAY and RAY features are
sign-indefinite -- odd-degree Rademacher products can be negative -- so the
linear-attention denominator can in principle go non-positive. Do NOT add an
epsilon to the *features* (that biases the estimator); a small epsilon is added
only to the *denominator* for division stability. Empirically well-conditioned
for ``b > 0``.

The random projections are drawn once with NumPy (seeded for determinism) and
stored as TensorFlow constants, so the feature maps themselves are pure TF ops.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf


__all__ = [
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _normalize(x: tf.Tensor, epsilon: float = 1e-6) -> tf.Tensor:
    """L2-normalizes the last axis (the spherical assumption)."""
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + epsilon)
    return x / norm


def maclaurin_coeffs(b: float, eps: float, nmax: int) -> np.ndarray:
    """Maclaurin coefficients of ``kappa`` on the sphere.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` (the coefficients of ``1/(C - 2 s)``), all
    ``>= 0`` for ``b >= 0``.

    Args:
        b: Per-head bias.
        eps: YAT epsilon (``C = 2 + eps``).
        nmax: Truncation degree.

    Returns:
        NumPy array ``a`` of shape ``[nmax + 1]``.
    """
    C = 2.0 + eps
    n = np.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = b * b * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a  # [nmax + 1], all >= 0 for b >= 0


# --------------------------------------------------------------------------
# MAY -- Random Maclaurin features
# --------------------------------------------------------------------------
def create_maclaurin_projection(
    head_dim: int,
    num_features: int = 256,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    nmax: int = 40,
    dtype: tf.DType = tf.float32,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Precompute the fixed projection for MAY (Random Maclaurin) features.

    A degree ``N_r ~ Categorical(a_n / Z)`` is sampled for each of the ``M``
    output features, and ``nmax`` Rademacher vectors are drawn per feature (only
    the first ``N_r`` are used). The projection is shared by queries and keys.

    Args:
        head_dim: Per-head input dimension ``d``.
        num_features: ``M`` -- the number of output features.
        bias: Per-head bias (deployment regime ``bias > 0``).
        epsilon: YAT epsilon (``C = 2 + epsilon``).
        nmax: Maclaurin truncation degree (default 40).
        dtype: TF dtype for the stored tensors.
        seed: Optional integer seed for deterministic projections.

    Returns:
        Dict with ``omegas`` ``[M, nmax, d]`` (Rademacher), ``degrees`` ``[M]``
        (int32), the scalar ``Z``, and the integers ``num_features`` / ``nmax``.
    """
    rng = np.random.default_rng(seed)
    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = float(a.sum())
    p = a / a.sum()

    degrees = rng.choice(len(a), size=num_features, p=p).astype(np.int32)  # [M]
    omegas = rng.choice([-1.0, 1.0], size=(num_features, nmax, head_dim))   # [M, nmax, d]

    return {
        "omegas": tf.constant(omegas, dtype=dtype),       # [M, nmax, d]
        "degrees": tf.constant(degrees, dtype=tf.int32),  # [M]
        "Z": Z,
        "num_features": num_features,
        "nmax": nmax,
    }


def maclaurin_features(
    x: tf.Tensor,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> tf.Tensor:
    """Compute MAY features ``phi(x) ∈ ℝ^M`` for each token.

    ``phi_r(x̂) = sqrt(Z / M) * prod_{l < N_r} (omega[r, l] · x̂)``. Unused
    factors (``l >= N_r``) are neutralized to 1.0 so degree-0 draws give the
    constant feature.

    Args:
        x: ``(..., d)`` (typically ``(B, L, H, d)``).
        params: Output of :func:`create_maclaurin_projection`.
        normalize: L2-normalize ``x`` first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., M)``.
    """
    if normalize:
        x = _normalize(x, epsilon)

    omegas = params["omegas"]          # [M, nmax, d]
    degrees = params["degrees"]        # [M]
    M = params["num_features"]
    nmax = params["nmax"]
    Z = params["Z"]

    x = tf.cast(x, omegas.dtype)

    # proj[..., m, l] = sum_d x[..., d] * omegas[m, l, d]  -> (..., M, nmax)
    proj = tf.einsum("...d,mld->...ml", x, omegas)

    # mask[m, l] = l < N_m  -> (M, nmax); neutralize unused factors to 1.0
    l = tf.range(nmax, dtype=tf.int32)                       # [nmax]
    # Cast degrees to int32 too so the comparison never mixes int32/int64.
    mask = tf.expand_dims(l, 0) < tf.cast(tf.expand_dims(degrees, 1), tf.int32)  # [M, nmax]
    mask = tf.reshape(mask, [1] * (len(proj.shape) - 2) + [M, nmax])
    proj = tf.where(mask, proj, tf.ones_like(proj))

    feat = tf.reduce_prod(proj, axis=-1)                     # (..., M)
    scale = tf.cast(math.sqrt(Z / float(M)), feat.dtype)
    return scale * feat


def maclaurin_yat_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> tf.Tensor:
    """Linear-complexity spherical-YAT attention via MAY features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, v_dim)``.
        params: Output of :func:`create_maclaurin_projection`.
        causal: If ``True``, use prefix sums (each query sees keys up to its
            own position).
        epsilon: Denominator stability constant (added to the *denominator*
            only -- never to the features).

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    q_feat = maclaurin_features(query, params, normalize=True, epsilon=epsilon)
    k_feat = maclaurin_features(key, params, normalize=True, epsilon=epsilon)
    return _linear_attention(q_feat, k_feat, value, causal=causal, epsilon=epsilon)


# --------------------------------------------------------------------------
# RAY -- sketched degree-2 modulation × radial RFF
# --------------------------------------------------------------------------
def create_radial_projection(
    head_dim: int,
    sketch_m: int = 128,
    num_radial: int = 8,
    radial_dim: int = 64,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    dtype: tf.DType = tf.float32,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Precompute the fixed projection for RAY (radial) features.

    Two random projections ``W1, W2`` sketch the degree-2 modulation
    ``(z_q·z_k)^2``; a Gauss-Laguerre radial RFF approximates
    ``1/(eps + ||z_q - z_k||^2)`` over the exponential-integral representation.

    Args:
        head_dim: Per-head input dimension ``d`` (augmented to ``d + 1``).
        sketch_m: ``m`` -- degree-2 sketch width.
        num_radial: Number of Gauss-Laguerre nodes.
        radial_dim: RFF features per node.
        bias: Per-head bias (used to augment ``z = [x̂, sqrt(bias)]``).
        epsilon: YAT epsilon.
        dtype: TF dtype for the stored tensors.
        seed: Optional integer seed.

    Returns:
        Dict with ``W1``/``W2`` ``[m, d+1]``, ``omegas`` ``[num_radial, radial_dim, d+1]``,
        ``phases`` ``[num_radial, radial_dim]``, ``coef`` ``[num_radial]``, plus
        scalars/integers ``b``, ``sketch_m``, ``radial_dim``, ``num_radial``.
    """
    rng = np.random.default_rng(seed)
    da = head_dim + 1  # augmented z = [x̂, sqrt(bias)]

    # degree-2 sketch: two independent projections -> approximates (z·z')^2
    W1 = rng.standard_normal((sketch_m, da))
    W2 = rng.standard_normal((sketch_m, da))

    # radial RFF for 1/(eps + ||z - z'||^2) via Gauss-Laguerre over the
    # exponential-integral representation 1/(eps + r^2) = ∫ e^{-t(eps + r^2)} dt.
    tau, wq = np.polynomial.laguerre.laggauss(num_radial)
    t_nodes = tau / epsilon              # t_j = tau_j / epsilon
    coef = (1.0 / epsilon) * wq          # (1/epsilon) w_j

    omegas, phases = [], []
    for tj in t_nodes:
        omegas.append(rng.standard_normal((radial_dim, da)) * np.sqrt(2.0 * tj))
        phases.append(rng.uniform(0, 2 * np.pi, radial_dim))

    return {
        "W1": tf.constant(W1, dtype=dtype),                 # [m, d+1]
        "W2": tf.constant(W2, dtype=dtype),                 # [m, d+1]
        "omegas": tf.constant(np.stack(omegas), dtype=dtype),   # [num_radial, radial_dim, d+1]
        "phases": tf.constant(np.stack(phases), dtype=dtype),   # [num_radial, radial_dim]
        "coef": tf.constant(coef, dtype=dtype),             # [num_radial]
        "b": float(bias),
        "sketch_m": sketch_m,
        "radial_dim": radial_dim,
        "num_radial": num_radial,
    }


def radial_features(
    x: tf.Tensor,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> tf.Tensor:
    """Compute RAY features for each token: ``psi(z) ⊗ phi_rad(z)``.

    Args:
        x: ``(..., d)``.
        params: Output of :func:`create_radial_projection`.
        normalize: L2-normalize ``x`` first.
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., sketch_m * num_radial * radial_dim)``.
    """
    if normalize:
        x = _normalize(x, epsilon)

    W1 = params["W1"]
    x = tf.cast(x, W1.dtype)
    W2 = params["W2"]
    omegas = params["omegas"]          # [num_radial, radial_dim, d+1]
    phases = params["phases"]          # [num_radial, radial_dim]
    coef = params["coef"]              # [num_radial]
    sketch_m = params["sketch_m"]
    radial_dim = params["radial_dim"]
    num_radial = params["num_radial"]

    leading = tf.shape(x)[:-1]

    # Augment z = [x̂, sqrt(b)]  -> (..., d+1)
    sqrt_b = tf.cast(math.sqrt(params["b"]), x.dtype)
    pad_shape = tf.concat([leading, [1]], axis=0)
    z = tf.concat([x, tf.fill(pad_shape, sqrt_b)], axis=-1)

    # degree-2 sketch psi(z): (..., sketch_m), approximates (z·z')^2
    psi = (
        tf.einsum("...d,md->...m", z, W1)
        * tf.einsum("...d,md->...m", z, W2)
        / tf.cast(math.sqrt(float(sketch_m)), z.dtype)
    )

    # radial RFF per node: (..., num_radial, radial_dim)
    proj = tf.einsum("...d,jrd->...jr", z, omegas) + phases
    rff = tf.cast(math.sqrt(2.0 / float(radial_dim)), z.dtype) * tf.cos(proj)
    sqrt_coef = tf.sqrt(tf.maximum(coef, 0.0))                  # [num_radial]
    rff = rff * tf.reshape(sqrt_coef, [num_radial, 1])
    # phi_rad: flatten (num_radial, radial_dim) -> (..., num_radial * radial_dim)
    rff_shape = tf.concat([leading, [num_radial * radial_dim]], axis=0)
    phi_rad = tf.reshape(rff, rff_shape)

    # tensor product psi ⊗ phi_rad -> (..., sketch_m * num_radial * radial_dim)
    out = tf.einsum("...s,...r->...sr", psi, phi_rad)
    out_shape = tf.concat([leading, [sketch_m * num_radial * radial_dim]], axis=0)
    return tf.reshape(out, out_shape)


def radial_yat_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> tf.Tensor:
    """Linear-complexity spherical-YAT attention via RAY (radial) features.

    Args mirror :func:`maclaurin_yat_attention`. RAY is the weaker method;
    expect it to underperform MAY.
    """
    q_feat = radial_features(query, params, normalize=True, epsilon=epsilon)
    k_feat = radial_features(key, params, normalize=True, epsilon=epsilon)
    return _linear_attention(q_feat, k_feat, value, causal=causal, epsilon=epsilon)


# --------------------------------------------------------------------------
# Generic linear-attention readout (feature-map-agnostic)
# --------------------------------------------------------------------------
def _linear_attention(
    q_feat: tf.Tensor,
    k_feat: tf.Tensor,
    value: tf.Tensor,
    causal: bool = False,
    epsilon: float = 1e-5,
) -> tf.Tensor:
    """Linear-attention readout from feature maps.

    ``q_feat`` / ``k_feat``: ``(B, *, H, F)``; ``value``: ``(B, K, H, dv)``.
    A small ``epsilon`` is added to the *denominator* only.
    """
    value = tf.cast(value, q_feat.dtype)

    if causal:
        # kv[b, k, h, f, d] = k_feat[b, k, h, f] * value[b, k, h, d]
        kv = tf.einsum("bkhf,bkhd->bkhfd", k_feat, value)
        kv_cum = tf.cumsum(kv, axis=1)
        k_cum = tf.cumsum(k_feat, axis=1)

        num = tf.einsum("bqhf,bqhfd->bqhd", q_feat, kv_cum)
        den = tf.einsum("bqhf,bqhf->bqh", q_feat, k_cum)
        den = den[..., None] + epsilon
        return num / den

    # Non-causal: O(n) via the associativity trick.
    kv = tf.einsum("bkhf,bkhd->bhfd", k_feat, value)
    num = tf.einsum("bqhf,bhfd->bqhd", q_feat, kv)

    k_sum = tf.reduce_sum(k_feat, axis=1)                  # (B, H, F)
    den = tf.einsum("bqhf,bhf->bqh", q_feat, k_sum)
    den = den[..., None] + epsilon
    return num / den
