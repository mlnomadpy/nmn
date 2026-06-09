"""Bias-aware linear-attention feature maps for spherical YAT (Keras).

This module implements two **bias-aware** random feature maps that linearize
spherical YAT attention, plus a generic linear-attention readout that consumes
them. It is the Keras counterpart of the nnx ``spherical_yat_performer`` and the
mlx ``performer`` modules, but unlike those (which approximate the *bias-free*
``b = 0`` kernel via the anchor / SLAY method), the maps here carry the
per-head bias ``b`` directly.

Spherical YAT kernel
--------------------
For L2-normalized query/key (``||q̂|| = ||k̂|| = 1``) with ``s = q̂·k̂``::

    kappa(s) = (s + b)^2 / (C - 2 s),   C = 2 + eps

where ``b`` is the per-head bias (deployment regime is ``b > 0``) and ``eps``
is the YAT epsilon. In benchmarks ``eps`` is set to the *median squared
distance* between normalized q and k (not ``1e-5``) — that is the regime where
the Maclaurin map (MAY) shines.

MAY — Random Maclaurin features (headline method)
-------------------------------------------------
The Maclaurin coefficients of ``kappa`` on the sphere are all non-negative for
``b >= 0``::

    beta_n = (1/C) (2/C)^n                                # coeffs of 1/(C - 2 s)
    a_n    = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n     # (beta_{-1}=beta_{-2}=0)

Truncating at ``nmax`` and writing ``Z = sum_n a_n``, ``p_n = a_n / Z``, the
randomized estimator draws ``M`` degrees ``N_r ~ Categorical(p)`` and, for each,
``N_r`` Rademacher vectors ``omega in {-1, +1}^d``. The feature is the product
over the drawn Rademacher projections::

    phi_r(x̂) = sqrt(Z / M) * prod_{l < N_r} (omega[r, l] · x̂)

with degree-0 draws contributing the constant feature ``sqrt(Z / M)`` (this is
exactly what lets MAY attend *diffusely* — the ``b = 0`` anchor method cannot).
The estimator is **unbiased**: ``E[phi(q̂)·phi(k̂)] = kappa(q̂·k̂)``.

RAY — sketched degree-2 modulation × radial RFF (secondary method)
------------------------------------------------------------------
Augmenting ``z = [x̂, sqrt(b)]`` gives ``z_q·z_k = s + b`` and
``||z_q - z_k||^2 = ||q̂ - k̂||^2``, so::

    kappa = (z_q·z_k)^2 · 1 / (eps + ||z_q - z_k||^2)

a product of two PSD kernels. ``(z_q·z_k)^2`` is sketched by two independent
random projections; ``1/(eps + r^2)`` is approximated by Gauss-Laguerre
quadrature over the exponential-integral representation, each node giving a
Gaussian RBF realized as a random Fourier feature. RAY targets the hard
sharp-``eps`` regime and is expected to underperform MAY (comparable to SLAY).

Sign-indefinite caveat
----------------------
Both MAY and RAY features are **sign-indefinite**: odd-degree Rademacher
products (MAY) and cosine RFFs (RAY) can be negative, so the linear-attention
denominator can in principle go non-positive. Do **not** add an epsilon to the
features themselves — that biases the (unbiased) estimator. A small epsilon is
added only to the *denominator* for division stability. Empirically the maps are
well-conditioned at ``b > 0``.

All projections are sampled once at construction with NumPy (so they are fixed
and shared by q and k); the per-token feature maps and the attention readout use
``keras.src.ops`` and therefore run on whatever Keras backend is active.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from keras.src import ops

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
def _normalize(x, epsilon: float = 1e-6):
    """L2-normalize over the last axis (the spherical assumption)."""
    norm = ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True) + epsilon)
    return x / norm


def maclaurin_coeffs(b: float, eps: float, nmax: int) -> np.ndarray:
    """Maclaurin coefficients ``a_n`` of ``kappa`` on the sphere.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` and ``C = 2 + eps``. All entries are ``>= 0`` for
    ``b >= 0`` (this is what makes the categorical sampling well-defined).

    Returns:
        ``a`` of shape ``(nmax + 1,)``.
    """
    C = 2.0 + eps
    n = np.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = b * b * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a


# --------------------------------------------------------------------------
# MAY — Random Maclaurin features
# --------------------------------------------------------------------------
def create_maclaurin_projection(
    head_dim: int,
    num_features: int = 256,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    nmax: int = 40,
    seed: Optional[int] = None,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Precompute the fixed projection for MAY (Random Maclaurin) features.

    Args:
        head_dim: Per-head dimension ``d``.
        num_features: ``M`` — number of Maclaurin features produced per token.
        bias: Per-head bias (deployment regime ``bias > 0``). The map is bias-aware.
        epsilon: YAT epsilon (``C = 2 + epsilon``). Set to the median squared
            distance between normalized q/k for the benchmark regime.
        nmax: Maclaurin truncation order (default 40).
        seed: Optional integer seed for deterministic projections.
        dtype: Float dtype string for the emitted feature arrays.

    Returns:
        Dict with ``omegas`` ``(M, nmax, d)`` Rademacher tensor, ``degrees``
        ``(M,)`` integer degrees ``N_r``, scalar ``Z``, and the integers
        ``num_features`` (``M``) and ``nmax``.
    """
    rng = np.random.default_rng(seed)
    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = float(a.sum())
    p = a / Z
    # Degrees N_r ~ Categorical(a_n / Z); degree 0 → constant feature.
    degrees = rng.choice(len(a), size=num_features, p=p).astype(np.int64)
    # Rademacher vectors: (M, nmax, d); only the first N_r are actually used.
    omegas = rng.choice([-1.0, 1.0], size=(num_features, nmax, head_dim))

    return {
        "omegas": ops.convert_to_tensor(omegas.astype(dtype)),   # (M, nmax, d)
        "degrees": ops.convert_to_tensor(degrees),               # (M,)
        "Z": Z,
        "num_features": num_features,
        "nmax": nmax,
        "head_dim": head_dim,
        "b": float(bias),
        "epsilon": epsilon,
    }


def maclaurin_features(
    x,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
):
    """Compute MAY features ``phi(x) in R^M`` for every ``(..., d)`` token.

    Faithful to the verified NumPy reference::

        proj[..., r, l] = omega[r, l] · x̂
        mask[r, l]      = (l < N_r)
        proj            = where(mask, proj, 1.0)          # neutralize l >= N_r
        phi_r(x̂)        = sqrt(Z / M) * prod_l proj[..., r, l]

    Args:
        x: ``(..., d)`` (e.g. ``(B, T, H, d)``); normalized to the unit sphere.
        params: Output of :func:`create_maclaurin_projection`.
        normalize: L2-normalize ``x`` first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., M)``. **Sign-indefinite** — do not add an
        epsilon to these (it would bias the unbiased estimator).
    """
    if normalize:
        x = _normalize(x, epsilon)

    omegas = params["omegas"]          # (M, nmax, d)
    degrees = params["degrees"]        # (M,)
    M = params["num_features"]
    nmax = params["nmax"]
    Z = params["Z"]

    x = ops.cast(x, omegas.dtype)

    # proj[..., m, l] = sum_d x[..., d] * omega[m, l, d]  →  (..., M, nmax)
    proj = ops.einsum("...d,mld->...ml", x, omegas)

    # mask[m, l] = l < N_m  →  (M, nmax); neutralize unused factors with 1.0.
    # Cast both sides to int32 so the comparison never mixes int32/int64.
    l = ops.cast(ops.arange(nmax), "int32")
    mask = ops.expand_dims(l, 0) < ops.cast(ops.expand_dims(degrees, 1), "int32")
    mask = ops.cast(mask, "bool")
    proj = ops.where(mask, proj, ops.ones_like(proj))

    # Product over the nmax (factor) axis → (..., M).
    feat = ops.prod(proj, axis=-1)
    scale = math.sqrt(Z / M)
    return feat * scale


# --------------------------------------------------------------------------
# RAY — sketched degree-2 modulation × radial RFF
# --------------------------------------------------------------------------
def create_radial_projection(
    head_dim: int,
    sketch_m: int = 128,
    num_radial: int = 8,
    radial_dim: int = 64,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    seed: Optional[int] = None,
    dtype: str = "float32",
) -> Dict[str, Any]:
    """Precompute the fixed projection for RAY (radial) features.

    Builds the degree-2 sketch (two independent Gaussian projections that
    approximate ``(z_q·z_k)^2``) and the radial RFF (Gauss-Laguerre over the
    exponential-integral representation of ``1/(eps + r^2)``).

    Args:
        head_dim: Per-head dimension ``d``. The augmented dim is ``d + 1``.
        sketch_m: Width ``m`` of the degree-2 sketch.
        num_radial: Number of Gauss-Laguerre nodes.
        radial_dim: RFF width per node.
        bias: Per-head bias; folded into the augmented vector ``z = [x̂, sqrt(bias)]``.
        epsilon: YAT epsilon (the ``r^2`` shift inside ``1/(eps + r^2)``).
        seed: Optional integer seed.
        dtype: Float dtype string for the emitted arrays.

    Returns:
        Dict with ``W1`` ``(m, d+1)``, ``W2`` ``(m, d+1)``,
        ``omegas`` ``(num_radial, radial_dim, d+1)``,
        ``phases`` ``(num_radial, radial_dim)``, ``coef`` ``(num_radial,)``,
        plus scalars ``b``, ``sketch_m``, ``radial_dim``, ``num_radial``.
    """
    rng = np.random.default_rng(seed)
    da = head_dim + 1  # augmented z = [x̂, sqrt(b)]

    # Degree-2 sketch: two independent projections → approximates (z·z')^2.
    W1 = rng.standard_normal((sketch_m, da))
    W2 = rng.standard_normal((sketch_m, da))

    # Radial RFF for 1/(eps + r^2) via 1/(eps+r^2) = ∫ e^{-t(eps+r^2)} dt,
    # Gauss-Laguerre over tau with the substitution t = tau / eps.
    tau, wq = np.polynomial.laguerre.laggauss(num_radial)
    t_nodes = tau / epsilon                 # t_j = tau_j / eps
    coef = (1.0 / epsilon) * wq             # (1/eps) w_j
    omegas, phases = [], []
    for tj in t_nodes:
        omegas.append(rng.standard_normal((radial_dim, da)) * np.sqrt(2.0 * tj))
        phases.append(rng.uniform(0.0, 2.0 * np.pi, radial_dim))

    return {
        "W1": ops.convert_to_tensor(W1.astype(dtype)),
        "W2": ops.convert_to_tensor(W2.astype(dtype)),
        "omegas": ops.convert_to_tensor(np.stack(omegas).astype(dtype)),
        "phases": ops.convert_to_tensor(np.stack(phases).astype(dtype)),
        "coef": ops.convert_to_tensor(coef.astype(dtype)),
        "b": float(bias),
        "sketch_m": sketch_m,
        "radial_dim": radial_dim,
        "num_radial": num_radial,
        "head_dim": head_dim,
        "epsilon": epsilon,
    }


def radial_features(
    x,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
):
    """Compute RAY features ``phi(x)`` for every ``(..., d)`` token.

    Faithful to the verified NumPy reference: augment ``z = [x̂, sqrt(b)]``,
    sketch ``psi(z)`` for ``(z·z')^2``, build per-node radial RFFs, and return
    the tensor product ``psi(z) ⊗ phi_rad(z)`` flattened to the last axis.

    Args:
        x: ``(..., d)``; normalized to the unit sphere.
        params: Output of :func:`create_radial_projection`.
        normalize: L2-normalize ``x`` first.
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., sketch_m * num_radial * radial_dim)``. Also
        **sign-indefinite** (cosine RFFs can be negative).
    """
    if normalize:
        x = _normalize(x, epsilon)

    W1 = params["W1"]
    W2 = params["W2"]
    omegas = params["omegas"]      # (num_radial, radial_dim, d+1)
    phases = params["phases"]      # (num_radial, radial_dim)
    coef = params["coef"]          # (num_radial,)
    sketch_m = params["sketch_m"]
    radial_dim = params["radial_dim"]
    num_radial = params["num_radial"]
    b = params["b"]

    x = ops.cast(x, W1.dtype)
    sqrt_b = math.sqrt(b)

    # z = [x̂, sqrt(b)] → (..., d+1). Build the sqrt(b) pad from x so it carries
    # the same (dynamic) leading shape without manual shape juggling.
    pad = ops.zeros_like(x[..., :1]) + sqrt_b              # (..., 1) filled with sqrt(b)
    z = ops.concatenate([x, pad], axis=-1)

    # Degree-2 sketch psi(z): (..., sketch_m), approximates (z·z')^2.
    psi = (ops.matmul(z, ops.transpose(W1)) * ops.matmul(z, ops.transpose(W2)))
    psi = psi / math.sqrt(sketch_m)

    # Radial RFF per node: (..., num_radial, radial_dim).
    # proj[..., j, c] = sum_e z[..., e] * omega[j, c, e]
    proj = ops.einsum("...e,jce->...jc", z, omegas)
    proj = proj + phases                                   # broadcast (num_radial, radial_dim)
    rff = math.sqrt(2.0 / radial_dim) * ops.cos(proj)      # (..., num_radial, radial_dim)
    sqrt_coef = ops.sqrt(ops.maximum(coef, 0.0))           # (num_radial,)
    rff = rff * ops.reshape(sqrt_coef, (num_radial, 1))    # scale each node

    # Flatten the (num_radial, radial_dim) tail → phi_rad (..., R*radial_dim).
    # ops.shape(...) returns a tuple of (possibly dynamic) dims; keep the leading
    # dims as-is and collapse the trailing two into a single Python int.
    rad_total = num_radial * radial_dim
    lead = tuple(ops.shape(rff)[:-2])
    phi_rad = ops.reshape(rff, lead + (rad_total,))

    # Tensor product psi ⊗ phi_rad → (..., sketch_m, rad_total) → flatten.
    out = ops.expand_dims(psi, -1) * ops.expand_dims(phi_rad, -2)
    lead_out = tuple(ops.shape(out)[:-2])
    return ops.reshape(out, lead_out + (sketch_m * rad_total,))


# --------------------------------------------------------------------------
# Linear-attention readout (feature-map-agnostic)
# --------------------------------------------------------------------------
def _linear_attention(q_feat, k_feat, value, causal: bool, epsilon: float):
    """Generic linear attention from feature maps.

    q_feat: ``(B, Q, H, F)``; k_feat: ``(B, K, H, F)``; value: ``(B, K, H, dv)``.
    Returns ``(B, Q, H, dv)``. A small epsilon is added to the *denominator*
    only (never to the features — that would bias the estimator).
    """
    value = ops.cast(value, q_feat.dtype)

    if causal:
        # kv[b, k, h, f, d] = k_feat[b, k, h, f] * value[b, k, h, d]
        kv = ops.einsum("bkhf,bkhd->bkhfd", k_feat, value)
        kv_cum = ops.cumsum(kv, axis=1)
        k_cum = ops.cumsum(k_feat, axis=1)
        num = ops.einsum("bqhf,bqhfd->bqhd", q_feat, kv_cum)
        den = ops.einsum("bqhf,bqhf->bqh", q_feat, k_cum)
        den = ops.expand_dims(den, -1) + epsilon
        return num / den

    # Non-causal: O(n) via the associativity trick.
    kv = ops.einsum("bkhf,bkhd->bhfd", k_feat, value)
    num = ops.einsum("bqhf,bhfd->bqhd", q_feat, kv)
    k_sum = ops.sum(k_feat, axis=1)
    den = ops.einsum("bqhf,bhf->bqh", q_feat, k_sum)
    den = ops.expand_dims(den, -1) + epsilon
    return num / den


def maclaurin_yat_attention(
    query,
    key,
    value,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-9,
    normalize: bool = True,
):
    """Linear-complexity spherical YAT attention via MAY features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, dv)``.
        params: Output of :func:`create_maclaurin_projection`.
        causal: If ``True``, use prefix sums (each query sees keys up to its
            own position).
        epsilon: Stability constant added to the denominator only.
        normalize: L2-normalize q/k first (the spherical assumption).

    Returns:
        ``(B, Q, H, dv)``.
    """
    q_feat = maclaurin_features(query, params, normalize=normalize)
    k_feat = maclaurin_features(key, params, normalize=normalize)
    return _linear_attention(q_feat, k_feat, value, causal, epsilon)


def radial_yat_attention(
    query,
    key,
    value,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-9,
    normalize: bool = True,
):
    """Linear-complexity spherical YAT attention via RAY features.

    Same signature as :func:`maclaurin_yat_attention`, using radial features.
    RAY is the weaker method (sharp-``eps`` regime); expect it to underperform
    MAY and be comparable to the bias-free anchor (SLAY) baseline.
    """
    q_feat = radial_features(query, params, normalize=normalize)
    k_feat = radial_features(key, params, normalize=normalize)
    return _linear_attention(q_feat, k_feat, value, causal, epsilon)
