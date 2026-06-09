"""Bias-aware linear-attention feature maps for spherical YAT (Flax Linen).

This module provides **MAY** (Random Maclaurin) and **RAY** (radial) feature
maps that linearize the spherical YAT attention kernel, together with a generic
linear-attention readout. They are the Linen-framework counterpart of the nnx
``spherical_yat_performer`` (SLAY / anchor) and the mlx ``performer`` modules,
but unlike SLAY they are **bias-aware**.

The kernel
----------
On unit vectors (``s = q̂·k̂`` with ``||q̂|| = ||k̂|| = 1``):

    kappa(s) = (s + b)^2 / (C - 2 s),    C = 2 + eps

where ``b`` is the per-head bias (the deployment regime is ``b > 0``) and
``eps`` is the YAT epsilon. In benchmarks ``eps`` is set to the **median squared
distance** between the normalized q and k (NOT 1e-5); that is the regime where
MAY shines.

The existing **SLAY** (anchor) method approximates only the ``b = 0`` kernel and
cannot carry the bias. **MAY** and **RAY** approximate the full ``(s + b)^2``
numerator, so they remain accurate as ``b`` grows. At ``b >= 1`` MAY clearly
beats SLAY.

MAY — Random Maclaurin features (headline method)
-------------------------------------------------
The Maclaurin coefficients of ``kappa`` on the sphere are all non-negative for
``b >= 0``::

    beta_n = (1/C) (2/C)^n                               # coeffs of 1/(C - 2 s)
    a_n    = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n     # (beta_{-1}=beta_{-2}=0)

Truncating at ``nmax`` and writing ``Z = sum_n a_n``, ``p_n = a_n / Z``, the
Random-Maclaurin estimator samples a degree ``N_r ~ Categorical(p)`` per
feature and forms a product of ``N_r`` Rademacher projections::

    phi_r(x̂) = sqrt(Z / M) * prod_{l < N_r} (omega[r, l] · x̂)

Degree-0 draws yield the **constant** feature, which is exactly what lets MAY
attend diffusely (SLAY's ``b = 0`` design cannot). The estimator is unbiased::

    E[phi(q̂) · phi(k̂)] = kappa(q̂ · k̂)

**Caveat (sign-indefinite):** MAY features can be negative (odd-degree products),
so the linear-attention denominator can in principle go non-positive. Do NOT add
an epsilon to the *features* (that biases the estimator); add a small epsilon
only to the *denominator* for division stability. Empirically well-conditioned
for ``b > 0``.

RAY — sketched degree-2 modulation x radial RFF (secondary method)
------------------------------------------------------------------
Augment ``z = [x̂, sqrt(b)]`` in R^{d+1} so that ``z_q·z_k = s + b`` and
``||z_q - z_k||^2 = ||q̂ - k̂||^2``. Then ``kappa = (z_q·z_k)^2 / (eps + ||z_q -
z_k||^2)`` is a product of two PSD kernels, sketched as:

* a degree-2 sketch ``psi(z)`` with ``E[psi(z_q)·psi(z_k)] = (z_q·z_k)^2``;
* a radial RFF for ``1/(eps + r^2)`` via Gauss-Laguerre quadrature of the
  exponential-integral representation ``1/(eps+r^2) = ∫_0^∞ e^{-t(eps+r^2)} dt``.

The RAY feature is the tensor product ``psi(z) ⊗ phi_rad(z)``, and is also
sign-indefinite. RAY is the weaker method (the sharp-eps regime is hard); it is
expected to underperform MAY and be comparable-ish to SLAY.

Linear-attention readout
-------------------------
Given ``phi_q [.., Q, H, F]``, ``phi_k [.., K, H, F]``, ``v [.., K, H, dv]``::

    kv  = sum_k phi_k ⊗ v          # [.., H, F, dv]
    num = phi_q · kv               # [.., Q, H, dv]
    den = phi_q · sum_k phi_k      # [.., Q, H]
    out = num / (den + eps_div)

Causal attention uses prefix sums. All projection constructors are plain
``create_*_projection(key, ...)`` functions returning a parameter dict; the
feature maps and the attention readout are pure ``jax``/``jnp`` functions.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import Array


__all__ = [
    "spherical_kappa",
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    "linear_attention",
]


# --------------------------------------------------------------------------
# Exact kernel (for reference / testing)
# --------------------------------------------------------------------------
def spherical_kappa(s: Array, b: float, eps: float = 1e-5) -> Array:
    """Exact spherical-YAT kernel ``kappa(s) = (s + b)^2 / (C - 2 s)``.

    Args:
        s: Cosine similarity ``q̂·k̂`` of unit vectors.
        b: Per-head bias.
        eps: YAT epsilon (``C = 2 + eps``).

    Returns:
        ``kappa(s)`` with the same shape as ``s``.
    """
    C = 2.0 + eps
    return (s + b) ** 2 / (C - 2.0 * s)


def _normalize(x: Array, eps: float = 1e-6) -> Array:
    """L2-normalize along the last axis."""
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


# --------------------------------------------------------------------------
# Maclaurin coefficients
# --------------------------------------------------------------------------
def maclaurin_coeffs(b: float, eps: float, nmax: int) -> Array:
    """Maclaurin coefficients of ``kappa`` on the sphere.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b^2 beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` and ``beta_{-1} = beta_{-2} = 0``. All entries are
    non-negative for ``b >= 0``.

    Args:
        b: Per-head bias.
        eps: YAT epsilon (``C = 2 + eps``).
        nmax: Highest Maclaurin degree to retain (returns ``nmax + 1`` coeffs).

    Returns:
        ``a`` of shape ``[nmax + 1]``.
    """
    C = 2.0 + eps
    n = jnp.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = b * b * beta
    # a[1:] += 2 b beta[:-1]
    a = a.at[1:].add(2.0 * b * beta[:-1])
    # a[2:] += beta[:-2]
    a = a.at[2:].add(beta[:-2])
    return a  # [nmax + 1], all >= 0 for b >= 0


# --------------------------------------------------------------------------
# MAY — Random Maclaurin features
# --------------------------------------------------------------------------
def create_maclaurin_projection(
    key: Array,
    head_dim: int,
    num_features: int = 256,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    nmax: int = 40,
    dtype: Any = jnp.float32,
) -> dict[str, Any]:
    """Build a (fixed) Random-Maclaurin projection shared by q and k.

    Args:
        key: JAX PRNG key.
        head_dim: Per-head dimension ``d``.
        num_features: ``M`` — number of output features.
        bias: Per-head bias used to build the Maclaurin coefficients.
        epsilon: YAT epsilon (``C = 2 + epsilon``).
        nmax: Maclaurin truncation degree (default 40).
        dtype: dtype for the stored Rademacher tensor.

    Returns:
        Dict with ``omegas`` ``[M, nmax, d]`` (Rademacher), ``degrees`` ``[M]``
        (sampled ``N_r``), the scalar normalizer ``Z``, and the integers ``M``,
        ``nmax``, ``head_dim``.
    """
    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = jnp.sum(a)
    p = a / Z

    k_deg, k_omega = jax.random.split(key)
    # N_r ~ Categorical(p), r = 1..M
    degrees = jax.random.choice(
        k_deg, a.shape[0], shape=(num_features,), p=p
    )
    # Rademacher vectors: [M, nmax, d]; only the first N_r are used per feature.
    omegas = jnp.where(
        jax.random.bernoulli(k_omega, 0.5, (num_features, nmax, head_dim)),
        1.0,
        -1.0,
    ).astype(dtype)

    return {
        "omegas": omegas,          # (M, nmax, d)
        "degrees": degrees,        # (M,)
        "Z": Z,                    # scalar
        "M": num_features,
        "nmax": nmax,
        "head_dim": head_dim,
    }


def maclaurin_features(
    x: Array,
    params: dict[str, Any],
    normalize: bool = True,
    eps: float = 1e-6,
) -> Array:
    """Random-Maclaurin feature map ``phi(x) ∈ R^M``.

    Args:
        x: ``(..., d)`` (e.g. ``(..., seq, heads, d)``).
        params: Output of :func:`create_maclaurin_projection`.
        normalize: L2-normalize ``x`` to the unit sphere first.
        eps: Stability constant for the normalization denominator.

    Returns:
        Features ``(..., M)``. May be negative (sign-indefinite); do NOT add an
        epsilon to these features (it biases the estimator).
    """
    if normalize:
        x = _normalize(x, eps)

    omegas = params["omegas"]          # (M, nmax, d)
    degrees = params["degrees"]        # (M,)
    Z = params["Z"]
    M = params["M"]
    nmax = omegas.shape[1]

    # proj[..., m, l] = omega[m, l] · x̂
    proj = jnp.einsum("...d,mld->...ml", x, omegas)   # (..., M, nmax)
    l = jnp.arange(nmax)                               # (nmax,)
    mask = l[None, :] < degrees[:, None]               # (M, nmax)
    # neutralize unused factors (l >= N_r) by setting them to 1.0
    proj = jnp.where(mask, proj, 1.0)
    feat = jnp.prod(proj, axis=-1)                     # (..., M)
    return jnp.sqrt(Z / M) * feat


# --------------------------------------------------------------------------
# RAY — sketched degree-2 modulation x radial RFF
# --------------------------------------------------------------------------
def _laggauss(n: int) -> tuple[Array, Array]:
    """Gauss-Laguerre nodes/weights (computed in NumPy, returned as arrays)."""
    import numpy as np

    nodes, weights = np.polynomial.laguerre.laggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


def create_radial_projection(
    key: Array,
    head_dim: int,
    sketch_m: int = 128,
    num_radial: int = 8,
    radial_dim: int = 64,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    dtype: Any = jnp.float32,
) -> dict[str, Any]:
    """Build a RAY projection (degree-2 sketch x radial RFF).

    Augments ``z = [x̂, sqrt(bias)]`` so that ``z_q·z_k = s + bias`` and
    ``||z_q - z_k||^2 = ||q̂ - k̂||^2``.

    Args:
        key: JAX PRNG key.
        head_dim: Per-head dimension ``d`` (augmented to ``d + 1``).
        sketch_m: Width ``m`` of the degree-2 sketch.
        num_radial: Number of Gauss-Laguerre nodes for the radial RFF.
        radial_dim: RFF features per radial node.
        bias: Per-head bias (enters the augmentation).
        epsilon: YAT epsilon (radial-kernel scale).
        dtype: dtype for stored arrays.

    Returns:
        Dict with ``W1``, ``W2`` ``[m, d+1]``, ``omegas`` ``[num_radial,
        radial_dim, d+1]``, ``phases`` ``[num_radial, radial_dim]``, ``coef``
        ``[num_radial]``, plus ``bias``, ``sketch_m``, ``radial_dim``,
        ``num_radial``, ``head_dim``.
    """
    da = head_dim + 1   # augmented z = [x̂, sqrt(bias)]
    k_w1, k_w2, k_om, k_ph = jax.random.split(key, 4)

    # degree-2 sketch: two independent projections -> approximates (z·z')^2
    W1 = jax.random.normal(k_w1, (sketch_m, da), dtype=dtype)
    W2 = jax.random.normal(k_w2, (sketch_m, da), dtype=dtype)

    # radial RFF for 1/(eps + r^2) via Gauss-Laguerre over
    # 1/(eps + r^2) = ∫ e^{-t(eps + r^2)} dt, substitute t = tau / eps.
    tau, wq = _laggauss(num_radial)
    t_nodes = tau / epsilon              # t_j = tau_j / epsilon
    coef = (1.0 / epsilon) * wq          # (1/epsilon) w_j

    # omega_j ~ N(0, 2 t_j I) ; phase ~ U(0, 2 pi)
    base = jax.random.normal(k_om, (num_radial, radial_dim, da), dtype=dtype)
    scale = jnp.sqrt(2.0 * t_nodes).astype(dtype)[:, None, None]
    omegas = base * scale
    phases = jax.random.uniform(
        k_ph, (num_radial, radial_dim), dtype=dtype, minval=0.0, maxval=2.0 * math.pi
    )

    return {
        "W1": W1,                           # (m, d+1)
        "W2": W2,                           # (m, d+1)
        "omegas": omegas,                   # (num_radial, radial_dim, d+1)
        "phases": phases,                   # (num_radial, radial_dim)
        "coef": coef.astype(dtype),         # (num_radial,)
        "bias": float(bias),
        "sketch_m": sketch_m,
        "radial_dim": radial_dim,
        "num_radial": num_radial,
        "head_dim": head_dim,
    }


def radial_features(
    x: Array,
    params: dict[str, Any],
    normalize: bool = True,
    eps: float = 1e-6,
) -> Array:
    """RAY feature map ``phi(x) = psi(z) ⊗ phi_rad(z)``.

    Args:
        x: ``(..., d)``.
        params: Output of :func:`create_radial_projection`.
        normalize: L2-normalize ``x`` first.
        eps: Stability constant for the normalization denominator.

    Returns:
        Features ``(..., sketch_m * num_radial * radial_dim)``. Sign-indefinite.
    """
    if normalize:
        x = _normalize(x, eps)

    bias = params["bias"]
    sketch_m = params["sketch_m"]
    radial_dim = params["radial_dim"]
    num_radial = params["num_radial"]
    W1, W2 = params["W1"], params["W2"]
    omegas, phases, coef = params["omegas"], params["phases"], params["coef"]

    # augment z = [x̂, sqrt(bias)]
    sqrt_b = jnp.full(x.shape[:-1] + (1,), math.sqrt(bias), dtype=x.dtype)
    z = jnp.concatenate([x, sqrt_b], axis=-1)            # (..., d+1)

    # degree-2 sketch psi(z): (..., sketch_m), approximates (z·z')^2
    proj1 = jnp.einsum("...a,ma->...m", z, W1)
    proj2 = jnp.einsum("...a,ma->...m", z, W2)
    psi = proj1 * proj2 / math.sqrt(sketch_m)            # (..., m)

    # radial RFF per node: (..., num_radial, radial_dim)
    proj = jnp.einsum("...a,jra->...jr", z, omegas) + phases
    rff = jnp.sqrt(2.0 / radial_dim) * jnp.cos(proj)     # (..., num_radial, radial_dim)
    sqrt_coef = jnp.sqrt(jnp.maximum(coef, 0.0))         # (num_radial,)
    rff = rff * sqrt_coef[:, None]                       # broadcast over radial_dim
    phi_rad = rff.reshape(x.shape[:-1] + (num_radial * radial_dim,))

    # tensor product psi (x) phi_rad -> (..., sketch_m * radial_total)
    out = jnp.einsum("...s,...r->...sr", psi, phi_rad)
    return out.reshape(x.shape[:-1] + (sketch_m * num_radial * radial_dim,))


# --------------------------------------------------------------------------
# Linear-attention readout (feature-map agnostic)
# --------------------------------------------------------------------------
def linear_attention(
    phi_q: Array,
    phi_k: Array,
    value: Array,
    causal: bool = False,
    eps_div: float = 1e-6,
) -> Array:
    """Generic linear-attention readout from feature maps.

    Args:
        phi_q: ``(B, Q, H, F)`` query features.
        phi_k: ``(B, K, H, F)`` key features.
        value: ``(B, K, H, dv)``.
        causal: If True, use prefix sums so each query only sees keys up to its
            own position.
        eps_div: Small epsilon added to the *denominator* only (do not add to
            the features — that would bias sign-indefinite MAY/RAY estimators).

    Returns:
        ``(B, Q, H, dv)``.
    """
    if causal:
        # kv[b, k, h, f, d] = phi_k[b, k, h, f] * value[b, k, h, d]
        kv = jnp.einsum("bkhf,bkhd->bkhfd", phi_k, value)
        kv_cum = jnp.cumsum(kv, axis=1)
        k_cum = jnp.cumsum(phi_k, axis=1)

        num = jnp.einsum("bqhf,bqhfd->bqhd", phi_q, kv_cum)
        den = jnp.einsum("bqhf,bqhf->bqh", phi_q, k_cum)
        return num / (den[..., None] + eps_div)

    kv = jnp.einsum("bkhf,bkhd->bhfd", phi_k, value)
    num = jnp.einsum("bqhf,bhfd->bqhd", phi_q, kv)

    k_sum = jnp.sum(phi_k, axis=1)
    den = jnp.einsum("bqhf,bhf->bqh", phi_q, k_sum)
    return num / (den[..., None] + eps_div)


# --------------------------------------------------------------------------
# Convenience attention wrappers
# --------------------------------------------------------------------------
def maclaurin_yat_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict[str, Any],
    causal: bool = False,
    eps_div: float = 1e-6,
) -> Array:
    """Linear-complexity bias-aware YAT attention via MAY features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, dv)``.
        params: Output of :func:`create_maclaurin_projection`.
        causal: Causal masking via prefix sums.
        eps_div: Denominator stability constant.

    Returns:
        ``(B, Q, H, dv)``.
    """
    phi_q = maclaurin_features(query, params, normalize=True)
    phi_k = maclaurin_features(key, params, normalize=True)
    return linear_attention(phi_q, phi_k, value, causal=causal, eps_div=eps_div)


def radial_yat_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict[str, Any],
    causal: bool = False,
    eps_div: float = 1e-6,
) -> Array:
    """Linear-complexity bias-aware YAT attention via RAY features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, dv)``.
        params: Output of :func:`create_radial_projection`.
        causal: Causal masking via prefix sums.
        eps_div: Denominator stability constant.

    Returns:
        ``(B, Q, H, dv)``.
    """
    phi_q = radial_features(query, params, normalize=True)
    phi_k = radial_features(key, params, normalize=True)
    return linear_attention(phi_q, phi_k, value, causal=causal, eps_div=eps_div)
