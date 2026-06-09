"""MAY — Random Maclaurin features for spherical Yat-attention (bias-aware).

This module implements the **MAY** ("Maclaurin Yat") linear-attention feature
map, a bias-aware unbiased estimator of the spherical Yat kernel.

Kernel
------
On unit vectors (s = q̂·k̂, with ‖q̂‖ = ‖k̂‖ = 1) the spherical Yat kernel is

    kappa(s) = (s + b)² / (C - 2 s),    C = 2 + eps

where ``b`` is the per-head bias (deployment regime **b > 0**) and ``eps`` is
the Yat epsilon. In benchmarks ``eps`` is set to the *median squared distance*
between normalized q and k (NOT 1e-5) — this is the regime where MAY shines.

Why MAY (vs. SLAY)
------------------
The existing **SLAY** anchor performer
(:mod:`nmn.nnx.layers.attention.spherical_yat_performer`) approximates only the
**b = 0** kernel ``s² / (C - 2 s)``; it structurally *cannot* carry the bias
term. MAY/RAY are **bias-aware**: they approximate the full ``(s + b)²``
numerator, so at ``b >= 1`` MAY clearly beats SLAY.

Random Maclaurin construction
-----------------------------
Expand ``kappa`` as a power series in ``s`` on the sphere. The geometric base
``1 / (C - 2 s)`` has coefficients

    beta_n = (1/C) (2/C)^n            (all > 0)

Multiplying by ``(s + b)² = s² + 2 b s + b²`` shifts indices:

    a_n = beta_{n-2} + 2 b beta_{n-1} + b² beta_n     (beta_{-1}=beta_{-2}=0)

so ``a_n >= 0`` for all ``b >= 0``. Truncate at ``nmax`` (default 40), let
``Z = sum_n a_n`` and ``p_n = a_n / Z``.

The Random Maclaurin map draws, per feature r = 1..M, a degree
``N_r ~ Categorical(p)`` and ``N_r`` Rademacher vectors ``omega ∈ {-1,+1}^d``,
then forms the product of projections:

    proj[n, r, l] = omega[r, l] · x̂                  ([..., M, nmax])
    mask[r, l]    = (l < N_r)                          (neutralize unused factors)
    phi_r(x̂)      = sqrt(Z / M) * prod_l where(mask, proj, 1.0)

Degree-0 draws produce the **constant feature** — this is exactly what lets MAY
attend diffusely (SLAY's ``b = 0`` design cannot).

Guarantee
---------
``E[phi(q̂)·phi(k̂)] = kappa(q̂·k̂)`` (an *unbiased* estimator). The pointwise
inner product therefore tracks ``kappa`` (rising with ``s``), and more features
raise the cosine similarity to exact attention.

Sign-indefinite caveat
----------------------
MAY features are **sign-indefinite**: odd-degree products can be negative, so
the linear-attention denominator can in principle go non-positive. Do **NOT**
add an epsilon to the *features* (that biases the estimator). Add a small
epsilon only to the *denominator* for division stability. Empirically the map
is well-conditioned at ``b > 0``.

Public API (mirrors the SLAY file's signature style):
    create_maclaurin_projection(key, head_dim, ...) -> dict
    maclaurin_features(x, params) -> Array
    maclaurin_yat_attention(q, k, v, params, causal=...) -> Array
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array


def maclaurin_coeffs(b: float, epsilon: float, nmax: int) -> np.ndarray:
    """Maclaurin coefficients ``a_n`` of ``kappa(s) = (s+b)²/(C-2s)``.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b² beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` and ``C = 2 + eps``. All ``a_n >= 0`` for
    ``b >= 0``.

    Args:
        b: Per-head bias.
        epsilon: Yat epsilon (``C = 2 + epsilon``).
        nmax: Truncation order (returns ``nmax + 1`` coefficients).

    Returns:
        NumPy array ``a`` of shape ``[nmax + 1]``, all non-negative for b >= 0.
    """
    C = 2.0 + epsilon
    n = np.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = (b * b) * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a  # [nmax+1]


def create_maclaurin_projection(
    key: Array,
    head_dim: int,
    num_features: int = 256,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    nmax: int = 40,
    dtype: Dtype = jnp.float32,
) -> dict:
    """Create parameters for MAY (Random Maclaurin) Yat features.

    The projection is fixed at construction and shared by q and k. We sample
    ``num_features`` degrees ``N_r ~ Categorical(a_n / Z)`` and, for each, a
    dense ``[num_features, nmax, head_dim]`` Rademacher tensor. Only the first
    ``N_r`` Rademacher rows are used per feature (the rest are masked out in
    :func:`maclaurin_features`).

    Args:
        key: JAX random key.
        head_dim: Dimension of each attention head ``d``.
        num_features: Number of output features ``M`` (the feature budget).
        bias: Per-head bias ``b`` (deployment regime ``b > 0``).
        epsilon: Yat epsilon (``C = 2 + epsilon``). For benchmarks set this to
            the median squared distance between normalized q and k.
        nmax: Maclaurin truncation order (default 40).
        dtype: Data type.

    Returns:
        Dictionary with projection parameters:
            ``omegas``  — Rademacher tensor ``[M, nmax, d]`` in {-1, +1}.
            ``degrees`` — integer degrees ``[M]`` (the sampled ``N_r``).
            ``Z``       — normalization constant ``sum_n a_n`` (scalar).
            ``num_features``, ``nmax``, ``head_dim``, ``bias``, ``epsilon``.
    """
    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = float(a.sum())
    p = a / a.sum()

    key, deg_key, rad_key = random.split(key, 3)
    # Sample degrees N_r ~ Categorical(p) via Gumbel-free categorical.
    degrees = random.choice(
        deg_key, len(a), shape=(num_features,), p=jnp.asarray(p, dtype=jnp.float32)
    ).astype(jnp.int32)
    # Rademacher vectors: [M, nmax, d] in {-1, +1}.
    omegas = jnp.where(
        random.bernoulli(rad_key, 0.5, (num_features, nmax, head_dim)),
        jnp.array(1.0, dtype=dtype),
        jnp.array(-1.0, dtype=dtype),
    )

    return {
        'omegas': omegas,              # [M, nmax, d]
        'degrees': degrees,            # [M]
        'Z': jnp.asarray(Z, dtype=dtype),
        'num_features': num_features,
        'nmax': nmax,
        'head_dim': head_dim,
        'bias': bias,
        'epsilon': epsilon,
    }


def maclaurin_features(
    x: Array,
    params: dict,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Compute MAY (Random Maclaurin) features.

    For each feature r:

        proj[..., r, l] = omega[r, l] · x̂
        phi_r(x̂)        = sqrt(Z / M) * prod_{l < N_r} proj[..., r, l]

    where factors with ``l >= N_r`` are neutralized to 1.0 (degree-0 features
    therefore produce the constant ``sqrt(Z / M)``). The result has exactly
    ``M`` features.

    Args:
        x: Input tensor ``[..., head_dim]`` (e.g. ``[..., seq, heads, d]``).
        params: Parameters from :func:`create_maclaurin_projection`.
        normalize: If True, L2-normalize ``x`` to unit norm first.
        epsilon: Stability constant for the normalization.

    Returns:
        Feature tensor ``[..., M]``.
    """
    if normalize:
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + epsilon)

    omegas = params['omegas']          # [M, nmax, d]
    degrees = params['degrees']        # [M]
    Z = params['Z']
    M = params['num_features']

    x = x.astype(omegas.dtype)
    # proj[..., m, l] = sum_d x[..., d] * omega[m, l, d]
    proj = jnp.einsum("...d,mld->...ml", x, omegas)   # [..., M, nmax]

    nmax = omegas.shape[1]
    l = jnp.arange(nmax)                               # [nmax]
    mask = l[None, :] < degrees[:, None]               # [M, nmax]
    proj = jnp.where(mask, proj, jnp.array(1.0, dtype=proj.dtype))  # neutralize l>=N_r

    feat = jnp.prod(proj, axis=-1)                     # [..., M]  product over factors
    scale = jnp.sqrt(Z / M).astype(feat.dtype)
    return scale * feat


def _linear_readout(
    q_feat: Array,
    k_feat: Array,
    value: Array,
    causal: bool,
    epsilon: float,
    precision: PrecisionLike,
) -> Array:
    """Feature-map-agnostic linear-attention readout.

    Given phi_q ``[.., Q, H, F]``, phi_k ``[.., K, H, F]``, v ``[.., K, H, dv]``:

        kv  = sum_k phi_k ⊗ v          [.., H, F, dv]
        num = phi_q · kv               [.., Q, H, dv]
        den = phi_q · sum_k phi_k      [.., Q, H]
        out = num / (den + eps_div)

    NOTE: the features are sign-indefinite, so a small epsilon is added only to
    the **denominator** (never to the features) for division stability.
    """
    if causal:
        kv = jnp.einsum("...khm,...khd->...khmd", k_feat, value, precision=precision)
        kv_cum = jnp.cumsum(kv, axis=-4)
        k_cum = jnp.cumsum(k_feat, axis=-3)

        num = jnp.einsum("...qhm,...qhmd->...qhd", q_feat, kv_cum, precision=precision)
        den = jnp.einsum("...qhm,...qhm->...qh", q_feat, k_cum, precision=precision)
        den = den[..., None]
    else:
        kv = jnp.einsum("...khm,...khd->...hmd", k_feat, value, precision=precision)
        num = jnp.einsum("...qhm,...hmd->...qhd", q_feat, kv, precision=precision)

        k_sum = jnp.sum(k_feat, axis=-3)
        den = jnp.einsum("...qhm,...hm->...qh", q_feat, k_sum, precision=precision)
        den = den[..., None]

    # Stabilize the denominator without biasing the (sign-indefinite) estimator:
    # push |den| away from zero while preserving its sign.
    den = den + jnp.sign(den) * epsilon + (den == 0) * epsilon
    return num / den


def maclaurin_yat_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict,
    causal: bool = False,
    epsilon: float = 1e-6,
    precision: PrecisionLike = None,
    gradient_scaling: bool = True,
) -> Array:
    """Compute spherical Yat attention using MAY (Random Maclaurin) features.

    Uses the unbiased estimator ``E[phi(q̂)·phi(k̂)] = kappa(q̂·k̂)`` to perform
    O(n) linear attention via associativity reordering. The features carry the
    bias term ``b``, so this is faithful for the deployment regime ``b > 0``
    (where SLAY's bias-free approximation fails).

    Args:
        query: ``[..., q_length, num_heads, head_dim]``.
        key: ``[..., kv_length, num_heads, head_dim]``.
        value: ``[..., kv_length, num_heads, v_dim]``.
        params: Parameters from :func:`create_maclaurin_projection`.
        causal: If True, use causal attention via prefix sums.
        epsilon: Denominator stability constant (added to the denominator only).
        precision: JAX precision.
        gradient_scaling: Unused, kept for API compatibility with SLAY.

    Returns:
        Output ``[..., q_length, num_heads, v_dim]``.
    """
    query, key, value = promote_dtype((query, key, value), dtype=None)

    q_feat = maclaurin_features(query, params, normalize=True)
    k_feat = maclaurin_features(key, params, normalize=True)

    return _linear_readout(q_feat, k_feat, value, causal, epsilon, precision)
