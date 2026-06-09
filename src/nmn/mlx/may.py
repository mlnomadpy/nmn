"""MAY — Random Maclaurin features for *bias-aware* spherical YAT attention (MLX).

The spherical YAT kernel on unit vectors ``s = q̂·k̂`` (``‖q̂‖ = ‖k̂‖ = 1``) is

    kappa(s) = (s + b)² / (C − 2 s),    C = 2 + ε

where ``b`` is the per-head bias (the deployment regime is ``b > 0``) and ``ε``
is the YAT epsilon. In benchmarks ``ε`` is set to the **median squared distance**
between normalized ``q`` and ``k`` (NOT ``1e-5``) — that is the regime where MAY
shines.

The existing **SLAY** anchor method (:mod:`nmn.mlx.performer`) only approximates
the ``b = 0`` kernel; it structurally cannot carry the bias. **MAY** is
**bias-aware**: it expands ``kappa`` in a Maclaurin series and draws Random
Maclaurin features whose expected inner product reproduces ``kappa`` *including*
the ``b``-dependent terms.

Maclaurin expansion (all coefficients non-negative for ``b ≥ 0``)::

    beta_n = (1/C) (2/C)^n                              # coeffs of 1/(C − 2s)
    a_n    = beta_{n-2} + 2 b beta_{n-1} + b² beta_n     # (beta_{-1}=beta_{-2}=0)

Truncate at ``nmax`` (default 40), let ``Z = Σ_n a_n`` and ``p_n = a_n / Z``.
The Random Maclaurin construction samples ``M`` degrees ``N_r ~ Categorical(p)``
and, for each, ``N_r`` Rademacher vectors ``ω ∈ {−1,+1}^d``. The feature is the
product of the corresponding ``ω·x̂`` factors scaled by ``√(Z/M)``::

    phi_r(x̂) = √(Z/M) · Π_{l < N_r} (ω_{r,l} · x̂)

so ``phi(x)`` has exactly ``M`` features and
``E[phi(q̂)·phi(k̂)] = kappa(q̂·k̂)``. Degree-0 draws give a constant feature —
this is what lets MAY attend diffusely (SLAY's ``b = 0`` cannot).

**Caveat (sign-indefinite).** MAY features are sign-indefinite: odd-degree
products can be negative, so the linear-attention denominator can in principle go
non-positive. We do **not** add an epsilon to the features themselves (that would
bias the unbiased estimator); a small epsilon is added only to the attention
denominator for division stability. Empirically the construction is
well-conditioned at ``b > 0``.

Mirrors the signature style of :mod:`nmn.mlx.performer` (the SLAY file):
numpy-seeded ``mx.array`` parameters, no jax key. The exact attention reference
is :func:`nmn.mlx.attention.yat_attention_normalized`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import numpy as np


__all__ = [
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
]


def maclaurin_coeffs(b: float, epsilon: float, nmax: int) -> np.ndarray:
    """Maclaurin coefficients of ``kappa`` on the sphere.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b² beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` and ``C = 2 + epsilon``. All entries are ``≥ 0``
    for ``b ≥ 0``.

    Args:
        b: Per-head bias.
        epsilon: YAT epsilon (``C = 2 + epsilon``).
        nmax: Truncation degree (returns ``nmax + 1`` coefficients).

    Returns:
        ``np.ndarray`` of shape ``(nmax + 1,)``.
    """
    C = 2.0 + epsilon
    n = np.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = (b * b) * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a  # (nmax + 1,), all >= 0 for b >= 0


def create_maclaurin_projection(
    head_dim: int,
    num_features: int = 256,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    nmax: int = 40,
    dtype: mx.Dtype = mx.float32,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Precompute the (fixed) Random Maclaurin projection shared by q and k.

    Args:
        head_dim: Per-head dimension ``d``.
        num_features: ``M`` — number of output features (one per random draw).
        bias: Per-head bias ``b`` baked into the Maclaurin coefficients. The
            deployment regime is ``b > 0`` (where MAY beats SLAY).
        epsilon: YAT epsilon (``C = 2 + epsilon``). In benchmarks set this to
            the median squared distance between normalized q and k.
        nmax: Maclaurin truncation degree (default 40).
        dtype: MLX dtype for the precomputed arrays.
        seed: Optional integer seed for deterministic projections.

    Returns:
        Dict with the Rademacher tensor ``omegas`` ``(M, nmax, d)``, the integer
        ``degrees`` ``(M,)``, the partition constant ``Z``, and the bookkeeping
        integers ``head_dim``, ``num_features`` (``M``), ``nmax``.
    """
    rng = np.random.default_rng(seed)

    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = float(a.sum())
    p = a / a.sum()

    # Degrees N_r ~ Categorical(p), r = 1..M.
    degrees_np = rng.choice(len(a), size=num_features, p=p).astype(np.int32)

    # Dense Rademacher tensor: only the first N_r rows are used per feature.
    omegas_np = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(num_features, nmax, head_dim),
    ).astype(np.float32)

    return {
        "omegas": mx.array(omegas_np).astype(dtype),   # (M, nmax, d)
        "degrees": mx.array(degrees_np),               # (M,) int32
        "Z": Z,
        "head_dim": head_dim,
        "num_features": num_features,
        "nmax": nmax,
    }


def maclaurin_features(
    x: mx.array,
    params: dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> mx.array:
    """Compute the Random Maclaurin feature ``phi(x) ∈ ℝ^M`` per token.

    ``phi_r(x̂) = √(Z/M) · Π_{l < N_r} (ω_{r,l} · x̂)``. Factors with ``l ≥ N_r``
    are neutralized to ``1`` (so they drop out of the product). Degree-0 draws
    yield the constant feature ``√(Z/M)``.

    Args:
        x: ``(..., seq_len, num_heads, head_dim)``.
        params: Output of :func:`create_maclaurin_projection`.
        normalize: L2-normalize inputs first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., seq_len, num_heads, M)``. Sign-indefinite.
    """
    if normalize:
        x_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + epsilon)
        x = x / x_norm

    omegas = params["omegas"]        # (M, nmax, d)
    degrees = params["degrees"]      # (M,)
    Z = params["Z"]
    M = params["num_features"]
    nmax = params["nmax"]

    # proj[..., r, l] = ω_{r,l} · x̂  →  (..., M, nmax)
    proj = mx.einsum("...d,mld->...ml", x, omegas)

    # mask[r, l] = (l < N_r)  →  neutralize unused factors to 1.
    l = mx.arange(nmax).reshape((1, nmax))                 # (1, nmax)
    mask = l < degrees.reshape((M, 1))                     # (M, nmax) bool
    proj = mx.where(mask, proj, mx.array(1.0, dtype=proj.dtype))

    feat = mx.prod(proj, axis=-1)                          # (..., M)
    return math.sqrt(Z / M) * feat


def maclaurin_yat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    params: dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> mx.array:
    """Linear-complexity *bias-aware* YAT attention via MAY features.

    Linear-attention readout (feature-map-agnostic, mirrors the SLAY
    ``yat_tp_attention``): ``out = (phi_q · Σ_k phi_k ⊗ v) / (phi_q · Σ_k phi_k)``.
    A small ``epsilon`` is added only to the denominator (NOT to the features) —
    MAY features are sign-indefinite, so the denominator is stabilized rather
    than the unbiased estimator perturbed.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, v_dim)``.
        params: Output of :func:`create_maclaurin_projection`.
        causal: If ``True``, use prefix sums so each query only sees keys up to
            its own position.
        epsilon: Denominator stability constant.

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    if query.dtype != value.dtype:
        value = value.astype(query.dtype)
    if key.dtype != query.dtype:
        key = key.astype(query.dtype)

    q_feat = maclaurin_features(query, params, normalize=True)
    k_feat = maclaurin_features(key, params, normalize=True)

    if causal:
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
