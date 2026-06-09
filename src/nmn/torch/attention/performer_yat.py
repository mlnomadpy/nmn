"""Bias-aware linear-attention feature maps for spherical YAT (PyTorch).

This module implements two **unbiased** random-feature maps that linearize
the spherical YAT attention kernel, so that attention runs in ``O(seq_len)``
via the standard ``Σ_k φ(k) vᵀ`` reordering trick.

Kernel (on unit vectors ``s = q̂·k̂``, with ``‖q̂‖ = ‖k̂‖ = 1``):

    kappa(s) = (s + b)² / (C − 2 s),     C = 2 + eps

where ``b`` is the per-head **bias** (the deployment regime is ``b > 0``) and
``eps`` is the YAT epsilon (in the benchmark set to the *median squared
distance* between normalized q/k, not ``1e-5``).

Contrast with **SLAY** (the anchor/performer method in other backends): SLAY
approximates only the ``b = 0`` kernel and therefore **cannot carry the bias**.
The two maps here are **bias-aware**:

* **MAY** — Random Maclaurin features (the headline method). Expands ``kappa``
  in its Maclaurin series on the sphere and samples monomials. The degree-0
  draw produces a constant feature, which is exactly what lets MAY attend
  diffusely; SLAY's ``b = 0`` design cannot. ``E[φ(q̂)·φ(k̂)] = kappa(q̂·k̂)``.
* **RAY** — sketched degree-2 modulation ⊗ radial RFF. Writes ``kappa`` as a
  product of two PSD kernels and sketches each. Correct but weaker (the
  sharp-eps regime is hard); expected to be comparable-ish to SLAY, not to win.

Maclaurin coefficients (all ``≥ 0`` for ``b ≥ 0``):

    beta_n = (1/C) (2/C)^n                               # coeffs of 1/(C − 2 s)
    a_n    = beta_{n-2} + 2 b beta_{n-1} + b² beta_n      # beta_{-1}=beta_{-2}=0

so ``kappa(s) = Σ_n a_n sⁿ``.

Sign-indefinite caveat (documented per the spec): MAY/RAY features are
**sign-indefinite** — odd-degree monomial products can be negative, so the
linear-attention denominator can in principle go non-positive. We do **not**
add an epsilon to the *features* (that would bias the estimator); we add a
small epsilon only to the *denominator* for division stability. Empirically
the estimator is well-conditioned at ``b > 0``.

Public API (mirrors the SLAY/performer style in other backends):

* ``create_maclaurin_projection(...)`` / ``maclaurin_features(x, params)`` /
  ``maclaurin_yat_attention(q, k, v, params, causal=...)``
* ``create_radial_projection(...)`` / ``radial_features(x, params)`` /
  ``radial_yat_attention(q, k, v, params, causal=...)``

The exact (quadratic) reference these approximate is
:func:`nmn.torch.attention.yat_attention.yat_attention_normalized`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor


__all__ = [
    "maclaurin_coeffs",
    "create_maclaurin_projection",
    "maclaurin_features",
    "maclaurin_yat_attention",
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
    "linear_attention_readout",
]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _normalize(x: Tensor, epsilon: float = 1e-6) -> Tensor:
    """L2-normalize along the last axis (the spherical assumption)."""
    norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + epsilon)
    return x / norm


def maclaurin_coeffs(b: float, eps: float, nmax: int) -> np.ndarray:
    """Maclaurin coefficients ``a_n`` of ``kappa`` on the unit sphere.

    ``a_n = beta_{n-2} + 2 b beta_{n-1} + b² beta_n`` with
    ``beta_n = (1/C)(2/C)^n`` and ``C = 2 + eps``. All ``a_n ≥ 0`` for
    ``b ≥ 0``, so they define a valid (unnormalized) categorical distribution
    over monomial degrees.

    Args:
        b: Per-head bias.
        eps: YAT epsilon (``C = 2 + eps``).
        nmax: Maximum monomial degree (inclusive). Returns ``nmax + 1`` coeffs.

    Returns:
        ``np.ndarray`` of shape ``(nmax + 1,)``.
    """
    C = 2.0 + eps
    n = np.arange(nmax + 1)
    beta = (1.0 / C) * (2.0 / C) ** n
    a = (b * b) * beta.copy()
    a[1:] += 2.0 * b * beta[:-1]
    a[2:] += beta[:-2]
    return a  # [nmax+1], all >= 0 for b >= 0


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
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Precompute the (fixed) MAY projection, shared by q and k.

    Samples ``M = num_features`` monomial degrees ``N_r ~ Categorical(a_n / Z)``
    and a dense ``[M, nmax + 1, head_dim]`` Rademacher tensor (only the first
    ``N_r`` rows of each feature are actually used; the rest are neutralized to
    ``1`` at feature time). Degree-0 draws yield the constant feature.

    Args:
        head_dim: Per-head dimension ``d``.
        num_features: ``M`` — number of output features (each token maps to
            exactly ``M`` features).
        bias: Per-head bias (deployment regime ``bias > 0``).
        epsilon: YAT epsilon (``C = 2 + epsilon``).
        nmax: Maclaurin truncation degree (default 40).
        seed: Optional integer seed for deterministic projections.
        dtype: Torch dtype for the stored tensors.
        device: Optional torch device for the stored tensors.

    Returns:
        Dict of plain tensors / scalars: ``omegas`` ``[M, nmax+1, d]``,
        ``degrees`` ``[M]`` (int64), ``Z`` (float), ``num_features``,
        ``nmax``, ``b``, ``eps``, ``head_dim``.
    """
    rng = np.random.default_rng(seed)
    a = maclaurin_coeffs(bias, epsilon, nmax)
    Z = float(a.sum())
    p = a / Z
    degrees_np = rng.choice(len(a), size=num_features, p=p)
    # Rademacher vectors: [M, nmax+1, d]; only first N_r used per feature.
    omegas_np = rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32),
        size=(num_features, nmax + 1, head_dim),
    )

    omegas = torch.as_tensor(omegas_np, dtype=dtype, device=device)
    degrees = torch.as_tensor(degrees_np, dtype=torch.int64, device=device)

    return {
        "omegas": omegas,              # (M, nmax+1, d)
        "degrees": degrees,            # (M,)
        "Z": Z,
        "num_features": num_features,
        "nmax": nmax,
        "b": bias,
        "eps": epsilon,
        "head_dim": head_dim,
    }


def maclaurin_features(
    x: Tensor,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute MAY features ``φ(x) ∈ ℝ^M`` for each token.

    ``φ_r(x̂) = sqrt(Z / M) · Π_{l < N_r} (omega[r, l] · x̂)``, with unused
    factors (``l ≥ N_r``) neutralized to ``1``. The product over degree-0
    features is empty and equals the constant ``sqrt(Z / M)``.

    Args:
        x: ``(..., head_dim)`` (any leading shape, e.g. ``(B, S, H, d)``).
        params: Output of :func:`create_maclaurin_projection`.
        normalize: L2-normalize ``x`` first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., M)``. **Sign-indefinite** (see module doc).
    """
    if normalize:
        x = _normalize(x, epsilon)

    omegas = params["omegas"].to(device=x.device, dtype=x.dtype)  # (M, L, d)
    degrees = params["degrees"].to(device=x.device)              # (M,)
    M = params["num_features"]
    Z = params["Z"]
    L = omegas.shape[1]

    # proj[..., r, l] = omega[r, l] · x̂   ->  (..., M, L)
    proj = torch.einsum("...d,mld->...ml", x, omegas)

    # mask[r, l] = (l < N_r); neutralize unused factors to 1.0
    l_idx = torch.arange(L, device=x.device)               # (L,)
    mask = l_idx[None, :] < degrees[:, None]                # (M, L)
    proj = torch.where(mask, proj, torch.ones_like(proj))

    feat = proj.prod(dim=-1)                                # (..., M)
    return math.sqrt(Z / M) * feat


def maclaurin_yat_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> Tensor:
    """Linear-complexity bias-aware spherical-YAT attention via MAY features.

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, v_dim)``.
        params: Output of :func:`create_maclaurin_projection`.
        causal: If ``True``, use prefix sums so each query only sees keys up to
            its own position.
        epsilon: Numerical stability constant added to the **denominator only**
            (never to the features — that would bias the estimator).

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    q_feat = maclaurin_features(query, params, normalize=True)
    k_feat = maclaurin_features(key, params, normalize=True)
    return linear_attention_readout(
        q_feat, k_feat, value, causal=causal, epsilon=epsilon
    )


# --------------------------------------------------------------------------
# RAY — sketched degree-2 modulation ⊗ radial RFF
# --------------------------------------------------------------------------
def create_radial_projection(
    head_dim: int,
    sketch_m: int = 128,
    num_radial: int = 8,
    radial_dim: int = 64,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Precompute the (fixed) RAY projection, shared by q and k.

    Augments ``z = [x̂, sqrt(b)] ∈ ℝ^{d+1}`` so that ``z_q·z_k = s + b`` and
    ``‖z_q − z_k‖² = ‖q̂ − k̂‖²``. Then
    ``kappa = (z_q·z_k)² · 1/(eps + ‖z_q − z_k‖²)``, a product of two PSD
    kernels: a **degree-2 sketch** (two independent Gaussian projections) for
    ``(z_q·z_k)²`` and a **radial RFF** (Gauss-Laguerre over the
    exponential-integral representation of ``1/(eps + r²)``) for the radial
    factor.

    Args:
        head_dim: Per-head dimension ``d`` (augmented dim is ``d + 1``).
        sketch_m: ``m`` — degree-2 sketch width.
        num_radial: Gauss-Laguerre node count for the radial factor.
        radial_dim: RFF dimension per radial node.
        bias: Per-head bias.
        epsilon: YAT epsilon (the radial kernel is ``1/(epsilon + r²)``).
        seed: Optional integer seed.
        dtype: Torch dtype for the stored tensors.
        device: Optional torch device.

    Returns:
        Dict of plain tensors / scalars.
    """
    rng = np.random.default_rng(seed)
    da = head_dim + 1  # augmented z = [x_hat, sqrt(b)]

    # degree-2 sketch: two independent projections -> approximates (z·z')²
    W1 = rng.standard_normal((sketch_m, da)).astype(np.float32)
    W2 = rng.standard_normal((sketch_m, da)).astype(np.float32)

    # radial RFF for 1/(eps + ‖z - z'‖²) via Gauss-Laguerre over
    # 1/(eps + r²) = ∫_0^∞ e^{−t(eps + r²)} dt, substitution t = tau/eps.
    tau, wq = np.polynomial.laguerre.laggauss(num_radial)
    t_nodes = tau / epsilon              # t_j = tau_j / epsilon
    coef = (1.0 / epsilon) * wq          # (1/epsilon) w_j
    omegas, phases = [], []
    for tj in t_nodes:
        omegas.append(
            rng.standard_normal((radial_dim, da)).astype(np.float32) * np.sqrt(2.0 * tj)
        )
        phases.append(rng.uniform(0, 2 * np.pi, radial_dim).astype(np.float32))

    def _t(arr):
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)

    return {
        "W1": _t(W1),                                  # (m, da)
        "W2": _t(W2),                                  # (m, da)
        "omegas": _t(np.stack(omegas)),                # (num_radial, radial_dim, da)
        "phases": _t(np.stack(phases)),                # (num_radial, radial_dim)
        "coef": _t(coef.astype(np.float32)),           # (num_radial,)
        "b": bias,
        "eps": epsilon,
        "sketch_m": sketch_m,
        "radial_dim": radial_dim,
        "num_radial": num_radial,
        "head_dim": head_dim,
    }


def radial_features(
    x: Tensor,
    params: Dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute RAY features ``φ(x) = psi(z) ⊗ φ_rad(z)`` for each token.

    Args:
        x: ``(..., head_dim)``.
        params: Output of :func:`create_radial_projection`.
        normalize: L2-normalize ``x`` first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., sketch_m * num_radial * radial_dim)``.
        **Sign-indefinite** (see module doc).
    """
    if normalize:
        x = _normalize(x, epsilon)

    W1 = params["W1"].to(device=x.device, dtype=x.dtype)
    W2 = params["W2"].to(device=x.device, dtype=x.dtype)
    omegas = params["omegas"].to(device=x.device, dtype=x.dtype)   # (J, radial_dim, da)
    phases = params["phases"].to(device=x.device, dtype=x.dtype)   # (J, radial_dim)
    coef = params["coef"].to(device=x.device, dtype=x.dtype)       # (J,)
    sketch_m = params["sketch_m"]
    radial_dim = params["radial_dim"]
    b = params["b"]

    leading = x.shape[:-1]

    # z = [x̂, sqrt(b)]  ->  (..., da)
    sqrt_b = torch.full(
        leading + (1,), math.sqrt(b), dtype=x.dtype, device=x.device
    )
    z = torch.cat([x, sqrt_b], dim=-1)

    # degree-2 sketch psi(z): (..., sketch_m), approximates (z·z')²
    psi = (
        torch.einsum("...d,md->...m", z, W1)
        * torch.einsum("...d,md->...m", z, W2)
    ) / math.sqrt(sketch_m)

    # radial RFF per node: (..., J, radial_dim)
    proj = torch.einsum("...d,jrd->...jr", z, omegas) + phases  # (..., J, radial_dim)
    rff = math.sqrt(2.0 / radial_dim) * torch.cos(proj)
    sqrt_coef = torch.sqrt(torch.clamp(coef, min=0.0))          # (J,)
    rff = rff * sqrt_coef[..., :, None]
    phi_rad = rff.reshape(leading + (-1,))                      # (..., J*radial_dim)

    # tensor product psi(z) ⊗ φ_rad(z)  ->  (..., sketch_m * J*radial_dim)
    out = psi[..., :, None] * phi_rad[..., None, :]             # (..., m, J*radial_dim)
    return out.reshape(leading + (-1,))


def radial_yat_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    params: Dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> Tensor:
    """Linear-complexity bias-aware spherical-YAT attention via RAY features.

    See :func:`maclaurin_yat_attention`; same signature, RAY feature map.
    """
    q_feat = radial_features(query, params, normalize=True)
    k_feat = radial_features(key, params, normalize=True)
    return linear_attention_readout(
        q_feat, k_feat, value, causal=causal, epsilon=epsilon
    )


# --------------------------------------------------------------------------
# Feature-map-agnostic linear-attention readout
# --------------------------------------------------------------------------
def linear_attention_readout(
    q_feat: Tensor,
    k_feat: Tensor,
    value: Tensor,
    causal: bool = False,
    epsilon: float = 1e-5,
) -> Tensor:
    """L1-normalized linear-attention readout from feature maps.

    ``kv = Σ_k φ_k ⊗ v``; ``num = φ_q · kv``; ``den = φ_q · Σ_k φ_k``;
    ``out = num / (den + epsilon)``. The epsilon is added to the **denominator
    only** (the features are unbiased and must not be shifted).

    Args:
        q_feat: ``(B, Q, H, F)``.
        k_feat: ``(B, K, H, F)``.
        value: ``(B, K, H, v_dim)``.
        causal: If ``True``, use prefix sums (each query sees only keys up to
            its own position).
        epsilon: Denominator stability constant.

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    if k_feat.dtype != q_feat.dtype:
        k_feat = k_feat.to(q_feat.dtype)
    if value.dtype != q_feat.dtype:
        value = value.to(q_feat.dtype)

    if causal:
        # kv[b, k, h, f, d] = k_feat[b, k, h, f] * value[b, k, h, d]
        kv = torch.einsum("bkhf,bkhd->bkhfd", k_feat, value)
        kv_cum = torch.cumsum(kv, dim=1)
        k_cum = torch.cumsum(k_feat, dim=1)

        num = torch.einsum("bqhf,bqhfd->bqhd", q_feat, kv_cum)
        den = torch.einsum("bqhf,bqhf->bqh", q_feat, k_cum)
        den = den[..., None]
        return num / (den + epsilon)

    # Non-causal: O(n) via the associativity trick.
    kv = torch.einsum("bkhf,bkhd->bhfd", k_feat, value)
    num = torch.einsum("bqhf,bhfd->bqhd", q_feat, kv)

    k_sum = k_feat.sum(dim=1)
    den = torch.einsum("bqhf,bhf->bqh", q_feat, k_sum)
    den = den[..., None]
    return num / (den + epsilon)
