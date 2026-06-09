"""RAY — sketched degree-2 modulation × radial RFF for spherical YAT (MLX).

A secondary, bias-aware feature map for the spherical YAT kernel

    kappa(s) = (s + b)² / (C − 2 s),    C = 2 + ε.

RAY factorizes ``kappa`` as a **product of two PSD kernels** by augmenting the
unit input ``x̂`` with the bias coordinate::

    z = [x̂, √b] ∈ ℝ^{d+1}   ⇒   z_q·z_k = s + b,   ‖z_q − z_k‖² = ‖q̂ − k̂‖²

so that ``kappa = (z_q·z_k)² · 1/(ε + ‖z_q − z_k‖²)``.

* **Degree-2 sketch** ``psi(z)`` approximates ``(z_q·z_k)²`` with two independent
  Gaussian projections ``W1, W2 ∈ ℝ^{m×(d+1)}``::

      psi_i(z) = (W1_i·z)(W2_i·z) / √m   ⇒   E[psi(z_q)·psi(z_k)] = (z_q·z_k)².

* **Radial RFF** approximates ``1/(ε + r²)`` via the exponential-integral
  representation ``1/(ε + r²) = ∫₀^∞ e^{−t(ε + r²)} dt``, discretized by
  Gauss-Laguerre over ``τ`` (substituting ``t = τ/ε``): nodes ``t_j = τ_j/ε``,
  coefficients ``coef_j = (1/ε) w_j``. For each node ``e^{−t_j r²}`` is a Gaussian
  RBF kernel, approximated by random Fourier features
  ``ω_j ~ N(0, 2 t_j I)``, ``phase ~ U(0, 2π)``,
  ``rff_j(z) = √(2/D) cos(ω_j·z + phase)`` scaled by ``√coef_j``.

The **RAY feature** is the tensor product ``psi(z) ⊗ phi_rad(z)``. Like MAY it is
**sign-indefinite** (the degree-2 sketch and the RFFs can be negative), so a small
epsilon is added only to the attention denominator, never to the features.

RAY is the weaker method: the sharp-epsilon regime is genuinely hard, so it is
expected to underperform MAY and land comparable-ish to SLAY. It is implemented
faithfully; it is not expected to win.

Mirrors the signature style of :mod:`nmn.mlx.performer` (numpy-seeded
``mx.array`` parameters, no jax key). Exact reference:
:func:`nmn.mlx.attention.yat_attention_normalized`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import numpy as np


__all__ = [
    "create_radial_projection",
    "radial_features",
    "radial_yat_attention",
]


def create_radial_projection(
    head_dim: int,
    sketch_m: int = 8,
    num_radial: int = 4,
    radial_dim: int = 8,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    dtype: mx.Dtype = mx.float32,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Precompute the (fixed) RAY projection shared by q and k.

    Args:
        head_dim: Per-head dimension ``d`` (the augmented dim is ``d + 1``).
        sketch_m: ``m`` — width of the degree-2 sketch for ``(z_q·z_k)²``.
        num_radial: Number of Gauss-Laguerre nodes for the radial integral.
        radial_dim: ``D`` — RFF features per radial node.
        bias: Per-head bias ``b`` (augmented coordinate ``√b``). Deployment
            regime is ``b > 0``.
        epsilon: YAT epsilon (the sharp ``1/(ε + r²)`` scale). In benchmarks set
            to the median squared distance between normalized q and k.
        dtype: MLX dtype for the precomputed arrays.
        seed: Optional integer seed for deterministic projections.

    Returns:
        Dict with ``W1``/``W2`` ``(m, d+1)``, the stacked radial ``omegas``
        ``(num_radial, D, d+1)``, ``phases`` ``(num_radial, D)``, the Laguerre
        ``coef`` ``(num_radial,)``, plus bookkeeping (``bias``, ``sketch_m``,
        ``radial_dim``, ``num_radial``). The output feature dim is
        ``sketch_m · num_radial · radial_dim``.
    """
    rng = np.random.default_rng(seed)
    da = head_dim + 1  # augmented z = [x̂, √b]

    # Degree-2 sketch: two independent projections approximate (z·z')².
    W1 = rng.standard_normal((sketch_m, da)).astype(np.float32)
    W2 = rng.standard_normal((sketch_m, da)).astype(np.float32)

    # Radial RFF for 1/(ε + r²) via Gauss-Laguerre over the exponential-integral
    # representation 1/(ε + r²) = ∫ e^{−t(ε + r²)} dt.
    tau, wq = np.polynomial.laguerre.laggauss(num_radial)
    t_nodes = tau / epsilon                      # t_j = τ_j / ε
    coef = (1.0 / epsilon) * wq                  # (1/ε) w_j

    omegas, phases = [], []
    for tj in t_nodes:
        omegas.append(
            rng.standard_normal((radial_dim, da)).astype(np.float32)
            * np.sqrt(2.0 * tj)
        )
        phases.append(rng.uniform(0.0, 2.0 * np.pi, radial_dim).astype(np.float32))

    return {
        "W1": mx.array(W1).astype(dtype),                       # (m, d+1)
        "W2": mx.array(W2).astype(dtype),                       # (m, d+1)
        "omegas": mx.array(np.stack(omegas)).astype(dtype),     # (num_radial, D, d+1)
        "phases": mx.array(np.stack(phases)).astype(dtype),     # (num_radial, D)
        "coef": mx.array(coef.astype(np.float32)).astype(dtype),  # (num_radial,)
        "bias": float(bias),
        "head_dim": head_dim,
        "sketch_m": sketch_m,
        "radial_dim": radial_dim,
        "num_radial": num_radial,
    }


def radial_features(
    x: mx.array,
    params: dict[str, Any],
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> mx.array:
    """Compute the RAY feature ``psi(z) ⊗ phi_rad(z)`` per token.

    Args:
        x: ``(..., seq_len, num_heads, head_dim)``.
        params: Output of :func:`create_radial_projection`.
        normalize: L2-normalize inputs first (the spherical assumption).
        epsilon: Stability constant for the normalization denominator.

    Returns:
        Features of shape ``(..., seq_len, num_heads, sketch_m · num_radial ·
        radial_dim)``. Sign-indefinite.
    """
    if normalize:
        x_norm = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True) + epsilon)
        x = x / x_norm

    W1 = params["W1"]
    W2 = params["W2"]
    omegas = params["omegas"]        # (num_radial, D, d+1)
    phases = params["phases"]        # (num_radial, D)
    coef = params["coef"]            # (num_radial,)
    b = params["bias"]
    m = params["sketch_m"]
    D = params["radial_dim"]
    num_radial = params["num_radial"]

    # Augment z = [x̂, √b] → (..., d+1).
    sqrt_b = mx.full(x.shape[:-1] + (1,), math.sqrt(b), dtype=x.dtype)
    z = mx.concatenate([x, sqrt_b], axis=-1)

    # Degree-2 sketch psi(z): (..., m), approximates (z·z')².
    psi = (mx.einsum("...d,md->...m", z, W1)
           * mx.einsum("...d,md->...m", z, W2)) / math.sqrt(m)

    # Radial RFF per node: (..., num_radial, D).
    proj = mx.einsum("...d,jrd->...jr", z, omegas) + phases   # (..., num_radial, D)
    rff = math.sqrt(2.0 / D) * mx.cos(proj)
    scale = mx.sqrt(mx.maximum(coef, 0.0)).reshape((num_radial, 1))
    rff = rff * scale                                          # (..., num_radial, D)
    leading = rff.shape[:-2]
    phi_rad = mx.reshape(rff, leading + (num_radial * D,))     # (..., num_radial*D)

    # Tensor product psi ⊗ phi_rad → (..., m * num_radial * D).
    out = psi[..., :, None] * phi_rad[..., None, :]            # (..., m, R*D)
    return mx.reshape(out, leading + (m * num_radial * D,))


def radial_yat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    params: dict[str, Any],
    causal: bool = False,
    epsilon: float = 1e-5,
) -> mx.array:
    """Linear-complexity *bias-aware* YAT attention via RAY features.

    Same linear-attention readout as :func:`nmn.mlx.may.maclaurin_yat_attention`
    and the SLAY ``yat_tp_attention``; a small ``epsilon`` is added only to the
    denominator (RAY features are sign-indefinite).

    Args:
        query: ``(B, Q, H, d)``.
        key: ``(B, K, H, d)``.
        value: ``(B, K, H, v_dim)``.
        params: Output of :func:`create_radial_projection`.
        causal: If ``True``, use prefix sums for causal masking.
        epsilon: Denominator stability constant.

    Returns:
        ``(B, Q, H, v_dim)``.
    """
    if query.dtype != value.dtype:
        value = value.astype(query.dtype)
    if key.dtype != query.dtype:
        key = key.astype(query.dtype)

    q_feat = radial_features(query, params, normalize=True)
    k_feat = radial_features(key, params, normalize=True)

    if causal:
        kv = mx.einsum("bkhm,bkhd->bkhmd", k_feat, value)
        kv_cum = mx.cumsum(kv, axis=1)
        k_cum = mx.cumsum(k_feat, axis=1)

        num = mx.einsum("bqhm,bqhmd->bqhd", q_feat, kv_cum)
        den = mx.einsum("bqhm,bqhm->bqh", q_feat, k_cum)
        den = den[..., None] + epsilon
        return num / den

    kv = mx.einsum("bkhm,bkhd->bhmd", k_feat, value)
    num = mx.einsum("bqhm,bhmd->bqhd", q_feat, kv)

    k_sum = mx.sum(k_feat, axis=1)
    den = mx.einsum("bqhm,bhm->bqh", q_feat, k_sum)
    den = den[..., None] + epsilon
    return num / den
