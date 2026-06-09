"""RAY — sketched degree-2 modulation × radial RFF for spherical Yat-attention.

This module implements the **RAY** ("Radial Yat") linear-attention feature map,
a secondary bias-aware estimator of the spherical Yat kernel.

Kernel and augmentation
-----------------------
On unit vectors (s = q̂·k̂) the spherical Yat kernel is

    kappa(s) = (s + b)² / (C - 2 s),    C = 2 + eps.

Augment ``z = [x̂, sqrt(b)] ∈ R^{d+1}`` so that

    z_q · z_k     = s + b
    ‖z_q - z_k‖²  = ‖q̂ - k̂‖² = 2(1 - s)

and therefore

    kappa = (z_q · z_k)² · 1 / (eps + ‖z_q - z_k‖²)

is a **product of two PSD kernels**. RAY approximates each factor:

1. Degree-2 sketch ``psi(z)``: two independent random projections
   ``W1, W2 ∈ R^{m×(d+1)}`` give ``psi_i(z) = (W1_i·z)(W2_i·z)/sqrt(m)`` with
   ``E[psi(z_q)·psi(z_k)] = (z_q·z_k)²``.
2. Radial RFF for ``1/(eps + r²)`` via the exponential-integral identity
   ``1/(eps + r²) = ∫_0^∞ e^{-t(eps + r²)} dt``. Gauss-Laguerre over ``tau``
   (substituting ``t = tau/eps``) gives nodes ``t_j = tau_j/eps`` and coeffs
   ``coeff_j = (1/eps) w_j``. For node ``j``, ``e^{-t_j r²}`` is a Gaussian RBF
   kernel, approximated by random Fourier features
   ``omega_j ~ N(0, 2 t_j I)``, ``phase ~ U(0, 2pi)``,
   ``rff_j(z) = sqrt(2/D) cos(omega_j·z + phase)`` scaled by ``sqrt(coeff_j)``.

The **RAY feature** is the tensor product ``psi(z) ⊗ phi_rad(z)``.

Bias-aware, sign-indefinite, weaker than MAY
--------------------------------------------
RAY carries the bias through the augmentation, so it is bias-aware. Like MAY it
is **sign-indefinite** (the degree-2 sketch can be negative), so add an epsilon
only to the linear-attention *denominator*, never to the features. RAY is the
weaker method: the sharp-eps regime is hard, so it is expected to underperform
MAY and be comparable-ish to SLAY. Implement it correctly; don't expect it to
win.

Public API (mirrors the SLAY / MAY signature style):
    create_radial_projection(key, head_dim, ...) -> dict
    radial_features(x, params) -> Array
    radial_yat_attention(q, k, v, params, causal=...) -> Array
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array

from .maclaurin_yat import _linear_readout


def create_radial_projection(
    key: Array,
    head_dim: int,
    sketch_m: int = 128,
    num_radial: int = 8,
    radial_dim: int = 64,
    bias: float = 1.0,
    epsilon: float = 1e-5,
    dtype: Dtype = jnp.float32,
) -> dict:
    """Create parameters for RAY (radial) Yat features.

    Builds the degree-2 sketch projections and the radial RFF (Gauss-Laguerre)
    parameters. The total feature dimension is ``sketch_m * num_radial *
    radial_dim``.

    Args:
        key: JAX random key.
        head_dim: Dimension of each attention head ``d`` (augmented to ``d+1``).
        sketch_m: Number of degree-2 sketch features ``m``.
        num_radial: Number of Gauss-Laguerre radial nodes.
        radial_dim: Number of random Fourier features ``D`` per radial node.
        bias: Per-head bias ``b`` (used in the augmentation ``sqrt(b)``).
        epsilon: Yat epsilon. For benchmarks set to the median squared distance
            between normalized q and k.
        dtype: Data type.

    Returns:
        Dictionary with projection parameters:
            ``W1``, ``W2`` — degree-2 sketch projections ``[m, d+1]``.
            ``omegas``     — radial RFF frequencies ``[num_radial, D, d+1]``.
            ``phases``     — radial RFF phases ``[num_radial, D]``.
            ``coef``       — radial quadrature coefficients ``[num_radial]``.
            ``bias``, ``sketch_m``, ``radial_dim``, ``num_radial``, ``head_dim``.
    """
    da = head_dim + 1  # augmented z = [x_hat, sqrt(b)]

    key, k1, k2 = random.split(key, 3)
    W1 = random.normal(k1, (sketch_m, da), dtype=dtype)
    W2 = random.normal(k2, (sketch_m, da), dtype=dtype)

    # Radial RFF via Gauss-Laguerre over the exponential-integral representation
    # 1/(eps + r^2) = int_0^inf e^{-t(eps + r^2)} dt.
    tau, wq = np.polynomial.laguerre.laggauss(num_radial)
    t_nodes = tau / epsilon                 # t_j = tau_j / eps
    coef = (1.0 / epsilon) * wq             # (1/eps) w_j

    omegas_list, phases_list = [], []
    for tj in t_nodes:
        key, ko, kp = random.split(key, 3)
        omegas_list.append(
            random.normal(ko, (radial_dim, da), dtype=dtype) * jnp.sqrt(2.0 * tj).astype(dtype)
        )
        phases_list.append(
            random.uniform(kp, (radial_dim,), dtype=dtype, minval=0.0, maxval=2.0 * np.pi)
        )

    return {
        'W1': W1,
        'W2': W2,
        'omegas': jnp.stack(omegas_list),     # [num_radial, D, d+1]
        'phases': jnp.stack(phases_list),     # [num_radial, D]
        'coef': jnp.asarray(coef, dtype=dtype),
        'bias': bias,
        'sketch_m': sketch_m,
        'radial_dim': radial_dim,
        'num_radial': num_radial,
        'head_dim': head_dim,
        'epsilon': epsilon,
    }


def radial_features(
    x: Array,
    params: dict,
    normalize: bool = True,
    epsilon: float = 1e-6,
) -> Array:
    """Compute RAY (radial) features as the tensor product psi(z) ⊗ phi_rad(z).

    Args:
        x: Input tensor ``[..., head_dim]`` (e.g. ``[..., seq, heads, d]``).
        params: Parameters from :func:`create_radial_projection`.
        normalize: If True, L2-normalize ``x`` to unit norm first.
        epsilon: Stability constant for the normalization.

    Returns:
        Feature tensor ``[..., sketch_m * num_radial * radial_dim]``.
    """
    if normalize:
        x = x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + epsilon)

    W1 = params['W1']
    dtype = W1.dtype
    x = x.astype(dtype)

    # Augment z = [x_hat, sqrt(b)]: append a constant sqrt(b) channel.
    sqrt_b = jnp.sqrt(jnp.array(params['bias'], dtype=dtype))
    pad = jnp.broadcast_to(sqrt_b, x.shape[:-1] + (1,))
    z = jnp.concatenate([x, pad], axis=-1)          # [..., d+1]

    sketch_m = params['sketch_m']
    radial_dim = params['radial_dim']

    # Degree-2 sketch psi(z): [..., sketch_m], E[psi·psi'] = (z·z')^2.
    p1 = jnp.einsum("...a,ma->...m", z, W1)
    p2 = jnp.einsum("...a,ma->...m", z, params['W2'])
    psi = (p1 * p2) / jnp.sqrt(jnp.array(sketch_m, dtype=dtype))   # [..., m]

    # Radial RFF per node: [..., num_radial, D].
    proj = jnp.einsum("...a,jDa->...jD", z, params['omegas'])      # [..., num_radial, D]
    proj = proj + params['phases']                                # broadcast [num_radial, D]
    rff = jnp.sqrt(jnp.array(2.0 / radial_dim, dtype=dtype)) * jnp.cos(proj)
    coef = jnp.sqrt(jnp.maximum(params['coef'], 0.0))             # [num_radial]
    rff = rff * coef[:, None]                                      # scale each node
    phi_rad = rff.reshape(rff.shape[:-2] + (params['num_radial'] * radial_dim,))

    # Tensor product psi ⊗ phi_rad -> [..., sketch_m * num_radial * radial_dim].
    out = jnp.einsum("...s,...r->...sr", psi, phi_rad)
    return out.reshape(out.shape[:-2] + (psi.shape[-1] * phi_rad.shape[-1],))


def radial_yat_attention(
    query: Array,
    key: Array,
    value: Array,
    params: dict,
    causal: bool = False,
    epsilon: float = 1e-6,
    precision: PrecisionLike = None,
    gradient_scaling: bool = True,
) -> Array:
    """Compute spherical Yat attention using RAY (radial) features.

    Args:
        query: ``[..., q_length, num_heads, head_dim]``.
        key: ``[..., kv_length, num_heads, head_dim]``.
        value: ``[..., kv_length, num_heads, v_dim]``.
        params: Parameters from :func:`create_radial_projection`.
        causal: If True, use causal attention via prefix sums.
        epsilon: Denominator stability constant (added to the denominator only).
        precision: JAX precision.
        gradient_scaling: Unused, kept for API compatibility with SLAY.

    Returns:
        Output ``[..., q_length, num_heads, v_dim]``.
    """
    query, key, value = promote_dtype((query, key, value), dtype=None)

    q_feat = radial_features(query, params, normalize=True)
    k_feat = radial_features(key, params, normalize=True)

    return _linear_readout(q_feat, k_feat, value, causal, epsilon, precision)
