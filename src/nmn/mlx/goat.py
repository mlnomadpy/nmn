"""GOAT — Geometry-Only Attention Transformer (MLX backend).

Value-only / projection-free YAT-kernel attention. Unlike
:class:`~nmn.mlx.attention.MultiHeadYatAttention` (full Q/K/V projections +
softmax), GOAT attention drops the query/key projections entirely: the YAT
kernel

.. math::

    k(u, u') = \\frac{(u \\cdot u' + b)^2}{\\lVert u - u' \\rVert^2 + \\varepsilon}

is evaluated on the *raw* head slices of the residual stream, and the
attention weights are the :math:`\\ell_1`-normalised kernel (a
Nadaraya--Watson kernel smoother) rather than a softmax. Two tiers, following
the GOAT family:

* ``variant="v"``   — GOAT-V: value-only. Only ``W_V`` (and ``W_O``) remain;
  halves the attention-side projection count vs. standard QKV.
* ``variant="nov"`` — GOAT-noV: projection-free. Only ``W_O`` remains; values
  are the raw head slices and ``W_O`` recovers V's role.

Per-head ``(b, eps)`` are learnable and softplus-reparameterised (strictly
positive :math:`\\varepsilon`, non-negative ``b``), so each head has its own
kernel sharpness/offset -- the diversity that the dropped Q/K projections
would otherwise provide. The kernel is PSD (a Schur product of the polynomial
and inverse-multiquadric kernels), so the layer is a finite expansion in one
RKHS and the attention weights ARE exact attribution via the representer
theorem -- no surrogate needed.

.. warning::

    The self-mask is **mandatory** for self-attention. Because
    :math:`k(u,u) = (\\lVert u \\rVert^2 + b)^2 / \\varepsilon` is the row
    maximum, an unmasked :math:`\\ell_1` layer collapses onto the diagonal and
    performs no cross-token mixing. ``self_mask=True`` (the default) zeroes the
    diagonal before normalisation.

Mask convention matches the rest of ``nmn.mlx``: ``True`` = attend,
``False`` = mask out.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "goat_yat_attention_weights",
    "goat_yat_attention",
    "GoatYatAttention",
]


def _yat_scores(q: mx.array, k: mx.array, b: mx.array, eps: mx.array) -> mx.array:
    """Raw (un-normalised) YAT kernel scores.

    ``q``/``k`` are ``(B, H, Lq, D)`` / ``(B, H, Lk, D)``; ``b``/``eps`` are
    ``(H,)`` (already positive). Returns ``(B, H, Lq, Lk)``, all non-negative.
    """
    b = b.reshape(1, -1, 1, 1)
    eps = eps.reshape(1, -1, 1, 1)
    dot = q @ mx.swapaxes(k, -1, -2)                       # (B, H, Lq, Lk)
    qn = mx.sum(q * q, axis=-1, keepdims=True)             # (B, H, Lq, 1)
    kn = mx.sum(k * k, axis=-1, keepdims=True)             # (B, H, Lk, 1)
    dist2 = mx.maximum(qn + mx.swapaxes(kn, -1, -2) - 2.0 * dot, 0.0)
    return (dot + b) ** 2 / (dist2 + eps)


def goat_yat_attention_weights(
    q: mx.array,
    k: mx.array,
    b: mx.array,
    eps: mx.array,
    mask: Optional[mx.array] = None,
    self_mask: bool = True,
    floor: float = 1e-8,
) -> mx.array:
    """:math:`\\ell_1`-normalised YAT-kernel attention weights on raw head slices.

    Args:
        q, k: ``(B, L, H, D)`` head-split tensors (no Q/K projection — these are
            the raw residual-stream slices).
        b, eps: ``(H,)`` per-head kernel scalars, already strictly positive
            (i.e. post-softplus).
        mask: optional ``(B, H, Lq, Lk)`` (or broadcastable) boolean mask,
            ``True`` = attend.
        self_mask: if ``True`` (default) and the map is square, zero the
            diagonal before normalising. **Required for self-attention.**
        floor: numerical floor on the row sum.

    Returns:
        ``(B, H, Lq, Lk)`` row-stochastic weights.
    """
    qh = mx.transpose(q, (0, 2, 1, 3))                     # (B, H, Lq, D)
    kh = mx.transpose(k, (0, 2, 1, 3))                     # (B, H, Lk, D)
    scores = _yat_scores(qh, kh, b, eps)                   # (B, H, Lq, Lk) >= 0
    if self_mask and scores.shape[-2] == scores.shape[-1]:
        n = scores.shape[-1]
        scores = scores * (1.0 - mx.eye(n, dtype=scores.dtype))[None, None]
    if mask is not None:
        scores = scores * mask.astype(scores.dtype)
    return scores / (mx.sum(scores, axis=-1, keepdims=True) + floor)


def goat_yat_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    b: mx.array,
    eps: mx.array,
    mask: Optional[mx.array] = None,
    self_mask: bool = True,
) -> mx.array:
    """Apply GOAT weights to values.

    ``q``/``k``/``v`` are ``(B, L, H, D)``. Returns ``(B, Lq, H, D)``.
    """
    w = goat_yat_attention_weights(q, k, b, eps, mask=mask, self_mask=self_mask)
    vh = mx.transpose(v, (0, 2, 1, 3))                     # (B, H, Lk, D)
    out = w @ vh                                           # (B, H, Lq, D)
    return mx.transpose(out, (0, 2, 1, 3))                 # (B, Lq, H, D)


class GoatYatAttention(nn.Module):
    """Multi-head GOAT self-attention (value-only or projection-free).

    Args:
        embed_dim: total embedding dimension; must be divisible by ``num_heads``.
        num_heads: number of attention heads.
        variant: ``"v"`` (GOAT-V, value-only) or ``"nov"`` (GOAT-noV,
            projection-free).
        use_out_proj: whether to apply the output projection ``W_O``.
        use_bias: whether ``W_V`` / ``W_O`` carry a bias.
        epsilon: initial value of the (learnable, per-head) denominator
            constant; stored as ``softplus``-inverse so that the effective
            ``eps`` starts here and stays positive.
        dtype: parameter/computation dtype.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        variant: str = "v",
        use_out_proj: bool = True,
        use_bias: bool = False,
        epsilon: float = 1.0,
        dtype: mx.Dtype = mx.float32,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if variant not in ("v", "nov"):
            raise ValueError(f"variant must be 'v' or 'nov', got {variant!r}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.variant = variant
        self.use_out_proj = use_out_proj
        self.use_bias = use_bias
        self.dtype = dtype

        # Per-head learnable (b, eps), softplus-reparameterised.
        # b_raw = 0 -> softplus(0) = log 2; eps_raw chosen so softplus = epsilon.
        raw_eps = math.log(math.exp(epsilon) - 1.0)
        self.b_raw = mx.zeros((num_heads,), dtype=dtype)
        self.eps_raw = mx.full((num_heads,), raw_eps, dtype=dtype)

        scale = embed_dim ** -0.5
        if variant == "v":
            self.v_kernel = mx.random.normal((embed_dim, embed_dim)).astype(dtype) * scale
            if use_bias:
                self.v_bias = mx.zeros((embed_dim,), dtype=dtype)
        if use_out_proj:
            self.out_kernel = mx.random.normal((embed_dim, embed_dim)).astype(dtype) * scale
            if use_bias:
                self.out_bias = mx.zeros((embed_dim,), dtype=dtype)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        self_mask: bool = True,
    ) -> mx.array:
        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        B, L, _ = x.shape
        H, D = self.num_heads, self.head_dim

        # Kernel runs on the RAW head slices: query == key, no Q/K projection.
        qk = mx.reshape(x, (B, L, H, D))

        if self.variant == "v":
            v = x @ self.v_kernel
            if getattr(self, "v_bias", None) is not None:
                v = v + self.v_bias
            v = mx.reshape(v, (B, L, H, D))
        else:  # "nov": values are the raw head slices; W_O recovers V's role.
            v = qk

        b = nn.softplus(self.b_raw)
        eps = nn.softplus(self.eps_raw)

        out = goat_yat_attention(qk, qk, v, b, eps, mask=mask, self_mask=self_mask)
        out = mx.reshape(out, (B, L, self.embed_dim))

        if self.use_out_proj and getattr(self, "out_kernel", None) is not None:
            out = out @ self.out_kernel
            if getattr(self, "out_bias", None) is not None:
                out = out + self.out_bias
        return out
