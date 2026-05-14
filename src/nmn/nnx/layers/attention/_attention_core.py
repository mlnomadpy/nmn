"""Shared tail for attention weight computations.

The three attention weight functions in this package
(``dot_product_attention_weights``, ``yat_attention_weights``,
``yat_attention_normalized``) all converge to the same tail once the
raw per-pair scores are computed:

    optional alpha multiply → bias add → mask → normalize → sow → dropout

This module holds that tail as a single helper so the three callers
stay short and any future fix (e.g. a new mask convention or dropout
broadcast variant) only has to land in one place.

Notes on parameters
-------------------
``alpha`` and ``bias`` are added as-is — the caller is responsible for
casting them to the same dtype as ``attn_weights`` (in particular,
yat_attention_weights runs the score math in float32 and casts bias to
float32 before the helper call).

``normalization`` is one of ``"softmax"`` / ``"l1"`` / ``"softermax"``.
``"l1"`` uses the YAT-friendly "scores are already non-negative" path:
masked positions are zeroed (not set to -inf) and the final divide is
``score / (sum + epsilon)``. Only ``yat_attention_weights`` exposes this
normalization mode today; the other two callers always pass
``"softmax"`` (and the ``use_softermax`` shortcut for backward
compatibility).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array, random
from flax import nnx
from flax.nnx.module import Module

from nmn.nnx.layers.squashers import softermax


__all__ = ["finalize_attention_weights"]


def finalize_attention_weights(
    attn_weights: Array,
    *,
    dtype,
    key: Array,
    alpha: Optional[Array] = None,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    normalization: str = "softmax",
    use_softermax: bool = False,
    power: float = 1.0,
    epsilon: float = 1e-5,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    module: Optional[Module] = None,
) -> Array:
    """Apply alpha / bias / mask / normalize / sow / dropout to raw scores.

    Args:
        attn_weights: Raw per-pair scores, shape ``[..., num_heads, q, k]``.
        dtype: The dtype the caller wants the *final* (post-softmax) weights
            in. The mask + dropout cast back to this dtype.
        key: The key tensor — only used to derive the broadcast shape for
            broadcast-dropout.
        alpha: Optional scaling multiplier applied before bias.
        bias: Optional additive bias.
        mask: Optional boolean mask (True = keep).
        normalization: ``"softmax"`` (default), ``"l1"``, or ``"softermax"``.
        use_softermax: Legacy alias — if True forces softermax.
        power: ``softermax`` power parameter (used iff softermax is selected).
        epsilon: L1 denominator epsilon (used iff ``normalization="l1"``).
        broadcast_dropout / dropout_rng / dropout_rate / deterministic:
            Standard dropout knobs. Dropout is broadcast across batch dims
            when ``broadcast_dropout=True``.
        module: Optional Flax NNX module — when supplied, the post-norm
            weights are sown as ``"attention_weights"``.

    Returns:
        Normalized attention weights of the same shape, cast to ``dtype``.
    """
    if alpha is not None:
        attn_weights = attn_weights * alpha

    if bias is not None:
        attn_weights = attn_weights + bias

    if mask is not None:
        if normalization == "l1":
            # L1: scores are non-negative; mask by zeroing masked positions.
            attn_weights = jnp.where(mask, attn_weights, 0.0)
        else:
            big_neg = jnp.finfo(attn_weights.dtype).min
            attn_weights = jnp.where(mask, attn_weights, big_neg)

    if normalization == "l1":
        attn_sum = jnp.sum(attn_weights, axis=-1, keepdims=True)
        attn_weights = (attn_weights / (attn_sum + epsilon)).astype(dtype)
    elif use_softermax or normalization == "softermax":
        attn_weights = softermax(attn_weights, n=power).astype(dtype)
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if module is not None:
        module.sow(nnx.Intermediate, "attention_weights", attn_weights)

    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    return attn_weights
