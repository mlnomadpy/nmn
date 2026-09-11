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
``"l1"`` uses the YAT-friendly non-negative path: after additive bias,
negative scores are clipped to zero, masked positions are zeroed, and each
nonzero row is divided by its sum.  This keeps the attention measure
non-negative even when callers supply signed relative-position biases.

Boolean masks and negative-infinity additive-bias entries follow one policy:
masked weights are exact zero, and a query row with no eligible keys has zero
weights and a zero readout with finite gradients.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.module import Module
from jax import Array, random

from nmn._validation import validate_rate
from nmn.nnx.layers.squashers import softermax

SUPPORTED_ATTENTION_NORMALIZATIONS = ("softmax", "l1", "softermax")

__all__ = [
    "SUPPORTED_ATTENTION_NORMALIZATIONS",
    "finalize_attention_weights",
    "validate_attention_normalization",
]


def validate_attention_normalization(normalization: str) -> None:
    """Validate the shared public attention normalization enum."""
    if normalization not in SUPPORTED_ATTENTION_NORMALIZATIONS:
        choices = ", ".join(repr(name) for name in SUPPORTED_ATTENTION_NORMALIZATIONS)
        raise ValueError(
            f"normalization must be one of {choices}; got {normalization!r}"
        )


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
        epsilon: Retained for API compatibility with callers that also use it
            in YAT score construction.
        broadcast_dropout / dropout_rng / dropout_rate / deterministic:
            Standard dropout knobs. Dropout is broadcast across batch dims
            when ``broadcast_dropout=True``.
        module: Optional Flax NNX module — when supplied, the post-norm
            weights are sown as ``"attention_weights"``.

    Returns:
        Normalized attention weights of the same shape, cast to ``dtype``.
    """
    validate_attention_normalization(normalization)
    dropout_rate = validate_rate(dropout_rate, "dropout_rate")
    if alpha is not None:
        attn_weights = attn_weights * alpha

    additive_mask = None
    if bias is not None:
        attn_weights = attn_weights + bias
        # Negative infinity is the standard additive-mask sentinel.  Treat it
        # exactly like a False boolean mask so an entirely disabled row has a
        # defined zero result instead of NaNs.
        additive_mask = jnp.logical_not(jnp.isneginf(bias))

    effective_mask = mask
    if additive_mask is not None:
        effective_mask = (
            additive_mask
            if effective_mask is None
            else jnp.logical_and(effective_mask, additive_mask)
        )

    if effective_mask is not None:
        effective_mask = jnp.broadcast_to(effective_mask, attn_weights.shape)

    if effective_mask is not None:
        if normalization == "l1" or use_softermax or normalization == "softermax":
            # L1/softermax operate on non-negative scores; zero is their mask
            # identity and naturally makes a fully masked row all-zero.
            attn_weights = jnp.where(effective_mask, attn_weights, 0.0)
        else:
            big_neg = jnp.finfo(attn_weights.dtype).min
            row_has_key = jnp.any(effective_mask, axis=-1, keepdims=True)
            attn_weights = jnp.where(effective_mask, attn_weights, big_neg)
            attn_weights = jnp.where(row_has_key, attn_weights, 0.0)

    if normalization == "l1":
        # Additive attention biases are conventionally signed.  Clipping here
        # defines L1 attention as a non-negative measure instead of allowing a
        # negative bias to create negative "weights".  Scale by the row maximum
        # before summing to avoid overflow for large but finite scores.
        attn_weights = jnp.maximum(attn_weights, 0.0)
        row_max = jnp.max(attn_weights, axis=-1, keepdims=True)
        safe_max = jnp.where(row_max > 0.0, row_max, 1.0)
        scaled_weights = attn_weights / safe_max
        attn_sum = jnp.sum(scaled_weights, axis=-1, keepdims=True)
        safe_sum = jnp.where(attn_sum > 0.0, attn_sum, 1.0)
        normalized_weights = scaled_weights / safe_sum

        # ReLU can erase an entire otherwise-valid row when every signed
        # biased score is non-positive.  L1 normalization is undefined there,
        # so use the unique uninformative probability measure: uniform over
        # eligible keys.  A truly all-masked row has no eligible keys and stays
        # exact zero, preserving the public masking policy.
        eligible = (
            jnp.ones_like(attn_weights, dtype=jnp.bool_)
            if effective_mask is None
            else effective_mask
        )
        eligible_count = jnp.sum(eligible, axis=-1, keepdims=True)
        safe_count = jnp.where(eligible_count > 0, eligible_count, 1)
        uniform_weights = eligible.astype(attn_weights.dtype) / safe_count
        attn_weights = jnp.where(
            attn_sum > 0.0, normalized_weights, uniform_weights
        ).astype(dtype)
    elif use_softermax or normalization == "softermax":
        attn_weights = softermax(attn_weights, n=power).astype(dtype)
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if effective_mask is not None:
        # Exact zeros are important both for the readout and its value
        # gradient.  This also turns the finite placeholder distribution used
        # for a fully masked softmax row into the documented zero policy.
        attn_weights = jnp.where(
            effective_mask, attn_weights, jnp.zeros_like(attn_weights)
        )

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
