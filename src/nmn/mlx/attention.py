"""YAT attention for MLX.

Implements:

* Functional ``yat_attention_weights`` / ``yat_attention`` /
  ``yat_attention_normalized`` — the score
  ``(q·k)² / (‖q − k‖² + ε)`` plus softmax normalization.
* ``MultiHeadYatAttention`` — a four-projection (Q, K, V, out) module
  using YAT attention internally.

Mirrors ``nmn.tf.attention``. The mask convention is ``True = attend,
False = mask out`` (consistent with the other backends).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


__all__ = [
    "normalize_qk",
    "yat_attention_weights",
    "yat_attention",
    "yat_attention_normalized",
    "MultiHeadYatAttention",
]


DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)
_MASK_FILL = -1e9


def normalize_qk(
    query: mx.array,
    key: mx.array,
    epsilon: float = 1e-6,
) -> Tuple[mx.array, mx.array]:
    """L2-normalize ``query`` and ``key`` along the last axis."""
    q_norm = mx.sqrt(mx.sum(query * query, axis=-1, keepdims=True) + epsilon)
    k_norm = mx.sqrt(mx.sum(key * key, axis=-1, keepdims=True) + epsilon)
    return query / q_norm, key / k_norm


def yat_attention_weights(
    query: mx.array,
    key: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> mx.array:
    """Softmax over the YAT score.

    Inputs use the (batch, length, num_heads, head_dim) convention, the
    output mask the (batch, heads, q_len, kv_len) convention.
    """
    head_dim = query.shape[-1]
    dim_scale = math.sqrt(head_dim)

    if spherical:
        query, key = normalize_qk(query, key, epsilon)
        dot_product = mx.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = dot_product * dot_product
        distance_sq = mx.maximum(2.0 - 2.0 * dot_product, 0.0)
        attn = squared_dot / ((distance_sq + epsilon) * dim_scale)
    else:
        dot_product = mx.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = dot_product * dot_product
        q_sq = mx.sum(query * query, axis=-1)  # (B, Q, H)
        k_sq = mx.sum(key * key, axis=-1)      # (B, K, H)
        q_sq = mx.transpose(q_sq, (0, 2, 1))[..., :, None]  # (B, H, Q, 1)
        k_sq = mx.transpose(k_sq, (0, 2, 1))[..., None, :]  # (B, H, 1, K)
        dist_sq = mx.maximum(q_sq + k_sq - 2.0 * dot_product, 0.0)
        attn = squared_dot / ((dist_sq + epsilon) * dim_scale)

    if alpha is not None:
        attn = attn * alpha
    elif scale is not None:
        attn = attn * scale

    if mask is not None:
        attn = mx.where(
            mask,
            attn,
            mx.array(_MASK_FILL, dtype=attn.dtype),
        )

    attn = mx.softmax(attn, axis=-1)

    if dropout_rate > 0.0 and training:
        keep = 1.0 - dropout_rate
        rnd = mx.random.uniform(shape=attn.shape)
        mask_dropout = rnd < keep
        attn = mx.where(mask_dropout, attn / keep, mx.zeros_like(attn))
    return attn


def yat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> mx.array:
    """Softmax(YAT score) @ V. See ``yat_attention_weights`` for conventions."""
    weights = yat_attention_weights(
        query, key,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        spherical=spherical,
    )
    return mx.einsum("bhqk,bkhd->bqhd", weights, value)


def yat_attention_normalized(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
) -> mx.array:
    """``yat_attention`` with the QK normalization shortcut (``spherical=True``)."""
    return yat_attention(
        query, key, value,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        spherical=True,
    )


class MultiHeadYatAttention(nn.Module):
    """Multi-head YAT attention.

    Architecture::

        Input → Linear(Q) ─┐
        Input → Linear(K) ─┼─→ yat_attention(Q, K, V) → Linear(out) → Output
        Input → Linear(V) ─┘

    Args:
        embed_dim: Total embedding dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate (only used in training mode).
        use_bias: Whether the four projections use a bias.
        use_alpha: Whether to apply alpha scaling. Ignored if ``constant_alpha`` set.
        constant_alpha: ``True`` → √2; ``float`` → that value; ``None`` →
            learnable scalar.
        normalize_qk: If ``True``, L2-normalize Q and K (without the spherical
            distance shortcut).
        spherical: If ``True``, use the spherical YAT score
            ``(2 − 2·q·k)`` for the distance, faster but assumes unit-norm Q/K.
        use_out_proj: Whether to apply the output projection.
        epsilon: YAT stability constant.
        dtype: MLX dtype for parameters and computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        normalize_qk: bool = False,
        spherical: bool = False,
        use_out_proj: bool = True,
        epsilon: float = 1e-5,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.epsilon = epsilon
        self.use_out_proj = use_out_proj
        self._normalize_qk = normalize_qk
        self._spherical = spherical
        self.dtype = dtype
        self.use_bias = use_bias

        # Alpha config.
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        elif use_alpha:
            self.alpha = mx.ones((1,), dtype=dtype)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        self.is_built = False

    # ------------------------------------------------------------------
    # Build / forward
    # ------------------------------------------------------------------

    def build(self, input_dim: int) -> None:
        if self.is_built:
            return
        std = math.sqrt(2.0 / (input_dim + self.embed_dim))

        def make_kernel(shape):
            return (mx.random.normal(shape=shape) * std).astype(self.dtype)

        self.q_kernel = make_kernel((input_dim, self.embed_dim))
        self.k_kernel = make_kernel((input_dim, self.embed_dim))
        self.v_kernel = make_kernel((input_dim, self.embed_dim))
        if self.use_bias:
            self.q_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)
            self.k_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)
            self.v_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)

        if self.use_out_proj:
            self.out_kernel = make_kernel((self.embed_dim, self.embed_dim))
            if self.use_bias:
                self.out_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)

        self.is_built = True

    def _linear(self, x: mx.array, kernel: mx.array, bias: Optional[mx.array]) -> mx.array:
        y = x @ kernel
        if bias is not None:
            y = y + bias
        return y

    def __call__(
        self,
        query: mx.array,
        key: Optional[mx.array] = None,
        value: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        training: bool = False,
    ) -> mx.array:
        if key is None:
            key = query
        if value is None:
            value = key

        if query.dtype != self.dtype:
            query = query.astype(self.dtype)
        if key.dtype != self.dtype:
            key = key.astype(self.dtype)
        if value.dtype != self.dtype:
            value = value.astype(self.dtype)

        if not self.is_built:
            self.build(int(query.shape[-1]))

        B, Q, _ = query.shape
        K_len = key.shape[1]

        q = self._linear(query, self.q_kernel, getattr(self, "q_bias", None))
        k = self._linear(key, self.k_kernel, getattr(self, "k_bias", None))
        v = self._linear(value, self.v_kernel, getattr(self, "v_bias", None))

        q = mx.reshape(q, (B, Q, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, K_len, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, K_len, self.num_heads, self.head_dim))

        if self._normalize_qk:
            q, _ = normalize_qk(q, q, epsilon=1e-6)  # normalize q (key reused below)
            k, _ = normalize_qk(k, k, epsilon=1e-6)

        alpha_val = None
        scale_val = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                scale_val = self._constant_alpha_value
            elif getattr(self, "alpha", None) is not None:
                alpha_val = self.alpha

        x = yat_attention(
            q, k, v,
            mask=mask,
            dropout_rate=self.dropout if training else 0.0,
            training=training,
            epsilon=self.epsilon,
            alpha=alpha_val,
            scale=scale_val,
            spherical=self._spherical,
        )

        x = mx.reshape(x, (B, Q, self.embed_dim))

        if self.use_out_proj and getattr(self, "out_kernel", None) is not None:
            x = self._linear(x, self.out_kernel, getattr(self, "out_bias", None))
        return x
