"""YAT Attention for Keras (backend-agnostic).

Implements the YAT attention mechanism and MultiHeadYatAttention layer:
    score(q, k) = (q.k)^2 / (||q - k||^2 + epsilon)

Supports learnable/constant alpha scaling, spherical mode, and QK normalization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

from keras.src import initializers, ops
from keras.src.layers.layer import Layer

__all__ = [
    "normalize_qk",
    "yat_attention_weights",
    "yat_attention",
    "yat_attention_normalized",
    "MultiHeadYatAttention",
]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def normalize_qk(query, key, epsilon: float = 1e-6):
    """L2-normalizes query and key to unit vectors."""
    q_norm = ops.sqrt(ops.sum(ops.square(query), axis=-1, keepdims=True) + epsilon)
    k_norm = ops.sqrt(ops.sum(ops.square(key), axis=-1, keepdims=True) + epsilon)
    return query / q_norm, key / k_norm


def yat_attention_weights(
    query,
    key,
    mask=None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha=None,
    scale: Optional[float] = None,
    spherical: bool = False,
):
    """Computes YAT attention weights: softmax((Q.K)^2 / (||Q-K||^2 + eps))

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        mask: Optional boolean mask (batch, num_heads, q_len, kv_len).
            True = attend, False = mask out.
        dropout_rate: Dropout probability.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha (multiply before softmax).
        scale: Optional constant scale factor.
        spherical: If True, normalize Q/K and use simplified formula.

    Returns:
        Attention weights (batch, num_heads, q_len, kv_len).
    """
    # Scale by 1/√head_dim to match standard attention's score growth profile.
    head_dim = query.shape[-1]
    dim_scale = ops.sqrt(ops.cast(head_dim, query.dtype))

    if spherical:
        query, key = normalize_qk(query, key, epsilon)
        dot_product = ops.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = ops.square(dot_product)
        # Clamp: bf16 rounding can make q·k > 1 after normalization
        distance_sq = ops.maximum(2.0 - 2.0 * dot_product, 0.0)
        attn_weights = squared_dot / ((distance_sq + epsilon) * dim_scale)
    else:
        dot_product = ops.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = ops.square(dot_product)

        q_norm_sq = ops.sum(ops.square(query), axis=-1, keepdims=True)
        k_norm_sq = ops.sum(ops.square(key), axis=-1, keepdims=True)

        q_norm_sq = ops.transpose(q_norm_sq, axes=[0, 2, 1, 3])
        k_norm_sq = ops.transpose(k_norm_sq, axes=[0, 2, 1, 3])
        k_norm_sq = ops.swapaxes(k_norm_sq, -2, -1)

        # Clamp to zero: bf16 cancellation can make distance negative when q ≈ k
        squared_dist = ops.maximum(q_norm_sq + k_norm_sq - 2.0 * dot_product, 0.0)
        attn_weights = squared_dot / ((squared_dist + epsilon) * dim_scale)

    if alpha is not None:
        attn_weights = attn_weights * alpha
    elif scale is not None:
        attn_weights = attn_weights * scale

    if mask is not None:
        attn_weights = ops.where(mask, attn_weights, -1e9)

    attn_weights = ops.softmax(attn_weights, axis=-1)

    if dropout_rate > 0.0 and training:
        from keras.src import random
        keep = random.dropout(
            ops.ones_like(attn_weights), rate=dropout_rate
        )
        attn_weights = attn_weights * keep

    return attn_weights


def yat_attention(
    query,
    key,
    value,
    mask=None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha=None,
    scale: Optional[float] = None,
    spherical: bool = False,
):
    """Computes YAT attention: softmax((Q.K)^2 / (||Q-K||^2 + eps)) . V

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        value: (batch, kv_len, num_heads, v_dim)
        mask: Optional boolean mask.
        dropout_rate: Dropout probability.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha.
        scale: Optional constant scale factor.
        spherical: If True, normalize Q/K first.

    Returns:
        Output (batch, q_len, num_heads, v_dim).
    """
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
    return ops.einsum("bhqk,bkhd->bqhd", weights, value)


def yat_attention_normalized(
    query,
    key,
    value,
    mask=None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha=None,
    scale: Optional[float] = None,
):
    """YAT attention with normalized Q/K (optimized)."""
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


class MultiHeadYatAttention(Layer):
    """Multi-head attention using the YAT formula (Keras Layer).

    Architecture:
        Input -> Dense(Q) -|
        Input -> Dense(K) -+-> YAT_attention(Q, K, V) -> Dense(out) -> Output
        Input -> Dense(V) -|

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability (default: 0.0).
        use_bias: Whether to use bias in projections (default: True).
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
        normalize_qk: Whether to L2-normalize Q and K (default: False).
        spherical: If True, use spherical YAT formula (default: False).
        use_out_proj: Whether to use output projection (default: True).
        epsilon: Numerical stability constant (default: 1e-5).
        kernel_initializer: Initializer for projection kernels.
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
        kernel_initializer: str = "glorot_normal",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.epsilon = epsilon
        self.use_out_proj = use_out_proj
        self._normalize_qk = normalize_qk
        self._spherical = spherical
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    def build(self, input_shape):
        input_dim = input_shape[-1]

        def _make_kernel(name):
            return self.add_weight(
                name=name,
                shape=(input_dim, self.embed_dim),
                initializer=self.kernel_initializer,
                trainable=True,
            )

        def _make_bias(name):
            return self.add_weight(
                name=name,
                shape=(self.embed_dim,),
                initializer="zeros",
                trainable=True,
            ) if self.use_bias else None

        self.q_kernel = _make_kernel("q_kernel")
        self.k_kernel = _make_kernel("k_kernel")
        self.v_kernel = _make_kernel("v_kernel")
        self.q_bias = _make_bias("q_bias")
        self.k_bias = _make_bias("k_bias")
        self.v_bias = _make_bias("v_bias")

        if self.use_out_proj:
            self.out_kernel = _make_kernel("out_kernel")
            self.out_bias = _make_bias("out_bias")
        else:
            self.out_kernel = None
            self.out_bias = None

        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha_param = self.add_weight(
                name="alpha", shape=(1,), initializer="ones", trainable=True
            )
        else:
            self.alpha_param = None

        self.built = True

    def _linear(self, x, kernel, bias):
        y = ops.matmul(x, kernel)
        if bias is not None:
            y = y + bias
        return y

    def call(
        self,
        query,
        key=None,
        value=None,
        mask=None,
        training=False,
    ):
        """Applies multi-head YAT attention.

        Args:
            query: (batch, q_len, embed_dim)
            key: (batch, kv_len, embed_dim). Defaults to query.
            value: (batch, kv_len, embed_dim). Defaults to key.
            mask: Boolean mask (batch, num_heads, q_len, kv_len).
            training: Whether in training mode.

        Returns:
            Output (batch, q_len, embed_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size = ops.shape(query)[0]
        q_len = ops.shape(query)[1]
        kv_len = ops.shape(key)[1]

        q = self._linear(query, self.q_kernel, self.q_bias)
        k = self._linear(key, self.k_kernel, self.k_bias)
        v = self._linear(value, self.v_kernel, self.v_bias)

        q = ops.reshape(q, (batch_size, q_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, kv_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, kv_len, self.num_heads, self.head_dim))

        if self._normalize_qk:
            q, k = normalize_qk(q, k)

        alpha_val = None
        scale_val = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                scale_val = self._constant_alpha_value
            elif self.alpha_param is not None:
                alpha_val = self.alpha_param

        dropout = self.dropout_rate if training else 0.0
        x = yat_attention(
            q, k, v,
            mask=mask,
            dropout_rate=dropout,
            training=training,
            epsilon=self.epsilon,
            alpha=alpha_val,
            scale=scale_val,
            spherical=self._spherical,
        )

        x = ops.reshape(x, (batch_size, q_len, self.embed_dim))

        if self.out_kernel is not None:
            x = self._linear(x, self.out_kernel, self.out_bias)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.embed_dim,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "use_bias": self.use_bias,
                "use_alpha": self.use_alpha,
                "constant_alpha": self.constant_alpha,
                "normalize_qk": self._normalize_qk,
                "spherical": self._spherical,
                "use_out_proj": self.use_out_proj,
                "epsilon": self.epsilon,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
