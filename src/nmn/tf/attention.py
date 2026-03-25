"""YAT Attention for TensorFlow.

Implements the YAT attention mechanism and MultiHeadYatAttention module:
    score(q, k) = (q.k)^2 / (||q - k||^2 + epsilon)

Supports learnable/constant alpha scaling, spherical mode, and QK normalization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import tensorflow as tf

__all__ = [
    "normalize_qk",
    "yat_attention_weights",
    "yat_attention",
    "yat_attention_normalized",
    "MultiHeadYatAttention",
]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def normalize_qk(
    query: tf.Tensor,
    key: tf.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """L2-normalizes query and key to unit vectors.

    When Q, K are unit vectors, ||q-k||^2 = 2(1 - q.k).
    """
    q_norm = tf.sqrt(tf.reduce_sum(tf.square(query), axis=-1, keepdims=True) + epsilon)
    k_norm = tf.sqrt(tf.reduce_sum(tf.square(key), axis=-1, keepdims=True) + epsilon)
    return query / q_norm, key / k_norm


def yat_attention_weights(
    query: tf.Tensor,
    key: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[tf.Tensor] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> tf.Tensor:
    """Computes YAT attention weights: softmax((Q.K)^2 / (||Q-K||^2 + eps))

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        mask: Optional boolean mask (batch, num_heads, q_len, kv_len).
            True = attend, False = mask out.
        dropout_rate: Dropout probability on attention weights.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha (simple multiply before softmax).
        scale: Optional constant scale factor (e.g. sqrt(2)).
        spherical: If True, normalize Q/K and use simplified formula.

    Returns:
        Attention weights (batch, num_heads, q_len, kv_len).
    """
    # Scale by 1/√head_dim to match standard attention's score growth profile.
    head_dim = query.shape[-1]
    dim_scale = tf.sqrt(tf.cast(head_dim, query.dtype))

    if spherical:
        query, key = normalize_qk(query, key, epsilon)
        dot_product = tf.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = tf.square(dot_product)
        # Clamp: bf16 rounding can make q·k > 1 after normalization
        distance_sq = tf.maximum(2.0 - 2.0 * dot_product, 0.0)
        attn_weights = squared_dot / ((distance_sq + epsilon) * dim_scale)
    else:
        dot_product = tf.einsum("bqhd,bkhd->bhqk", query, key)
        squared_dot = tf.square(dot_product)

        q_norm_sq = tf.reduce_sum(tf.square(query), axis=-1, keepdims=True)
        k_norm_sq = tf.reduce_sum(tf.square(key), axis=-1, keepdims=True)

        # Transpose to (batch, heads, seq, 1)
        q_norm_sq = tf.transpose(q_norm_sq, perm=[0, 2, 1, 3])
        k_norm_sq = tf.transpose(k_norm_sq, perm=[0, 2, 1, 3])
        k_norm_sq = tf.linalg.matrix_transpose(k_norm_sq)

        # Clamp to zero: bf16 cancellation can make distance negative when q ≈ k
        squared_dist = tf.maximum(q_norm_sq + k_norm_sq - 2.0 * dot_product, 0.0)
        attn_weights = squared_dot / ((squared_dist + epsilon) * dim_scale)

    # Alpha scaling (before softmax)
    if alpha is not None:
        attn_weights = attn_weights * alpha
    elif scale is not None:
        attn_weights = attn_weights * scale

    # Mask
    if mask is not None:
        attn_weights = tf.where(mask, attn_weights, tf.constant(-1e9, dtype=attn_weights.dtype))

    # Softmax
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)

    # Dropout
    if dropout_rate > 0.0 and training:
        attn_weights = tf.nn.dropout(attn_weights, rate=dropout_rate)

    return attn_weights


def yat_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[tf.Tensor] = None,
    scale: Optional[float] = None,
    spherical: bool = False,
) -> tf.Tensor:
    """Computes YAT attention: softmax((Q.K)^2 / (||Q-K||^2 + eps)) . V

    Args:
        query: (batch, q_len, num_heads, head_dim)
        key: (batch, kv_len, num_heads, head_dim)
        value: (batch, kv_len, num_heads, v_dim)
        mask: Optional boolean mask (batch, num_heads, q_len, kv_len).
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
    return tf.einsum("bhqk,bkhd->bqhd", weights, value)


def yat_attention_normalized(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: Optional[tf.Tensor] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[tf.Tensor] = None,
    scale: Optional[float] = None,
) -> tf.Tensor:
    """YAT attention with normalized Q/K (optimized).

    Normalizes Q, K to unit vectors, then: (q.k)^2 / (2(1-q.k) + eps)
    Only needs ONE dot product.

    Args:
        query: (batch, q_len, num_heads, head_dim) — will be normalized.
        key: (batch, kv_len, num_heads, head_dim) — will be normalized.
        value: (batch, kv_len, num_heads, v_dim)
        mask: Optional boolean mask.
        dropout_rate: Dropout probability.
        training: Whether in training mode.
        epsilon: Numerical stability constant.
        alpha: Optional learnable alpha.
        scale: Optional constant scale factor.

    Returns:
        Output (batch, q_len, num_heads, v_dim).
    """
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


class MultiHeadYatAttention(tf.Module):
    """Multi-head attention using the YAT formula (tf.Module).

    Architecture:
        Input -> Linear(Q) -|
        Input -> Linear(K) -+-> YAT_attention(Q, K, V) -> Linear(out) -> Output
        Input -> Linear(V) -|

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability (default: 0.0).
        use_bias: Whether to use bias in projections (default: True).
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
            If None, use learnable alpha.
        normalize_qk: Whether to L2-normalize Q and K (default: False).
        spherical: If True, use spherical YAT formula (default: False).
        use_out_proj: Whether to use output projection (default: True).
        epsilon: Numerical stability constant (default: 1e-5).
        dtype: TensorFlow dtype (default: tf.float32).
        name: Optional name scope.
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
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

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

        # Alpha configuration
        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
            self.alpha = None
        elif use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=dtype), trainable=True, name="alpha"
            )
        else:
            self.alpha = None
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        # Projection weights will be created lazily
        self.is_built = False

    def build(self, input_dim: int) -> None:
        """Creates projection weights."""
        if self.is_built:
            return

        std = math.sqrt(2.0 / (input_dim + self.embed_dim))
        factory = lambda shape: tf.Variable(
            tf.random.normal(shape, stddev=std, dtype=self.dtype),
            trainable=True,
        )

        self.q_kernel = factory((input_dim, self.embed_dim))
        self.k_kernel = factory((input_dim, self.embed_dim))
        self.v_kernel = factory((input_dim, self.embed_dim))

        if self.use_bias:
            bias_factory = lambda: tf.Variable(
                tf.zeros([self.embed_dim], dtype=self.dtype), trainable=True
            )
            self.q_bias = bias_factory()
            self.k_bias = bias_factory()
            self.v_bias = bias_factory()
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if self.use_out_proj:
            self.out_kernel = factory((self.embed_dim, self.embed_dim))
            self.out_bias = (
                tf.Variable(
                    tf.zeros([self.embed_dim], dtype=self.dtype), trainable=True
                )
                if self.use_bias
                else None
            )
        else:
            self.out_kernel = None
            self.out_bias = None

        self.is_built = True

    def _linear(self, x, kernel, bias):
        y = tf.matmul(x, kernel)
        if bias is not None:
            y = y + bias
        return y

    @tf.Module.with_name_scope
    def __call__(
        self,
        query: tf.Tensor,
        key: Optional[tf.Tensor] = None,
        value: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
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

        query = tf.cast(query, self.dtype)
        key = tf.cast(key, self.dtype)
        value = tf.cast(value, self.dtype)

        if not self.is_built:
            self.build(int(query.shape[-1]))

        batch_size = tf.shape(query)[0]
        q_len = tf.shape(query)[1]
        kv_len = tf.shape(key)[1]

        # Project
        q = self._linear(query, self.q_kernel, self.q_bias)
        k = self._linear(key, self.k_kernel, self.k_bias)
        v = self._linear(value, self.v_kernel, self.v_bias)

        # Reshape to multi-head: (B, L, E) -> (B, L, H, D)
        q = tf.reshape(q, (batch_size, q_len, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, kv_len, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, kv_len, self.num_heads, self.head_dim))

        # QK normalization
        if self._normalize_qk:
            q = tf.math.l2_normalize(q, axis=-1)
            k = tf.math.l2_normalize(k, axis=-1)

        # Get alpha
        alpha_val = None
        scale_val = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                scale_val = self._constant_alpha_value
            elif self.alpha is not None:
                alpha_val = self.alpha

        # Apply YAT attention
        dropout_rate = self.dropout if training else 0.0
        x = yat_attention(
            q, k, v,
            mask=mask,
            dropout_rate=dropout_rate,
            training=training,
            epsilon=self.epsilon,
            alpha=alpha_val,
            scale=scale_val,
            spherical=self._spherical,
        )

        # Reshape back: (B, Q, H, D) -> (B, Q, E)
        x = tf.reshape(x, (batch_size, q_len, self.embed_dim))

        # Output projection
        if self.out_kernel is not None:
            x = self._linear(x, self.out_kernel, self.out_bias)

        return x
