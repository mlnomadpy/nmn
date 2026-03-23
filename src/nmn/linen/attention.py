"""YAT Attention for Flax Linen.

Implements MultiHeadAttention module using the YAT formula:
    score(q, k) = (q.k)^2 / (||q - k||^2 + epsilon)

Attention functions are re-exported from NNX (both use JAX).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

import jax.numpy as jnp
from jax import Array

from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact

# Re-export attention functions from NNX (same JAX backend)
from nmn.nnx.layers.attention.yat_attention import (
    normalize_qk,
    yat_attention_weights as _nnx_yat_attention_weights,
    yat_attention as _nnx_yat_attention,
    yat_attention_normalized as _nnx_yat_attention_normalized,
)

__all__ = [
    "normalize_qk",
    "yat_attention_weights",
    "yat_attention",
    "yat_attention_normalized",
    "MultiHeadAttention",
]

DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def yat_attention_weights(
    query: Array,
    key: Array,
    mask: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Array] = None,
    spherical: bool = False,
) -> Array:
    """Computes YAT attention weights.

    Thin wrapper around NNX implementation with a simpler signature
    suitable for Linen usage.

    Args:
        query: (..., q_len, num_heads, head_dim)
        key: (..., kv_len, num_heads, head_dim)
        mask: Optional boolean mask (..., num_heads, q_len, kv_len).
        dropout_rate: Dropout probability (requires deterministic=False).
        deterministic: If True, no dropout.
        epsilon: Numerical stability constant.
        alpha: Optional alpha scaling parameter.
        spherical: If True, normalize Q/K first.

    Returns:
        Attention weights (..., num_heads, q_len, kv_len).
    """
    if spherical:
        query, key = normalize_qk(query, key, epsilon)

    return _nnx_yat_attention_weights(
        query, key,
        mask=mask,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        epsilon=epsilon,
        alpha=alpha,
    )


def yat_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Array] = None,
    spherical: bool = False,
) -> Array:
    """Computes YAT attention: softmax((Q.K)^2 / (||Q-K||^2 + eps)) . V

    Args:
        query: (..., q_len, num_heads, head_dim)
        key: (..., kv_len, num_heads, head_dim)
        value: (..., kv_len, num_heads, v_dim)
        mask: Optional boolean mask.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        epsilon: Numerical stability constant.
        alpha: Optional alpha scaling.
        spherical: If True, normalize Q/K first.

    Returns:
        Output (..., q_len, num_heads, v_dim).
    """
    if spherical:
        query, key = normalize_qk(query, key, epsilon)

    return _nnx_yat_attention(
        query, key, value,
        mask=mask,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        epsilon=epsilon,
        alpha=alpha,
    )


def yat_attention_normalized(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    epsilon: float = 1e-5,
    alpha: Optional[Array] = None,
) -> Array:
    """YAT attention with normalized Q/K (optimized)."""
    return yat_attention(
        query, key, value,
        mask=mask,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        epsilon=epsilon,
        alpha=alpha,
        spherical=True,
    )


class MultiHeadAttention(Module):
    """Multi-head attention using the YAT formula (Flax Linen).

    Architecture:
        Input -> Dense(Q) -|
        Input -> Dense(K) -+-> YAT_attention(Q, K, V) -> Dense(out) -> Output
        Input -> Dense(V) -|

    Attributes:
        num_heads: Number of attention heads.
        qkv_features: Dimension of Q/K/V projections (default: input features).
        out_features: Output dimension (default: qkv_features).
        dropout_rate: Attention dropout probability (default: 0.0).
        use_bias: Whether to use bias in projections (default: True).
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2). If float, use that value.
        normalize_qk: Whether to L2-normalize Q and K (default: False).
        spherical: If True, use spherical YAT formula (default: False).
        epsilon: Numerical stability constant (default: 1e-5).
        dtype: Computation dtype.
        param_dtype: Parameter dtype (default: float32).
        kernel_init: Initializer for projection kernels.
        bias_init: Initializer for biases.
    """

    num_heads: int = 1
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    use_alpha: bool = True
    constant_alpha: Optional[Any] = None
    normalize_qk: bool = False
    spherical: bool = False
    epsilon: float = 1e-5
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.xavier_normal()
    bias_init: Any = nn.initializers.zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Optional[Array] = None,
        inputs_v: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = True,
    ) -> Array:
        """Applies multi-head YAT attention.

        Args:
            inputs_q: (batch, q_len, features)
            inputs_k: (batch, kv_len, features). Defaults to inputs_q.
            inputs_v: (batch, kv_len, features). Defaults to inputs_k.
            mask: Boolean mask (batch, num_heads, q_len, kv_len).
            deterministic: If True, no dropout.

        Returns:
            Output (batch, q_len, out_features).
        """
        if inputs_k is None:
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        in_features = inputs_q.shape[-1]
        qkv_features = self.qkv_features or in_features
        out_features = self.out_features or qkv_features

        if qkv_features % self.num_heads != 0:
            raise ValueError(
                f"qkv_features ({qkv_features}) must be divisible by "
                f"num_heads ({self.num_heads})."
            )
        head_dim = qkv_features // self.num_heads

        # Q/K/V projections using Linen Dense
        q = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="query",
        )(inputs_q)
        k = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="key",
        )(inputs_k)
        v = nn.Dense(
            qkv_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="value",
        )(inputs_v)

        # Reshape to multi-head: (B, L, E) -> (B, L, H, D)
        q = q.reshape(q.shape[:-1] + (self.num_heads, head_dim))
        k = k.reshape(k.shape[:-1] + (self.num_heads, head_dim))
        v = v.reshape(v.shape[:-1] + (self.num_heads, head_dim))

        # QK normalization
        if self.normalize_qk:
            q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
            k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)

        # Alpha
        _constant_alpha_value = None
        alpha_val = None
        if self.constant_alpha is not None:
            if self.constant_alpha is True:
                _constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                _constant_alpha_value = float(self.constant_alpha)
        elif self.use_alpha:
            alpha_val = self.param("alpha", self.alpha_init, (1,), self.param_dtype)

        q, k, v, alpha_val = promote_dtype(q, k, v, alpha_val, dtype=self.dtype)

        # YAT attention (using NNX functions under the hood)
        x = _nnx_yat_attention(
            q, k, v,
            mask=mask,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            epsilon=self.epsilon,
            alpha=alpha_val,
        )

        # Apply constant alpha
        if _constant_alpha_value is not None:
            x = x * _constant_alpha_value

        # Reshape back: (B, Q, H, D) -> (B, Q, E)
        x = x.reshape(x.shape[:-2] + (qkv_features,))

        # Output projection
        x = nn.Dense(
            out_features,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="out",
        )(x)

        return x
