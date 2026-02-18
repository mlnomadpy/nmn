"""Multi-Head Attention Module.

This module implements the MultiHeadAttention class, which provides
a flexible multi-head attention mechanism that can use either:
- YAT attention (default): softmax((Q·K)² / (||Q-K||² + ε)) · V
- Standard scaled dot-product attention: softmax(Q·K / sqrt(d_k)) · V

The architecture uses:
- Linear projections for Q, K, V
- Configurable attention mechanism (YAT or standard)
- Optional QK normalization for training stability
- Support for autoregressive (cached) decoding
- Optional alpha scaling for YAT attention (learnable or constant)
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Union

import jax.numpy as jnp
from jax import lax

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import LinearGeneral, default_kernel_init
from flax.nnx.nn.normalization import LayerNorm
from flax.typing import (
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
)
from jax import Array

from .yat_attention import yat_attention
from .masks import combine_masks

# Default constant alpha value (sqrt(2)), same as NMN
DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)


class MultiHeadAttention(Module):
    """Multi-head attention with YAT or standard dot-product attention.

    This layer projects the inputs into multi-headed query, key, and value
    vectors, applies attention (YAT by default), and reshapes the output.

    Architecture:
        Input → Linear(Q) ─┐
        Input → Linear(K) ─┼→ attention(Q, K, V) → Output
        Input → Linear(V) ─┘

    YAT attention computes: softmax((Q·K)² / (||Q-K||² + ε)) · V
    Standard attention computes: softmax(Q·K / sqrt(d_k)) · V

    With optional alpha scaling (for YAT attention):
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Attributes:
        num_heads: Number of attention heads.
        in_features: Input feature dimension.
        qkv_features: Dimension of Q, K, V projections.
        out_features: Output dimension (same as qkv_features by default).
        head_dim: Dimension per head (qkv_features // num_heads).
        epsilon: Numerical stability constant for YAT attention.
        use_softermax: Whether to use softermax instead of softmax.
        power: Power parameter for softermax.
        use_alpha: Whether alpha scaling is enabled.
        alpha: Learnable alpha parameter (if use_alpha=True and constant_alpha=None).

    Example:
        >>> rngs = nnx.Rngs(0)
        >>> # Learnable alpha (default)
        >>> attn = MultiHeadAttention(
        ...     num_heads=8,
        ...     in_features=512,
        ...     rngs=rngs,
        ...     decode=False,
        ... )
        >>> # Constant alpha = sqrt(2)
        >>> attn = MultiHeadAttention(
        ...     num_heads=8,
        ...     in_features=512,
        ...     constant_alpha=True,
        ...     rngs=rngs,
        ...     decode=False,
        ... )
        >>> # No alpha scaling
        >>> attn = MultiHeadAttention(
        ...     num_heads=8,
        ...     in_features=512,
        ...     use_alpha=False,
        ...     rngs=rngs,
        ...     decode=False,
        ... )
        >>> x = jnp.zeros((2, 10, 512))  # (batch, seq_len, features)
        >>> output = attn(x, deterministic=True)
        >>> output.shape
        (2, 10, 512)
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        *,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        out_kernel_init: Initializer | None = None,
        bias_init: Initializer = initializers.zeros_init(),
        out_bias_init: Initializer | None = None,
        use_bias: bool = True,
        attention_fn: Callable[..., Array] = yat_attention,
        decode: bool | None = None,
        normalize_qk: bool = False,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        alpha_init: Initializer = initializers.ones_init(),
        use_dropconnect: bool = False,
        dropconnect_rate: float = 0.0,
        qkv_dot_general: DotGeneralT | None = None,
        out_dot_general: DotGeneralT | None = None,
        qkv_dot_general_cls: Any = None,
        out_dot_general_cls: Any = None,
        rngs: rnglib.Rngs,
        epsilon: float = 1e-5,
        use_softermax: bool = False,
        power: float = 1.0,
    ):
        """Initializes the MultiHeadAttention module.

        Args:
            num_heads: Number of attention heads.
            in_features: Input feature dimension.
            qkv_features: Dimension of Q, K, V projections (default: in_features).
            out_features: Output dimension (default: in_features).
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            broadcast_dropout: Whether to broadcast dropout across batch dims.
            dropout_rate: Attention dropout probability.
            deterministic: If True, no dropout is applied.
            precision: JAX precision for matrix operations.
            kernel_init: Initializer for Q, K, V projection kernels.
            out_kernel_init: Initializer for output projection kernel.
            bias_init: Initializer for biases.
            out_bias_init: Initializer for output projection bias.
            use_bias: Whether to use bias in projections.
            attention_fn: Attention function to use (default: yat_attention).
            decode: Whether to use autoregressive decoding mode.
            normalize_qk: Whether to apply layer norm to Q and K.
            use_alpha: Whether to use alpha scaling for YAT attention. Ignored if
                constant_alpha is set.
            constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
                use that value. If None (default), use learnable alpha when
                use_alpha=True.
            alpha_init: Initializer for learnable alpha (only used if use_alpha=True
                and constant_alpha=None).
            use_dropconnect: Whether to use DropConnect (for training).
            dropconnect_rate: DropConnect probability.
            qkv_dot_general: (Deprecated).
            out_dot_general: (Deprecated).
            qkv_dot_general_cls: (Deprecated).
            out_dot_general_cls: (Deprecated).
            rngs: Random number generator container.
            epsilon: Numerical stability constant for YAT attention.
            use_softermax: Whether to use softermax instead of softmax.
            power: Power parameter for softermax.
        """
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features if qkv_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.deterministic = deterministic
        self.precision = precision
        self.kernel_init = kernel_init
        self.out_kernel_init = out_kernel_init
        self.bias_init = bias_init
        self.out_bias_init = out_bias_init
        self.use_bias = use_bias
        self.attention_fn = attention_fn
        self.decode = decode
        self.normalize_qk = normalize_qk
        self.qkv_dot_general = qkv_dot_general
        self.out_dot_general = out_dot_general
        self.qkv_dot_general_cls = qkv_dot_general_cls
        self.out_dot_general_cls = out_dot_general_cls
        self.epsilon = epsilon
        self.use_softermax = use_softermax
        self.power = power
        self.use_dropconnect = use_dropconnect
        self.dropconnect_rate = dropconnect_rate

        # Handle alpha configuration (same logic as YatNMN)
        # Priority: constant_alpha > use_alpha
        #
        # Options:
        #   1. constant_alpha=True -> use sqrt(2) as constant
        #   2. constant_alpha=<float> -> use that value as constant
        #   3. use_alpha=True (default) -> learnable alpha parameter
        #   4. use_alpha=False -> no alpha scaling
        self.alpha: nnx.Param[Array] | None
        
        if constant_alpha is not None:
            # Use constant alpha (no learnable parameter)
            if constant_alpha is True:
                self._constant_alpha_value = float(DEFAULT_CONSTANT_ALPHA)
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.alpha = None
            use_alpha = True  # Alpha scaling is enabled (but constant)
        else:
            self._constant_alpha_value = None
            if use_alpha:
                # Use learnable alpha
                alpha_key = rngs.params()
                self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
            else:
                # No alpha scaling
                self.alpha = None
        
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha
        self.alpha_init = alpha_init

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.qkv_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.qkv_features // self.num_heads

        # Use standard linear projections for Q, K, V
        # The YAT mechanism is applied in the attention computation, not projections
        linear = functools.partial(
            LinearGeneral,
            in_features=self.in_features,
            out_features=self.qkv_features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
        )

        # Create Q, K, V projection layers
        self.query = linear(rngs=rngs)
        self.key = linear(rngs=rngs)
        self.value = linear(rngs=rngs)

        # Optional QK normalization (ViT-22B style)
        self.query_ln: LayerNorm | None
        self.key_ln: LayerNorm | None
        if self.normalize_qk:
            self.query_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
            self.key_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                rngs=rngs,
            )
        else:
            self.query_ln = None
            self.key_ln = None

        # Autoregressive decoding cache
        self.cached_key: nnx.Cache[Array] | None = None
        self.cached_value: nnx.Cache[Array] | None = None
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        *,
        mask: Array | None = None,
        deterministic: bool | None = None,
        rngs: rnglib.Rngs | None = None,
        sow_weights: bool = False,
        decode: bool | None = None,
    ) -> Array:
        """Applies multi-head attention on the input data.

        Projects inputs into Q, K, V, applies attention, and returns output.

        Self-attention: Pass only inputs_q (inputs_k and inputs_v will copy it).
        Cross-attention: Pass inputs_q as queries, inputs_k as keys/values.

        Args:
            inputs_q: Query input of shape [batch..., length, features].
            inputs_k: Key input. If None, copies inputs_q.
            inputs_v: Value input. If None, copies inputs_k.
            mask: Attention mask of shape [batch..., num_heads, q_len, kv_len].
                False values are masked out.
            deterministic: If True, no dropout is applied.
            rngs: Random number generators for dropout.
            sow_weights: If True, sow attention weights for introspection.
            decode: If True, use autoregressive decoding mode.

        Returns:
            Output of shape [batch..., length, qkv_features].
        """
        # Handle self-attention and cross-attention
        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    "`inputs_k` cannot be None if `inputs_v` is not None. "
                    "Pass the value to `inputs_k` and leave `inputs_v` as None."
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible input dimension, got {inputs_q.shape[-1]} "
                f"but module expects {self.in_features}."
            )

        # Determine if we should use deterministic mode
        is_deterministic: bool = False
        if self.dropout_rate > 0.0 or (
            self.use_dropconnect and self.dropconnect_rate > 0.0
        ):
            is_deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg=(
                    "No `deterministic` argument was provided to MultiHeadAttention "
                    "as either a __call__ argument, class attribute, or nnx.flag."
                ),
            )
        else:
            is_deterministic = True

        # Apply linear projections
        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        # Reshape to multi-head format: [batch..., length, num_heads, head_dim]
        query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))
        key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_dim))
        value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_dim))

        # Optional QK normalization (stabilizes training with higher LR)
        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            query = self.query_ln(query)
            key = self.key_ln(key)

        # Handle autoregressive decoding
        decode = first_from(
            decode,
            self.decode,
            error_msg=(
                "No `decode` argument was provided to MultiHeadAttention "
                "as either a __call__ argument, class attribute, or nnx.flag."
            ),
        )

        if decode:
            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    "Autoregressive cache not initialized, call `init_cache` first."
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key.value.shape

            # Validate query shape
            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    f"Autoregressive cache shape error, "
                    f"expected query shape {expected_shape} instead got {query.shape}."
                )

            # Update cache
            cur_index = self.cache_index.value
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
            value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
            self.cached_key.value = key
            self.cached_value.value = value
            self.cache_index.value += 1

            # Causal mask for cached decoding
            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
            )

        # Get dropout RNG if needed
        dropout_rng = None
        if self.dropout_rate > 0.0 and not is_deterministic:
            if rngs is None:
                raise ValueError("'rngs' must be provided for dropout.")
            dropout_rng = rngs.dropout()

        # Get alpha value (either learnable or constant)
        # For constant alpha, we apply it directly after the attention call
        # rather than passing it to the attention function (which uses it as an exponent)
        alpha_value = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                # Will be applied as direct scale after attention
                pass
            elif self.alpha is not None:
                alpha_value = self.alpha.value

        # Apply attention (YAT by default)
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=is_deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
            epsilon=self.epsilon,
            use_softermax=self.use_softermax,
            power=self.power,
            alpha=alpha_value,
        )

        # Apply constant alpha as direct scale (e.g. sqrt(2))
        if self._constant_alpha_value is not None:
            x = x * self._constant_alpha_value

        # Reshape back: [batch..., length, num_heads, head_dim] -> [batch..., length, qkv_features]
        x = x.reshape(x.shape[:-2] + (self.qkv_features,))
        return x

    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32):
        """Initializes the cache for autoregressive decoding.

        Args:
            input_shape: Shape of the input, used to determine cache dimensions.
            dtype: Data type for the cache arrays.
        """
        cache_shape = (*input_shape[:-1], self.num_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))



