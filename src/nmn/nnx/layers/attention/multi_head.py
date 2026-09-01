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

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import LinearGeneral, default_kernel_init
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    PrecisionLike,
    Shape,
)
from jax import Array, lax

from .._numerics import fp32_if_low_precision, inverse_softplus
from .masks import combine_masks
from .yat_attention import yat_attention

# Default constant alpha value (sqrt(2)), same as NMN
DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)
_L2_NORMALIZE_EPSILON = 1e-12


def _l2_normalize_per_head(x: Array) -> Array:
    """Match ``torch.nn.functional.normalize(..., p=2, eps=1e-12)``."""
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    epsilon = jnp.asarray(_L2_NORMALIZE_EPSILON, dtype=norm.dtype)
    return x / jnp.maximum(norm, epsilon)


def _raise_cache_overflow(index: Array, max_length: int) -> None:
    if int(index) >= max_length:
        raise ValueError(f"Autoregressive cache is full at length {max_length}.")


def _check_cache_capacity(index: Array, max_length: int) -> None:
    """Fail before a cache write in eager mode and under ``nnx.jit``."""
    if isinstance(index, jax.core.Tracer):
        jax.debug.callback(
            functools.partial(_raise_cache_overflow, max_length=max_length),
            index,
            ordered=True,
        )
    else:
        _raise_cache_overflow(index, max_length)


def _validate_decode_mask(mask: Array | None, expected_shape: tuple[int, ...]) -> None:
    if mask is None:
        return
    try:
        broadcast_shape = jnp.broadcast_shapes(tuple(mask.shape), expected_shape)
    except ValueError as exc:
        raise ValueError(
            f"Decode mask shape {mask.shape} is not broadcastable to "
            f"{expected_shape}."
        ) from exc
    if broadcast_shape != expected_shape:
        raise ValueError(
            f"Decode mask shape {mask.shape} would expand attention shape "
            f"{expected_shape} to {broadcast_shape}."
        )


def _linear_general_with_kernel(
    layer: LinearGeneral, inputs: Array, kernel: Array
) -> Array:
    """Apply ``LinearGeneral`` with an ephemeral DropConnect kernel."""
    ndim = inputs.ndim
    axis = tuple(ax if ax >= 0 else ndim + ax for ax in layer.axis)
    batch_axes = tuple(ax if ax >= 0 else ndim + ax for ax in layer.batch_axis.keys())
    n_batch_dims = len(batch_axes)
    expanded_batch_shape = tuple(
        inputs.shape[ax] if ax in batch_axes else 1
        for ax in range(ndim)
        if ax not in axis
    )
    bias = layer.bias[...] if layer.bias is not None else None
    inputs, kernel, bias = layer.promote_dtype(
        (inputs, kernel, bias), dtype=layer.dtype
    )
    dot_general = layer.dot_general or lax.dot_general
    if layer.dot_general_cls is not None:
        dot_general = layer.dot_general_cls()
    dot_general_kwargs = {"out_sharding": None}
    if layer.preferred_element_type is not None:
        dot_general_kwargs["preferred_element_type"] = layer.preferred_element_type
    output = dot_general(
        inputs,
        kernel,
        (
            (axis, tuple(range(n_batch_dims, len(axis) + n_batch_dims))),
            (batch_axes, tuple(range(n_batch_dims))),
        ),
        precision=layer.precision,
        **dot_general_kwargs,
    )
    if bias is not None:
        output += jnp.reshape(bias, (*expanded_batch_shape, *layer.out_features))
    return output


def _dropconnect(kernel: Array, key: Array, rate: float) -> Array:
    keep_probability = 1.0 - rate
    mask = jax.random.bernoulli(key, keep_probability, kernel.shape)
    return lax.select(mask, kernel / keep_probability, jnp.zeros_like(kernel))


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
        out_features: Output dimension (same as in_features by default).
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
        learnable_epsilon: bool = False,
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
            normalize_qk: Whether to L2-normalize Q and K per head.
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
        # Ordinary calls are non-decoding unless explicitly requested.  This
        # keeps ``decode`` optional as documented while preserving the call-time
        # override used by autoregressive clients.
        self.decode = False if decode is None else decode
        self.normalize_qk = normalize_qk
        self.qkv_dot_general = qkv_dot_general
        self.out_dot_general = out_dot_general
        self.qkv_dot_general_cls = qkv_dot_general_cls
        self.out_dot_general_cls = out_dot_general_cls
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.epsilon_param: nnx.Param[Array] | None
        if learnable_epsilon:
            self.epsilon_param = nnx.Param(inverse_softplus(epsilon, param_dtype))
        else:
            self.epsilon_param = None
        self.use_softermax = use_softermax
        self.power = power
        self.use_dropconnect = use_dropconnect
        self.dropconnect_rate = dropconnect_rate
        if not 0.0 <= dropconnect_rate < 1.0:
            raise ValueError(
                "dropconnect_rate must be in the half-open interval [0, 1), "
                f"got {dropconnect_rate}"
            )
        self.dropconnect_rng = (
            rngs.dropout.fork() if use_dropconnect else nnx.data(None)
        )

        # Handle alpha configuration (same logic as YatNMN)
        # Priority: constant_alpha > use_alpha
        #
        # Options:
        #   1. constant_alpha=True -> use sqrt(2) as constant
        #   2. constant_alpha=<float> -> use that value as constant
        #   3. use_alpha=True (default) -> learnable alpha parameter
        #   4. use_alpha=False -> no alpha scaling
        self.alpha: nnx.Param[Array] | None

        if constant_alpha is not None and constant_alpha is not False:
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

        # Project the headed attention result back to the requested output
        # dimension.  Keeping the head axes intact matches Flax's
        # MultiHeadAttention parameter layout and avoids an unnecessary reshape.
        self.out = LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            rngs=rngs,
        )

        # normalize_qk has the same per-head L2 semantics as the other NMN
        # backends.  Retain data placeholders for stable NNX graph structure.
        self.query_ln = nnx.data(None)
        self.key_ln = nnx.data(None)

        # Autoregressive decoding cache
        self.cached_key: nnx.Cache[Array] | None = nnx.data(None)
        self.cached_value: nnx.Cache[Array] | None = nnx.data(None)
        self.cache_index: nnx.Cache[Array] | None = nnx.data(None)

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
            Output of shape [batch..., length, out_features].
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

        # Resolve and validate decoding before any stochastic projection.  A
        # rejected call must not advance DropConnect streams or mutate caches.
        decode = first_from(
            decode,
            self.decode,
            error_msg=(
                "No `decode` argument was provided to MultiHeadAttention "
                "as either a __call__ argument, class attribute, or nnx.flag."
            ),
        )
        decode_context = None
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
            ) = self.cached_key[...].shape
            expected_input_shape = tuple(batch_dims) + (1, self.in_features)
            for name, tensor in (
                ("query", inputs_q),
                ("key", inputs_k),
                ("value", inputs_v),
            ):
                if expected_input_shape != tensor.shape:
                    raise ValueError(
                        f"Autoregressive cache shape error, expected {name} input "
                        f"shape {expected_input_shape} instead got {tensor.shape}."
                    )
            decode_mask_shape = tuple(batch_dims) + (
                self.num_heads,
                1,
                max_length,
            )
            _validate_decode_mask(mask, decode_mask_shape)
            cur_index = self.cache_index[...]
            _check_cache_capacity(cur_index, max_length)
            decode_context = (
                tuple(batch_dims),
                max_length,
                num_heads,
                depth_per_head,
                cur_index,
            )

        # Apply linear projections
        apply_dropconnect = (
            self.use_dropconnect
            and self.dropconnect_rate > 0.0
            and not is_deterministic
        )
        pending_dropconnect_count = None
        if apply_dropconnect:
            assert self.dropconnect_rng is not None
            base_key = self.dropconnect_rng.key[...]
            count = self.dropconnect_rng.count[...]
            dropconnect_keys = tuple(
                jax.random.fold_in(base_key, count + offset) for offset in range(4)
            )
            pending_dropconnect_count = count + 4
            query = _linear_general_with_kernel(
                self.query,
                inputs_q,
                _dropconnect(
                    self.query.kernel[...],
                    dropconnect_keys[0],
                    self.dropconnect_rate,
                ),
            )
            key = _linear_general_with_kernel(
                self.key,
                inputs_k,
                _dropconnect(
                    self.key.kernel[...],
                    dropconnect_keys[1],
                    self.dropconnect_rate,
                ),
            )
            value = _linear_general_with_kernel(
                self.value,
                inputs_v,
                _dropconnect(
                    self.value.kernel[...],
                    dropconnect_keys[2],
                    self.dropconnect_rate,
                ),
            )
        else:
            query = self.query(inputs_q)
            key = self.key(inputs_k)
            value = self.value(inputs_v)

        # Reshape to multi-head format: [batch..., length, num_heads, head_dim]
        query = query.reshape(query.shape[:-1] + (self.num_heads, self.head_dim))
        key = key.reshape(key.shape[:-1] + (self.num_heads, self.head_dim))
        value = value.reshape(value.shape[:-1] + (self.num_heads, self.head_dim))

        # Optional QK normalization (stabilizes training with higher LR)
        if self.normalize_qk:
            query = _l2_normalize_per_head(query)
            key = _l2_normalize_per_head(key)

        if decode:
            assert decode_context is not None
            (
                batch_dims,
                max_length,
                num_heads,
                depth_per_head,
                cur_index,
            ) = decode_context
            assert self.cached_key is not None
            assert self.cached_value is not None
            assert self.cache_index is not None

            expected_shape = batch_dims + (1, num_heads, depth_per_head)
            for name, tensor in (
                ("query", query),
                ("key", key),
                ("value", value),
            ):
                if expected_shape != tensor.shape:
                    raise ValueError(
                        f"Autoregressive cache shape error, expected {name} shape "
                        f"{expected_shape} instead got {tensor.shape}."
                    )

            # Build the next cache state locally.  It is committed only after
            # attention and output projection complete successfully.
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            next_key = lax.dynamic_update_slice(self.cached_key[...], key, indices)
            next_value = lax.dynamic_update_slice(
                self.cached_value[...], value, indices
            )
            key = next_key
            value = next_value

            causal_mask = jnp.broadcast_to(
                jnp.arange(max_length) <= cur_index,
                batch_dims + (1, 1, max_length),
            )
            mask = combine_masks(mask, causal_mask)
            pending_cache = (next_key, next_value, cur_index + 1)
        else:
            pending_cache = None

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
                alpha_value = self.alpha[...]

        # Resolve effective epsilon (learnable via softplus, or constant)
        if self.learnable_epsilon and self.epsilon_param is not None:
            (raw_epsilon,) = fp32_if_low_precision(self.epsilon_param[...])
            effective_epsilon = jax.nn.softplus(raw_epsilon)
        else:
            effective_epsilon = self.epsilon

        # Materialize the user mask at the actual score shape before both the
        # attention call and the post-projection zero-row policy.  Reducing an
        # unbroadcast rank-1/2 mask would otherwise confuse query/head axes.
        effective_mask = None
        if mask is not None:
            effective_mask = jnp.broadcast_to(
                mask,
                query.shape[:-3] + (query.shape[-2], query.shape[-3], key.shape[-3]),
            )

        # Apply attention (YAT by default)
        x = self.attention_fn(
            query,
            key,
            value,
            # Preserve the user/custom attention_fn mask ABI; the built-in
            # attention core materializes broadcasting internally.
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=is_deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
            epsilon=effective_epsilon,
            use_softermax=self.use_softermax,
            power=self.power,
            alpha=alpha_value,
        )

        # Apply constant alpha as direct scale (e.g. sqrt(2))
        if self._constant_alpha_value is not None:
            x = x * self._constant_alpha_value

        if apply_dropconnect:
            assert self.dropconnect_rng is not None
            output = _linear_general_with_kernel(
                self.out,
                x,
                _dropconnect(
                    self.out.kernel[...],
                    dropconnect_keys[3],
                    self.dropconnect_rate,
                ),
            )
        else:
            output = self.out(x)

        if effective_mask is not None:
            query_has_key = jnp.any(effective_mask, axis=(-3, -1))
            output = jnp.where(query_has_key[..., None], output, jnp.zeros_like(output))

        if pending_cache is not None:
            assert self.cached_key is not None
            assert self.cached_value is not None
            assert self.cache_index is not None
            next_key, next_value, next_index = pending_cache
            self.cached_key[...] = next_key
            self.cached_value[...] = next_value
            self.cache_index[...] = next_index

        if pending_dropconnect_count is not None:
            assert self.dropconnect_rng is not None
            self.dropconnect_rng.count[...] = pending_dropconnect_count

        return output

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
