"""Rotary YAT Attention Module.

This module combines Rotary Position Embeddings (RoPE) with YAT attention,
providing position-aware attention using the geometric YAT formula.

RoPE encodes absolute position through rotation and naturally captures
relative position through the rotation angle difference. Combined with
YAT's distance-similarity tradeoff, this creates a powerful position-aware
attention mechanism.

The formula:
    1. Apply RoPE: q' = RoPE(q, pos), k' = RoPE(k, pos)
    2. Compute YAT: softmax((q'·k')² / (||q' - k'||² + ε)) · V

Performer Mode:
    When use_performer=True, uses FAVOR+ random feature approximation for
    O(n) time complexity instead of O(n²).

Reference:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding
      (https://arxiv.org/abs/2104.09864)
    - Rethinking Attention with Performers (https://arxiv.org/abs/2009.14794)
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Callable, Any, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import LinearGeneral, default_kernel_init
from flax.nnx.nn.normalization import LayerNorm
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import (
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
)
from jax import Array

from .yat_attention import (
    yat_attention_weights,
    normalize_qk,
)
from .spherical_yat_performer import (
    yat_tp_attention,
    create_yat_tp_projection,
)
from .multi_head import DEFAULT_CONSTANT_ALPHA
from .masks import combine_masks
from nmn.nnx.squashers import softermax


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: Dtype = jnp.float32,
) -> Tuple[Array, Array]:
    """Precomputes cosine and sine frequencies for RoPE.

    This computes the rotation frequencies used in Rotary Position Embeddings.
    The frequencies are computed once and can be reused for all forward passes.

    Args:
        dim: Dimension of the head (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base for the frequency computation (default: 10000.0).
        dtype: Data type for the output arrays.

    Returns:
        Tuple of (cos_freqs, sin_freqs), each of shape [max_seq_len, dim//2].

    Example:
        >>> freqs_cos, freqs_sin = precompute_freqs_cis(64, 2048)
        >>> freqs_cos.shape
        (2048, 32)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))

    # Create position indices
    t = np.arange(max_seq_len)

    # Outer product: [max_seq_len, dim//2]
    freqs = np.outer(t, inv_freq)

    return jnp.array(np.cos(freqs), dtype=dtype), jnp.array(np.sin(freqs), dtype=dtype)


def apply_rotary_emb(
    x: Array,
    freqs_cos: Array,
    freqs_sin: Array,
    position_offset: int = 0,
) -> Array:
    """Applies Rotary Position Embeddings to input tensor.

    Rotates the input tensor using precomputed cos/sin frequencies.
    This encoding captures absolute position through rotation and
    relative position through rotation angle difference.

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, head_dim].
        freqs_cos: Cosine frequencies of shape [max_seq_len, head_dim//2].
        freqs_sin: Sine frequencies of shape [max_seq_len, head_dim//2].
        position_offset: Starting position for the sequence (for caching).

    Returns:
        Rotated tensor of same shape as input.

    Example:
        >>> x = jnp.ones((2, 10, 8, 64))  # (batch, seq, heads, dim)
        >>> freqs_cos, freqs_sin = precompute_freqs_cis(64, 100)
        >>> x_rotated = apply_rotary_emb(x, freqs_cos, freqs_sin)
        >>> x_rotated.shape
        (2, 10, 8, 64)
    """
    seq_len = x.shape[-3]

    # Slice frequencies for the current sequence
    # freqs shape: [seq_len, head_dim//2]
    freqs_cos = freqs_cos[position_offset : position_offset + seq_len]
    freqs_sin = freqs_sin[position_offset : position_offset + seq_len]

    # Split x into even and odd components
    # x shape: [..., seq_len, num_heads, head_dim]
    x_even = x[..., 0::2]  # [..., seq_len, num_heads, head_dim//2]
    x_odd = x[..., 1::2]  # [..., seq_len, num_heads, head_dim//2]

    # Reshape freqs for broadcasting: [1, seq_len, 1, head_dim//2]
    freqs_cos = freqs_cos.reshape(1, seq_len, 1, -1)
    freqs_sin = freqs_sin.reshape(1, seq_len, 1, -1)

    # Apply rotation using complex multiplication in real form:
    # (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    x_out_even = x_even * freqs_cos - x_odd * freqs_sin
    x_out_odd = x_even * freqs_sin + x_odd * freqs_cos

    # Interleave back: stack and reshape
    x_out = jnp.stack([x_out_even, x_out_odd], axis=-1)
    x_out = x_out.reshape(x.shape)

    return x_out


def rotary_yat_attention_weights(
    query: Array,
    key: Array,
    freqs_cos: Array,
    freqs_sin: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    epsilon: float = 1e-5,
    use_softermax: bool = False,
    power: float = 1.0,
    position_offset: int = 0,
    alpha: Optional[Array] = None,
) -> Array:
    """Computes Rotary YAT attention weights.

    First applies RoPE to query and key, then computes YAT attention weights.

    Args:
        query: Queries of shape [..., q_length, num_heads, head_dim].
        key: Keys of shape [..., kv_length, num_heads, head_dim].
        freqs_cos: Cosine frequencies for RoPE.
        freqs_sin: Sine frequencies for RoPE.
        bias: Optional attention bias.
        mask: Optional attention mask.
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing weights.
        epsilon: Numerical stability constant.
        use_softermax: Whether to use softermax.
        power: Softermax power parameter.
        position_offset: Starting position for RoPE.
        alpha: Optional alpha scaling parameter.

    Returns:
        Attention weights of shape [..., num_heads, q_length, kv_length].
    """
    # Apply RoPE to query and key
    query_rotated = apply_rotary_emb(query, freqs_cos, freqs_sin, position_offset)
    key_rotated = apply_rotary_emb(key, freqs_cos, freqs_sin, position_offset)

    # Compute YAT attention weights on rotated Q, K
    return yat_attention_weights(
        query_rotated,
        key_rotated,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
        epsilon=epsilon,
        use_softermax=use_softermax,
        power=power,
        alpha=alpha,
    )


def rotary_yat_attention(
    query: Array,
    key: Array,
    value: Array,
    freqs_cos: Array,
    freqs_sin: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    epsilon: float = 1e-5,
    use_softermax: bool = False,
    power: float = 1.0,
    position_offset: int = 0,
    alpha: Optional[Array] = None,
) -> Array:
    """Computes Rotary YAT attention: RoPE + YAT formula.

    Combines Rotary Position Embeddings with YAT attention:
        1. Apply RoPE: q' = RoPE(q), k' = RoPE(k)
        2. Compute YAT: softmax((q'·k')² / (||q' - k'||² + ε)) · V

    With optional alpha scaling:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Args:
        query: Queries of shape [..., q_length, num_heads, head_dim].
        key: Keys of shape [..., kv_length, num_heads, head_dim].
        value: Values of shape [..., kv_length, num_heads, v_dim].
        freqs_cos: Cosine frequencies for RoPE.
        freqs_sin: Sine frequencies for RoPE.
        bias: Optional attention bias.
        mask: Optional attention mask.
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing weights.
        epsilon: Numerical stability constant.
        use_softermax: Whether to use softermax.
        power: Softermax power parameter.
        position_offset: Starting position for RoPE.
        alpha: Optional alpha scaling parameter.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim].
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    # Compute attention weights with RoPE + YAT
    attn_weights = rotary_yat_attention_weights(
        query,
        key,
        freqs_cos,
        freqs_sin,
        bias=bias,
        mask=mask,
        broadcast_dropout=broadcast_dropout,
        dropout_rng=dropout_rng,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dtype=dtype,
        precision=precision,
        module=module,
        epsilon=epsilon,
        use_softermax=use_softermax,
        power=power,
        position_offset=position_offset,
        alpha=alpha,
    )

    # Return weighted sum over values
    return jnp.einsum(
        "...hqk,...khd->...qhd", attn_weights, value, precision=precision
    )


def rotary_yat_performer_attention(
    query: Array,
    key: Array,
    value: Array,
    freqs_cos: Array,
    freqs_sin: Array,
    performer_params: dict,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[Array] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[Module] = None,
    epsilon: float = 1e-5,
    position_offset: int = 0,
    causal: bool = False,
    normalize_inputs: bool = True,
    alpha: Optional[Array] = None,
    gradient_scaling: bool = True,
) -> Array:
    """Computes Rotary YAT Performer attention using Multi-Scale TP-PRF.

    Combines:
    - RoPE for position encoding
    - Multi-Scale FAVOR+ features for YAT kernel approximation
    - O(n) linear complexity attention
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    # Apply RoPE first
    query_rotated = apply_rotary_emb(query, freqs_cos, freqs_sin, position_offset)
    key_rotated = apply_rotary_emb(key, freqs_cos, freqs_sin, position_offset)

    # Normalize to unit vectors (required for YAT approximation)
    if normalize_inputs:
        query_rotated, key_rotated = normalize_qk(query_rotated, key_rotated, epsilon)

    # Use the new Spherical YAT Linear Attention
    output = yat_tp_attention(
        query_rotated,
        key_rotated,
        value,
        performer_params,
        causal=causal,
        epsilon=epsilon,
        precision=precision,
        gradient_scaling=gradient_scaling,
    )

    # Apply alpha scaling
    head_dim = query.shape[-1]
    if alpha is not None:
        alpha_val = jnp.asarray(alpha, dtype=dtype)
        # Simple learnable alpha scaling
        output = output * alpha_val

    # Apply dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep = random.bernoulli(dropout_rng, keep_prob, output.shape)
        output = output * keep / keep_prob

    return output


class RotaryYatAttention(Module):
    """Multi-head Rotary YAT Attention with optional Performer mode.

    Combines Rotary Position Embeddings with YAT attention for
    position-aware geometric attention. Supports Performer mode for
    O(n) linear complexity.

    Architecture:
        Input → Linear(Q) → RoPE ─┐
        Input → Linear(K) → RoPE ─┼→ YAT_attention → Linear(out) → Output
        Input → Linear(V) ───────┘

    The YAT formula after RoPE:
        softmax((RoPE(Q)·RoPE(K))² / (||RoPE(Q) - RoPE(K)||² + ε)) · V

    Performer Mode:
        When use_performer=True, uses random feature approximation for O(n)
        complexity instead of O(n²). This is useful for very long sequences.

    Example:
        >>> from flax import nnx
        >>> from nmn.nnx.attention import RotaryYatAttention
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> # Standard quadratic attention
        >>> attn = RotaryYatAttention(
        ...     embed_dim=256,
        ...     num_heads=8,
        ...     max_seq_len=512,
        ...     rngs=rngs,
        ... )
        >>>
        >>> # Performer mode for linear complexity
        >>> attn_performer = RotaryYatAttention(
        ...     embed_dim=256,
        ...     num_heads=8,
        ...     max_seq_len=8192,
        ...     use_performer=True,
        ...     num_features=256,
        ...     rngs=rngs,
        ... )
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        *,
        theta: float = 10000.0,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = initializers.zeros_init(),
        alpha_init: Initializer = initializers.ones_init(),
        use_bias: bool = False,
        normalize_qk: bool = False,
        use_out_proj: bool = True,
        epsilon: float = 1e-5,
        use_softermax: bool = False,
        power: float = 1.0,
        use_performer: bool = False,
        num_features: int | None = None,
        causal: bool = False,
        performer_normalize: bool = True,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        rngs: rnglib.Rngs,
    ):
        """Initializes RotaryYatAttention.

        Args:
            embed_dim: Total embedding dimension.
            num_heads: Number of attention heads.
            max_seq_len: Maximum sequence length for RoPE precomputation.
            theta: Base for RoPE frequency computation.
            dtype: Computation dtype.
            param_dtype: Parameter dtype.
            broadcast_dropout: Whether to broadcast dropout.
            dropout_rate: Attention dropout probability.
            precision: JAX precision.
            kernel_init: Weight initializer.
            bias_init: Bias initializer.
            alpha_init: Initializer for learnable alpha.
            use_bias: Whether to use bias in projections.
            normalize_qk: Whether to apply QK layer normalization.
            use_out_proj: Whether to use output projection.
            epsilon: Numerical stability for YAT.
            use_softermax: Whether to use softermax (only for non-Performer).
            power: Softermax power parameter.
            use_performer: If True, use Performer mode for O(n) complexity.
            num_features: Number of random features for Performer (default: head_dim).
            causal: If True, use causal attention (Performer mode).
            performer_normalize: If True (default), normalize Q/K to unit vectors
                in Performer mode. This enables the optimized YAT formula:
                (q·k)² / (2(1 - q·k) + ε) which only needs ONE dot product!
            use_alpha: Whether to use alpha scaling for YAT attention.
            constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
                use that value. If None (default), use learnable alpha when
                use_alpha=True.
            rngs: Random number generators.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.broadcast_dropout = broadcast_dropout
        self.dropout_rate = dropout_rate
        self.precision = precision
        self.use_bias = use_bias
        self.normalize_qk = normalize_qk
        self.use_out_proj = use_out_proj
        self.epsilon = epsilon
        self.use_softermax = use_softermax
        self.power = power
        self.use_performer = use_performer
        self.causal = causal
        self.performer_normalize = performer_normalize

        # Handle alpha configuration (same logic as MultiHeadAttention)
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

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE."
            )

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.head_dim, max_seq_len, theta, dtype=param_dtype
        )
        self.freqs_cos = nnx.Cache(freqs_cos)
        self.freqs_sin = nnx.Cache(freqs_sin)

        # Performer random projection
        self.perf_projections: nnx.Cache | None
        self.perf_scales: nnx.Cache | None
        if use_performer:
            total_features = num_features if num_features is not None else 64
            self.num_scales = 8
            
            # Split features across scales for memory efficiency
            # Ensure at least 4 features per scale
            self.num_features_per_scale = max(4, total_features // self.num_scales)
            # Update total features to reflect actual count
            self.num_features = self.num_features_per_scale * self.num_scales
            
            # Create Multi-Scale params
            params = create_yat_tp_projection(
                rngs.params(),
                self.head_dim,
                num_prf_features=self.num_features_per_scale,
                num_quad_nodes=self.num_scales,
                dtype=param_dtype,
            )
            
            self.perf_projections = nnx.Cache(params['projections'])
            self.perf_scales = nnx.Cache(params['scales'])
            self.perf_head_dim = params['head_dim']
        else:
            self.num_features = None
            self.num_scales = None
            self.num_features_per_scale = None
            self.perf_projections = None
            self.perf_scales = None
            self.perf_head_dim = None

        # Q, K, V projections
        linear_kwargs = dict(
            in_features=embed_dim,
            out_features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=kernel_init,
            bias_init=bias_init,
            use_bias=use_bias,
            precision=precision,
        )
        self.q_proj = LinearGeneral(**linear_kwargs, rngs=rngs)
        self.k_proj = LinearGeneral(**linear_kwargs, rngs=rngs)
        self.v_proj = LinearGeneral(**linear_kwargs, rngs=rngs)

        # Optional output projection
        self.o_proj: LinearGeneral | None
        if use_out_proj:
            self.o_proj = LinearGeneral(**linear_kwargs, rngs=rngs)
        else:
            self.o_proj = None

        # Optional QK normalization
        self.query_ln: LayerNorm | None
        self.key_ln: LayerNorm | None
        if normalize_qk:
            self.query_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.key_ln = LayerNorm(
                self.head_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
        else:
            self.query_ln = None
            self.key_ln = None

        # Cache for autoregressive decoding
        self.cached_key: nnx.Cache[Array] | None = None
        self.cached_value: nnx.Cache[Array] | None = None
        self.cache_index: nnx.Cache[Array] | None = None

    def __call__(
        self,
        x: Array,
        mask: Array | None = None,
        *,
        deterministic: bool = True,
        decode: bool = False,
        position_offset: int = 0,
        rngs: rnglib.Rngs | None = None,
        sow_weights: bool = False,
    ) -> Array:
        """Applies Rotary YAT attention.

        Args:
            x: Input of shape [batch, seq_len, embed_dim].
            mask: Optional attention mask.
            deterministic: If True, no dropout.
            decode: If True, use autoregressive decoding.
            position_offset: Starting position for RoPE.
            rngs: Random number generators.
            sow_weights: If True, sow attention weights (not supported in Performer mode).

        Returns:
            Output of shape [batch, seq_len, embed_dim].
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to multi-head format: [batch, seq, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Optional QK normalization
        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            q = self.query_ln(q)
            k = self.key_ln(k)

        # Handle autoregressive decoding
        if decode:
            if self.cached_key is None or self.cached_value is None:
                raise ValueError(
                    "Cache not initialized. Call init_cache first."
                )

            cur_index = self.cache_index.value
            # Update cache
            indices = (0, cur_index, 0, 0)
            k_cached = jax.lax.dynamic_update_slice(
                self.cached_key.value, k, indices
            )
            v_cached = jax.lax.dynamic_update_slice(
                self.cached_value.value, v, indices
            )
            self.cached_key.value = k_cached
            self.cached_value.value = v_cached
            self.cache_index.value += 1

            k = k_cached
            v = v_cached

            # Causal mask for decoding
            max_length = k.shape[1]
            mask = combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    (batch_size, 1, 1, max_length),
                ),
            )
            position_offset = 0  # Use full cache positions

        # Get RoPE frequencies
        freqs_cos = jax.device_put(self.freqs_cos.value)
        freqs_sin = jax.device_put(self.freqs_sin.value)

        # Dropout RNG
        dropout_rng = None
        if self.dropout_rate > 0.0 and not deterministic:
            if rngs is None:
                raise ValueError("rngs required for dropout")
            dropout_rng = rngs.dropout()

        # Get alpha value (either learnable or constant)
        # For constant alpha, we apply it directly after attention
        # rather than passing it to the attention function (which uses it as an exponent)
        alpha_value = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                pass  # Will be applied as direct scale after attention
            elif self.alpha is not None:
                alpha_value = self.alpha.value

        # Apply Rotary YAT attention
        if self.use_performer:
            # Performer mode: O(n) complexity
            
            # Reconstruct params dict
            performer_params = {
                'projections': jax.device_put(self.perf_projections.value),
                'scales': jax.device_put(self.perf_scales.value),
                'head_dim': self.perf_head_dim,
                'num_prf_features': self.num_features_per_scale,
                'num_scales': self.num_scales,
            }
            
            output = rotary_yat_performer_attention(
                q,
                k,
                v,
                freqs_cos,
                freqs_sin,
                performer_params,
                bias=None,  # Not supported in Performer
                mask=mask,
                broadcast_dropout=self.broadcast_dropout,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=self.precision,
                epsilon=self.epsilon,
                position_offset=position_offset,
                causal=self.causal,
                normalize_inputs=self.performer_normalize,
                alpha=alpha_value,
            )
        else:
            # Standard O(n²) attention
            output = rotary_yat_attention(
                q,
                k,
                v,
                freqs_cos,
                freqs_sin,
                mask=mask,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=deterministic,
                dtype=self.dtype,
                precision=self.precision,
                module=self if sow_weights else None,
                epsilon=self.epsilon,
                use_softermax=self.use_softermax,
                power=self.power,
                position_offset=position_offset,
                alpha=alpha_value,
            )

        # Apply constant alpha as direct scale (e.g. sqrt(2))
        if self._constant_alpha_value is not None:
            output = output * self._constant_alpha_value

        # Reshape back: [batch, seq, num_heads, head_dim] -> [batch, seq, embed_dim]
        output = output.reshape(batch_size, seq_len, self.embed_dim)

        # Optional output projection
        if self.o_proj is not None:
            output = self.o_proj(output)

        return output

    def init_cache(self, batch_size: int, max_length: int, dtype: Dtype = jnp.float32):
        """Initializes cache for autoregressive decoding.

        Args:
            batch_size: Batch size for the cache.
            max_length: Maximum sequence length.
            dtype: Data type for cache arrays.
        """
        cache_shape = (batch_size, max_length, self.num_heads, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype))
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

