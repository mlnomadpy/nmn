"""Rotary YAT Attention for MLX.

Combines Rotary Position Embeddings (RoPE) with the YAT attention
formula:

    1. Apply RoPE: q' = RoPE(q, pos), k' = RoPE(k, pos)
    2. Compute YAT: softmax((q'·k')² / (‖q' − k'‖² + ε)) · V

Mirrors ``nmn.nnx.layers.attention.rotary_yat`` (without the autoregressive
KV-cache, Performer, and LayerNorm-QK paths — those are deferred to a
future PR).

References:
    - RoFormer (https://arxiv.org/abs/2104.09864)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import yat_attention_weights


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "rotary_yat_attention_weights",
    "rotary_yat_attention",
    "RotaryYatAttention",
]


DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def _splice(cache: mx.array, new: mx.array, start: int) -> mx.array:
    """Insert ``new`` into ``cache`` along axis 1 starting at ``start``.

    Equivalent to ``jax.lax.dynamic_update_slice`` on a ``(B, T, H, D)``
    cache: returns ``cache`` with ``cache[:, start:start + L]`` replaced
    by ``new``. Implemented as concatenate-of-slices since MLX doesn't
    expose an in-place scatter that lives nicely with autograd.
    """
    L = new.shape[1]
    prefix = cache[:, :start]
    suffix = cache[:, start + L:]
    return mx.concatenate([prefix, new, suffix], axis=1)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE.

    Args:
        dim: Per-head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base for the frequency computation.
        dtype: Output dtype.

    Returns:
        (cos_freqs, sin_freqs), each of shape ``(max_seq_len, dim // 2)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2] / dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq)
    return (
        mx.array(np.cos(freqs).astype(np.float32)).astype(dtype),
        mx.array(np.sin(freqs).astype(np.float32)).astype(dtype),
    )


def apply_rotary_emb(
    x: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    position_offset: int = 0,
) -> mx.array:
    """Apply RoPE to a tensor of shape ``(..., seq_len, num_heads, head_dim)``.

    The rotation pair is taken from the even / odd interleaved channels
    of the last axis, matching the reference implementation.
    """
    seq_len = x.shape[-3]
    max_seq_len = freqs_cos.shape[0]
    if position_offset + seq_len > max_seq_len:
        raise ValueError(
            f"Rotary frequencies precomputed for max_seq_len={max_seq_len}, "
            f"but position_offset + seq_len = {position_offset + seq_len} "
            f"exceeds it."
        )

    cos = freqs_cos[position_offset:position_offset + seq_len]
    sin = freqs_sin[position_offset:position_offset + seq_len]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Reshape cos / sin to (1, seq_len, 1, dim//2) — broadcasts across batch
    # and heads.
    cos = mx.reshape(cos, (1, seq_len, 1, -1))
    sin = mx.reshape(sin, (1, seq_len, 1, -1))

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave back: stack on a new last axis, then flatten the last
    # two dims.
    out = mx.stack([rotated_even, rotated_odd], axis=-1)
    return mx.reshape(out, x.shape)


def rotary_yat_attention_weights(
    query: mx.array,
    key: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    position_offset: int = 0,
) -> mx.array:
    """Softmax-normalized YAT attention weights after applying RoPE to Q/K."""
    q_rot = apply_rotary_emb(query, freqs_cos, freqs_sin, position_offset)
    k_rot = apply_rotary_emb(key, freqs_cos, freqs_sin, position_offset)
    return yat_attention_weights(
        q_rot, k_rot,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        spherical=False,
    )


def rotary_yat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    position_offset: int = 0,
) -> mx.array:
    """RoPE → YAT attention → V."""
    weights = rotary_yat_attention_weights(
        query, key,
        freqs_cos, freqs_sin,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        position_offset=position_offset,
    )
    return mx.einsum("bhqk,bkhd->bqhd", weights, value)


class RotaryYatAttention(nn.Module):
    """Multi-head YAT attention with Rotary Position Embeddings.

    Architecture::

        Input → Linear(Q) → RoPE ─┐
        Input → Linear(K) → RoPE ─┼─→ yat_attention(Q', K', V) → Linear(out)
        Input → Linear(V) ────────┘

    The RoPE frequency tables are computed once at construction and
    cached as module attributes (``freqs_cos`` / ``freqs_sin``).

    Args:
        embed_dim: Total embedding dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length supported (defines the RoPE
            table size).
        theta: RoPE base frequency.
        dropout: Attention-dropout rate (only used in training mode).
        use_bias: Whether the projections carry a bias.
        use_alpha / constant_alpha: Same as ``MultiHeadYatAttention``.
        use_out_proj: Whether to apply the output projection.
        epsilon: YAT stability constant.
        dtype: MLX dtype for parameters and computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        *,
        theta: float = 10000.0,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
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
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE."
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_out_proj = use_out_proj
        self.epsilon = epsilon
        self.dtype = dtype

        # Alpha configuration.
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

        # RoPE frequency tables — cached on the module so MLX places them
        # on the active device.
        cos_freqs, sin_freqs = precompute_freqs_cis(head_dim, max_seq_len, theta, dtype)
        self.freqs_cos = cos_freqs
        self.freqs_sin = sin_freqs

        # KV cache for autoregressive decode (populated by init_cache).
        self.cached_key: Optional[mx.array] = None
        self.cached_value: Optional[mx.array] = None
        self.cache_index: int = 0

        self.is_built = False

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

    def _linear(
        self, x: mx.array, kernel: mx.array, bias: Optional[mx.array]
    ) -> mx.array:
        y = x @ kernel
        if bias is not None:
            y = y + bias
        return y

    # ------------------------------------------------------------------
    # KV cache for autoregressive decoding
    # ------------------------------------------------------------------

    def init_cache(self, batch_size: int, max_length: Optional[int] = None) -> None:
        """Allocate a KV cache for autoregressive decoding.

        Args:
            batch_size: Batch size of the decoding stream.
            max_length: Maximum number of tokens to decode. Defaults to
                ``max_seq_len``.
        """
        if not self.is_built:
            raise ValueError(
                "Module is not built yet — call it once with a real input "
                "first so the projection shapes are known."
            )
        if max_length is None:
            max_length = self.max_seq_len
        if max_length > self.max_seq_len:
            raise ValueError(
                f"max_length={max_length} exceeds RoPE table size "
                f"max_seq_len={self.max_seq_len}."
            )
        shape = (batch_size, max_length, self.num_heads, self.head_dim)
        self.cached_key = mx.zeros(shape, dtype=self.dtype)
        self.cached_value = mx.zeros(shape, dtype=self.dtype)
        self.cache_index = 0

    def reset_cache(self) -> None:
        """Clear the KV cache."""
        self.cached_key = None
        self.cached_value = None
        self.cache_index = 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        *,
        training: bool = False,
        position_offset: int = 0,
        decode: bool = False,
    ) -> mx.array:
        """Run multi-head Rotary YAT attention.

        When ``decode=True`` the layer appends the new tokens' K and V to
        an internal cache (populated by ``init_cache``) and uses the
        ``cache_index`` as the implicit ``position_offset``.
        """
        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        if not self.is_built:
            self.build(int(x.shape[-1]))

        B, L, _ = x.shape

        if decode:
            if self.cached_key is None or self.cached_value is None:
                raise ValueError(
                    "decode=True requires init_cache() to have been called."
                )
            if self.cache_index + L > self.cached_key.shape[1]:
                raise ValueError(
                    f"KV cache full: cache_index={self.cache_index} + "
                    f"new tokens={L} exceeds capacity "
                    f"{self.cached_key.shape[1]}."
                )
            # RoPE positions for the new tokens start at cache_index.
            effective_offset = self.cache_index
        else:
            effective_offset = position_offset

        if effective_offset + L > self.max_seq_len:
            raise ValueError(
                f"position_offset + seq_len = {effective_offset + L} exceeds "
                f"max_seq_len={self.max_seq_len}. Recreate the layer with a "
                f"larger max_seq_len."
            )

        q = self._linear(x, self.q_kernel, getattr(self, "q_bias", None))
        k = self._linear(x, self.k_kernel, getattr(self, "k_bias", None))
        v = self._linear(x, self.v_kernel, getattr(self, "v_bias", None))

        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_heads, self.head_dim))

        if decode:
            cache_old = self.cache_index
            # Splice the new K / V into the cache and read back the full
            # accumulated history.
            new_key = _splice(self.cached_key, k, cache_old)
            new_value = _splice(self.cached_value, v, cache_old)
            self.cached_key = new_key
            self.cached_value = new_value
            self.cache_index = cache_old + L
            cache_new = self.cache_index

            # Rotate Q by the new tokens' positions (cache_old..cache_old+L-1)
            # and the *full* cached K by absolute positions 0..cache_new-1.
            q_rot = apply_rotary_emb(
                q, self.freqs_cos, self.freqs_sin, cache_old
            )
            k_full = self.cached_key[:, :cache_new]
            v_full = self.cached_value[:, :cache_new]
            k_rot = apply_rotary_emb(k_full, self.freqs_cos, self.freqs_sin, 0)

            # Build a causal mask over (L, cache_new) so token i in the
            # chunk only sees keys up to position cache_old + i. Single-
            # token decode (L=1) collapses to "see everything" which is
            # already correct.
            if L > 1:
                i_idx = np.arange(L)[:, None]                 # (L, 1)
                j_idx = np.arange(cache_new)[None, :]         # (1, K)
                causal = (j_idx <= cache_old + i_idx)[None, None]
                user_mask_np = None if mask is None else np.array(mask)
                if user_mask_np is None:
                    full_mask = causal
                else:
                    full_mask = np.logical_and(user_mask_np, causal)
                full_mask = mx.array(
                    np.broadcast_to(full_mask, (B, self.num_heads, L, cache_new))
                )
            else:
                full_mask = mask

            alpha_val, scale_val = self._alpha_pair()
            from .attention import yat_attention_weights as _yaw

            weights = _yaw(
                q_rot, k_rot,
                mask=full_mask,
                dropout_rate=self.dropout if training else 0.0,
                training=training,
                epsilon=self.epsilon,
                alpha=alpha_val,
                scale=scale_val,
                spherical=False,
            )
            out = mx.einsum("bhqk,bkhd->bqhd", weights, v_full)
        else:
            alpha_val, scale_val = self._alpha_pair()
            out = rotary_yat_attention(
                q, k, v,
                self.freqs_cos, self.freqs_sin,
                mask=mask,
                dropout_rate=self.dropout if training else 0.0,
                training=training,
                epsilon=self.epsilon,
                alpha=alpha_val,
                scale=scale_val,
                position_offset=effective_offset,
            )

        out = mx.reshape(out, (B, L, self.embed_dim))
        if self.use_out_proj and getattr(self, "out_kernel", None) is not None:
            out = self._linear(out, self.out_kernel, getattr(self, "out_bias", None))
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _alpha_pair(self) -> Tuple[Optional[mx.array], Optional[float]]:
        """Resolve (learnable alpha, constant scale) for the forward call."""
        alpha_val: Optional[mx.array] = None
        scale_val: Optional[float] = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                scale_val = self._constant_alpha_value
            elif getattr(self, "alpha", None) is not None:
                alpha_val = self.alpha
        return alpha_val, scale_val
