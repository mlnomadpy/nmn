"""Multi-Head YAT Attention Module (PyTorch).

Implements MultiHeadYatAttention using the YAT formula:
    softmax((Q·K)² / (||Q-K||² + ε)) · V

Supports:
- Learnable or constant alpha scaling
- QK L2 normalization
- Self-attention and cross-attention
- Attention dropout
- dtype / param_dtype separation
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .yat_attention import yat_attention

__all__ = ["MultiHeadYatAttention"]

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class MultiHeadYatAttention(nn.Module):
    """Multi-head attention using the YAT formula.

    Architecture:
        Input → Linear(Q) ─┐
        Input → Linear(K) ─┼→ YAT_attention(Q, K, V) → Linear(out) → Output
        Input → Linear(V) ─┘

    YAT attention: softmax((Q·K)² / (||Q-K||² + ε)) · V

    Alpha scaling options:
        - constant_alpha=True → y = y * sqrt(2) (direct scale)
        - constant_alpha=<float> → y = y * <float> (direct scale)
        - use_alpha=True (default) → learnable: y * (sqrt(d)/log(1+d))^alpha
        - use_alpha=False → no scaling

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability (default: 0.0).
        bias: Whether to use bias in Q/K/V/out projections (default: True).
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2) as constant scale. If a float,
            use that value. If None, use learnable alpha. (default: None)
        normalize_qk: Whether to L2-normalize Q and K (default: False).
        spherical: If True, use spherical YAT with L2-normalized Q/K and
            simplified formula (default: False).
        use_out_proj: Whether to use output projection (default: True).
        epsilon: Numerical stability constant (default: 1e-5).
        device: Device for parameters.
        dtype: Computation dtype.
        param_dtype: Parameter storage dtype (default: None, uses dtype).

    Example::

        >>> attn = MultiHeadYatAttention(embed_dim=256, num_heads=8)
        >>> x = torch.randn(2, 10, 256)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 10, 256])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        normalize_qk: bool = False,
        spherical: bool = False,
        use_out_proj: bool = True,
        epsilon: float = 1e-5,
        device=None,
        dtype=None,
        param_dtype=None,
    ):
        super().__init__()

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

        storage_dtype = param_dtype if param_dtype is not None else dtype
        self.compute_dtype = dtype
        self.param_dtype = storage_dtype

        factory_kwargs = {"device": device, "dtype": storage_dtype}

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Output projection
        if use_out_proj:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        else:
            self.out_proj = None

        # Optional QK L2 normalization
        self._normalize_qk = normalize_qk
        self._spherical = spherical

        # Alpha configuration
        # Priority: constant_alpha > use_alpha
        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.register_parameter("alpha", None)
            use_alpha = True
        elif use_alpha:
            self.alpha = Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("alpha", None)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

    def _promote_dtype(self, *tensors):
        """Promote tensors to computation dtype."""
        if self.compute_dtype is not None:
            target = self.compute_dtype
        else:
            target = None
            for t in tensors:
                if t is not None:
                    target = t.dtype
                    break
            if target is None:
                target = torch.float32
        return tuple(
            t.to(target) if t is not None else None for t in tensors
        )

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        *,
        deterministic: bool = False,
    ) -> Tensor:
        """Applies multi-head YAT attention.

        Self-attention: pass only query (key/value default to query).
        Cross-attention: pass query + key (value defaults to key).

        Args:
            query: (batch, q_len, embed_dim)
            key: (batch, kv_len, embed_dim). Defaults to query.
            value: (batch, kv_len, embed_dim). Defaults to key.
            mask: Boolean mask (batch, num_heads, q_len, kv_len) or
                (batch, 1, q_len, kv_len). True = attend, False = mask out.
            deterministic: If True, disable dropout.

        Returns:
            Output of shape (batch, q_len, embed_dim).
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to multi-head: (B, L, E) → (B, L, H, D)
        q = q.reshape(batch_size, q_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, kv_len, self.num_heads, self.head_dim)

        # Optional QK L2 normalization
        if self._normalize_qk:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # Get learnable alpha (constant alpha passed as scale)
        alpha = None
        scale = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                # Constant alpha: pass as direct pre-softmax scale
                scale = self._constant_alpha_value
            elif self.alpha is not None:
                alpha = self.alpha

        # Promote dtypes
        (q, k, v, alpha) = self._promote_dtype(q, k, v, alpha)

        # Apply YAT attention
        dropout_p = self.dropout if self.training and not deterministic else 0.0
        x = yat_attention(
            q, k, v,
            mask=mask,
            dropout_p=dropout_p,
            training=self.training and not deterministic,
            epsilon=self.epsilon,
            alpha=alpha,
            scale=scale,
            spherical=self._spherical,
        )

        # Reshape back: (B, Q, H, D) → (B, Q, E)
        x = x.reshape(batch_size, q_len, self.embed_dim)

        # Output projection
        if self.out_proj is not None:
            x = self.out_proj(x)

        return x
