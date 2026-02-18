"""YAT Attention Functions.

This module implements the YAT attention mechanism, which replaces the standard
scaled dot-product attention with a distance-similarity tradeoff formula:

    ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

This balances:
- Similarity: squared dot product (numerator)
- Proximity: squared Euclidean distance (denominator)

The result is attention that activates when queries and keys are both
similar in direction AND close in Euclidean space.

Alpha Scaling:
    Optional alpha parameter scales the attention scores:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha
    
    Alpha can be:
    - Learnable: Pass alpha as a trainable parameter
    - Constant: Pass alpha as a float value (e.g., sqrt(2) ≈ 1.414)
    - Disabled: Pass alpha=None (default)

Performer Mode:
    When use_performer=True, uses FAVOR+ random feature approximation for
    O(n) time complexity instead of O(n²). This approximates the YAT attention
    by using random features to approximate the normalized attention scores.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jax import random

from flax import nnx
from flax.nnx.module import Module
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
from jax import Array

from nmn.nnx.squashers import softermax


def yat_attention_weights(
    query: Array,
    key: Array,
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
    alpha: Optional[Array] = None,
) -> Array:
    """Computes YAT attention weights: softmax((Q·K)² / (||Q-K||² + ε))

    Uses the YAT formula to compute attention scores that balance similarity
    (squared dot product) with proximity (squared Euclidean distance).

    The YAT attention score: ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

    With optional alpha scaling:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Args:
        query: Queries with shape [..., q_length, num_heads, head_dim]
        key: Keys with shape [..., kv_length, num_heads, head_dim]
        bias: Optional bias for attention weights, broadcastable to
            [..., num_heads, q_length, kv_length]
        mask: Optional boolean mask, broadcastable to
            [..., num_heads, q_length, kv_length]. False values are masked out.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability (0.0 = no dropout).
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype (inferred from inputs if None).
        precision: JAX precision for einsum operations.
        module: Optional Module to sow attention weights for introspection.
        epsilon: Small constant for numerical stability in denominator.
        use_softermax: If True, use softermax instead of standard softmax.
        power: Power parameter for softermax (only used if use_softermax=True).
        alpha: Optional alpha scaling parameter. Can be a scalar Array (learnable)
            or a float value (constant). If None, no alpha scaling is applied.

    Returns:
        Attention weights of shape [..., num_heads, q_length, kv_length]
    """
    query, key = promote_dtype((query, key), dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    head_dim = query.shape[-1]

    # YAT-style attention: softmax((q·k)² / (||q-k||² + ε)) · V
    #
    # query shape: [..., q_length, num_heads, head_dim]
    # key shape: [..., kv_length, num_heads, head_dim]
    #
    # The YAT formula: ⵟ(q, k) = (q·k)² / (||q - k||² + ε)
    # where ||q - k||² = ||q||² - 2(q·k) + ||k||²

    # Calculate dot product: q·k
    # Output shape: [..., num_heads, q_length, kv_length]
    dot_product = jnp.einsum(
        "...qhd,...khd->...hqk", query, key, precision=precision
    )

    # Squared dot product for numerator: (q·k)²
    squared_dot_product = jnp.square(dot_product)

    # Calculate squared norms
    # q_norm: [..., q_length, num_heads, 1]
    # k_norm: [..., kv_length, num_heads, 1]
    q_norm_sq = jnp.sum(jnp.square(query), axis=-1, keepdims=True)
    k_norm_sq = jnp.sum(jnp.square(key), axis=-1, keepdims=True)

    # We need to compute ||q - k||² for each (q, k) pair
    # ||q - k||² = ||q||² + ||k||² - 2*(q·k)
    #
    # Reshape norms to broadcast correctly with dot_product shape
    # [..., num_heads, q_length, kv_length]
    batch_dims = query.ndim - 3

    # Transpose q_norm: [..., q_length, num_heads, 1] -> [..., num_heads, q_length, 1]
    q_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
    q_norm_transposed = q_norm_sq.transpose(q_axes)

    # Transpose k_norm: [..., kv_length, num_heads, 1] -> [..., num_heads, 1, kv_length]
    k_axes = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
    k_norm_transposed = k_norm_sq.transpose(k_axes)
    k_norm_transposed = jnp.swapaxes(k_norm_transposed, -2, -1)

    # Squared Euclidean distance: ||q||² + ||k||² - 2*(q·k)
    squared_dist = q_norm_transposed + k_norm_transposed - 2.0 * dot_product

    # YAT attention scores: (q·k)² / (||q - k||² + ε)
    attn_weights = squared_dot_product / (squared_dist + epsilon)

    # Apply alpha scaling: y *= alpha (simple multiply)
    if alpha is not None:
        alpha_val = jnp.asarray(alpha, dtype=dtype)
        attn_weights = attn_weights * alpha_val

    # Apply attention bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # Apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # Normalize the attention weights
    if use_softermax:
        attn_weights = softermax(attn_weights, n=power).astype(dtype)
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Sow attention weights for introspection
    if module:
        module.sow(nnx.Intermediate, "attention_weights", attn_weights)

    # Apply attention dropout
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


def yat_attention(
    query: Array,
    key: Array,
    value: Array,
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
    alpha: Optional[Array] = None,
) -> Array:
    """Computes YAT attention: softmax((Q·K)² / (||Q-K||² + ε)) · V

    This replaces the standard scaled dot-product attention with the YAT formula,
    which balances similarity (dot product) with proximity (Euclidean distance).

    The YAT attention score for each (query, key) pair is:
        ⵟ(q, k) = (q·k)² / (||q - k||² + ε)

    With optional alpha scaling:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Args:
        query: Queries with shape [..., q_length, num_heads, head_dim]
        key: Keys with shape [..., kv_length, num_heads, head_dim]
        value: Values with shape [..., kv_length, num_heads, v_dim]
        bias: Optional bias for attention weights.
        mask: Optional boolean mask for attention weights.
        broadcast_dropout: If True, dropout is broadcast across batch dims.
        dropout_rng: JAX PRNGKey for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout is applied.
        dtype: Computation dtype.
        precision: JAX precision for einsum operations.
        module: Optional Module for sowing attention weights.
        epsilon: Small constant for numerical stability in denominator.
        use_softermax: If True, use softermax instead of standard softmax.
        power: Power parameter for softermax.
        alpha: Optional alpha scaling parameter. Can be a scalar Array (learnable)
            or a float value (constant). If None, no alpha scaling is applied.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim]
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # Compute attention weights using YAT formula
    attn_weights = yat_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        module,
        epsilon,
        use_softermax,
        power,
        alpha,
    )

    # Return weighted sum over values for each query position
    return jnp.einsum(
        "...hqk,...khd->...qhd", attn_weights, value, precision=precision
    )


# =============================================================================
# Performer-style YAT Attention (Linear Complexity)
# =============================================================================


def normalize_qk(
    query: Array,
    key: Array,
    epsilon: float = 1e-6,
) -> tuple[Array, Array]:
    """Normalizes query and key to unit vectors.

    When Q and K are unit vectors, the YAT formula simplifies significantly:
        ||q - k||² = ||q||² + ||k||² - 2(q·k) = 1 + 1 - 2(q·k) = 2(1 - q·k)

    So YAT becomes: (q·k)² / (2(1 - q·k) + ε)

    This eliminates separate norm computations, requiring only ONE dot product.

    Args:
        query: Query tensor of any shape [..., head_dim].
        key: Key tensor of any shape [..., head_dim].
        epsilon: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_query, normalized_key).
    """
    q_norm = jnp.sqrt(jnp.sum(query ** 2, axis=-1, keepdims=True) + epsilon)
    k_norm = jnp.sqrt(jnp.sum(key ** 2, axis=-1, keepdims=True) + epsilon)
    return query / q_norm, key / k_norm


def yat_attention_normalized(
    query: Array,
    key: Array,
    value: Array,
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
    alpha: Optional[Array] = None,
) -> Array:
    """Computes YAT attention with normalized Q and K (optimized).

    When Q and K are normalized to unit vectors, the YAT formula simplifies:
        (q·k)² / (2(1 - q·k) + ε)

    This is faster because we only need ONE dot product instead of computing
    separate squared norms. The normalization is O(n) and very fast.

    With optional alpha scaling:
        scaled_attn = attn * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Args:
        query: Queries [..., q_length, num_heads, head_dim] (will be normalized).
        key: Keys [..., kv_length, num_heads, head_dim] (will be normalized).
        value: Values [..., kv_length, num_heads, v_dim].
        bias: Optional attention bias.
        mask: Optional attention mask.
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing.
        epsilon: Numerical stability constant.
        use_softermax: Whether to use softermax.
        power: Softermax power parameter.
        alpha: Optional alpha scaling parameter. Can be a scalar Array (learnable)
            or a float value (constant). If None, no alpha scaling is applied.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim].
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    head_dim = query.shape[-1]

    # Normalize Q and K to unit vectors
    query_normalized, key_normalized = normalize_qk(query, key, epsilon)

    # Compute dot product: q·k (only need this one operation!)
    # Output shape: [..., num_heads, q_length, kv_length]
    dot_product = jnp.einsum(
        "...qhd,...khd->...hqk", query_normalized, key_normalized, precision=precision
    )

    # Squared dot product: (q·k)²
    squared_dot_product = jnp.square(dot_product)

    # Simplified distance: 2(1 - q·k) since ||q|| = ||k|| = 1
    # ||q - k||² = ||q||² + ||k||² - 2(q·k) = 1 + 1 - 2(q·k) = 2 - 2(q·k)
    distance_sq = 2.0 - 2.0 * dot_product

    # YAT attention scores: (q·k)² / (2(1 - q·k) + ε)
    attn_weights = squared_dot_product / (distance_sq + epsilon)

    # Apply alpha scaling: y *= alpha (simple multiply)
    if alpha is not None:
        alpha_val = jnp.asarray(alpha, dtype=dtype)
        attn_weights = attn_weights * alpha_val

    # Apply bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # Apply mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # Normalize
    if use_softermax:
        attn_weights = softermax(attn_weights, n=power).astype(dtype)
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    # Sow weights
    if module:
        module.sow(nnx.Intermediate, "attention_weights", attn_weights)

    # Dropout
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        attn_weights = attn_weights * multiplier

    # Weighted sum
    return jnp.einsum(
        "...hqk,...khd->...qhd", attn_weights, value, precision=precision
    )


def yat_performer_feature_map(
    x: Array,
    projection: Array,
    epsilon: float = 1e-6,
    pre_normalized: bool = False,
) -> Array:
    """Applies YAT-adapted feature map for Performer approximation.

    This feature map implements the Spherical Yat-Performer approach:
    - Normalizes inputs to unit sphere
    - Uses FAVOR+ style positive random features
    - Designed to produce stable, positive features for linear attention

    For the spherical YAT kernel: (q·k)² / (C - 2(q·k)) where C = 2 + ε
    
    We use a simplified FAVOR+ approximation that produces stable gradients:
        φ(x) = |W @ x / sqrt(d)| * exp(W @ x / sqrt(d) - 0.5) / sqrt(m)
    
    The absolute value ensures positivity while the exponential captures
    alignment information.

    Args:
        x: Input tensor of shape [..., seq_len, num_heads, head_dim].
        projection: Random projection matrix [num_features, head_dim].
        epsilon: Numerical stability constant.
        pre_normalized: If True, assumes x is already normalized to unit vectors.

    Returns:
        Feature-mapped tensor of shape [..., seq_len, num_heads, num_features].
    """
    head_dim = x.shape[-1]
    num_features = projection.shape[0]

    # Normalize to unit sphere if not already
    if pre_normalized:
        x_normalized = x
    else:
        x_norm = jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + epsilon)
        x_normalized = x / x_norm

    # Project normalized vectors: W @ x / sqrt(d)
    # For orthogonal W with ||w_i|| = sqrt(d): E[(w·q)(w·k)] = q·k
    x_proj = jnp.einsum("...d,md->...m", x_normalized, projection)
    x_proj = x_proj / jnp.sqrt(head_dim).astype(x.dtype)
    
    # FAVOR+ style: exp(wx - ||wx||²/2) = exp(wx - 0.5) for unit x
    # Modified with absolute value for guaranteed positivity
    # This ensures positive attention weights in linear attention
    exp_features = jnp.exp(x_proj - 0.5)
    
    # Use combination of linear and exp for better gradient flow
    # Linear term captures alignment, exp term weights similar vectors higher
    linear_features = jnp.abs(x_proj) + epsilon
    
    # Combine: use product to get positive features that capture alignment
    features = linear_features * exp_features
    
    # Normalize by sqrt(num_features) for proper scaling
    features = features / jnp.sqrt(num_features).astype(x.dtype)

    return features


def yat_performer_attention(
    query: Array,
    key: Array,
    value: Array,
    projection: Array,
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
    causal: bool = False,
    normalize_inputs: bool = True,
    alpha: Optional[Array] = None,
) -> Array:
    """Computes YAT attention with Performer-style linear complexity.

    Uses random feature approximation to compute YAT attention in O(n) time
    instead of O(n²). The approximation captures the geometric properties
    of YAT attention while enabling efficient computation.

    When normalize_inputs=True (default), Q and K are normalized to unit vectors
    first, enabling the optimized YAT formula:
        (q·k)² / (2(1 - q·k) + ε)

    This is faster because:
    1. We only need ONE dot product instead of computing separate norms
    2. The feature map can be simpler for unit vectors
    3. Better numerical stability

    With optional alpha scaling:
        scaled_output = output * (sqrt(head_dim) / log(1 + head_dim))^alpha

    Args:
        query: Queries with shape [..., q_length, num_heads, head_dim].
        key: Keys with shape [..., kv_length, num_heads, head_dim].
        value: Values with shape [..., kv_length, num_heads, v_dim].
        projection: Random projection matrix [num_features, head_dim].
        bias: Optional attention bias (not used in efficient mode).
        mask: Optional attention mask (limited support in efficient mode).
        broadcast_dropout: Whether to broadcast dropout.
        dropout_rng: RNG for dropout.
        dropout_rate: Dropout probability.
        deterministic: If True, no dropout.
        dtype: Computation dtype.
        precision: JAX precision.
        module: Optional module for sowing.
        epsilon: Numerical stability constant.
        causal: If True, use causal attention.
        normalize_inputs: If True (default), normalize Q and K to unit vectors.
        alpha: Optional alpha scaling parameter. Can be a scalar Array (learnable)
            or a float value (constant). If None, no alpha scaling is applied.

    Returns:
        Output of shape [..., q_length, num_heads, v_dim].
    """
    query, key, value = promote_dtype((query, key, value), dtype=dtype)
    dtype = query.dtype

    head_dim = query.shape[-1]

    # Normalize Q and K for optimized computation
    if normalize_inputs:
        query, key = normalize_qk(query, key, epsilon)

    # Apply YAT-adapted feature maps (pre_normalized if we normalized)
    q_features = yat_performer_feature_map(
        query, projection, epsilon, pre_normalized=normalize_inputs
    )
    k_features = yat_performer_feature_map(
        key, projection, epsilon, pre_normalized=normalize_inputs
    )

    if causal:
        output = _yat_causal_performer(q_features, k_features, value, epsilon, precision)
    else:
        # Non-causal efficient computation
        # Step 1: Compute φ(K)^T @ V
        kv = jnp.einsum(
            "...khm,...khd->...hmd", k_features, value, precision=precision
        )

        # Step 2: Compute φ(Q) @ (φ(K)^T @ V)
        qkv = jnp.einsum(
            "...qhm,...hmd->...qhd", q_features, kv, precision=precision
        )

        # Step 3: Normalizer
        k_sum = jnp.sum(k_features, axis=-3)
        normalizer = jnp.einsum(
            "...qhm,...hm->...qh", q_features, k_sum, precision=precision
        )
        normalizer = normalizer[..., None] + epsilon

        output = qkv / normalizer

    # Apply alpha scaling: y *= alpha (simple multiply)
    if alpha is not None:
        alpha_val = jnp.asarray(alpha, dtype=dtype)
        output = output * alpha_val

    # Apply dropout to output if needed
    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep = random.bernoulli(dropout_rng, keep_prob, output.shape)
        output = output * keep / keep_prob

    return output


def _yat_causal_performer(
    q_features: Array,
    k_features: Array,
    value: Array,
    epsilon: float,
    precision: PrecisionLike,
) -> Array:
    """Causal YAT Performer attention using prefix sums."""
    # Outer product for kv
    kv = jnp.einsum("...khm,...khd->...khmd", k_features, value, precision=precision)

    # Cumulative sums
    kv_cumsum = jnp.cumsum(kv, axis=-4)
    k_cumsum = jnp.cumsum(k_features, axis=-3)

    # Compute output
    numerator = jnp.einsum(
        "...qhm,...qhmd->...qhd", q_features, kv_cumsum, precision=precision
    )
    denominator = jnp.einsum(
        "...qhm,...qhm->...qh", q_features, k_cumsum, precision=precision
    )
    denominator = denominator[..., None] + epsilon

    return numerator / denominator


def create_yat_projection(
    key: Array,
    num_features: int,
    head_dim: int,
    dtype: Dtype = jnp.float32,
    orthogonal: bool = True,
) -> Array:
    """Creates random projection matrix for YAT Performer.

    Args:
        key: JAX random key.
        num_features: Number of random features.
        head_dim: Dimension of each attention head.
        dtype: Data type.
        orthogonal: If True, use orthogonal random features.

    Returns:
        Projection matrix of shape [num_features, head_dim].
    """
    if orthogonal:
        # Orthogonal random features
        num_blocks = (num_features + head_dim - 1) // head_dim
        blocks = []
        for i in range(num_blocks):
            key, subkey = random.split(key)
            random_matrix = random.normal(subkey, (head_dim, head_dim), dtype=dtype)
            q, _ = jnp.linalg.qr(random_matrix)
            blocks.append(q)
        projection = jnp.concatenate(blocks, axis=0)[:num_features]
        return projection * jnp.sqrt(head_dim).astype(dtype)
    else:
        return random.normal(key, (num_features, head_dim), dtype=dtype)

