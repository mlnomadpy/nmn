"""Memory-efficient fused YAT L1 attention with custom_vjp.

Functional API that computes YAT attention with L1 normalization without
saving the full [B, H, Q, K] attention matrix across the forward-backward
boundary.  Only (Q, K, V, L1_sum, output) are saved as residuals; the
attention weights are recomputed from Q, K during backward.

The YAT L1 attention formula:

    scores  = (Q·K + b)² / (||Q-K||² + ε)
    weights = scores / sum(scores, axis=-1)
    output  = weights @ V

Supports learnable bias (per-head broadcastable) and learnable epsilon
(scalar, passed through softplus by the caller).
"""

from __future__ import annotations

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array


def fused_yat_l1_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    bias: Optional[Array] = None,
    epsilon: float | Array = 1e-5,
    mask: Optional[Array] = None,
    scale: Optional[float] = None,
    precision: jax.lax.Precision | None = None,
) -> Array:
  """Memory-efficient YAT attention with L1 normalization.

  Computes::

      scores  = (Q·K + b)² / (||Q-K||² + ε) / scale
      weights = scores / sum(scores, axis=-1)
      output  = weights @ V

  The full attention matrix is NOT saved across the forward-backward
  boundary — only Q, K, V and a per-query scalar normalizer are kept,
  and the attention weights are recomputed in the backward pass.

  Args:
      query: [..., q_length, num_heads, head_dim]
      key:   [..., kv_length, num_heads, head_dim]
      value: [..., kv_length, num_heads, v_dim]
      bias:  Optional, broadcastable to [..., num_heads, q_length, kv_length].
             Adds to the dot product before squaring.
      epsilon: Scalar float or learnable JAX array (scalar). Added to the
             squared-distance denominator for numerical stability. When a
             JAX array is passed, gradients flow back through it.
      mask:  Optional boolean, broadcastable to
             [..., num_heads, q_length, kv_length]. ``False`` = masked out.
      scale: Denominator scale. Defaults to ``sqrt(head_dim)``.
      precision: JAX einsum precision.

  Returns:
      Output of shape [..., q_length, num_heads, v_dim].
  """
  has_bias = bias is not None
  has_mask = mask is not None
  has_eps_grad = isinstance(epsilon, jnp.ndarray)

  b = bias if has_bias else jnp.zeros(1, dtype=jnp.float32)
  e = epsilon if has_eps_grad else jnp.array(epsilon, dtype=jnp.float32)
  m = mask if has_mask else jnp.zeros(1, dtype=jnp.bool_)

  if scale is None:
    import math
    scale = math.sqrt(float(query.shape[-1]))

  return _fused_yat_l1_attn(
      query, key, value, b, e, m,
      has_bias, has_mask, has_eps_grad, scale, precision,
  )


# ── Helpers to compute YAT scores from Q, K ────────────────────────────

def _yat_scores(q_f32, k_f32, bias, epsilon, mask, has_bias, has_mask, scale):
  """Compute raw YAT scores and the L1 normalizer.

  Returns (scores, L, dist) where L = sum(scores, axis=-1, keepdims=True).
  ``dist`` = squared_dist + eps (the denominator), kept for backward.
  """
  dot = jnp.einsum("...qhd,...khd->...hqk", q_f32, k_f32)

  q_sq = jnp.sum(q_f32 ** 2, axis=-1, keepdims=True)
  k_sq = jnp.sum(k_f32 ** 2, axis=-1, keepdims=True)

  batch_dims = q_f32.ndim - 3
  perm = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
  q_sq_t = q_sq.transpose(perm)
  k_sq_t = k_sq.transpose(perm)
  k_sq_t = jnp.swapaxes(k_sq_t, -2, -1)

  raw_dist = q_sq_t + k_sq_t - 2.0 * dot
  dist = jnp.maximum(raw_dist, 0.0) + epsilon

  num = (dot + bias) if has_bias else dot
  scores = num ** 2 / (dist * scale)

  if has_mask:
    scores = jnp.where(mask, scores, 0.0)

  L = jnp.sum(scores, axis=-1, keepdims=True)
  return scores, L, dist, dot, num


# ── custom_vjp implementation ──────────────────────────────────────────

@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _fused_yat_l1_attn(query, key, value, bias, epsilon, mask,
                        has_bias, has_mask, has_eps_grad, scale, precision):
  q_f32 = query.astype(jnp.float32)
  k_f32 = key.astype(jnp.float32)

  scores, L, _, _, _ = _yat_scores(
      q_f32, k_f32, bias.astype(jnp.float32) if has_bias else bias,
      epsilon.astype(jnp.float32), mask, has_bias, has_mask, scale)

  weights = (scores / (L + 1e-12)).astype(query.dtype)

  out = jnp.einsum("...hqk,...khd->...qhd", weights, value, precision=precision)
  return out


def _fused_yat_l1_attn_fwd(query, key, value, bias, epsilon, mask,
                            has_bias, has_mask, has_eps_grad, scale, precision):
  q_f32 = query.astype(jnp.float32)
  k_f32 = key.astype(jnp.float32)

  scores, L, _, _, _ = _yat_scores(
      q_f32, k_f32, bias.astype(jnp.float32) if has_bias else bias,
      epsilon.astype(jnp.float32), mask, has_bias, has_mask, scale)

  weights = (scores / (L + 1e-12)).astype(query.dtype)
  out = jnp.einsum("...hqk,...khd->...qhd", weights, value, precision=precision)

  # Residuals: save Q, K, V, bias, eps, mask, L, out — NOT weights/scores
  return out, (query, key, value, bias, epsilon, mask, L, out)


def _fused_yat_l1_attn_bwd(has_bias, has_mask, has_eps_grad, scale, precision,
                            res, g):
  query, key, value, bias, epsilon, mask, L, out = res

  # ── Recompute attention weights ────────────────────────────────────
  q_f32 = query.astype(jnp.float32)
  k_f32 = key.astype(jnp.float32)
  e_f32 = epsilon.astype(jnp.float32)
  b_f32 = bias.astype(jnp.float32) if has_bias else bias

  scores, _, dist, dot, num = _yat_scores(
      q_f32, k_f32, b_f32, e_f32, mask, has_bias, has_mask, scale)

  L_safe = L + 1e-12
  W = scores / L_safe                   # [..., H, Q, K]

  g_f32 = g.astype(jnp.float32)
  v_f32 = value.astype(jnp.float32)

  # ── dV: grad through output = W @ V ───────────────────────────────
  # out = einsum("...hqk,...khd->...qhd", W, V)
  g_v = jnp.einsum("...hqk,...qhd->...khd", W, g_f32, precision=precision)

  # ── dW: grad from output ──────────────────────────────────────────
  dW = jnp.einsum("...qhd,...khd->...hqk", g_f32, v_f32, precision=precision)

  # ── dS: backprop through L1 normalization ─────────────────────────
  # W_i = S_i / L,  L = sum(S_j)
  # dS_i = (dW_i - sum_j(dW_j * W_j)) / L
  dW_dot_W = jnp.sum(dW * W, axis=-1, keepdims=True)
  dS = (dW - dW_dot_W) / L_safe

  if has_mask:
    dS = jnp.where(mask, dS, 0.0)

  # ── dS through YAT score formula ──────────────────────────────────
  # scores = num² / (dist * scale),   num = dot + b,   dist = ||q-k||² + eps
  inv_dist_scale = 1.0 / (dist * scale)
  inv_dist_scale_sq = inv_dist_scale / dist

  # Partials (holding other intermediates constant):
  #   d(scores)/d(num)  = 2*num / (dist*scale)   [through numerator only]
  #   d(scores)/d(dist) = -num² / (dist²*scale)  [through denominator only]
  # The dist→dot path (d(dist)/d(dot) = -2) is assembled below.
  d_scores_d_num = 2 * num * inv_dist_scale
  d_scores_d_dist = -(num ** 2) * inv_dist_scale_sq

  g_num = dS * d_scores_d_num           # [..., H, Q, K]
  g_dist = dS * d_scores_d_dist         # [..., H, Q, K]

  # Total dot gradient = numerator path (d_num/d_dot=1) + dist path (d_dist/d_dot=-2)
  g_dot_total = g_num + g_dist * (-2.0)

  # ── g_dot_total -> dQ, dK through einsum ──────────────────────────
  # dot = einsum("...qhd,...khd->...hqk")
  dQ = jnp.einsum("...hqk,...khd->...qhd", g_dot_total, k_f32, precision=precision)
  dK = jnp.einsum("...hqk,...qhd->...khd", g_dot_total, q_f32, precision=precision)

  # ── g_dist -> dQ, dK through q_sq and k_sq ────────────────────────
  # Contribution from g_dist through q_sq path: d(q_sq)/d(q) = 2*q
  # g_dist shape: [..., H, Q, K], q shape: [..., Q, H, D]
  # q_sq_t shape: [..., H, Q, 1] -- was summed over K via broadcasting
  # So g_q_sq = sum(g_dist, axis=-1, keepdims=False)  -> [..., H, Q]
  g_q_sq = jnp.sum(g_dist, axis=-1)              # [..., H, Q]
  # Transpose back: [..., H, Q] -> [..., Q, H] -> [..., Q, H, 1]
  batch_dims = query.ndim - 3
  q_perm_back = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims)
  g_q_sq_t = g_q_sq.transpose(q_perm_back)[..., None]  # [..., Q, H, 1]
  dQ += 2.0 * q_f32 * g_q_sq_t

  # Contribution from g_dist through k_sq path
  g_k_sq = jnp.sum(g_dist, axis=-2)              # [..., H, K]
  k_perm_back = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims)
  g_k_sq_t = g_k_sq.transpose(k_perm_back)[..., None]  # [..., K, H, 1]
  dK += 2.0 * k_f32 * g_k_sq_t

  # ── dBias: d(scores)/d(bias) = 2*num / (dist*scale) ──────────────
  if has_bias:
    g_bias = jnp.sum(dS * 2 * num * inv_dist_scale,
                     axis=tuple(range(batch_dims)))
    # Reduce to match bias.shape via broadcasting rules
    while g_bias.ndim > bias.ndim:
      g_bias = jnp.sum(g_bias, axis=0)
    for i in range(bias.ndim):
      if bias.shape[i] == 1 and g_bias.shape[i] != 1:
        g_bias = jnp.sum(g_bias, axis=i, keepdims=True)
    g_bias = g_bias.astype(bias.dtype)
  else:
    g_bias = jnp.zeros_like(bias)

  # ── dEpsilon: d(scores)/d(eps) = -num² / (dist² * scale) ─────────
  if has_eps_grad:
    g_eps = jnp.sum(dS * (-(num ** 2) * inv_dist_scale_sq))
    g_eps = g_eps.reshape(epsilon.shape).astype(epsilon.dtype)
  else:
    g_eps = jnp.zeros_like(epsilon)

  # ── Cast back ─────────────────────────────────────────────────────
  dQ = dQ.astype(query.dtype)
  dK = dK.astype(key.dtype)
  g_v = g_v.astype(value.dtype)

  # mask is nondiff (in nondiff_argnums? No — mask is a diff arg but we
  # return zeros since it's boolean and not differentiable)
  g_mask = jnp.zeros_like(mask)

  return dQ, dK, g_v, g_bias, g_eps, g_mask


_fused_yat_l1_attn.defvjp(_fused_yat_l1_attn_fwd, _fused_yat_l1_attn_bwd)


# ═══════════════════════════════════════════════════════════════════════
# Self-YAT L1 attention (Q = K = x, no-self-attention variant)
# ═══════════════════════════════════════════════════════════════════════
#
# Key idea: the diagonal scores (self-YAT: (||x||² + b)² / ε) are included
# in the L1 normalizer — they act as a norm-aware regularizer that makes
# off-diagonal attention weights smaller — but are zeroed before the
# weighted sum with V so tokens do not attend to themselves.
#
# Forward::
#
#     scores  = (x·x + b)² / (||x - x||² + ε) / scale
#     L       = sum(scores, axis=-1)          # includes diagonal
#     weights = (scores - diag(scores)) / L   # diagonal zeroed
#     output  = weights @ V


def fused_yat_l1_self_attention(
    x: Array,
    value: Array,
    *,
    bias: Optional[Array] = None,
    epsilon: float | Array = 1e-5,
    scale: Optional[float] = None,
    precision: jax.lax.Precision | None = None,
) -> Array:
  """YAT L1 self-attention where Q = K = x (no learned QK projections).

  Computes::

      scores  = (x·x + b)² / (||x - x||² + ε) / scale
      L       = sum(scores, axis=-1)              # includes diagonal
      weights = scores_with_zero_diagonal / L
      output  = weights @ V

  The diagonal self-YAT scores contribute to the L1 normalizer but are
  excluded from the weighted sum with V — the self-score acts as a
  norm-aware regularizer that shrinks the off-diagonal weights in
  proportion to how "strong" the token is on its own.

  Requires ``x.shape[-3]`` (sequence length) equal on both sides, i.e.
  standard self-attention with ``kv_length == q_length``. For cross
  attention, use ``fused_yat_l1_attention`` instead.

  Args:
      x:     [..., seq_len, num_heads, head_dim] — shared Q and K.
      value: [..., seq_len, num_heads, v_dim]
      bias:  Optional, broadcastable to [..., num_heads, seq_len, seq_len].
             Added to ``x·x`` before squaring.
      epsilon: Scalar float or learnable JAX array (scalar).
      scale: Denominator scale. Defaults to ``sqrt(head_dim)``.
      precision: JAX einsum precision.

  Returns:
      Output of shape [..., seq_len, num_heads, v_dim].
  """
  has_bias = bias is not None
  has_eps_grad = isinstance(epsilon, jnp.ndarray)

  b = bias if has_bias else jnp.zeros(1, dtype=jnp.float32)
  e = epsilon if has_eps_grad else jnp.array(epsilon, dtype=jnp.float32)

  if scale is None:
    import math
    scale = math.sqrt(float(x.shape[-1]))

  return _fused_yat_l1_self_attn(
      x, value, b, e,
      has_bias, has_eps_grad, scale, precision,
  )


def _yat_self_scores(x_f32, bias, epsilon, has_bias, scale):
  """Compute raw YAT scores with zero diagonal, and the L1 normalizer that
  includes the diagonal.

  Returns (scores_off_diag, L, dist, dot, num) where:
    - scores_off_diag has zero diagonal
    - L = sum(scores_full, axis=-1, keepdims=True)   # includes diagonal
  """
  # dot = x @ x^T — symmetric matmul via einsum
  dot = jnp.einsum("...qhd,...khd->...hqk", x_f32, x_f32)

  x_sq = jnp.sum(x_f32 ** 2, axis=-1, keepdims=True)   # [..., seq, heads, 1]

  batch_dims = x_f32.ndim - 3
  perm = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims, batch_dims + 2)
  x_sq_t = x_sq.transpose(perm)                         # [..., heads, seq, 1]
  x_sq_k = jnp.swapaxes(x_sq_t, -2, -1)                # [..., heads, 1, seq]

  raw_dist = x_sq_t + x_sq_k - 2.0 * dot
  dist = jnp.maximum(raw_dist, 0.0) + epsilon

  num = (dot + bias) if has_bias else dot
  scores = num ** 2 / (dist * scale)                    # [..., heads, seq, seq]

  # L includes the diagonal (full sum)
  L = jnp.sum(scores, axis=-1, keepdims=True)

  # Zero out diagonal for the weights that multiply V
  seq_len = scores.shape[-1]
  eye = jnp.eye(seq_len, dtype=jnp.bool_)
  scores_off_diag = jnp.where(eye, 0.0, scores)

  return scores_off_diag, L, dist, dot, num


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _fused_yat_l1_self_attn(x, value, bias, epsilon,
                             has_bias, has_eps_grad, scale, precision):
  x_f32 = x.astype(jnp.float32)

  scores_off, L, _, _, _ = _yat_self_scores(
      x_f32, bias.astype(jnp.float32) if has_bias else bias,
      epsilon.astype(jnp.float32), has_bias, scale)

  weights = (scores_off / (L + 1e-12)).astype(x.dtype)
  out = jnp.einsum("...hqk,...khd->...qhd", weights, value, precision=precision)
  return out


def _fused_yat_l1_self_attn_fwd(x, value, bias, epsilon,
                                 has_bias, has_eps_grad, scale, precision):
  x_f32 = x.astype(jnp.float32)

  scores_off, L, _, _, _ = _yat_self_scores(
      x_f32, bias.astype(jnp.float32) if has_bias else bias,
      epsilon.astype(jnp.float32), has_bias, scale)

  weights = (scores_off / (L + 1e-12)).astype(x.dtype)
  out = jnp.einsum("...hqk,...khd->...qhd", weights, value, precision=precision)

  return out, (x, value, bias, epsilon, L, out)


def _fused_yat_l1_self_attn_bwd(has_bias, has_eps_grad, scale, precision, res, g):
  x, value, bias, epsilon, L, out = res

  x_f32 = x.astype(jnp.float32)
  e_f32 = epsilon.astype(jnp.float32)
  b_f32 = bias.astype(jnp.float32) if has_bias else bias

  scores_off, _, dist, dot, num = _yat_self_scores(
      x_f32, b_f32, e_f32, has_bias, scale)

  seq_len = scores_off.shape[-1]
  eye = jnp.eye(seq_len, dtype=jnp.bool_)

  L_safe = L + 1e-12
  W = scores_off / L_safe                              # [..., H, S, S]

  g_f32 = g.astype(jnp.float32)
  v_f32 = value.astype(jnp.float32)

  # dV: out = W @ V
  g_v = jnp.einsum("...hqk,...qhd->...khd", W, g_f32, precision=precision)

  # dW_off = dO @ V^T
  dW_off = jnp.einsum("...qhd,...khd->...hqk", g_f32, v_f32, precision=precision)

  # ── Backprop through L1 normalization ──
  # W = scores_off / L,  L = sum(scores_full)
  #   d(W_ij)/d(scores_full_ij)  = -scores_off_ij / L²       (from L in denom)
  #                              + {1/L if i!=j else 0}       (from numerator, zeroed on diag)
  #   d(W_ij)/d(scores_full_ik)  = -scores_off_ij / L²  for k!=j
  #
  # Equivalently, in terms of g_scores_full[i, k] = sum_j dW[i,j] * d(W_ij)/d(scores_full_ik):
  #   g_scores_off_ij = dW_ij / L               (numerator path, only if i!=j)
  #   g_scores_full   -= sum_j(dW_ij * W_ij) / L  (from L's denominator)
  # These both contribute to g_scores_full which includes diagonal entries.
  dW_dot_W = jnp.sum(dW_off * W, axis=-1, keepdims=True)

  # Gradient wrt the FULL score matrix (off-diag + diagonal via L):
  # off-diagonal contribution is dW_ij / L (already zero for i=j since dW_off is def. of off-diag)
  # BUT we need to NOT include numerator contribution for diagonal entries
  off_diag_mask = jnp.logical_not(eye)
  dS = jnp.where(off_diag_mask, dW_off / L_safe, 0.0)
  # Subtract the L-through-denominator contribution (applies to every score,
  # including diagonal, since L includes diagonal)
  dS = dS - dW_dot_W / L_safe

  # ── dS through YAT score formula ──
  inv_dist_scale = 1.0 / (dist * scale)
  inv_dist_scale_sq = inv_dist_scale / dist
  d_scores_d_num = 2 * num * inv_dist_scale
  d_scores_d_dist = -(num ** 2) * inv_dist_scale_sq

  g_num = dS * d_scores_d_num
  g_dist = dS * d_scores_d_dist

  # dot = x @ x^T:  d(dot_ij)/d(x_i) = x_j,  d(dot_ij)/d(x_j) = x_i
  # Both contribute to dx. Use symmetric accumulation.
  g_dot_total = g_num + g_dist * (-2.0)

  # dx through dot path (both roles of x — as q and as k)
  dx_from_dot_q = jnp.einsum("...hqk,...khd->...qhd", g_dot_total, x_f32, precision=precision)
  dx_from_dot_k = jnp.einsum("...hqk,...qhd->...khd", g_dot_total, x_f32, precision=precision)
  dx = dx_from_dot_q + dx_from_dot_k

  # dx through x_sq paths (both roles)
  batch_dims = x.ndim - 3
  q_perm_back = tuple(range(batch_dims)) + (batch_dims + 1, batch_dims)

  g_q_sq = jnp.sum(g_dist, axis=-1)                     # [..., H, S]
  g_q_sq_t = g_q_sq.transpose(q_perm_back)[..., None]   # [..., S, H, 1]
  dx = dx + 2.0 * x_f32 * g_q_sq_t

  g_k_sq = jnp.sum(g_dist, axis=-2)                     # [..., H, S]
  g_k_sq_t = g_k_sq.transpose(q_perm_back)[..., None]
  dx = dx + 2.0 * x_f32 * g_k_sq_t

  # dBias
  if has_bias:
    g_bias = jnp.sum(dS * 2 * num * inv_dist_scale,
                     axis=tuple(range(batch_dims)))
    while g_bias.ndim > bias.ndim:
      g_bias = jnp.sum(g_bias, axis=0)
    for i in range(bias.ndim):
      if bias.shape[i] == 1 and g_bias.shape[i] != 1:
        g_bias = jnp.sum(g_bias, axis=i, keepdims=True)
    g_bias = g_bias.astype(bias.dtype)
  else:
    g_bias = jnp.zeros_like(bias)

  # dEpsilon
  if has_eps_grad:
    g_eps = jnp.sum(dS * (-(num ** 2) * inv_dist_scale_sq))
    g_eps = g_eps.reshape(epsilon.shape).astype(epsilon.dtype)
  else:
    g_eps = jnp.zeros_like(epsilon)

  dx = dx.astype(x.dtype)
  g_v = g_v.astype(value.dtype)

  return dx, g_v, g_bias, g_eps


_fused_yat_l1_self_attn.defvjp(_fused_yat_l1_self_attn_fwd, _fused_yat_l1_self_attn_bwd)
