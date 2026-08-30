"""Pallas-fused YAT L1 attention — tiled, memory-efficient, GPU/TPU/CPU.

Implements the YAT attention score with L1 normalization as a Pallas kernel
following the FlashAttention-2 tiling strategy from JAX's reference
implementation (jax.experimental.pallas.ops.gpu.attention).

The key insight: L1 normalization is *simpler* than softmax for online
accumulation because YAT scores are non-negative — no max tracking or
log-sum-exp trick is needed. We just accumulate the running sum.

Forward:
    For each Q-block (resident in SRAM), stream K/V blocks and accumulate:
        scores_ij = (q_i · k_j)² / (||q_i - k_j||² + ε) / scale
        l_i += sum_j(scores_ij)
        o_i += scores_ij @ v_j
    Final: o_i /= l_i

Backward:
    Recompute scores tile-by-tile from saved (Q, K, V, L) — never
    materializes the full [Q, K] attention matrix in HBM.

Usage:
    out = pallas_yat_l1_attention(q, k, v, epsilon=1e-5, interpret=True)

Set interpret=False (default) for GPU/TPU compilation via Triton/Mosaic.
Set interpret=True for CPU testing or debugging.
"""

from __future__ import annotations

import functools
import math
import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel
# ═══════════════════════════════════════════════════════════════════════

def _yat_l1_fwd_kernel(
    q_ref, k_ref, v_ref,
    o_ref, l_ref,
    *,
    epsilon: float,
    scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
    causal: bool,
):
  kv_seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)
  head_dim_padded = q_ref.shape[-1]
  value_dim = v_ref.shape[-1]

  q = q_ref[...]                                       # [block_q, head_dim_padded]
  q_sq = jnp.sum(q * q, axis=-1)                       # [block_q]

  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  o = jnp.zeros((block_q, value_dim), dtype=jnp.float32)

  def body(start_k, carry):
    o_prev, l_prev = carry
    curr_k_slice = pl.dslice(start_k * block_k, block_k)

    k = k_ref[curr_k_slice, :]                          # [block_k, head_dim_padded]
    v = v_ref[curr_k_slice, :]                          # [block_k, value_dim]

    dot = pl.dot(q, k.T)                                # [block_q, block_k]
    k_sq = jnp.sum(k * k, axis=-1)                     # [block_k]

    dist = q_sq[:, None] + k_sq[None, :] - 2.0 * dot
    dist = jnp.maximum(dist, 0.0) + epsilon

    scores = (dot * dot) / (dist * scale)

    if causal:
      span_q = start_q * block_q + jnp.arange(block_q)
      span_k = start_k * block_k + jnp.arange(block_k)
      scores = jnp.where(span_q[:, None] >= span_k[None, :], scores, 0.0)

    l_next = l_prev + jnp.sum(scores, axis=-1)
    o_next = o_prev + pl.dot(scores.astype(v.dtype), v)
    return o_next, l_next

  if causal:
    upper_bound = jnp.minimum(
        lax.div(block_q * (start_q + 1) + block_k - 1, block_k),
        pl.cdiv(kv_seq_len, block_k),
    )
  else:
    upper_bound = pl.cdiv(kv_seq_len, block_k)

  o, l_i = lax.fori_loop(0, upper_bound, body, (o, l_i))

  o = o / jnp.maximum(l_i[:, None], 1e-12)
  o_ref[...] = o.astype(o_ref.dtype)
  l_ref[...] = l_i


# ═══════════════════════════════════════════════════════════════════════
# Backward kernel — dK, dV accumulation
# ═══════════════════════════════════════════════════════════════════════

def _yat_l1_bwd_dkv_kernel(
    q_ref, k_ref, v_ref, l_ref, out_ref, do_ref,
    dk_ref, dv_ref,
    *,
    epsilon: float,
    scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
    causal: bool,
):
  q_seq_len = q_ref.shape[0]
  start_k = pl.program_id(0)
  head_dim_padded = k_ref.shape[-1]
  value_dim = v_ref.shape[-1]

  curr_k_slice = pl.dslice(start_k * block_k, block_k)
  k = k_ref[curr_k_slice, :]                           # [block_k, head_dim_padded]
  v = v_ref[curr_k_slice, :]
  k_sq = jnp.sum(k * k, axis=-1)                       # [block_k]

  dk = jnp.zeros((block_k, head_dim_padded), dtype=jnp.float32)
  dv = jnp.zeros((block_k, value_dim), dtype=jnp.float32)

  def body(start_q, carry):
    dk_prev, dv_prev = carry
    curr_q_slice = pl.dslice(start_q * block_q, block_q)

    q = q_ref[curr_q_slice, :]                          # [block_q, head_dim_padded]
    do = do_ref[curr_q_slice, :]
    out = out_ref[curr_q_slice, :]
    l = l_ref[curr_q_slice]                             # [block_q]

    q_sq = jnp.sum(q * q, axis=-1)
    dot = pl.dot(q, k.T)                                # [block_q, block_k]
    dist = jnp.maximum(q_sq[:, None] + k_sq[None, :] - 2.0 * dot, 0.0) + epsilon
    scores = (dot * dot) / (dist * scale)

    if causal:
      span_q = start_q * block_q + jnp.arange(block_q)
      span_k = start_k * block_k + jnp.arange(block_k)
      mask = span_q[:, None] >= span_k[None, :]
      scores = jnp.where(mask, scores, 0.0)

    l_safe = jnp.maximum(l, 1e-12)
    W = scores / l_safe[:, None]

    # dW = dO @ V^T.  The normalization correction must be global
    # across all K/V tiles: sum_j(dW_j * W_j) == dO dot output.
    dW = pl.dot(do, v.T)                                # [block_q, block_k]
    delta = jnp.sum(do * out, axis=-1, keepdims=True)
    dS = (dW - delta) / l_safe[:, None]

    if causal:
      dS = jnp.where(mask, dS, 0.0)

    # dV += W^T @ dO
    dv_next = dv_prev + pl.dot(W.astype(do.dtype).T, do)

    # dK from dS through YAT score
    inv_dist_scale = 1.0 / (dist * scale)
    d_scores_d_num = 2.0 * dot * inv_dist_scale
    d_scores_d_dist = -(dot * dot) * inv_dist_scale / dist

    g_num = dS * d_scores_d_num
    g_dist = dS * d_scores_d_dist
    g_dot_total = g_num + g_dist * (-2.0)

    dk_next = dk_prev + pl.dot(g_dot_total.astype(q.dtype).T, q)
    dk_next = dk_next + 2.0 * k * jnp.sum(g_dist, axis=0, keepdims=True).T

    return dk_next, dv_next

  lower_bound = lax.div(start_k * block_k, block_q) if causal else 0
  dk, dv = lax.fori_loop(lower_bound, pl.cdiv(q_seq_len, block_q), body, (dk, dv))

  dk_ref[curr_k_slice, :] = dk.astype(dk_ref.dtype)
  dv_ref[curr_k_slice, :] = dv.astype(dv_ref.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Backward kernel — dQ accumulation
# ═══════════════════════════════════════════════════════════════════════

def _yat_l1_bwd_dq_kernel(
    q_ref, k_ref, v_ref, l_ref, out_ref, do_ref,
    dq_ref,
    *,
    epsilon: float,
    scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
    causal: bool,
):
  kv_seq_len = k_ref.shape[0]
  start_q = pl.program_id(0)
  head_dim_padded = q_ref.shape[-1]

  curr_q_slice = pl.dslice(start_q * block_q, block_q)
  q = q_ref[curr_q_slice, :]
  do = do_ref[curr_q_slice, :]
  out = out_ref[curr_q_slice, :]
  l = l_ref[curr_q_slice]
  q_sq = jnp.sum(q * q, axis=-1)

  dq = jnp.zeros((block_q, head_dim_padded), dtype=jnp.float32)

  def body(start_k, dq_prev):
    curr_k_slice = pl.dslice(start_k * block_k, block_k)
    k = k_ref[curr_k_slice, :]
    v = v_ref[curr_k_slice, :]
    k_sq = jnp.sum(k * k, axis=-1)

    dot = pl.dot(q, k.T)
    dist = jnp.maximum(q_sq[:, None] + k_sq[None, :] - 2.0 * dot, 0.0) + epsilon
    scores = (dot * dot) / (dist * scale)

    if causal:
      span_q = start_q * block_q + jnp.arange(block_q)
      span_k = start_k * block_k + jnp.arange(block_k)
      mask = span_q[:, None] >= span_k[None, :]
      scores = jnp.where(mask, scores, 0.0)

    l_safe = jnp.maximum(l, 1e-12)
    W = scores / l_safe[:, None]

    dW = pl.dot(do, v.T)
    delta = jnp.sum(do * out, axis=-1, keepdims=True)
    dS = (dW - delta) / l_safe[:, None]

    if causal:
      dS = jnp.where(mask, dS, 0.0)

    inv_dist_scale = 1.0 / (dist * scale)
    d_scores_d_num = 2.0 * dot * inv_dist_scale
    d_scores_d_dist = -(dot * dot) * inv_dist_scale / dist

    g_num = dS * d_scores_d_num
    g_dist = dS * d_scores_d_dist
    g_dot_total = g_num + g_dist * (-2.0)

    dq_next = dq_prev + pl.dot(g_dot_total.astype(k.dtype), k)
    dq_next = dq_next + 2.0 * q * jnp.sum(g_dist, axis=-1, keepdims=True)

    return dq_next

  if causal:
    upper_bound = jnp.minimum(
        pl.cdiv((start_q + 1) * block_q, block_k),
        pl.cdiv(kv_seq_len, block_k),
    )
  else:
    upper_bound = pl.cdiv(kv_seq_len, block_k)

  dq = lax.fori_loop(0, upper_bound, body, dq)
  dq_ref[curr_q_slice, :] = dq.astype(dq_ref.dtype)


# ═══════════════════════════════════════════════════════════════════════
# Public API with custom_vjp
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 7])
def pallas_yat_l1_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    epsilon: float = 1e-5,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    interpret: bool = False,
) -> jax.Array:
  """Pallas-fused YAT L1 attention.

  Args:
      q: [..., q_len, num_heads, head_dim]
      k: [..., kv_len, num_heads, head_dim]
      v: [..., kv_len, num_heads, v_dim]
      epsilon: Denominator stability constant.
      causal: Apply a top-left causal mask (a query at position ``i`` may
        attend to key positions ``j <= i``), including cross-attention where
        ``q_len`` and ``kv_len`` differ.
      block_q: Q tile size. Ragged sequence tails are padded internally.
      block_k: K/V tile size. Ragged sequence tails are padded internally.
      interpret: If True, run on CPU via JAX tracing (for testing).

  Returns:
      Output of shape [..., q_len, num_heads, v_dim].
  """
  _validate_inputs(q, k, v, block_q, block_k)
  q, k, v, batch_shape = _flatten_batch(q, k, v)

  out = _pallas_yat_l1_fwd(q, k, v, epsilon, causal, block_q, block_k, interpret)
  return _restore_batch(out, batch_shape)


def _validate_inputs(q, k, v, block_q, block_k):
  if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
    raise ValueError("q, k, and v must have shape [..., sequence, heads, features]")
  if q.shape[:-3] != k.shape[:-3] or q.shape[:-3] != v.shape[:-3]:
    raise ValueError("q, k, and v batch dimensions must match")
  if q.shape[-1] != k.shape[-1]:
    raise ValueError("q and k head_dim must match")
  if q.shape[-2] != k.shape[-2] or q.shape[-2] != v.shape[-2]:
    raise ValueError("q, k, and v number of heads must match")
  if k.shape[-3] != v.shape[-3]:
    raise ValueError("k and v sequence lengths must match")
  if q.shape[-3] == 0 or k.shape[-3] == 0:
    raise ValueError("q and k sequence lengths must be non-zero")
  if block_q <= 0 or block_k <= 0:
    raise ValueError("block_q and block_k must be positive")
  if q.dtype != k.dtype:
    raise ValueError("q and k dtypes must match")


def _flatten_batch(q, k, v):
  batch_shape = q.shape[:-3]
  if not batch_shape:
    return q[None], k[None], v[None], batch_shape
  if len(batch_shape) == 1:
    return q, k, v, batch_shape
  flat = math.prod(batch_shape)
  return (
      q.reshape(flat, *q.shape[-3:]),
      k.reshape(flat, *k.shape[-3:]),
      v.reshape(flat, *v.shape[-3:]),
      batch_shape,
  )


def _restore_batch(x, batch_shape):
  if not batch_shape:
    return x[0]
  if len(batch_shape) > 1:
    return x.reshape(*batch_shape, *x.shape[-3:])
  return x


def _pad_sequences(q, k, v, block_q, block_k):
  q_len, kv_len = q.shape[1], k.shape[1]
  q_pad = (-q_len) % block_q
  kv_pad = (-kv_len) % block_k
  q = jnp.pad(q, ((0, 0), (0, q_pad), (0, 0), (0, 0)))
  k = jnp.pad(k, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
  v = jnp.pad(v, ((0, 0), (0, kv_pad), (0, 0), (0, 0)))
  return q, k, v, q_len, kv_len


def _pallas_yat_l1_fwd(q, k, v, epsilon, causal, block_q, block_k, interpret):
  block_q = min(block_q, q.shape[1])
  block_k = min(block_k, k.shape[1])
  q, k, v, q_len, _ = _pad_sequences(q, k, v, block_q, block_k)
  out, _ = _pallas_yat_l1_fwd_padded(
      q, k, v, epsilon, causal, block_q, block_k, interpret)
  return out[:, :q_len]


def _pallas_yat_l1_fwd_padded(q, k, v, epsilon, causal, block_q, block_k, interpret):
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len, value_dim = k.shape[1], v.shape[-1]
  scale = math.sqrt(float(head_dim))

  kernel = functools.partial(
      _yat_l1_fwd_kernel,
      epsilon=epsilon, scale=scale,
      block_q=block_q, block_k=block_k,
      head_dim=head_dim, causal=causal,
  )

  grid = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

  in_specs = [
      pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
      pl.BlockSpec((None, kv_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
  ]
  out_specs = [
      pl.BlockSpec((None, block_q, None, value_dim), lambda i, j, k: (j, i, k, 0)),
      pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
  ]
  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, q_seq_len, num_heads, value_dim), v.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len), jnp.float32),
  ]

  out, l = pl.pallas_call(
      kernel,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      out_shape=out_shapes,
      interpret=interpret,
      name="yat_l1_fwd",
  )(q, k, v)
  return out, l


def _pallas_yat_l1_fwd_with_residuals(q, k, v, epsilon, causal, block_q, block_k, interpret):
  block_q_actual = min(block_q, q.shape[1])
  block_k_actual = min(block_k, k.shape[1])
  q, k, v, q_len, kv_len = _pad_sequences(
      q, k, v, block_q_actual, block_k_actual)
  out, l = _pallas_yat_l1_fwd_padded(
      q, k, v, epsilon, causal, block_q_actual, block_k_actual, interpret)
  return out[:, :q_len], (q, k, v, l, out, q_len, kv_len)


def _pallas_yat_l1_bwd(epsilon, causal, block_q, block_k, interpret, res, do):
  q, k, v, l, out = res
  batch_size, q_seq_len, num_heads, head_dim = q.shape
  kv_seq_len, value_dim = k.shape[1], v.shape[-1]
  scale = math.sqrt(float(head_dim))

  block_q_actual = min(block_q, q_seq_len)
  block_k_actual = min(block_k, kv_seq_len)

  common = dict(epsilon=epsilon, scale=scale, head_dim=head_dim, causal=causal)

  # ── dK, dV kernel ──
  dkv_kernel = functools.partial(
      _yat_l1_bwd_dkv_kernel,
      block_q=block_q_actual, block_k=block_k_actual, **common,
  )
  grid_dkv = (pl.cdiv(kv_seq_len, block_k_actual), batch_size, num_heads)

  dk, dv = pl.pallas_call(
      dkv_kernel,
      grid=grid_dkv,
      in_specs=[
          pl.BlockSpec((None, q_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, kv_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, None, q_seq_len), lambda _, j, k: (j, k, 0)),
          pl.BlockSpec((None, q_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, q_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
      ],
      out_specs=[
          pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, kv_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct(k.shape, k.dtype),
          jax.ShapeDtypeStruct(v.shape, v.dtype),
      ],
      interpret=interpret,
      name="yat_l1_bwd_dkv",
  )(q, k, v, l, out, do)

  # ── dQ kernel ──
  dq_kernel = functools.partial(
      _yat_l1_bwd_dq_kernel,
      block_q=block_q_actual, block_k=block_k_actual, **common,
  )
  grid_dq = (pl.cdiv(q_seq_len, block_q_actual), batch_size, num_heads)

  dq = pl.pallas_call(
      dq_kernel,
      grid=grid_dq,
      in_specs=[
          pl.BlockSpec((None, q_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, kv_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, None, q_seq_len), lambda _, j, k: (j, k, 0)),
          pl.BlockSpec((None, q_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
          pl.BlockSpec((None, q_seq_len, None, value_dim), lambda _, j, k: (j, 0, k, 0)),
      ],
      out_specs=pl.BlockSpec((None, q_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
      out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
      interpret=interpret,
      name="yat_l1_bwd_dq",
  )(q, k, v, l, out, do)

  return dq, dk, dv


def _pallas_vjp_fwd(q, k, v, epsilon, causal, block_q, block_k, interpret):
  _validate_inputs(q, k, v, block_q, block_k)
  q, k, v, batch_shape = _flatten_batch(q, k, v)

  out, residuals = _pallas_yat_l1_fwd_with_residuals(
      q, k, v, epsilon, causal, block_q, block_k, interpret)

  return _restore_batch(out, batch_shape), (residuals, batch_shape)


def _pallas_vjp_bwd(epsilon, causal, block_q, block_k, interpret, res, do):
  (q, k, v, l, out, q_len, kv_len), batch_shape = res

  if not batch_shape:
    do = do[None]
  elif len(batch_shape) > 1:
    flat = math.prod(batch_shape)
    do = do.reshape(flat, *do.shape[-3:])
  do = jnp.pad(do, ((0, 0), (0, q.shape[1] - q_len), (0, 0), (0, 0)))

  dq, dk, dv = _pallas_yat_l1_bwd(epsilon, causal, block_q, block_k, interpret, (q, k, v, l, out), do)
  dq, dk, dv = dq[:, :q_len], dk[:, :kv_len], dv[:, :kv_len]
  return (
      _restore_batch(dq, batch_shape),
      _restore_batch(dk, batch_shape),
      _restore_batch(dv, batch_shape),
  )


pallas_yat_l1_attention.defvjp(_pallas_vjp_fwd, _pallas_vjp_bwd)
