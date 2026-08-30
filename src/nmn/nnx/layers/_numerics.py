"""Shared numerical helpers for NNX YAT layers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def inverse_softplus(epsilon: float, dtype):
  """Return a stable inverse-softplus initializer in ``dtype``."""
  epsilon = float(epsilon)
  if not math.isfinite(epsilon) or epsilon <= 0.0:
    raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
  raw = epsilon + math.log(-math.expm1(-epsilon))
  return jnp.asarray([raw], dtype=dtype)


def fp32_if_low_precision(*values):
  """Upcast fp16/bf16 arrays for cancellation-sensitive YAT arithmetic."""
  low_precision = (jnp.float16, jnp.bfloat16)
  return tuple(
    _safe_fp32_upcast(value)
    if value is not None and value.dtype in low_precision
    else value
    for value in values
  )


@jax.custom_vjp
def _safe_fp32_upcast(value):
  return value.astype(jnp.float32)


def _safe_fp32_upcast_fwd(value):
  return value.astype(jnp.float32), jnp.zeros((), dtype=value.dtype)


def _safe_fp32_upcast_bwd(dtype_anchor, cotangent):
  info = jnp.finfo(dtype_anchor.dtype)
  cotangent = jnp.nan_to_num(
    cotangent, nan=0.0, posinf=info.max, neginf=info.min
  )
  cotangent = jnp.clip(cotangent, info.min, info.max)
  return (cotangent.astype(dtype_anchor.dtype),)


_safe_fp32_upcast.defvjp(_safe_fp32_upcast_fwd, _safe_fp32_upcast_bwd)


def finite_cast(value, dtype):
  """Cast to a low-precision output dtype without creating infinities."""
  if dtype not in (jnp.float16, jnp.bfloat16):
    return value.astype(dtype)
  info = jnp.finfo(dtype)
  value = jnp.nan_to_num(value, nan=0.0, posinf=info.max, neginf=info.min)
  return jnp.clip(value, info.min, info.max).astype(dtype)
