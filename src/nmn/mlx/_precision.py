"""Autodiff-safe precision helpers for MLX YAT layers."""

from __future__ import annotations

import mlx.core as mx


def _dtype_max(dtype: mx.Dtype) -> float:
    return 65504.0 if dtype == mx.float16 else 3.38953139e38


@mx.custom_function
def _reduction_safe_upcast(value: mx.array) -> mx.array:
    return value.astype(mx.float32)


@_reduction_safe_upcast.vjp
def _reduction_safe_upcast_vjp(primals, cotangent, output):
    del output
    value = primals
    limit = _dtype_max(value.dtype)
    return mx.clip(cotangent.astype(mx.float32), -limit, limit).astype(value.dtype)


def reduction_safe_upcast(value: mx.array) -> mx.array:
    """Upcast low-precision reductions and saturate the returning cotangent."""
    if value.dtype in (mx.float16, mx.bfloat16):
        return _reduction_safe_upcast(value)
    return value


def saturating_downcast(value: mx.array, dtype: mx.Dtype) -> mx.array:
    """Two-sided finite cast to a low-precision public output dtype."""
    if dtype in (mx.float16, mx.bfloat16):
        limit = _dtype_max(dtype)
        value = mx.clip(value, -limit, limit)
    return value.astype(dtype)
