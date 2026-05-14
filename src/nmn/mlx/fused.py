"""Fused YAT score for MLX via ``mx.fast.metal_kernel``.

Computes ``y = α · (x · Wᵀ + b)² / (‖x − W‖² + ε)`` in a single Metal
kernel launch when the active device is the GPU; falls back to standard
MLX ops on CPU so the same call site works everywhere.

Wrapped in :class:`mx.custom_function` so gradients flow through it the
same way they would through the unfused YAT path — the VJP uses
standard MLX ops (not the kernel) because the win here is *forward*
activation-memory reduction; doing a full backward kernel is much more
involved and out of scope for this PR.

Use via the convenience helper :func:`fused_yat_score`, or pass
``fused=True`` to :class:`nmn.mlx.YatNMN`.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx


__all__ = ["fused_yat_score", "is_gpu_available"]


_METAL_SOURCE = """
    uint o = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (o >= out_features || b >= batch_size) return;

    float dot = 0.0f;
    float dist = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float xv = (float)x[b * in_features + i];
        float wv = (float)w[o * in_features + i];
        dot += xv * wv;
        float d = xv - wv;
        dist += d * d;
    }
    float num = dot + (float)bias[o];
    float denom = dist + (float)eps[0];
    out[b * out_features + o] = (T)(alpha[0] * num * num / denom);
"""


# Lazily-compiled kernel handle. ``mx.fast.metal_kernel`` is cheap to
# construct but we still cache the handle so the JIT compile only fires
# once per process.
_kernel_cache: Optional[object] = None


def _get_kernel() -> object:
    global _kernel_cache
    if _kernel_cache is None:
        _kernel_cache = mx.fast.metal_kernel(
            name="nmn_fused_yat_score",
            input_names=["x", "w", "bias", "alpha", "eps"],
            output_names=["out"],
            source=_METAL_SOURCE,
        )
    return _kernel_cache


def is_gpu_available() -> bool:
    """``True`` iff the current default device is the Metal GPU."""
    return str(mx.default_device()) == "Device(gpu, 0)"


def _metal_forward(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    alpha: mx.array,
    eps: mx.array,
) -> mx.array:
    """Launch the fused-YAT Metal kernel for a 2-D input batch."""
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(
            f"fused metal kernel expects 2-D x and w; got x.shape={x.shape}, "
            f"w.shape={w.shape}. Flatten leading batch dims before calling."
        )
    B, In = x.shape
    Out, In2 = w.shape
    if In != In2:
        raise ValueError(f"in_features mismatch: x has {In}, w has {In2}")
    kernel = _get_kernel()
    outputs = kernel(
        inputs=[x, w, b, alpha, eps],
        template=[
            ("T", x.dtype),
            ("in_features", In),
            ("out_features", Out),
            ("batch_size", B),
        ],
        grid=(Out, B, 1),
        threadgroup=(min(32, Out), min(8, max(B, 1)), 1),
        output_shapes=[(B, Out)],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def _standard_forward(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    alpha: mx.array,
    eps: mx.array,
) -> mx.array:
    """Reference forward using standard MLX ops — used as the CPU fallback."""
    dot = x @ w.T
    x_sq = mx.sum(x * x, axis=-1, keepdims=True)
    w_sq = mx.sum(w * w, axis=-1)[None, :]
    dist = mx.maximum(x_sq + w_sq - 2.0 * dot, 0.0)
    num = dot + b
    return alpha * (num * num) / (dist + eps)


@mx.custom_function
def _fused_yat_core(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    alpha: mx.array,
    eps: mx.array,
) -> mx.array:
    """Functional fused YAT score; dispatches by device."""
    if is_gpu_available():
        return _metal_forward(x, w, b, alpha, eps)
    return _standard_forward(x, w, b, alpha, eps)


@_fused_yat_core.vjp
def _fused_yat_core_vjp(primals, cotangent, output):
    """Chain-rule the YAT score using standard MLX ops.

    out = α · num² / dist
    num = x · Wᵀ + b
    dist = ‖x‖² + ‖W‖² − 2·x·Wᵀ + ε

    ∂out/∂num  =  α · 2 · num / dist
    ∂out/∂dist = −α · num² / dist²
    """
    x, w, b, alpha, eps = primals
    g = cotangent

    raw_dot = x @ w.T
    x_sq = mx.sum(x * x, axis=-1, keepdims=True)
    w_sq = mx.sum(w * w, axis=-1)[None, :]
    dist_raw = mx.maximum(x_sq + w_sq - 2.0 * raw_dot, 0.0) + eps
    num = raw_dot + b
    inv = 1.0 / dist_raw

    g_num = g * (alpha * 2.0 * num * inv)
    g_dist = g * (-alpha * num * num * (inv * inv))

    # dot enters num (+1) and dist (−2·dot in the expansion).
    g_dot_total = g_num + g_dist * (-2.0)

    # ∂dist/∂x picks up 2·x via ‖x‖²; sum over the out-features axis.
    g_x = g_dot_total @ w + mx.sum(g_dist, axis=-1, keepdims=True) * (2.0 * x)
    # ∂dist/∂w picks up 2·w via ‖W‖²; sum the broadcast over the batch.
    g_w = g_dot_total.T @ x + mx.sum(g_dist, axis=0)[..., None] * (2.0 * w)

    g_b = mx.sum(g_num, axis=0)
    # raw_yat = num² / dist  →  dα = Σ g · raw_yat
    raw_yat = num * num * inv
    g_alpha = mx.sum(g * raw_yat).reshape(alpha.shape)
    g_eps = mx.sum(g * (-alpha * num * num * (inv * inv))).reshape(eps.shape)

    return g_x, g_w, g_b, g_alpha, g_eps


def fused_yat_score(
    x: mx.array,
    w: mx.array,
    bias: Optional[mx.array] = None,
    alpha: Optional[mx.array] = None,
    epsilon: float = 1e-5,
) -> mx.array:
    """Convenience wrapper that fills in defaults and flattens batch dims.

    Args:
        x: ``(..., in_features)``. Leading batch dims are flattened then
            restored on the way out.
        w: ``(out_features, in_features)`` — same layout as
            :class:`nmn.mlx.YatNMN.kernel`.
        bias: Optional ``(out_features,)``; defaults to all zeros.
        alpha: Optional scalar (or 1-element array); defaults to 1.0.
        epsilon: Stability constant.

    Returns:
        ``(..., out_features)``.
    """
    out_features = w.shape[0]
    in_features = w.shape[1]
    if x.shape[-1] != in_features:
        raise ValueError(
            f"x last dim ({x.shape[-1]}) must match w in_features ({in_features})"
        )

    leading = x.shape[:-1]
    flat_x = mx.reshape(x, (int(_prod(leading)), in_features))
    if bias is None:
        bias = mx.zeros((out_features,), dtype=x.dtype)
    if alpha is None:
        alpha = mx.ones((1,), dtype=x.dtype)
    elif alpha.ndim == 0:
        alpha = mx.reshape(alpha, (1,))
    eps_arr = mx.array([epsilon], dtype=x.dtype)

    out_flat = _fused_yat_core(flat_x, w, bias, alpha, eps_arr)
    return mx.reshape(out_flat, leading + (out_features,))


def _prod(shape) -> int:
    p = 1
    for d in shape:
        p *= int(d)
    return p
