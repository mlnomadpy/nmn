"""Framework-independent helpers for positive YAT epsilon parameters."""

from __future__ import annotations

import math


_FLOAT_LIMITS = {
    "float16": (2.0 ** -24, 65504.0),
    "bfloat16": (2.0 ** -133, float.fromhex("0x1.fep127")),
    # JAX and TensorFlow flush exp/softplus float32 subnormals.  One exponent
    # above the normal boundary also avoids rounding log(tiny) below it.
    "float32": (2.0 ** -125, float.fromhex("0x1.fffffep127")),
    "float64": (
        float.fromhex("0x1.0000000000000p-1022"),
        float.fromhex("0x1.fffffffffffffp1023"),
    ),
}


def validate_epsilon(epsilon: float) -> float:
    """Return ``epsilon`` as a finite, strictly positive Python float."""
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive and finite, got {epsilon}")
    return epsilon


def inverse_softplus(epsilon: float) -> float:
    """Compute ``softplus**-1(epsilon)`` without under/overflow.

    ``log(exp(epsilon) - 1)`` loses tiny values to cancellation and overflows
    for large finite values. This equivalent form is stable across the full
    positive range of Python floats.
    """
    epsilon = validate_epsilon(epsilon)
    return epsilon + math.log(-math.expm1(-epsilon))


def dtype_name(dtype) -> str:
    """Return a canonical floating dtype name without importing a framework."""
    name = getattr(dtype, "name", str(dtype)).lower()
    for candidate in ("bfloat16", "float16", "float32", "float64"):
        if candidate in name:
            return candidate
    raise TypeError(f"unsupported epsilon parameter dtype {dtype!r}")


def epsilon_parameter_dtype(dtype) -> str:
    """Use fp32 storage for fp16/bf16 epsilon parameters."""
    name = dtype_name(dtype)
    return "float32" if name in ("float16", "bfloat16") else name


def validate_epsilon_for_dtype(epsilon: float, dtype) -> float:
    """Validate that softplus can represent ``epsilon`` in ``dtype``.

    Learnable epsilon parameters use at least fp32 storage, so low-precision
    layers retain tiny values such as 1e-20 and large values such as 1e5.
    Values outside the storage/compute dtype's reliably nonzero softplus range
    are rejected instead of silently becoming zero or infinity.
    """
    epsilon = validate_epsilon(epsilon)
    parameter_dtype = epsilon_parameter_dtype(dtype)
    smallest, largest = _FLOAT_LIMITS[parameter_dtype]
    if epsilon < smallest or epsilon > largest:
        raise ValueError(
            f"epsilon {epsilon} is not representable as a finite, strictly "
            f"positive {parameter_dtype} learnable parameter"
        )
    return epsilon
