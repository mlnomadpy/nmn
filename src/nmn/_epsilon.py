"""Framework-independent helpers for positive YAT epsilon parameters."""

from __future__ import annotations

import math


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
