"""Import-light validation helpers shared by every NMN backend."""

from __future__ import annotations

import math
import operator
from typing import Any, TypeVar, cast

_T = TypeVar("_T")


def _has_invalid_rate_dtype(value) -> bool:
    """Detect non-real scalar arrays without importing their framework."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False

    # NumPy dtypes, and the NumPy-compatible dtypes used by JAX, expose a
    # one-character kind. Boolean, complex, and textual/object/void values are
    # not real probability scalars.
    kind = getattr(dtype, "kind", None)
    if kind in {"b", "c", "S", "U", "O", "V"}:
        return True

    # Torch and TensorFlow dtypes do not consistently expose ``kind``, but
    # provide stable names such as ``torch.bool``, ``bool`` and ``string``.
    dtype_name = str(getattr(dtype, "name", dtype)).lower()
    return (
        "bool" in dtype_name
        or "complex" in dtype_name
        or "string" in dtype_name
        or dtype_name.startswith("str")
        or "bytes" in dtype_name
        or "object" in dtype_name
    )


def validate_rate(value: _T, name: str) -> float | _T:
    """Validate a probability, preserving an abstract scalar tracer if needed."""
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        )

    # Reject vector/container configuration rather than relying on permissive
    # one-element conversions (for example ``float(np.array([0.5]))``).  A
    # zero-dimensional framework scalar remains valid.
    shape = getattr(value, "shape", None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        )
    if _has_invalid_rate_dtype(value):
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        )
    try:
        rate = float(cast(Any, value))
    except TypeError:
        # JAX transformations represent scalar arguments as abstract tracers;
        # coercing one to float raises ConcretizationTypeError (a TypeError).
        # Preserve that previously supported traced-scalar path without
        # importing JAX into this backend-neutral module. Concrete JAX arrays
        # still take the eager float/range-validation path above.
        aval = getattr(value, "aval", None)
        if aval is not None and tuple(getattr(aval, "shape", (None,))) == ():
            return value
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        ) from None
    except (ValueError, OverflowError, RuntimeError):
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        ) from None
    if not math.isfinite(rate) or not 0.0 <= rate < 1.0:
        raise ValueError(
            f"{name} must be a finite real number in [0, 1), got {value!r}"
        )
    return rate


def validate_positive_int(value, name: str) -> int:
    """Return *value* as a strictly positive integer, rejecting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        integer = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return integer
