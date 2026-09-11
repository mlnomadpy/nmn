"""MLX adapters for the shared learnable-epsilon numerical contract."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx

from nmn._epsilon import (
    epsilon_parameter_dtype,
    inverse_softplus,
    validate_epsilon_for_dtype,
)

__all__ = ["make_epsilon_parameter"]


def make_epsilon_parameter(
    epsilon: float,
    dtype: mx.Dtype,
    shape: Sequence[int] = (1,),
) -> mx.array:
    """Create a stable raw-softplus epsilon parameter with safe storage.

    Float16 and bfloat16 layers keep this scalar in float32, matching the
    other backends.  That prevents valid tiny or large initial values from
    silently becoming zero or infinity in low-precision parameter storage.
    """
    validate_epsilon_for_dtype(epsilon, dtype)
    parameter_dtype = getattr(mx, epsilon_parameter_dtype(dtype))
    return mx.full(tuple(shape), inverse_softplus(epsilon), dtype=parameter_dtype)
