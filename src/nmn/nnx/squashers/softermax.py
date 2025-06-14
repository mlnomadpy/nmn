import jax.numpy as jnp
from jax import Array
from typing import Optional

def softermax(
    x: Array,
    n: float = 1.0,
    epsilon: float = 1e-12,
    axis: Optional[int] = -1,
) -> Array:
    """
    Normalizes a set of non-negative scores using the Softermax function.

    The Softermax function is defined as:
    .. math::
        \\text{softermax}_n(x_k, \\{x_i\\}) = \\frac{x_k^n}{\\epsilon + \\sum_i x_i^n}

    The power `n` controls the sharpness of the distribution: `n=1` recovers
    the original Softermax, while `n > 1` makes the distribution harder (more
    peaked), and `0 < n < 1` makes it softer.

    Args:
        x (Array): A JAX array of non-negative scores.
        n (float, optional): The power to raise each score to. Defaults to 1.0.
        epsilon (float, optional): A small constant for numerical stability.
            Defaults to 1e-12.
        axis (Optional[int], optional): The axis to perform the sum over.
            Defaults to -1.

    Returns:
        Array: The normalized scores.
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")

    x_n = jnp.power(x, n)
    sum_x_n = jnp.sum(x_n, axis=axis, keepdims=True)
    return x_n / (epsilon + sum_x_n)
