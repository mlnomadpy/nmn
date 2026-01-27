import jax.numpy as jnp
from jax import Array

def softer_sigmoid(
    x: Array,
    n: float = 1.0,
) -> Array:
    """
    Squashes a non-negative score into the range [0, 1) using the soft-sigmoid function.

    The soft-sigmoid function is defined as:
    .. math::
        \\text{soft-sigmoid}_n(x) = \\frac{x^n}{1 + x^n}
    
    The power `n` modulates the softness: higher `n` makes the function approach
    zero faster for large `x`, while `n < 1` makes the decay slower.

    Args:
        x (Array): A JAX array of non-negative scores (x >= 0).
        n (float, optional): The power to raise the score to. Defaults to 1.0.

    Returns:
        Array: The squashed scores in the range [0, 1).
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")
    
    x_n = jnp.power(x, n)
    return x_n / (1.0 + x_n)
