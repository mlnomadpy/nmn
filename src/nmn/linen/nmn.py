import math
import jax.numpy as jnp
import jax.lax as lax
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from flax.typing import (
  PRNGKey as PRNGKey,
  Shape as Shape,
  DotGeneralT,
)
from flax import linen as nn
from flax.linen.initializers import zeros_init
from typing import Any, Optional

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)

class YatNMN(Module):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      use_alpha: whether to use alpha scaling (default: True).
      constant_alpha: if True, use sqrt(2) as constant alpha. If float, use that value.
        If None (default), use learnable alpha when use_alpha=True.
      positive_init: if True, apply abs() to initialized kernel weights.
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision`` for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      epsilon: small constant added to avoid division by zero (default: 1e-5).
    """
    features: int
    use_bias: bool = True
    use_alpha: bool = True
    constant_alpha: Optional[Any] = None
    positive_init: bool = False
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.xavier_normal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    dot_general: DotGeneralT | None = None
    dot_general_cls: Any = None
    return_weights: bool = False

    @compact
    def __call__(self, inputs: Any) -> Any:
        """Applies a transformation to the inputs along the last dimension using squared Euclidean distance.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (self.features, jnp.shape(inputs)[-1]),
            self.param_dtype,
        )

        # Apply positive init
        if self.positive_init:
            kernel = jnp.abs(kernel)

        # Handle alpha configuration
        _constant_alpha_value = None
        if self.constant_alpha is not None:
            if self.constant_alpha is True:
                _constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                _constant_alpha_value = float(self.constant_alpha)
            alpha = None
        elif self.use_alpha:
            alpha = self.param(
                'alpha',
                self.alpha_init,
                (1,),
                self.param_dtype,
            )
        else:
            alpha = None

        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        inputs, kernel, bias, alpha = promote_dtype(inputs, kernel, bias, alpha, dtype=self.dtype)

        # Compute dot product between input and kernel
        if self.dot_general_cls is not None:
          dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
          dot_general = self.dot_general
        else:
          dot_general = lax.dot_general
        y = dot_general(
          inputs,
          jnp.transpose(kernel),
          (((inputs.ndim - 1,), (0,)), ((), ())),
          precision=self.precision,
        )
        inputs_squared_sum = jnp.sum(inputs**2, axis=-1, keepdims=True)
        kernel_squared_sum = jnp.sum(kernel**2, axis=-1)
        distances = inputs_squared_sum + kernel_squared_sum - 2 * y

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        # YAT: (x·W + b)² / (||x - W||² + ε)
        y = y ** 2 / (distances + self.epsilon)

        # Apply alpha scaling
        if _constant_alpha_value is not None:
            y = y * _constant_alpha_value
        elif alpha is not None:
            # Simple learnable alpha scaling
            y = y * alpha

        if self.return_weights:
           return y, kernel
        return y
