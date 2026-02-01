import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from flax.linen.module import Module, compact
from flax.typing import (
  PRNGKey as PRNGKey,
  Shape as Shape,
  DotGeneralT,
)

from typing import (
  Any,
)
import jax.numpy as jnp
import jax.lax as lax
from flax.linen import Module, compact
from flax import linen as nn
from flax.linen.initializers import zeros_init, lecun_normal
from typing import Any, Optional

class YatNMN(Module):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.

    Attributes:
      features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision`` for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      epsilon: small constant added to avoid division by zero (default: 1e-6).
    """
    features: int
    use_bias: bool = True
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()

    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)  # Initialize alpha to 1.0
    epsilon: float = 1e-6
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
        if self.use_alpha:
            alpha = self.param(
                'alpha',
                self.alpha_init,
                (1,),  # Single scalar parameter
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
        # # Element-wise operation
        y = y ** 2 /  (distances + self.epsilon)

        if alpha is not None:
          scale = (jnp.sqrt(self.features) / jnp.log(1 + self.features)) ** alpha
          y = y * scale

        # Normalize y
        if self.return_weights:
           return y, kernel
        return y
