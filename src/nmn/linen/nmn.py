import math
import jax
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
      constant_bias: if a float, use that value as a fixed (non-learnable) bias
        constant. If None (default), use learnable bias when use_bias=True.
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
      learnable_epsilon: if True, epsilon becomes a learnable parameter passed
        through softplus to guarantee strict positivity (default: False).
      spherical: if True, normalize inputs and each kernel row (neuron) to unit
        norm before computing the YAT formula (default: False).
      weight_normalized: if True, normalize each kernel row (neuron) to unit
        norm at forward time. This optimization avoids recomputing kernel norms
        in YAT distance calculation since they are guaranteed to be 1.0
        (default: False).
    """
    features: int
    use_bias: bool = True
    constant_bias: Optional[float] = None
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
    learnable_epsilon: bool = False
    spherical: bool = False
    weight_normalized: bool = False
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
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

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

        # Bias: learnable, constant, or none
        if self.constant_bias is not None:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                'bias', self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps_init = math.log(math.exp(self.epsilon) - 1.0)
            epsilon_param = self.param(
                'epsilon_param',
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                self.param_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(inputs, kernel, bias, alpha, dtype=self.dtype)

        # Spherical mode: normalize inputs and each kernel row (neuron) to unit norm.
        # Linen kernel shape is (features, input_dim), so each row is a neuron → axis=-1.
        if self.spherical:
            inputs = inputs / (jnp.linalg.norm(inputs, axis=-1, keepdims=True) + 1e-8)
            kernel = kernel / (jnp.linalg.norm(kernel, axis=-1, keepdims=True) + 1e-8)

        # Weight normalization: normalize each kernel row at forward time.
        if self.weight_normalized:
            kernel = kernel / (
                jnp.sqrt(jnp.sum(kernel ** 2, axis=-1, keepdims=True)) + 1e-8
            )

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

        # Compute squared distances
        if self.spherical:
            # Both x and each W_row are unit vectors → ||x - W||² = 2 - 2(x·W)
            distances = jnp.maximum(2.0 - 2.0 * y, 0.0)
        else:
            inputs_squared_sum = jnp.sum(inputs ** 2, axis=-1, keepdims=True)
            if self.weight_normalized:
                # ||W_row||² = 1 for each neuron
                kernel_squared_sum = jnp.ones((self.features,), dtype=kernel.dtype)
            else:
                kernel_squared_sum = jnp.sum(kernel ** 2, axis=-1)
            # Clamp to zero: bf16 cancellation can make distance negative when x ≈ W
            distances = jnp.maximum(
                inputs_squared_sum + kernel_squared_sum - 2 * y, 0.0
            )

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))

        # Resolve effective epsilon (learnable via softplus, or constant)
        if epsilon_param is not None:
            eps = jax.nn.softplus(epsilon_param.astype(y.dtype))
        else:
            eps = self.epsilon

        # YAT: (x·W + b)² / (||x - W||² + ε)
        y = y ** 2 / (distances + eps)

        # Apply alpha scaling
        if _constant_alpha_value is not None:
            y = y * _constant_alpha_value
        elif alpha is not None:
            # Simple learnable alpha scaling
            y = y * alpha

        if self.return_weights:
           return y, kernel
        return y
