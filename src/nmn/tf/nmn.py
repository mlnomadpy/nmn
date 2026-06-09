"""YAT dense layer for TensorFlow (tf.Module-based, eager and graph compatible)."""

import tensorflow as tf
import math
from typing import Optional, Tuple, Union, List

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatNMN(tf.Module):
    """Dense layer implementing the ⵟ-product (YAT) transformation.

    Computes ``y = (x·Wᵀ + b)² / (‖x − W‖² + ε)`` over the last dimension of
    the input, providing intrinsic non-linearity without a separate activation
    function.

    Args:
        features: Number of output features.
        use_bias: Whether to add a bias term inside the numerator square
            (default: ``True``).
        constant_bias: If a ``float``, use that value as a fixed (non-learnable)
            bias constant. If ``None`` (default), use learnable bias when
            ``use_bias=True``.
        use_alpha: Whether to apply output scaling via an alpha parameter
            (default: ``True``).  Has no effect when ``constant_alpha`` is set.
        constant_alpha: If ``True``, applies a fixed √2 scale factor.  If a
            ``float``, applies that value as a fixed scale.  If ``None``
            (default), a learnable scalar ``alpha`` variable is created when
            ``use_alpha=True``.
        positive_init: Initialise kernel weights with their absolute values so
            all entries are non-negative (default: ``False``).
        dtype: TensorFlow dtype used for both parameters and computation
            (default: ``tf.float32``).
        epsilon: Small constant added to the denominator to avoid division by
            zero (default: ``1e-5``).
        learnable_epsilon: If ``True``, epsilon becomes a learnable parameter
            passed through softplus to guarantee strict positivity
            (default: ``False``).
        lazy: If ``True``, freeze ONLY the kernel (the feature directions): the
            kernel ``tf.Variable`` is created with ``trainable=False`` so it is
            excluded from gradients/optimizer updates. The ``bias``, ``alpha``,
            and learnable ``epsilon`` remain trainable. Composes with
            ``use_bias`` / ``use_alpha`` / ``learnable_epsilon``. Defaults to
            ``False`` (fully backward compatible). ``freeze_kernel`` is an alias.
        freeze_kernel: Alias for ``lazy``. If either is ``True`` the kernel is
            frozen.
        spherical: If ``True``, normalize inputs and each kernel row (neuron)
            to unit norm before computing the YAT formula (default: ``False``).
        weight_normalized: If ``True``, normalize each kernel row (neuron) to
            unit norm at forward time. This avoids recomputing kernel norms in
            the YAT distance calculation (default: ``False``).
        return_weights: If ``True``, ``__call__`` returns ``(output, kernel)``
            instead of just ``output`` (default: ``False``).
        name: Optional name scope for the module.

    Example::

        import tensorflow as tf
        from nmn.tf.nmn import YatNMN

        layer = YatNMN(features=128, constant_alpha=True)
        x = tf.random.normal([32, 256])
        y = layer(x)  # shape (32, 128) — no activation needed
    """
    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        positive_init: bool = False,
        dtype: tf.DType = tf.float32,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        spherical: bool = False,
        weight_normalized: bool = False,
        return_weights: bool = False,
        lazy: bool = False,
        freeze_kernel: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.features = features
        self.dtype = dtype
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        # Lazy mode: freeze ONLY the kernel (bias/alpha/epsilon stay trainable).
        self.lazy = bool(lazy or freeze_kernel)
        self.spherical = spherical
        self.weight_normalized = weight_normalized
        self.return_weights = return_weights
        self.positive_init = positive_init

        # Handle bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Handle alpha configuration
        self._constant_alpha_value = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        # Variables will be created in build
        self.is_built = False
        self.input_dim = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.
        
        Args:
            input_shape: Shape of the input tensor.
        """
        if self.is_built:
            return

        last_dim = int(input_shape[-1])
        self.input_dim = last_dim

        # Initialize kernel using Xavier normal initialization
        kernel_shape = (self.features, last_dim)
        fan_in = last_dim
        fan_out = self.features
        std = math.sqrt(2.0 / (fan_in + fan_out))
        initial_kernel = tf.random.normal(kernel_shape, stddev=std, dtype=self.dtype)
        if self.positive_init:
            initial_kernel = tf.abs(initial_kernel)
        # In lazy mode the kernel feature directions are frozen: creating the
        # Variable with trainable=False excludes it from gradients/optimizer
        # (it will not appear in self.trainable_variables). bias/alpha/epsilon
        # below stay trainable.
        self.kernel = tf.Variable(
            initial_kernel,
            trainable=not self.lazy,
            name='kernel',
            dtype=self.dtype
        )

        # Initialize alpha (only if learnable)
        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype),
                trainable=True,
                name='alpha'
            )

        # Initialize bias if needed (learnable only; constant bias has no Variable)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.features], dtype=self.dtype),
                trainable=True,
                name='bias'
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=self.dtype),
                trainable=True,
                name='epsilon_param',
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        """Builds the layer if it hasn't been built yet."""
        if not self.is_built:
            self.build(inputs.shape)
        elif self.input_dim != inputs.shape[-1]:
            raise ValueError(f'Input shape changed: expected last dimension '
                           f'{self.input_dim}, got {inputs.shape[-1]}')

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor or tuple of (output tensor, kernel weights) if return_weights is True.
        """
        # Ensure inputs are tensor
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)

        # Build if necessary
        self._maybe_build(inputs)

        kernel = self.kernel

        # Spherical mode: normalize inputs and each kernel row (neuron) to unit norm.
        # TF kernel shape is (features, last_dim) → each row is a neuron → axis=-1.
        if self.spherical:
            inputs = inputs / (
                tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)) + 1e-8
            )
            kernel = kernel / (
                tf.sqrt(tf.reduce_sum(tf.square(kernel), axis=-1, keepdims=True)) + 1e-8
            )

        # Weight normalization: normalize each kernel row at forward time.
        if self.weight_normalized:
            kernel = kernel / (
                tf.sqrt(tf.reduce_sum(tf.square(kernel), axis=-1, keepdims=True)) + 1e-8
            )

        # Compute dot product between input and transposed kernel
        y = tf.matmul(inputs, tf.transpose(kernel))

        # Compute squared distances
        if self.spherical:
            # Both x and each W_row are unit vectors → ||x - W||² = 2 - 2(x·W)
            distances = tf.maximum(2.0 - 2.0 * y, 0.0)
        else:
            inputs_squared_sum = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
            if self.weight_normalized:
                # ||W_row||² = 1 for each neuron
                kernel_squared_sum = tf.ones([self.features], dtype=kernel.dtype)
            else:
                kernel_squared_sum = tf.reduce_sum(tf.square(kernel), axis=-1)

            # Reshape kernel_squared_sum for broadcasting
            kernel_squared_sum = tf.reshape(
                kernel_squared_sum,
                [1] * (len(inputs.shape) - 1) + [self.features]
            )

            distances = tf.maximum(inputs_squared_sum + kernel_squared_sum - 2 * y, 0.0)

        # Add bias if used (learnable or constant) — inside the numerator square
        if self.use_bias:
            bias_shape = [1] * (len(y.shape) - 1) + [-1]
            if self._constant_bias_value is not None:
                bias_const = tf.constant(
                    self._constant_bias_value, shape=[self.features], dtype=self.dtype
                )
                y = y + tf.reshape(bias_const, bias_shape)
            else:
                y = y + tf.reshape(self.bias, bias_shape)

        # Resolve effective epsilon (learnable via softplus, or constant)
        if self.learnable_epsilon and self.epsilon_param is not None:
            eps = tf.nn.softplus(self.epsilon_param)
        else:
            eps = self.epsilon

        # Apply the transformation
        y = tf.square(y) / (distances + eps)

        # Apply scaling factor
        if self._constant_alpha_value is not None:
            y = y * tf.cast(self._constant_alpha_value, self.dtype)
        elif self.alpha is not None:
            # Simple learnable alpha scaling
            y = y * self.alpha


        if self.return_weights:
            return y, self.kernel
        return y

    def get_weights(self) -> List[tf.Tensor]:
        """Returns the current trainable weights of the layer.

        Returns:
            List of tensors: ``[kernel]``, ``[kernel, bias]``,
            ``[kernel, alpha]``, or ``[kernel, bias, alpha]`` depending on
            which parameters are enabled.  ``constant_alpha`` is not
            included because it is not a trainable variable.
        """
        if not self.is_built:
            raise ValueError("Layer must be built before weights can be retrieved.")
        weights = [self.kernel]
        if self.use_bias and self.bias is not None:
            weights.append(self.bias)
        if self.alpha is not None:
            weights.append(self.alpha)
        if self.epsilon_param is not None:
            weights.append(self.epsilon_param)
        return weights

    def set_weights(self, weights: List[tf.Tensor]) -> None:
        """Sets the trainable weights of the layer.

        Args:
            weights: List of tensors in the same order returned by
                :meth:`get_weights`.
        """
        if not self.is_built:
            raise ValueError("Layer must be built before weights can be set.")

        expected = self.get_weights()
        if len(weights) != len(expected):
            raise ValueError(
                f"Expected {len(expected)} weight tensors, got {len(weights)}"
            )

        self.kernel.assign(weights[0])
        idx = 1
        if self.use_bias and self.bias is not None:
            self.bias.assign(weights[idx])
            idx += 1
        if self.alpha is not None:
            self.alpha.assign(weights[idx])
            idx += 1
        if self.epsilon_param is not None:
            self.epsilon_param.assign(weights[idx])


# Alias for backward compatibility
YatDense = YatNMN
