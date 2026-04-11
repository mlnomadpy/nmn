from keras.src import activations, constraints, initializers, regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src import ops
import math
import numpy as np

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)

@keras_export("keras.layers.YatNMN")
class YatNMN(Layer):
    """A YAT densely-connected NN layer.

    This layer implements the operation:
    `output = scale * (dot(input, kernel)^2 / (squared_euclidean_distance + epsilon))`
    where:
    - `scale` is a dynamic scaling factor based on output dimension
    - `squared_euclidean_distance` is computed between input and kernel
    - `epsilon` is a small constant to prevent division by zero

    Note: This layer is activation-free. Any activation function should be applied
    as a separate layer after this layer.

    Args:
        units: Positive integer, dimensionality of the output space.
        use_bias: Boolean, whether the layer uses a bias vector.
        epsilon: Float, small constant added to denominator for numerical stability.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output.
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be a 2D input with shape
        `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units,
        use_bias=True,
        constant_bias=None,
        use_alpha=True,
        constant_alpha=None,
        positive_init=False,
        epsilon=1e-5,
        learnable_epsilon=False,
        spherical=False,
        weight_normalized=False,
        kernel_initializer="glorot_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = units
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.positive_init = positive_init
        self.spherical = spherical
        self.weight_normalized = weight_normalized

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value = None
        if constant_bias is not None:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Handle alpha configuration
        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

        # Bias: learnable parameter, or None if constant_bias is set / use_bias=False
        if self.use_bias and self._constant_bias_value is None:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        # Alpha parameter (only if learnable)
        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = self.add_weight(
                name="alpha",
                shape=(1,),
                initializer="ones",
                trainable=True,
            )
        else:
            self.alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = self.add_weight(
                name="epsilon_param",
                shape=(1,),
                initializer=initializers.Constant(raw_eps),
                trainable=True,
            )
        else:
            self.epsilon_param = None

        # Apply positive init: abs(kernel)
        if self.positive_init:
            self.kernel.assign(ops.abs(self.kernel))

        # Normalize each kernel column (neuron) to unit norm if weight_normalized.
        # Kernel shape is (input_dim, units) → each neuron is a column → axis=0.
        if self.weight_normalized:
            weight_norm = ops.sqrt(ops.sum(ops.square(self.kernel), axis=0, keepdims=True))
            self.kernel.assign(self.kernel / (weight_norm + 1e-8))

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        kernel = self.kernel

        # Spherical mode: normalize inputs and kernel columns (each neuron) to unit norm
        # Kernel shape is (input_dim, units), so each neuron is a column → axis=0
        if self.spherical:
            inputs = inputs / (ops.sqrt(ops.sum(ops.square(inputs), axis=-1, keepdims=True)) + 1e-8)
            kernel = kernel / (ops.sqrt(ops.sum(ops.square(kernel), axis=0, keepdims=True)) + 1e-8)

        # Weight normalization: normalize each kernel column (neuron) at forward time.
        # Kernel shape is (input_dim, units) → axis=0 to normalize per neuron.
        if self.weight_normalized:
            kernel = kernel / (ops.sqrt(ops.sum(ops.square(kernel), axis=0, keepdims=True)) + 1e-8)

        # Compute dot product
        dot_product = ops.matmul(inputs, kernel)

        # Compute squared distances — bias must NOT affect distance,
        # distance = ||x - W||² is purely geometric (independent of bias)
        if self.spherical:
            # Spherical: ||x||=1, ||W||=1, so dist = 2 - 2*(x·W)
            distances = ops.maximum(2 - 2 * dot_product, 0.0)
        else:
            inputs_squared_sum = ops.sum(ops.square(inputs), axis=-1, keepdims=True)
            if self.weight_normalized:
                kernel_squared_sum = ops.ones((self.units,), dtype=inputs.dtype)
            else:
                kernel_squared_sum = ops.sum(ops.square(kernel), axis=0)
            distances = ops.maximum(inputs_squared_sum + kernel_squared_sum - 2 * dot_product, 0.0)

        # Add bias inside the numerator square: (dot + bias)^2 / dist
        if self.use_bias:
            if self._constant_bias_value is not None:
                dot_product = dot_product + self._constant_bias_value
            else:
                dot_product = ops.add(dot_product, self.bias)

        # Resolve effective epsilon (learnable via softplus, or constant)
        if self.learnable_epsilon and self.epsilon_param is not None:
            eps = ops.softplus(self.epsilon_param)
        else:
            eps = self.epsilon

        # Compute inverse square attention
        outputs = dot_product ** 2 / (distances + eps)

        # Apply alpha scaling
        if self._constant_alpha_value is not None:
            outputs = outputs * self._constant_alpha_value
        elif self.alpha is not None:
            # Simple learnable alpha scaling
            outputs = outputs * self.alpha

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "constant_bias": self.constant_bias,
            "use_alpha": self.use_alpha,
            "constant_alpha": self.constant_alpha,
            "positive_init": self.positive_init,
            "epsilon": self.epsilon,
            "learnable_epsilon": self.learnable_epsilon,
            "spherical": self.spherical,
            "weight_normalized": self.weight_normalized,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        })
        return config


# Alias for backward compatibility
YatDense = YatNMN
