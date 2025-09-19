import tensorflow as tf
import math
from typing import Optional, Any, Tuple, Union, List, Callable
import numpy as np

def create_orthogonal_matrix(shape: Tuple[int, ...], dtype: tf.DType = tf.float32) -> tf.Tensor:
    """Creates an orthogonal matrix using the standard approach for rectangular matrices."""
    num_rows, num_cols = shape
    
    # Standard approach: sample from normal distribution and apply orthogonal transformation
    if num_rows >= num_cols:
        # Tall or square matrix
        random_matrix = tf.random.normal([num_rows, num_cols], dtype=dtype)
        q, r = tf.linalg.qr(random_matrix)
        # Make it uniform by adjusting signs
        d = tf.linalg.diag_part(r)
        ph = tf.cast(tf.sign(d), dtype)
        q *= ph[None, :]
        return q
    else:
        # Wide matrix: transpose approach
        random_matrix = tf.random.normal([num_cols, num_rows], dtype=dtype)
        q, r = tf.linalg.qr(random_matrix)
        # Make it uniform
        d = tf.linalg.diag_part(r)
        ph = tf.cast(tf.sign(d), dtype)
        q *= ph[None, :]
        return tf.transpose(q)

class YatNMN(tf.Module):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.

    Args:
        features: The number of output features.
        use_bias: Whether to add a bias to the output (default: True).
        dtype: The dtype of the computation (default: tf.float32).
        epsilon: Small constant added to avoid division by zero (default: 1e-6).
        return_weights: Whether to return the weight matrix along with output (default: False).
        name: Name of the module (default: None).
    """
    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        dtype: tf.DType = tf.float32,
        epsilon: float = 1e-6,
        return_weights: bool = False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.features = features
        self.use_bias = use_bias
        self.dtype = dtype
        self.epsilon = epsilon
        self.return_weights = return_weights
        
        # Variables will be created in build
        self.is_built = False
        self.input_dim = None
        self.kernel = None
        self.bias = None
        self.alpha = None

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

        # Initialize kernel using orthogonal initialization
        kernel_shape = (self.features, last_dim)
        initial_kernel = create_orthogonal_matrix(kernel_shape, dtype=self.dtype)
        self.kernel = tf.Variable(
            initial_kernel,
            trainable=True,
            name='kernel',
            dtype=self.dtype
        )

        # Initialize alpha to ones
        self.alpha = tf.Variable(
            tf.ones([1], dtype=self.dtype),
            trainable=True,
            name='alpha'
        )

        # Initialize bias if needed
        if self.use_bias:
            self.bias = tf.Variable(
                tf.zeros([self.features], dtype=self.dtype),
                trainable=True,
                name='bias'
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

        # Compute dot product between input and transposed kernel
        y = tf.matmul(inputs, tf.transpose(self.kernel))

        # Compute squared Euclidean distances
        inputs_squared_sum = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        kernel_squared_sum = tf.reduce_sum(tf.square(self.kernel), axis=-1)

        # Reshape kernel_squared_sum for broadcasting
        kernel_squared_sum = tf.reshape(
            kernel_squared_sum,
            [1] * (len(inputs.shape) - 1) + [self.features]
        )

        distances = inputs_squared_sum + kernel_squared_sum - 2 * y

        # Apply the transformation
        y = tf.square(y) / (distances + self.epsilon)

        # Apply scaling factor
        scale = tf.pow(
            tf.cast(
                tf.sqrt(float(self.features)) / tf.math.log(1. + float(self.features)),
                self.dtype
            ),
            self.alpha
        )
        y = y * scale

        # Add bias if used
        if self.use_bias:
            # Reshape bias for proper broadcasting
            bias_shape = [1] * (len(y.shape) - 1) + [-1]
            y = y + tf.reshape(self.bias, bias_shape)

        if self.return_weights:
            return y, self.kernel
        return y

    def get_weights(self) -> List[tf.Tensor]:
        """Returns the current weights of the layer."""
        weights = [self.kernel, self.alpha]
        if self.use_bias:
            weights.append(self.bias)
        return weights

    def set_weights(self, weights: List[tf.Tensor]) -> None:
        """Sets the weights of the layer.
        
        Args:
            weights: List of tensors with shapes matching the layer's variables.
        """
        if not self.is_built:
            raise ValueError("Layer must be built before weights can be set.")
            
        expected_num = 3 if self.use_bias else 2
        if len(weights) != expected_num:
            raise ValueError(f"Expected {expected_num} weight tensors, got {len(weights)}")

        self.kernel.assign(weights[0])
        self.alpha.assign(weights[1])
        if self.use_bias:
            self.bias.assign(weights[2])


# Alias for backward compatibility
YatDense = YatNMN
