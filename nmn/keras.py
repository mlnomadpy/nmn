import tensorflow as tf
from tensorflow import keras
from keras import layers
import math
from typing import Optional, Any, Tuple, Union
import numpy as np

class YatDense(layers.Layer):
    """A custom transformation applied over the last dimension of the input using squared Euclidean distance.

    Args:
        features: The number of output features.
        use_bias: Whether to add a bias to the output (default: True).
        dtype: The dtype of the computation (default: None).
        epsilon: Small constant added to avoid division by zero (default: 1e-6).
        kernel_initializer: Initializer for the kernel weights matrix (default: orthogonal).
        bias_initializer: Initializer for the bias vector (default: zeros).
        alpha_initializer: Initializer for the alpha parameter (default: ones).
        return_weights: Whether to return the weight matrix along with output (default: False).
    """
    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        dtype: Optional[Any] = None,
        epsilon: float = 1e-6,
        kernel_initializer: Union[str, Any] = 'orthogonal',
        bias_initializer: Union[str, Any] = 'zeros',
        alpha_initializer: Union[str, Any] = 'ones',
        return_weights: bool = False,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.features = features
        self.use_bias = use_bias
        self.epsilon = epsilon
        self.return_weights = return_weights
        
        # Initialize with provided or default initializers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.alpha_initializer = keras.initializers.get(alpha_initializer)

    def build(self, input_shape):
        """Creates the layer's weights."""
        last_dim = int(input_shape[-1])
        
        # Initialize kernel weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.features, last_dim),
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype
        )
        
        # Initialize alpha parameter
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer=self.alpha_initializer,
            trainable=True,
            dtype=self.dtype
        )
        
        # Initialize bias if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.features,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Forward pass of the layer.
        
        Args:
            inputs: Input tensor.
            
        Returns:
            Output tensor or tuple of (output tensor, kernel weights) if return_weights is True.
        """
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
            tf.cast(tf.sqrt(self.features) / tf.math.log(1. + tf.cast(self.features, tf.float32)), 
                   self.dtype),
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

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update({
            'features': self.features,
            'use_bias': self.use_bias,
            'epsilon': self.epsilon,
            'return_weights': self.return_weights,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'alpha_initializer': keras.initializers.serialize(self.alpha_initializer)
        })
        return config

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.
        
        Args:
            input_shape: Shape tuple (tuple of integers) or list of shape tuples.
            
        Returns:
            Output shape tuple.
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.features
        return tuple(output_shape)