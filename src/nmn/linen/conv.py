"""YAT convolution layers for Flax Linen."""

import jax
import jax.numpy as jnp
import jax.lax as lax
from flax import linen as nn
from flax.linen import Module, compact
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import zeros_init
from typing import Any, Optional, Sequence, Union, Tuple


class YatConv1D(Module):
    """1D YAT convolution layer for Flax Linen.
    
    This layer implements 1D convolution using the YAT algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).
    
    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (length,).
        strides: Stride of the convolution. Default (1,).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
        input_dilation: Input dilation rate. Default (1,).
        kernel_dilation: Kernel dilation rate. Default (1,).
        feature_group_count: Number of feature groups.
        use_bias: Whether to add a bias. Default True.
        use_alpha: Whether to use alpha scaling. Default True.
        dtype: The dtype of the computation.
        param_dtype: The dtype for parameters. Default float32.
        kernel_init: Initializer for kernel weights.
        bias_init: Initializer for bias.
        epsilon: Small constant for numerical stability.
    """
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] = (1,)
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID'
    input_dilation: Sequence[int] = (1,)
    kernel_dilation: Sequence[int] = (1,)
    feature_group_count: int = 1
    use_bias: bool = True
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-6

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 1D YAT convolution.
        
        Args:
            inputs: Input tensor of shape [batch, length, channels].
            
        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        
        # Kernel shape: [kernel_size, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, self.features)
        
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
            
        if self.use_alpha:
            alpha = self.param('alpha', self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None
        
        inputs, kernel, bias, alpha = promote_dtype(inputs, kernel, bias, alpha, dtype=self.dtype)
        
        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(inputs.shape, kernel.shape, ('NWC', 'WIO', 'NWC'))
        
        dot_prod_map = lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Compute ||input_patches||^2
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)
        
        patch_sq_sum_raw = lax.conv_general_dilated(
            inputs_squared,
            ones_kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Repeat to match output channels
        if self.feature_group_count > 1:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1)
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)
        
        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, -1))
        
        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        
        # YAT output
        y = dot_prod_map**2 / (distance_sq + self.epsilon)
        
        if bias is not None:
            y = y + bias.reshape((1, 1, -1))
        
        if alpha is not None:
            scale = (jnp.sqrt(float(self.features)) / jnp.log(1 + float(self.features))) ** alpha
            y = y * scale
        
        return y


class YatConv2D(Module):
    """2D YAT convolution layer for Flax Linen.
    
    This layer implements 2D convolution using the YAT algorithm.
    
    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (height, width).
        strides: Stride of the convolution. Default (1, 1).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
        input_dilation: Input dilation rate. Default (1, 1).
        kernel_dilation: Kernel dilation rate. Default (1, 1).
        feature_group_count: Number of feature groups.
        use_bias: Whether to add a bias. Default True.
        use_alpha: Whether to use alpha scaling. Default True.
        dtype: The dtype of the computation.
        param_dtype: The dtype for parameters. Default float32.
        kernel_init: Initializer for kernel weights.
        bias_init: Initializer for bias.
        epsilon: Small constant for numerical stability.
    """
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] = (1, 1)
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID'
    input_dilation: Sequence[int] = (1, 1)
    kernel_dilation: Sequence[int] = (1, 1)
    feature_group_count: int = 1
    use_bias: bool = True
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-6

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 2D YAT convolution.
        
        Args:
            inputs: Input tensor of shape [batch, height, width, channels].
            
        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        
        # Kernel shape: [height, width, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, self.features)
        
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
            
        if self.use_alpha:
            alpha = self.param('alpha', self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None
        
        inputs, kernel, bias, alpha = promote_dtype(inputs, kernel, bias, alpha, dtype=self.dtype)
        
        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(inputs.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
        
        dot_prod_map = lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Compute ||input_patches||^2
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)
        
        patch_sq_sum_raw = lax.conv_general_dilated(
            inputs_squared,
            ones_kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Repeat to match output channels
        if self.feature_group_count > 1:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1)
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)
        
        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, -1))
        
        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        
        # YAT output
        y = dot_prod_map**2 / (distance_sq + self.epsilon)
        
        if bias is not None:
            y = y + bias.reshape((1, 1, 1, -1))
        
        if alpha is not None:
            scale = (jnp.sqrt(float(self.features)) / jnp.log(1 + float(self.features))) ** alpha
            y = y * scale
        
        return y


class YatConv3D(Module):
    """3D YAT convolution layer for Flax Linen.
    
    This layer implements 3D convolution using the YAT algorithm.
    
    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (depth, height, width).
        strides: Stride of the convolution. Default (1, 1, 1).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
        input_dilation: Input dilation rate. Default (1, 1, 1).
        kernel_dilation: Kernel dilation rate. Default (1, 1, 1).
        feature_group_count: Number of feature groups.
        use_bias: Whether to add a bias. Default True.
        use_alpha: Whether to use alpha scaling. Default True.
        dtype: The dtype of the computation.
        param_dtype: The dtype for parameters. Default float32.
        kernel_init: Initializer for kernel weights.
        bias_init: Initializer for bias.
        epsilon: Small constant for numerical stability.
    """
    features: int
    kernel_size: Sequence[int]
    strides: Sequence[int] = (1, 1, 1)
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID'
    input_dilation: Sequence[int] = (1, 1, 1)
    kernel_dilation: Sequence[int] = (1, 1, 1)
    feature_group_count: int = 1
    use_bias: bool = True
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-6

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 3D YAT convolution.
        
        Args:
            inputs: Input tensor of shape [batch, depth, height, width, channels].
            
        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        
        # Kernel shape: [depth, height, width, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, self.features)
        
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.param_dtype)
        else:
            bias = None
            
        if self.use_alpha:
            alpha = self.param('alpha', self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None
        
        inputs, kernel, bias, alpha = promote_dtype(inputs, kernel, bias, alpha, dtype=self.dtype)
        
        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(inputs.shape, kernel.shape, ('NDHWC', 'DHWIO', 'NDHWC'))
        
        dot_prod_map = lax.conv_general_dilated(
            inputs,
            kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Compute ||input_patches||^2
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels // self.feature_group_count, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)
        
        patch_sq_sum_raw = lax.conv_general_dilated(
            inputs_squared,
            ones_kernel,
            window_strides=self.strides,
            padding=self.padding,
            lhs_dilation=self.input_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=dn,
            feature_group_count=self.feature_group_count,
        )
        
        # Repeat to match output channels
        if self.feature_group_count > 1:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1)
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)
        
        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, 1, -1))
        
        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        
        # YAT output
        y = dot_prod_map**2 / (distance_sq + self.epsilon)
        
        if bias is not None:
            y = y + bias.reshape((1, 1, 1, 1, -1))
        
        if alpha is not None:
            scale = (jnp.sqrt(float(self.features)) / jnp.log(1 + float(self.features))) ** alpha
            y = y * scale
        
        return y


# Aliases
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D

