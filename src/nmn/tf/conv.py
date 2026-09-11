"""YAT convolution layers for TensorFlow."""

from typing import Any, Callable, List, Optional, Tuple, Union

import tensorflow as tf

from nmn._conv_transpose import (
    canonical_same_crop_or_pad,
    canonical_transpose_config,
)
from nmn._epsilon import (
    epsilon_parameter_dtype,
    inverse_softplus,
    validate_epsilon,
    validate_epsilon_for_dtype,
)
from nmn._validation import validate_positive_int

from ._precision import reduction_safe_upcast
from ._yat_core import yat_score
from .saved_model import SingleInputSavedModelMixin


def _epsilon_variable_dtype(layer):
    dtype = tf.as_dtype(epsilon_parameter_dtype(layer.dtype))
    validate_epsilon_for_dtype(layer.epsilon, dtype)
    return dtype


def _validate_groups(filters: int, groups: int) -> None:
    """Validate the statically known grouped-convolution configuration."""
    validate_positive_int(filters, "filters")
    validate_positive_int(groups, "groups")
    if filters % groups != 0:
        raise ValueError(f"Filters ({filters}) must be divisible by groups ({groups})")


def _patch_norm_kernel(kernel_size, channels_per_group, groups, dtype):
    """Return a grouped-convolution kernel producing one norm per group."""
    return tf.ones(tuple(kernel_size) + (channels_per_group, groups), dtype=dtype)


def _grouped_convolution(inputs, kernel, groups, convolution):
    """Apply a convolution per channel group with portable gradients.

    TensorFlow's implicit grouped-convolution support differs by dimension and
    device (notably, grouped CPU gradients and grouped ``conv3d`` are not
    universally available). Explicit splitting removes those grouped-kernel
    limitations while preserving the usual contiguous group ordering. Device
    restrictions of the underlying ordinary convolution (for example some
    CPU dilated-convolution gradients) still apply.
    """
    if groups == 1:
        return convolution(inputs, kernel)
    input_groups = tf.split(inputs, groups, axis=-1)
    kernel_groups = tf.split(kernel, groups, axis=-1)
    return tf.concat(
        [
            convolution(group_inputs, group_kernel)
            for group_inputs, group_kernel in zip(input_groups, kernel_groups)
        ],
        axis=-1,
    )


def _upcast_yat_operands(inputs, kernel):
    """Accumulate low-precision convolutional YAT scores in float32."""
    if inputs.dtype in (tf.float16, tf.bfloat16):
        return reduction_safe_upcast(inputs), reduction_safe_upcast(kernel)
    return inputs, kernel


def _transpose_output_length(
    input_length, kernel_size, stride, padding, dilation, output_padding
):
    """TensorFlow-compatible implementation of the documented shape contract."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if output_padding is None:
        if padding == "SAME":
            return input_length * stride
        return input_length * stride + max(effective_kernel - stride, 0)
    if padding == "SAME":
        return input_length * stride + output_padding
    return (input_length - 1) * stride + effective_kernel + output_padding


def _adjust_transpose_same(value, adjustments):
    if adjustments is None:
        return value
    shape = tf.shape(value)
    begin = [0]
    begin.extend(max(low, 0) for low, _ in adjustments)
    begin.append(0)
    size = [shape[0]]
    size.extend(
        shape[axis] - max(low, 0) - max(high, 0)
        for axis, (low, high) in enumerate(adjustments, start=1)
    )
    size.append(shape[-1])
    value = tf.slice(value, begin, size)
    paddings = [[0, 0]]
    paddings.extend([[max(-low, 0), max(-high, 0)] for low, high in adjustments])
    paddings.append([0, 0])
    return tf.pad(value, paddings)


class YatConv1D(SingleInputSavedModelMixin, tf.Module):
    """1D YAT convolution module using TensorFlow operations.

    This module implements 1D convolution using the YAT  algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer, specifying the length of the 1D convolution window.
        strides: Integer, specifying the stride length of the convolution. Defaults to 1.
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer, dilation rate to use for dilated convolution. Defaults to 1.
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate
        _validate_groups(filters, groups)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Variables will be created in build
        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor [batch, length, channels].
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by groups ({self.groups})"
            )

        # Kernel shape: [kernel_size, input_channels_per_group, filters]
        channels_per_group = input_channels // self.groups
        kernel_shape = (self.kernel_size, channels_per_group, self.filters)

        # Initialize kernel using orthogonal initialization
        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        # Simple orthogonal-like initialization by normalizing
        kernel_init = kernel_init / tf.sqrt(
            tf.cast(channels_per_group * self.kernel_size, self.dtype)
        )

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Initialize bias (learnable only; constant bias has no Variable)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        # Initialize alpha
        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        """Builds the layer if it hasn't been built yet."""
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 1D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, length, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        assert self.input_channels is not None
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        # Compute dot product using standard convolution
        convolution = lambda x, kernel: tf.nn.conv1d(
            x,
            kernel,
            stride=self.strides,
            padding=self.padding,
            dilations=self.dilation_rate,
        )
        dot_prod_map = _grouped_convolution(inputs, kernel, self.groups, convolution)

        # Compute ||input_patches||^2 using convolution with ones kernel
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel = _patch_norm_kernel(
            (self.kernel_size,),
            self.input_channels // self.groups,
            self.groups,
            inputs.dtype,
        )

        patch_sq_sum_map_raw = _grouped_convolution(
            inputs_squared,
            ones_kernel,
            self.groups,
            convolution,
        )

        # The helper convolution emits one channel per group. Repeat each
        # group's patch norm for that group's contiguous output-filter block.
        patch_sq_sum_map = tf.repeat(
            patch_sq_sum_map_raw, self.filters // self.groups, axis=-1
        )

        # Compute ||kernel||^2 per filter
        kernel_sq_sum_per_filter = tf.reduce_sum(
            kernel**2, axis=[0, 1]
        )  # Sum over spatial and input channel dims

        # Reshape for broadcasting: [1, 1, filters]
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, -1])

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConv2D(SingleInputSavedModelMixin, tf.Module):
    """2D YAT convolution module using TensorFlow operations.

    This module implements 2D convolution using the YAT  algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple/list of 2 integers, specifying the height and width
            of the 2D convolution window.
        strides: Integer or tuple/list of 2 integers, specifying the strides of the convolution.
            Defaults to (1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer or tuple/list of 2 integers, dilation rate for dilated convolution.
            Defaults to (1, 1).
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "valid",
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size, kernel_size)
        )
        self.strides = (
            strides if isinstance(strides, (list, tuple)) else (strides, strides)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if isinstance(dilation_rate, (list, tuple))
            else (dilation_rate, dilation_rate)
        )
        _validate_groups(filters, groups)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Variables will be created in build
        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor [batch, height, width, channels].
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by groups ({self.groups})"
            )

        # Kernel shape: [kernel_height, kernel_width, input_channels_per_group, filters]
        channels_per_group = input_channels // self.groups
        kernel_shape = self.kernel_size + (channels_per_group, self.filters)

        # Initialize kernel using orthogonal initialization
        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        # Simple orthogonal-like initialization by normalizing
        kernel_init = kernel_init / tf.sqrt(
            tf.cast(
                channels_per_group * self.kernel_size[0] * self.kernel_size[1],
                self.dtype,
            )
        )

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Initialize bias (learnable only; constant bias has no Variable)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        # Initialize alpha
        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        """Builds the layer if it hasn't been built yet."""
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 2D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, height, width, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        assert self.input_channels is not None
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        # Compute dot product using standard convolution
        convolution = lambda x, kernel: tf.nn.conv2d(
            x,
            kernel,
            strides=[1] + list(self.strides) + [1],
            padding=self.padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )
        dot_prod_map = _grouped_convolution(inputs, kernel, self.groups, convolution)

        # Compute ||input_patches||^2 using convolution with ones kernel
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel = _patch_norm_kernel(
            self.kernel_size,
            self.input_channels // self.groups,
            self.groups,
            inputs.dtype,
        )

        patch_sq_sum_map_raw = _grouped_convolution(
            inputs_squared,
            ones_kernel,
            self.groups,
            convolution,
        )

        # The helper convolution emits one channel per group. Repeat each
        # group's patch norm for that group's contiguous output-filter block.
        patch_sq_sum_map = tf.repeat(
            patch_sq_sum_map_raw, self.filters // self.groups, axis=-1
        )

        # Compute ||kernel||^2 per filter
        kernel_sq_sum_per_filter = tf.reduce_sum(
            kernel**2, axis=[0, 1, 2]
        )  # Sum over spatial and input channel dims

        # Reshape for broadcasting: [1, 1, 1, filters]
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, 1, -1])

        # Compute YAT: distance_squared = ||patch||^2 + ||kernel||^2 - 2 * dot_product
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConv3D(SingleInputSavedModelMixin, tf.Module):
    """3D YAT convolution module using TensorFlow operations.

    This module implements 3D convolution using the YAT algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple/list of 3 integers, specifying the depth, height and width
            of the 3D convolution window.
        strides: Integer or tuple/list of 3 integers, specifying the strides of the convolution.
            Defaults to (1, 1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer or tuple/list of 3 integers, dilation rate for dilated convolution.
            Defaults to (1, 1, 1).
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: str = "valid",
        dilation_rate: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size, kernel_size, kernel_size)
        )
        self.strides = (
            strides
            if isinstance(strides, (list, tuple))
            else (strides, strides, strides)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if isinstance(dilation_rate, (list, tuple))
            else (dilation_rate, dilation_rate, dilation_rate)
        )
        _validate_groups(filters, groups)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Variables will be created in build
        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor [batch, depth, height, width, channels].
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by groups ({self.groups})"
            )

        # Kernel shape: [kernel_depth, kernel_height, kernel_width, input_channels_per_group, filters]
        channels_per_group = input_channels // self.groups
        kernel_shape = self.kernel_size + (channels_per_group, self.filters)

        # Initialize kernel using orthogonal initialization
        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        # Simple orthogonal-like initialization by normalizing
        fan_in = (
            channels_per_group
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
        )
        kernel_init = kernel_init / tf.sqrt(tf.cast(fan_in, self.dtype))

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Initialize bias (learnable only; constant bias has no Variable)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        # Initialize alpha
        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        """Builds the layer if it hasn't been built yet."""
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 3D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, depth, height, width, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        assert self.input_channels is not None
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        # Compute dot product using standard convolution
        convolution = lambda x, kernel: tf.nn.conv3d(
            x,
            kernel,
            strides=[1] + list(self.strides) + [1],
            padding=self.padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )
        dot_prod_map = _grouped_convolution(inputs, kernel, self.groups, convolution)

        # Compute ||input_patches||^2 using convolution with ones kernel
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel = _patch_norm_kernel(
            self.kernel_size,
            self.input_channels // self.groups,
            self.groups,
            inputs.dtype,
        )

        patch_sq_sum_map_raw = _grouped_convolution(
            inputs_squared,
            ones_kernel,
            self.groups,
            convolution,
        )

        # The helper convolution emits one channel per group. Repeat each
        # group's patch norm for that group's contiguous output-filter block.
        patch_sq_sum_map = tf.repeat(
            patch_sq_sum_map_raw, self.filters // self.groups, axis=-1
        )

        # Compute ||kernel||^2 per filter
        kernel_sq_sum_per_filter = tf.reduce_sum(
            kernel**2, axis=[0, 1, 2, 3]
        )  # Sum over spatial and input channel dims

        # Reshape for broadcasting: [1, 1, 1, 1, filters]
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, 1, 1, -1])

        # Compute YAT: distance_squared = ||patch||^2 + ||kernel||^2 - 2 * dot_product
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConvTranspose1D(SingleInputSavedModelMixin, tf.Module):
    """1D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 1D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer, specifying the length of the 1D convolution window.
        strides: Integer, specifying the stride length. Defaults to 1.
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to 1.
        output_padding: Optional high-side extension. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: int = 1,
        output_padding: Optional[int] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.dilation_rate = dilation_rate
        self.output_padding = output_padding
        if output_padding is not None:
            canonical_transpose_config(
                kernel_size, strides, self.padding, dilation_rate, output_padding
            )
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor ``[batch, length, channels]``.
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        # Kernel shape for transpose conv: [kernel_size, filters, input_channels]
        kernel_shape = (self.kernel_size, self.filters, input_channels)

        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        kernel_init = kernel_init / tf.sqrt(
            tf.cast(self.filters * self.kernel_size, self.dtype)
        )

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Learnable bias variable (skipped when constant_bias is set)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 1D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape ``[batch, length, channels]``.

        Returns:
            Output tensor after YAT transposed convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        input_length = input_shape[1]

        # Calculate output length - use Python int for static calculation
        strides = self.strides
        kernel_size = self.kernel_size

        output_length = _transpose_output_length(
            input_length,
            kernel_size,
            strides,
            self.padding,
            self.dilation_rate,
            self.output_padding,
        )
        same_adjustments = (
            canonical_same_crop_or_pad(
                kernel_size,
                strides,
                self.dilation_rate,
                self.output_padding,
            )
            if self.padding == "SAME" and self.output_padding is not None
            else None
        )
        native_output_length = (
            _transpose_output_length(
                input_length,
                kernel_size,
                strides,
                "VALID",
                self.dilation_rate,
                0,
            )
            if same_adjustments
            else output_length
        )
        native_padding = "VALID" if same_adjustments else self.padding

        # Build output shape as a 1D tensor
        output_shape = tf.concat(
            [
                tf.reshape(batch_size, [1]),
                tf.reshape(native_output_length, [1]),
                tf.constant([self.filters], dtype=tf.int32),
            ],
            axis=0,
        )

        output_shape_ones = tf.concat(
            [
                tf.reshape(batch_size, [1]),
                tf.reshape(native_output_length, [1]),
                tf.constant([1], dtype=tf.int32),
            ],
            axis=0,
        )

        # Transpose convolution
        dot_prod_map = tf.nn.conv1d_transpose(
            inputs,
            kernel,
            output_shape=output_shape,
            strides=strides,
            padding=native_padding,
            dilations=self.dilation_rate,
        )

        # For transpose conv, compute YAT distance calculation
        inputs_squared = inputs * inputs

        # Ones kernel for patch norms
        ones_kernel_shape = (kernel_size, 1, self.input_channels)
        ones_kernel = tf.ones(ones_kernel_shape, dtype=inputs.dtype)

        patch_sq_sum_map_raw = tf.nn.conv1d_transpose(
            inputs_squared,
            ones_kernel,
            output_shape=output_shape_ones,
            strides=strides,
            padding=native_padding,
            dilations=self.dilation_rate,
        )
        dot_prod_map = _adjust_transpose_same(dot_prod_map, same_adjustments)
        patch_sq_sum_map_raw = _adjust_transpose_same(
            patch_sq_sum_map_raw, same_adjustments
        )

        patch_sq_sum_map = tf.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum
        kernel_sq_sum_per_filter = tf.reduce_sum(kernel**2, axis=[0, 2])
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, -1])

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConvTranspose2D(SingleInputSavedModelMixin, tf.Module):
    """2D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 2D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple of 2 integers for kernel dimensions.
        strides: Integer or tuple of 2 integers. Defaults to (1, 1).
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to (1, 1).
        output_padding: Optional high-side extensions. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        output_padding: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size, kernel_size)
        )
        self.strides = (
            strides if isinstance(strides, (list, tuple)) else (strides, strides)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if isinstance(dilation_rate, (list, tuple))
            else (dilation_rate, dilation_rate)
        )
        self.output_padding = (
            None
            if output_padding is None
            else (
                tuple(output_padding)
                if isinstance(output_padding, (list, tuple))
                else (output_padding, output_padding)
            )
        )
        if self.output_padding is not None:
            canonical_transpose_config(
                self.kernel_size,
                self.strides,
                self.padding,
                self.dilation_rate,
                self.output_padding,
            )
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor ``[batch, height, width, channels]``.
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        # Kernel shape for transpose conv: [height, width, filters, input_channels]
        kernel_shape = self.kernel_size + (self.filters, input_channels)

        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        fan_in = self.filters * self.kernel_size[0] * self.kernel_size[1]
        kernel_init = kernel_init / tf.sqrt(tf.cast(fan_in, self.dtype))

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Learnable bias variable (skipped when constant_bias is set)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 2D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape ``[batch, height, width, channels]``.

        Returns:
            Output tensor after YAT transposed convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        batch_size = tf.shape(inputs)[0]
        input_height = tf.shape(inputs)[1]
        input_width = tf.shape(inputs)[2]

        output_height = _transpose_output_length(
            input_height,
            self.kernel_size[0],
            self.strides[0],
            self.padding,
            self.dilation_rate[0],
            None if self.output_padding is None else self.output_padding[0],
        )
        output_width = _transpose_output_length(
            input_width,
            self.kernel_size[1],
            self.strides[1],
            self.padding,
            self.dilation_rate[1],
            None if self.output_padding is None else self.output_padding[1],
        )

        same_adjustments = (
            canonical_same_crop_or_pad(
                self.kernel_size,
                self.strides,
                self.dilation_rate,
                self.output_padding,
            )
            if self.padding == "SAME" and self.output_padding is not None
            else None
        )
        native_output_height = (
            _transpose_output_length(
                input_height,
                self.kernel_size[0],
                self.strides[0],
                "VALID",
                self.dilation_rate[0],
                0,
            )
            if same_adjustments
            else output_height
        )
        native_output_width = (
            _transpose_output_length(
                input_width,
                self.kernel_size[1],
                self.strides[1],
                "VALID",
                self.dilation_rate[1],
                0,
            )
            if same_adjustments
            else output_width
        )
        native_padding = "VALID" if same_adjustments else self.padding

        output_shape = [
            batch_size,
            native_output_height,
            native_output_width,
            self.filters,
        ]

        # Transpose convolution
        dot_prod_map = tf.nn.conv2d_transpose(
            inputs,
            kernel,
            output_shape=output_shape,
            strides=[1] + list(self.strides) + [1],
            padding=native_padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )

        # For transpose conv, compute YAT distance calculation
        inputs_squared = inputs * inputs

        # Ones kernel for patch norms
        ones_kernel_shape = self.kernel_size + (1, self.input_channels)
        ones_kernel = tf.ones(ones_kernel_shape, dtype=inputs.dtype)

        patch_sq_sum_map_raw = tf.nn.conv2d_transpose(
            inputs_squared,
            ones_kernel,
            output_shape=[batch_size, native_output_height, native_output_width, 1],
            strides=[1] + list(self.strides) + [1],
            padding=native_padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )
        dot_prod_map = _adjust_transpose_same(dot_prod_map, same_adjustments)
        patch_sq_sum_map_raw = _adjust_transpose_same(
            patch_sq_sum_map_raw, same_adjustments
        )

        patch_sq_sum_map = tf.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum
        kernel_sq_sum_per_filter = tf.reduce_sum(kernel**2, axis=[0, 1, 3])
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, 1, -1])

        # YAT computation
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConvTranspose3D(SingleInputSavedModelMixin, tf.Module):
    """3D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 3D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple of 3 integers for kernel dimensions.
        strides: Integer or tuple of 3 integers. Defaults to (1, 1, 1).
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to (1, 1, 1).
        output_padding: Optional high-side extensions. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        output_padding: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size, kernel_size, kernel_size)
        )
        self.strides = (
            strides
            if isinstance(strides, (list, tuple))
            else (strides, strides, strides)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if isinstance(dilation_rate, (list, tuple))
            else (dilation_rate, dilation_rate, dilation_rate)
        )
        self.output_padding = (
            None
            if output_padding is None
            else (
                tuple(output_padding)
                if isinstance(output_padding, (list, tuple))
                else (output_padding, output_padding, output_padding)
            )
        )
        if self.output_padding is not None:
            canonical_transpose_config(
                self.kernel_size,
                self.strides,
                self.padding,
                self.dilation_rate,
                self.output_padding,
            )
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.is_built = False
        self.input_channels = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor
                ``[batch, depth, height, width, channels]``.
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        # Kernel shape for transpose conv: [depth, height, width, filters, input_channels]
        kernel_shape = self.kernel_size + (self.filters, input_channels)

        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        fan_in = (
            self.filters
            * self.kernel_size[0]
            * self.kernel_size[1]
            * self.kernel_size[2]
        )
        kernel_init = kernel_init / tf.sqrt(tf.cast(fan_in, self.dtype))

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Learnable bias variable (skipped when constant_bias is set)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 3D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape
                ``[batch, depth, height, width, channels]``.

        Returns:
            Output tensor after YAT transposed convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        batch_size = tf.shape(inputs)[0]
        input_depth = tf.shape(inputs)[1]
        input_height = tf.shape(inputs)[2]
        input_width = tf.shape(inputs)[3]

        output_depth = _transpose_output_length(
            input_depth,
            self.kernel_size[0],
            self.strides[0],
            self.padding,
            self.dilation_rate[0],
            None if self.output_padding is None else self.output_padding[0],
        )
        output_height = _transpose_output_length(
            input_height,
            self.kernel_size[1],
            self.strides[1],
            self.padding,
            self.dilation_rate[1],
            None if self.output_padding is None else self.output_padding[1],
        )
        output_width = _transpose_output_length(
            input_width,
            self.kernel_size[2],
            self.strides[2],
            self.padding,
            self.dilation_rate[2],
            None if self.output_padding is None else self.output_padding[2],
        )

        same_adjustments = (
            canonical_same_crop_or_pad(
                self.kernel_size,
                self.strides,
                self.dilation_rate,
                self.output_padding,
            )
            if self.padding == "SAME" and self.output_padding is not None
            else None
        )
        native_output_depth = (
            _transpose_output_length(
                input_depth,
                self.kernel_size[0],
                self.strides[0],
                "VALID",
                self.dilation_rate[0],
                0,
            )
            if same_adjustments
            else output_depth
        )
        native_output_height = (
            _transpose_output_length(
                input_height,
                self.kernel_size[1],
                self.strides[1],
                "VALID",
                self.dilation_rate[1],
                0,
            )
            if same_adjustments
            else output_height
        )
        native_output_width = (
            _transpose_output_length(
                input_width,
                self.kernel_size[2],
                self.strides[2],
                "VALID",
                self.dilation_rate[2],
                0,
            )
            if same_adjustments
            else output_width
        )
        native_padding = "VALID" if same_adjustments else self.padding

        output_shape = [
            batch_size,
            native_output_depth,
            native_output_height,
            native_output_width,
            self.filters,
        ]

        # Transpose convolution
        dot_prod_map = tf.nn.conv3d_transpose(
            inputs,
            kernel,
            output_shape=output_shape,
            strides=[1] + list(self.strides) + [1],
            padding=native_padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )

        # For transpose conv, compute YAT distance calculation
        inputs_squared = inputs * inputs

        # Ones kernel for patch norms
        ones_kernel_shape = self.kernel_size + (1, self.input_channels)
        ones_kernel = tf.ones(ones_kernel_shape, dtype=inputs.dtype)

        patch_sq_sum_map_raw = tf.nn.conv3d_transpose(
            inputs_squared,
            ones_kernel,
            output_shape=[
                batch_size,
                native_output_depth,
                native_output_height,
                native_output_width,
                1,
            ],
            strides=[1] + list(self.strides) + [1],
            padding=native_padding,
            dilations=[1] + list(self.dilation_rate) + [1],
        )
        dot_prod_map = _adjust_transpose_same(dot_prod_map, same_adjustments)
        patch_sq_sum_map_raw = _adjust_transpose_same(
            patch_sq_sum_map_raw, same_adjustments
        )

        patch_sq_sum_map = tf.repeat(patch_sq_sum_map_raw, self.filters, axis=-1)

        # Compute kernel squared sum
        kernel_sq_sum_per_filter = tf.reduce_sum(kernel**2, axis=[0, 1, 2, 4])
        kernel_sq_sum_reshaped = tf.reshape(kernel_sq_sum_per_filter, [1, 1, 1, 1, -1])

        # YAT computation
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


# DEPRECATED: lowercase aliases. The canonical names are the uppercase
# variants (YatConv1D, YatConv2D, ...) — they match the names exported
# from every other backend (torch / nnx / linen / keras). The lowercase
# aliases are kept for backward compatibility and will be removed in a
# future minor release.
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D
YatConvTranspose3d = YatConvTranspose3D
