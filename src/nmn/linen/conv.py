"""YAT convolution layers for Flax Linen."""

from typing import Any, Optional, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import Module, compact
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import zeros_init

from nmn._epsilon import (
    epsilon_parameter_dtype,
    inverse_softplus,
    validate_epsilon,
    validate_epsilon_for_dtype,
)

from ._yat_core import safe_kernel_init, upcast_yat_operands, yat_score


def _epsilon_dtype(param_dtype, epsilon):
    name = epsilon_parameter_dtype(param_dtype)
    if name == "float64" and not jax.config.x64_enabled:
        raise ValueError("float64 learnable epsilon requires jax_enable_x64")
    dtype = getattr(jnp, name)
    validate_epsilon_for_dtype(epsilon, dtype)
    return dtype


def _validate_feature_groups(
    input_channels: int, output_channels: int, feature_group_count: int
) -> None:
    """Validate grouped-convolution channel partitions with clear errors."""
    if feature_group_count <= 0:
        raise ValueError("feature_group_count must be a positive integer.")
    if input_channels % feature_group_count != 0:
        raise ValueError(
            f"Input channels ({input_channels}) must be divisible by "
            f"feature_group_count ({feature_group_count})."
        )
    if output_channels % feature_group_count != 0:
        raise ValueError(
            f"features ({output_channels}) must be divisible by "
            f"feature_group_count ({feature_group_count})."
        )


class _EpsilonValidatedModule(Module):
    """Linen module base that validates static epsilon at construction."""

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_epsilon(self.epsilon)


class YatConv1D(_EpsilonValidatedModule):
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    input_dilation: Sequence[int] = (1,)
    kernel_dilation: Sequence[int] = (1,)
    feature_group_count: int = 1
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 1D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, length, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        _validate_feature_groups(
            input_channels, self.features, self.feature_group_count
        )

        # Kernel shape: [kernel_size, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.features,
        )

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NWC", "WIO", "NWC")
        )

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
        # Grouped convolution needs one patch-norm output per input group.  A
        # single output feature is invalid in XLA when feature_group_count > 1
        # (rhs output features must be divisible by the group count), and would
        # lose the mapping between each output filter and its input group.
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.feature_group_count,
        )
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
            patch_sq_sum = jnp.repeat(
                patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1
            )
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, -1))

        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


class YatConv2D(_EpsilonValidatedModule):
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    input_dilation: Sequence[int] = (1, 1)
    kernel_dilation: Sequence[int] = (1, 1)
    feature_group_count: int = 1
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 2D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, height, width, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        _validate_feature_groups(
            input_channels, self.features, self.feature_group_count
        )

        # Kernel shape: [height, width, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.features,
        )

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NHWC", "HWIO", "NHWC")
        )

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
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.feature_group_count,
        )
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
            patch_sq_sum = jnp.repeat(
                patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1
            )
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, -1))

        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


class YatConv3D(_EpsilonValidatedModule):
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    input_dilation: Sequence[int] = (1, 1, 1)
    kernel_dilation: Sequence[int] = (1, 1, 1)
    feature_group_count: int = 1
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 3D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, depth, height, width, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        input_channels = inputs.shape[-1]
        _validate_feature_groups(
            input_channels, self.features, self.feature_group_count
        )

        # Kernel shape: [depth, height, width, input_channels // groups, features]
        kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.features,
        )

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute dot product using lax.conv_general_dilated
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NDHWC", "DHWIO", "NDHWC")
        )

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
        ones_kernel_shape = tuple(self.kernel_size) + (
            input_channels // self.feature_group_count,
            self.feature_group_count,
        )
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
            patch_sq_sum = jnp.repeat(
                patch_sq_sum_raw, self.features // self.feature_group_count, axis=-1
            )
        else:
            patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, 1, -1))

        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


class YatConvTranspose1D(_EpsilonValidatedModule):
    """1D YAT transposed convolution layer for Flax Linen.

    This layer implements 1D transposed convolution using the YAT algorithm.

    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (length,).
        strides: Stride of the transposed convolution. Default (1,).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 1D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape [batch, length, channels].

        Returns:
            Output tensor after YAT transposed convolution.
        """
        input_channels = inputs.shape[-1]

        # Kernel shape for transpose conv: [kernel_size, in_channels, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels, self.features)

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute transposed convolution using lax.conv_transpose
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NWC", "WIO", "NWC")
        )

        dot_prod_map = lax.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Compute ||input_patches||^2 using transposed conv with ones kernel
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_raw = lax.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Repeat to match output channels
        patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter — sum over all axes except the out_channels axis (last)
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, -1))

        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


class YatConvTranspose2D(_EpsilonValidatedModule):
    """2D YAT transposed convolution layer for Flax Linen.

    This layer implements 2D transposed convolution using the YAT algorithm.

    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (height, width).
        strides: Stride of the transposed convolution. Default (1, 1).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 2D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape [batch, height, width, channels].

        Returns:
            Output tensor after YAT transposed convolution.
        """
        input_channels = inputs.shape[-1]

        # Kernel shape for transpose conv: [height, width, in_channels, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels, self.features)

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute transposed convolution using lax.conv_transpose
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NHWC", "HWIO", "NHWC")
        )

        dot_prod_map = lax.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Compute ||input_patches||^2 using transposed conv with ones kernel
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_raw = lax.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Repeat to match output channels
        patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter — sum over all axes except the out_channels axis (last)
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, -1))

        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


class YatConvTranspose3D(_EpsilonValidatedModule):
    """3D YAT transposed convolution layer for Flax Linen.

    This layer implements 3D transposed convolution using the YAT algorithm.

    Attributes:
        features: Number of output features (filters).
        kernel_size: Size of the convolving kernel as a tuple (depth, height, width).
        strides: Stride of the transposed convolution. Default (1, 1, 1).
        padding: Padding algorithm. Either 'VALID' or 'SAME'.
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
    padding: Union[str, Sequence[Tuple[int, int]]] = "VALID"
    use_bias: bool = True
    constant_bias: Optional[float] = None
    use_alpha: bool = True
    dtype: Optional[Any] = None
    param_dtype: Any = jnp.float32
    kernel_init: Any = nn.initializers.orthogonal()
    bias_init: Any = zeros_init()
    alpha_init: Any = lambda key, shape, dtype: jnp.ones(shape, dtype)
    epsilon: float = 1e-5
    learnable_epsilon: bool = False

    @compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply 3D YAT transposed convolution.

        Args:
            inputs: Input tensor of shape [batch, depth, height, width, channels].

        Returns:
            Output tensor after YAT transposed convolution.
        """
        input_channels = inputs.shape[-1]

        # Kernel shape for transpose conv: [depth, height, width, in_channels, features]
        kernel_shape = tuple(self.kernel_size) + (input_channels, self.features)

        kernel = self.param(
            "kernel", safe_kernel_init(self.kernel_init), kernel_shape, self.param_dtype
        )

        if self.constant_bias is not None and self.constant_bias is not False:
            bias = jnp.full(
                (self.features,), float(self.constant_bias), dtype=self.param_dtype
            )
        elif self.use_bias:
            bias = self.param(
                "bias", self.bias_init, (self.features,), self.param_dtype
            )
        else:
            bias = None

        if self.use_alpha:
            alpha = self.param("alpha", self.alpha_init, (1,), self.param_dtype)
        else:
            alpha = None

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            epsilon_dtype = _epsilon_dtype(self.param_dtype, self.epsilon)
            raw_eps_init = inverse_softplus(self.epsilon)
            epsilon_param = self.param(
                "epsilon_param",
                lambda key, shape, dtype: jnp.full(shape, raw_eps_init, dtype=dtype),
                (1,),
                epsilon_dtype,
            )
        else:
            epsilon_param = None

        inputs, kernel, bias, alpha = promote_dtype(
            inputs, kernel, bias, alpha, dtype=self.dtype
        )
        inputs, kernel, bias, alpha, output_dtype = upcast_yat_operands(
            inputs, kernel, bias, alpha
        )

        # Compute transposed convolution using lax.conv_transpose
        dn = lax.conv_dimension_numbers(
            inputs.shape, kernel.shape, ("NDHWC", "DHWIO", "NDHWC")
        )

        dot_prod_map = lax.conv_transpose(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Compute ||input_patches||^2 using transposed conv with ones kernel
        inputs_squared = inputs * inputs
        ones_kernel_shape = tuple(self.kernel_size) + (input_channels, 1)
        ones_kernel = jnp.ones(ones_kernel_shape, dtype=kernel.dtype)

        patch_sq_sum_raw = lax.conv_transpose(
            inputs_squared,
            ones_kernel,
            strides=self.strides,
            padding=self.padding,
            dimension_numbers=dn,
        )

        # Repeat to match output channels
        patch_sq_sum = jnp.repeat(patch_sq_sum_raw, self.features, axis=-1)

        # Compute ||kernel||^2 per filter — sum over all axes except the out_channels axis (last)
        kernel_sq_sum = jnp.sum(kernel**2, axis=tuple(range(kernel.ndim - 1)))
        kernel_sq_sum = kernel_sq_sum.reshape((1, 1, 1, 1, -1))

        # YAT distance
        distance_sq = patch_sq_sum + kernel_sq_sum - 2 * dot_prod_map
        return yat_score(
            dot_prod_map,
            distance_sq,
            bias=bias,
            epsilon=self.epsilon,
            epsilon_param=epsilon_param,
            alpha=alpha,
            output_dtype=output_dtype,
        )


# DEPRECATED: lowercase aliases. The canonical names are the uppercase
# variants (YatConv1D, YatConv2D, ...) — they match the names exported
# from every other backend (torch / nnx / keras / tf). The lowercase
# aliases are kept for backward compatibility and will be removed in a
# future minor release.
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D
YatConvTranspose3d = YatConvTranspose3D
