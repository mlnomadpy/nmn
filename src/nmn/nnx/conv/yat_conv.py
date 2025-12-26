"""YAT Convolution Module.

This module implements the YatConv layer, which applies the YAT
 formula to convolution operations:

    y = (x * W)² / (||x - W||² + ε)

where * denotes the convolution operation, and the distance is computed
patch-wise between input patches and kernel weights.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from flax import nnx
from flax.nnx.module import Module
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import (
    Dtype,
    Initializer,
    PrecisionLike,
    ConvGeneralDilatedT,
    PaddingLike,
    PromoteDtypeFn,
)

from .utils import (
    canonicalize_padding,
    conv_dimension_numbers,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    DEFAULT_CONSTANT_ALPHA,
)

Array = jax.Array


class YatConv(Module):
    """YAT Convolution Module wrapping ``lax.conv_general_dilated``.

    Applies the YAT formula to convolution:
        y = (x * W)² / (||x - W||² + ε)

    Example usage::

        >>> from flax import nnx
        >>> from nmn.nnx.conv import YatConv
        >>> import jax.numpy as jnp

        >>> rngs = nnx.Rngs(0)
        >>> x = jnp.ones((1, 8, 3))  # (batch, length, channels)

        >>> # Valid padding
        >>> layer = YatConv(in_features=3, out_features=4, kernel_size=(3,),
        ...                 padding='VALID', rngs=rngs)
        >>> out = layer(x)
        >>> print(out.shape)
        (1, 6, 4)

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels (filters).
        kernel_size: Shape of the convolutional kernel. For 1D convolution,
            can be an integer. For 2D/3D, must be a sequence of integers.
        strides: Inter-window strides (default: 1).
        padding: 'SAME', 'VALID', 'CIRCULAR', 'REFLECT', 'CAUSAL', or
            sequence of (low, high) pairs.
        input_dilation: Dilation factor for inputs (default: 1).
        kernel_dilation: Dilation factor for kernel (default: 1).
        feature_group_count: For grouped convolution (default: 1).
        use_bias: Whether to add a bias (default: True).
        use_alpha: Whether to use alpha scaling (default: True).
        constant_alpha: If True, use sqrt(2) as constant alpha. If float,
            use that value. If None (default), use learnable alpha.
        use_dropconnect: Whether to use DropConnect (default: False).
        mask: Optional mask for masked convolution.
        dtype: Computation dtype.
        param_dtype: Parameter dtype (default: float32).
        precision: Numerical precision.
        kernel_init: Kernel initializer.
        bias_init: Bias initializer.
        alpha_init: Alpha initializer (for learnable alpha).
        epsilon: Small constant for numerical stability.
        drop_rate: DropConnect rate (default: 0.0).
        rngs: Random number generators.
    """

    __data__ = ("kernel", "bias", "alpha", "mask", "dropconnect_key")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: tp.Union[None, int, tp.Sequence[int]] = 1,
        *,
        padding: PaddingLike = "SAME",
        input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
        use_dropconnect: bool = False,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        alpha_init: Initializer = default_alpha_init,
        mask: tp.Optional[Array] = None,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        rngs: rnglib.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        else:
            kernel_size = tuple(kernel_size)

        self.kernel_shape = kernel_size + (
            in_features // feature_group_count,
            out_features,
        )
        kernel_key = rngs.params()
        self.kernel = nnx.Param(
            kernel_init(kernel_key, self.kernel_shape, param_dtype)
        )

        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            bias_shape = (out_features,)
            bias_key = rngs.params()
            self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
        else:
            self.bias = None

        # Handle alpha configuration
        self.alpha: nnx.Param[jax.Array] | None
        self._constant_alpha_value: tp.Optional[float] = None

        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = float(DEFAULT_CONSTANT_ALPHA)
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.alpha = None
            use_alpha = True
        else:
            if use_alpha:
                alpha_key = rngs.params()
                self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
            else:
                self.alpha = None

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.input_dilation = input_dilation
        self.kernel_dilation = kernel_dilation
        self.feature_group_count = feature_group_count
        self.use_bias = use_bias
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha
        self.use_dropconnect = use_dropconnect
        self.mask = mask
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.conv_general_dilated = conv_general_dilated
        self.promote_dtype = promote_dtype
        self.epsilon = epsilon
        self.drop_rate = drop_rate

        if use_dropconnect:
            self.dropconnect_key = rngs.params()
        else:
            self.dropconnect_key = None

    def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
        """Applies YAT convolution to the inputs.

        Args:
            inputs: Input tensor of shape (batch, spatial..., channels).
            deterministic: If True, DropConnect is disabled.

        Returns:
            Output tensor of shape (batch, spatial..., out_features).
        """
        assert isinstance(self.kernel_size, tuple)

        def maybe_broadcast(
            x: tp.Optional[tp.Union[int, tp.Sequence[int]]],
        ) -> tuple[int, ...]:
            if x is None:
                x = 1
            if isinstance(x, int):
                return (x,) * len(self.kernel_size)
            return tuple(x)

        num_batch_dimensions = inputs.ndim - (len(self.kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            total_batch_size = int(np.prod(input_batch_shape))
            flat_input_shape = (total_batch_size,) + inputs.shape[
                num_batch_dimensions:
            ]
            inputs_flat = jnp.reshape(inputs, flat_input_shape)
        else:
            inputs_flat = inputs
            input_batch_shape = ()

        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
        if padding_lax in ("CIRCULAR", "REFLECT"):
            assert isinstance(padding_lax, str)
            kernel_size_dilated = [
                (k - 1) * d + 1 for k, d in zip(self.kernel_size, kernel_dilation)
            ]
            zero_pad: tp.List[tuple[int, int]] = [(0, 0)]
            pads = (
                zero_pad
                + [((k - 1) // 2, k // 2) for k in kernel_size_dilated]
                + [(0, 0)]
            )
            padding_mode = {"CIRCULAR": "wrap", "REFLECT": "reflect"}[padding_lax]
            inputs_flat = jnp.pad(inputs_flat, pads, mode=padding_mode)
            padding_lax = "VALID"
        elif padding_lax == "CAUSAL":
            if len(self.kernel_size) != 1:
                raise ValueError(
                    "Causal padding is only implemented for 1D convolutions."
                )
            left_pad = kernel_dilation[0] * (self.kernel_size[0] - 1)
            pads = [(0, 0), (left_pad, 0), (0, 0)]
            inputs_flat = jnp.pad(inputs_flat, pads)
            padding_lax = "VALID"

        dimension_numbers = conv_dimension_numbers(inputs_flat.shape)
        assert self.in_features % self.feature_group_count == 0

        kernel_val = self.kernel.value

        # Apply DropConnect if enabled
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(
                self.dropconnect_key, p=keep_prob, shape=kernel_val.shape
            )
            kernel_val = (kernel_val * mask) / keep_prob

        # Apply mask if provided
        current_mask = self.mask
        if current_mask is not None:
            if current_mask.shape != self.kernel_shape:
                raise ValueError(
                    "Mask needs to have the same shape as weights. "
                    f"Shapes are: {current_mask.shape}, {self.kernel_shape}"
                )
            kernel_val *= current_mask

        bias_val = self.bias.value if self.bias is not None else None

        # Get alpha value
        if self._constant_alpha_value is not None:
            alpha = jnp.array(self._constant_alpha_value, dtype=self.param_dtype)
        elif self.alpha is not None:
            alpha = self.alpha.value
        else:
            alpha = None

        inputs_promoted, kernel_promoted, bias_promoted = self.promote_dtype(
            (inputs_flat, kernel_val, bias_val), dtype=self.dtype
        )
        inputs_flat = inputs_promoted
        kernel_val = kernel_promoted
        bias_val = bias_promoted

        # Compute dot product via convolution
        dot_prod_map = self.conv_general_dilated(
            inputs_flat,
            kernel_val,
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        # Compute ||x||² for each patch
        inputs_flat_squared = inputs_flat**2
        kernel_in_channels_for_sum_sq = self.kernel_shape[-2]
        kernel_for_patch_sq_sum_shape = self.kernel_size + (
            kernel_in_channels_for_sum_sq,
            1,
        )
        kernel_for_patch_sq_sum = jnp.ones(
            kernel_for_patch_sq_sum_shape, dtype=kernel_val.dtype
        )

        patch_sq_sum_map_raw = self.conv_general_dilated(
            inputs_flat_squared,
            kernel_for_patch_sq_sum,
            strides,
            padding_lax,
            lhs_dilation=input_dilation,
            rhs_dilation=kernel_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=self.precision,
        )

        if self.feature_group_count > 1:
            num_out_channels_per_group = self.out_features // self.feature_group_count
            if num_out_channels_per_group == 0:
                raise ValueError(
                    "out_features must be a multiple of feature_group_count."
                )
            patch_sq_sum_map = jnp.repeat(
                patch_sq_sum_map_raw, num_out_channels_per_group, axis=-1
            )
        else:
            patch_sq_sum_map = patch_sq_sum_map_raw

        # Compute ||W||² for each filter
        reduce_axes_for_kernel_sq = tuple(range(kernel_val.ndim - 1))
        kernel_sq_sum_per_filter = jnp.sum(
            kernel_val**2, axis=reduce_axes_for_kernel_sq
        )

        # YAT formula: (x·W)² / (||x - W||² + ε)
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_per_filter - 2 * dot_prod_map
        y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        # Add bias
        if self.use_bias and bias_val is not None:
            bias_reshape_dims = (1,) * (y.ndim - 1) + (-1,)
            y += jnp.reshape(bias_val, bias_reshape_dims)

        # Apply alpha scaling
        if alpha is not None:
            scale = (
                jnp.sqrt(self.out_features) / jnp.log(1 + self.out_features)
            ) ** alpha
            y = y * scale

        # Reshape output if needed
        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)

        return y

