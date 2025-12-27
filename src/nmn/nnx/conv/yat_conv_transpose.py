"""YAT Transposed Convolution Module.

This module implements the YatConvTranspose layer, which applies the YAT
 formula to transposed convolution (deconvolution) operations.

Transposed convolution is commonly used for upsampling in neural networks,
such as in decoder parts of autoencoders and generative models.
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
    PaddingLike,
    PromoteDtypeFn,
)

from .utils import (
    canonicalize_padding,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    DEFAULT_CONSTANT_ALPHA,
)

Array = jax.Array


class YatConvTranspose(Module):
    """YAT Transposed Convolution Module wrapping ``lax.conv_transpose``.

    Applies the YAT formula to transposed convolution for upsampling:
        y = (x *^T W)² / (||x - W||² + ε)

    where *^T denotes the transposed convolution operation.

    Example usage::

        >>> from flax import nnx
        >>> from nmn.nnx.conv import YatConvTranspose
        >>> import jax.numpy as jnp

        >>> rngs = nnx.Rngs(0)
        >>> x = jnp.ones((1, 4, 4, 8))  # (batch, H, W, channels)

        >>> # Upsample by 2x
        >>> layer = YatConvTranspose(
        ...     in_features=8, out_features=16,
        ...     kernel_size=(4, 4), strides=(2, 2),
        ...     padding='SAME', rngs=rngs
        ... )
        >>> out = layer(x)
        >>> print(out.shape)
        (1, 8, 8, 16)

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels.
        kernel_size: Shape of the convolutional kernel.
        strides: Inter-window strides (default: 1).
        padding: 'SAME', 'VALID', 'CIRCULAR', or sequence of (low, high) pairs.
        kernel_dilation: Dilation factor for kernel (default: 1).
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
        transpose_kernel: If True, flips spatial axes and swaps channels.
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
        strides: int | tp.Sequence[int] | None = None,
        *,
        padding: PaddingLike = "SAME",
        kernel_dilation: int | tp.Sequence[int] | None = None,
        use_bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
        use_dropconnect: bool = False,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike | None = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        alpha_init: Initializer = default_alpha_init,
        transpose_kernel: bool = False,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        rngs: rnglib.Rngs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        else:
            kernel_size = tuple(kernel_size)

        self.kernel_size = kernel_size
        self.in_features = in_features
        self.out_features = out_features
        self.strides = strides
        self.padding = padding
        self.kernel_dilation = kernel_dilation
        self.use_bias = use_bias
        self.use_dropconnect = use_dropconnect
        self.mask = mask
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.alpha_init = alpha_init
        self.transpose_kernel = transpose_kernel
        self.promote_dtype = promote_dtype
        self.epsilon = epsilon
        self.drop_rate = drop_rate

        if self.transpose_kernel:
            kernel_shape = kernel_size + (self.out_features, in_features)
        else:
            kernel_shape = kernel_size + (in_features, self.out_features)

        self.kernel_shape = kernel_shape
        self.kernel = nnx.Param(
            self.kernel_init(rngs.params(), kernel_shape, self.param_dtype)
        )

        self.bias: nnx.Param | None
        if self.use_bias:
            self.bias = nnx.Param(
                self.bias_init(rngs.params(), (self.out_features,), self.param_dtype)
            )
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

        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        if use_dropconnect:
            self.dropconnect_key = rngs.params()
        else:
            self.dropconnect_key = None

    def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
        """Applies YAT transposed convolution to the inputs.

        Args:
            inputs: Input tensor of shape (batch, spatial..., channels).
            deterministic: If True, DropConnect is disabled.

        Returns:
            Upsampled output tensor of shape (batch, spatial..., out_features).
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
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
        if padding_lax == "CIRCULAR":
            padding_lax = "VALID"

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

        # Compute transposed convolution (dot product)
        dot_prod_map = lax.conv_transpose(
            inputs_flat,
            kernel_val,
            strides,
            padding_lax,
            rhs_dilation=kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
        )

        # Compute ||x||² for each patch via transposed convolution
        inputs_flat_squared = inputs_flat**2
        if self.transpose_kernel:
            patch_kernel_in_features = self.out_features
        else:
            patch_kernel_in_features = self.in_features

        kernel_for_patch_sq_sum_shape = self.kernel_size + (patch_kernel_in_features, 1)
        kernel_for_patch_sq_sum = jnp.ones(
            kernel_for_patch_sq_sum_shape, dtype=kernel_val.dtype
        )

        patch_sq_sum_map_raw = lax.conv_transpose(
            inputs_flat_squared,
            kernel_for_patch_sq_sum,
            strides,
            padding_lax,
            rhs_dilation=kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
        )

        if self.out_features > 1:
            patch_sq_sum_map = jnp.repeat(
                patch_sq_sum_map_raw, self.out_features, axis=-1
            )
        else:
            patch_sq_sum_map = patch_sq_sum_map_raw

        # Compute ||W||² for each filter
        if self.transpose_kernel:
            reduce_axes_for_kernel_sq = tuple(range(len(self.kernel_size))) + (
                len(self.kernel_size) + 1,
            )
        else:
            reduce_axes_for_kernel_sq = tuple(range(len(self.kernel_size))) + (
                len(self.kernel_size),
            )

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

        # Handle circular padding
        if self.padding == "CIRCULAR":
            scaled_x_dims = [
                x_dim * stride
                for x_dim, stride in zip(jnp.shape(inputs_flat)[1:-1], strides)
            ]
            size_diffs = [
                -(y_dim - x_dim) % (2 * x_dim)
                for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)
            ]
            if self.transpose_kernel:
                total_pad = [
                    (size_diff // 2, (size_diff + 1) // 2) for size_diff in size_diffs
                ]
            else:
                total_pad = [
                    ((size_diff + 1) // 2, size_diff // 2) for size_diff in size_diffs
                ]
            y = jnp.pad(y, [(0, 0)] + total_pad + [(0, 0)])
            for i in range(1, y.ndim - 1):
                y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) + y.shape[i + 1 :])
                y = y.sum(axis=i)

        # Reshape output if needed
        if num_batch_dimensions != 1:
            output_shape = input_batch_shape + y.shape[1:]
            y = jnp.reshape(y, output_shape)

        return y




