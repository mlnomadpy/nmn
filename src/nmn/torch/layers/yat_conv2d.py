# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

from torch.nn import Conv2d

__all__ = ["YatConv2d"]

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatConv2d(Conv2d):
    """2D YAT convolution layer implementing the YAT algorithm.

    Computes: y = (x * W + b)² / (||x - W||² + ε), with optional alpha scaling.

    Args:
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
            use that value. If None (default), use learnable alpha when
            use_alpha=True.
        param_dtype: dtype for parameter initialization (default: None, uses
            PyTorch Conv2d default). Separate from computation dtype.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
        param_dtype=None,
    ) -> None:
        storage_dtype = param_dtype if param_dtype is not None else dtype
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            storage_dtype,
        )

        self.compute_dtype = dtype
        self.param_dtype = storage_dtype
        self.use_dropconnect = use_dropconnect
        self.epsilon = epsilon
        self.drop_rate = drop_rate

        factory_kwargs = {"device": device, "dtype": storage_dtype}

        # Handle alpha configuration
        # Priority: constant_alpha > use_alpha
        self._constant_alpha_value = None
        if constant_alpha is not None:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.register_parameter("alpha", None)
            use_alpha = True
        elif use_alpha:
            self.alpha = Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("alpha", None)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", None)

    def _promote_dtype(self, *tensors):
        """Promote tensors to computation dtype."""
        if self.compute_dtype is not None:
            target = self.compute_dtype
        else:
            target = None
            for t in tensors:
                if t is not None:
                    target = t.dtype
                    break
            if target is None:
                target = torch.float32
        return tuple(
            t.to(target) if t is not None else None for t in tensors
        )

    def _yat_forward(self, input: Tensor, conv_fn: callable, deterministic: bool = False) -> Tensor:
        weight = self.weight
        bias_val = self.bias

        # Get alpha value (constant or learnable)
        if self._constant_alpha_value is not None:
            alpha = torch.tensor(self._constant_alpha_value, device=input.device)
        elif self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = None

        # Promote to computation dtype
        input, weight, bias_val, alpha = self._promote_dtype(input, weight, bias_val, alpha)

        # Apply DropConnect if enabled and not in deterministic mode
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            if self.training:
                dropout_mask = torch.rand_like(weight) > self.drop_rate
                weight = weight * dropout_mask

        # Apply mask if provided
        if self.mask is not None:
            weight = weight * self.mask

        # Compute dot product using standard convolution: input * weight
        dot_prod_map = self._conv_forward(input, weight, None)

        # Compute ||input_patches||^2 using convolution with ones kernel
        input_squared = input * input

        # For grouped convolution, we need one kernel per group
        # Each kernel sums over the input channels in that group
        in_channels_per_group = self.in_channels // self.groups
        ones_kernel_shape = (self.groups, in_channels_per_group) + self.kernel_size
        ones_kernel = torch.ones(ones_kernel_shape, device=input.device, dtype=input.dtype)

        if self.padding_mode != "zeros":
            patch_sq_sum_map_raw = conv_fn(
                F.pad(
                    input_squared,
                    self._reversed_padding_repeated_twice,
                    mode=self.padding_mode,
                ),
                ones_kernel,
                None,
                self.stride,
                [0] * len(self.kernel_size),
                self.dilation,
                self.groups,
            )
        else:
            patch_sq_sum_map_raw = conv_fn(
                input_squared,
                ones_kernel,
                None,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        # Handle grouped convolution: repeat patch sums for each output channel in each group
        if self.groups > 1:
            if self.out_channels % self.groups != 0:
                raise ValueError("out_channels must be divisible by groups")
            num_out_channels_per_group = self.out_channels // self.groups
            patch_sq_sum_map = patch_sq_sum_map_raw.repeat_interleave(
                num_out_channels_per_group, dim=1
            )
        else:
            # For groups=1, need to repeat across all output channels
            patch_sq_sum_map = patch_sq_sum_map_raw.repeat(1, self.out_channels, *([1] * (patch_sq_sum_map_raw.dim() - 2)))

        # Compute ||kernel||^2 per filter (sum over all dimensions except output channel)
        # Weight shape: (out_channels, in_channels_per_group, *kernel_size)
        reduce_dims = tuple(range(1, weight.dim()))
        kernel_sq_sum_per_filter = torch.sum(weight**2, dim=reduce_dims)

        # Reshape for broadcasting: (1, out_channels, 1, 1, ...)
        view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
        kernel_sq_sum_reshaped = kernel_sq_sum_per_filter.view(*view_shape)

        # Compute distance squared: ||patch||^2 + ||kernel||^2 - 2 * dot_product
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map

        # Add bias before squaring: (x·W + b)² / (dist + ε)
        if bias_val is not None:
            dot_prod_map = dot_prod_map + bias_val.view(*view_shape)

        # YAT computation: (dot_product + bias)^2 / (distance_squared + epsilon)
        y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        # Apply alpha scaling if enabled
        if self._constant_alpha_value is not None:
            # Constant alpha: use directly as the scale factor (e.g. sqrt(2))
            y = y * self._constant_alpha_value
        elif alpha is not None:
            scale = (math.sqrt(self.out_channels) / math.log(1.0 + self.out_channels)) ** alpha
            y = y * scale

        return y

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        return self._yat_forward(input, F.conv2d, deterministic)
