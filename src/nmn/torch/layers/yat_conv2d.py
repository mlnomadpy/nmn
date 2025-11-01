# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.utils import _single, _pair, _triple

from ..base import _ConvNd, _ConvTransposeNd, YatConvNd, YatConvTransposeNd, _LazyConvXdMixin, convolution_notes
from .conv2d import Conv2d

__all__ = ["YatConv2d"]


class YatConv2d(Conv2d):
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
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        # Call the parent Conv2d constructor
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
            dtype,
        )
        
        # YAT-specific attributes
        self.use_alpha = use_alpha
        self.use_dropconnect = use_dropconnect
        self.epsilon = epsilon
        self.drop_rate = drop_rate
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        if self.use_alpha:
            self.alpha = Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("alpha", None)
            
        # Register mask as buffer (not a parameter)
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", None)

    def _yat_forward(self, input: Tensor, conv_fn: callable, deterministic: bool = False) -> Tensor:
        # Apply DropConnect and masking to weights
        weight = self.weight
        
        # Apply DropConnect if enabled and not in deterministic mode
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            if self.training:
                # Generate dropout mask
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
        
        # YAT computation: (dot_product)^2 / (distance_squared + epsilon)
        y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias.view(*view_shape)

        # Apply alpha scaling if enabled
        if self.use_alpha and self.alpha is not None:
            scale = (math.sqrt(self.out_channels) / math.log(1.0 + self.out_channels)) ** self.alpha
            y = y * scale

        return y

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        return self._yat_forward(input, F.conv2d, deterministic)


