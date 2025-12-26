# mypy: allow-untyped-defs
import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair

from .conv_transpose2d import ConvTranspose2d

__all__ = ["YatConvTranspose2d"]


class YatConvTranspose2d(ConvTranspose2d):
    """2D YAT transposed convolution layer implementing the YAT algorithm."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        
        self.use_alpha = use_alpha
        self.use_dropconnect = use_dropconnect
        self.epsilon = epsilon
        self.drop_rate = drop_rate
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        if self.use_alpha:
            self.alpha = Parameter(torch.ones(1, **factory_kwargs))
        else:
            self.register_parameter("alpha", None)
            
        if mask is not None:
            self.register_buffer("mask", mask)
        else:
            self.register_buffer("mask", None)

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None, *, deterministic: bool = False) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for YatConvTranspose2d")
        
        weight = self.weight
        
        # Apply DropConnect
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0 and self.training:
            keep_prob = 1.0 - self.drop_rate
            drop_mask = torch.bernoulli(torch.full_like(weight, keep_prob))
            weight = (weight * drop_mask) / keep_prob
        
        # Apply mask
        if self.mask is not None:
            weight = weight * self.mask
        
        # Compute output_padding
        num_spatial_dims = 2
        if output_size is not None:
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding,
                self.kernel_size, num_spatial_dims, self.dilation
            )
        else:
            output_padding = self.output_padding
        
        # Compute dot product using transposed convolution
        dot_prod_map = F.conv_transpose2d(
            input, weight, None, self.stride, self.padding,
            output_padding, self.groups, self.dilation
        )
        
        # Compute ||input||^2 contribution using transposed convolution with ones kernel
        # For transpose conv, weight shape is (in_channels, out_channels/groups, *kernel_size)
        input_squared = input * input
        
        # We need ones kernel that matches the weight layout for transpose conv
        in_channels_per_group = self.in_channels // self.groups
        out_channels_per_group = self.out_channels // self.groups
        ones_kernel_shape = (in_channels_per_group, out_channels_per_group) + self.kernel_size
        ones_kernel = torch.ones(ones_kernel_shape, device=input.device, dtype=input.dtype)
        
        patch_sq_sum_map = F.conv_transpose2d(
            input_squared, ones_kernel, None, self.stride, self.padding,
            output_padding, self.groups, self.dilation
        )

        # Compute ||kernel||^2 per output filter
        # Weight shape: (in_channels, out_channels/groups, *kernel_size)
        # Sum over in_channels (dim 0) and spatial dims, keep out_channels
        reduce_dims = (0,) + tuple(range(2, weight.dim()))
        kernel_sq_sum_per_filter = torch.sum(weight**2, dim=reduce_dims)

        view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
        kernel_sq_sum_reshaped = kernel_sq_sum_per_filter.view(*view_shape)

        # YAT computation
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        if self.bias is not None:
            y = y + self.bias.view(*view_shape)

        if self.use_alpha and self.alpha is not None:
            scale = (math.sqrt(self.out_channels) / math.log(1.0 + self.out_channels)) ** self.alpha
            y = y * scale

        return y
