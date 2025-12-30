# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single

from torch.nn import Conv1d

__all__ = ["YatConv1d"]


class YatConv1d(Conv1d):
    """1D YAT convolution layer implementing the YAT algorithm."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
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

    def _yat_forward(self, input: Tensor, conv_fn: callable, deterministic: bool = False) -> Tensor:
        weight = self.weight
        
        # Apply DropConnect if enabled and not in deterministic mode
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            if self.training:
                keep_prob = 1.0 - self.drop_rate
                drop_mask = torch.bernoulli(torch.full_like(weight, keep_prob))
                weight = (weight * drop_mask) / keep_prob
        
        # Apply mask if provided
        if self.mask is not None:
            weight = weight * self.mask
        
        # Compute dot product using standard convolution
        dot_prod_map = self._conv_forward(input, weight, None)

        # Compute ||input_patches||^2 using convolution with ones kernel
        input_squared = input * input
        
        # For grouped convolution, we need one kernel per group
        in_channels_per_group = self.in_channels // self.groups
        ones_kernel_shape = (self.groups, in_channels_per_group) + self.kernel_size
        ones_kernel = torch.ones(ones_kernel_shape, device=input.device, dtype=input.dtype)
        
        if self.padding_mode != "zeros":
            patch_sq_sum_map_raw = conv_fn(
                F.pad(input_squared, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                ones_kernel, None, self.stride, _single(0), self.dilation, self.groups,
            )
        else:
            patch_sq_sum_map_raw = conv_fn(
                input_squared, ones_kernel, None, self.stride, self.padding, self.dilation, self.groups,
            )
        
        # Handle grouped convolution
        if self.groups > 1:
            num_out_channels_per_group = self.out_channels // self.groups
            patch_sq_sum_map = patch_sq_sum_map_raw.repeat_interleave(num_out_channels_per_group, dim=1)
        else:
            patch_sq_sum_map = patch_sq_sum_map_raw.repeat(1, self.out_channels, *([1] * (patch_sq_sum_map_raw.dim() - 2)))

        # Compute ||kernel||^2 per filter
        reduce_dims = tuple(range(1, weight.dim()))
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

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        return self._yat_forward(input, F.conv1d, deterministic)
