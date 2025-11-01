# mypy: allow-untyped-defs
"""YAT (Yet Another Transformation) Convolutional layers for PyTorch."""
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple

# Import base classes from conv module
from .conv import (
    _ConvNd,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)


__all__ = [
    "YatConvNd",
    "YatConv1d",
    "YatConv2d",
    "YatConv3d",
    "YatConvTransposeNd",
    "YatConvTranspose1d",
    "YatConvTranspose2d",
    "YatConvTranspose3d",
]


class YatConvNd(_ConvNd):
    """Base class for YAT convolutional layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: Union[str, tuple[int, ...]],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        use_alpha: bool = True,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
            **kwargs,
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
                keep_prob = 1.0 - self.drop_rate
                mask = torch.bernoulli(torch.full_like(weight, keep_prob))
                weight = (weight * mask) / keep_prob
        
        # Apply mask if provided
        if self.mask is not None:
            if self.mask.shape != weight.shape:
                raise ValueError(
                    f'Mask needs to have the same shape as weights. '
                    f'Shapes are: {self.mask.shape}, {weight.shape}'
                )
            weight = weight * self.mask
        
        # Compute dot product using standard convolution: input * weight
        dot_prod_map = self._conv_forward(input, weight, None)

        # Compute ||input_patches||^2 using convolution with ones kernel
        input_squared = input * input
        
        # For grouped convolution, the kernel shape for computing patch sums is:
        # kernel_size + (in_channels_per_group, 1)
        in_channels_per_group = self.in_channels // self.groups
        ones_kernel_shape = self.kernel_size + (in_channels_per_group, 1)
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


class YatConv1d(YatConvNd, Conv1d):
    """1D YAT convolutional layer."""
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
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            use_alpha,
            use_dropconnect,
            mask,
            epsilon,
            drop_rate,
            device,
            dtype,
        )

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        return self._yat_forward(input, F.conv1d, deterministic)


class YatConv2d(Conv2d):
    """2D YAT convolutional layer."""
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


class YatConv3d(YatConvNd, Conv3d):
    """3D YAT convolutional layer."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
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
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            use_alpha,
            use_dropconnect,
            mask,
            epsilon,
            drop_rate,
            device,
            dtype,
        )

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        return self._yat_forward(input, F.conv3d, deterministic)


class YatConvTransposeNd(YatConvNd):
    """Base class for YAT transposed convolution layers."""
    
    def _yat_transpose_forward(self, input: Tensor, conv_transpose_fn: callable, deterministic: bool = False, output_size: Optional[list[int]] = None) -> Tensor:
        # Apply DropConnect and masking to weights
        weight = self.weight
        
        # Apply DropConnect if enabled and not in deterministic mode
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            if self.training:
                keep_prob = 1.0 - self.drop_rate
                mask = torch.bernoulli(torch.full_like(weight, keep_prob))
                weight = (weight * mask) / keep_prob
        
        # Apply mask if provided
        if self.mask is not None:
            if self.mask.shape != weight.shape:
                raise ValueError(
                    f'Mask needs to have the same shape as weights. '
                    f'Shapes are: {self.mask.shape}, {weight.shape}'
                )
            weight = weight * self.mask
        
        # Compute dot product using transposed convolution
        if hasattr(self, '_output_padding'):
            # For ConvTranspose classes, need to compute output_padding
            if output_size is not None:
                num_spatial_dims = len(self.kernel_size)
                output_padding = self._output_padding(
                    input, output_size, self.stride, self.padding, 
                    self.kernel_size, num_spatial_dims, self.dilation
                )
            else:
                output_padding = self.output_padding
                
            dot_prod_map = conv_transpose_fn(
                input, weight, None, self.stride, self.padding, 
                output_padding, self.groups, self.dilation
            )
        else:
            dot_prod_map = conv_transpose_fn(
                input, weight, None, self.stride, self.padding, 
                self.dilation, self.groups
            )

        # Compute ||input_patches||^2 using transposed convolution with ones kernel
        input_squared = input * input
        
        # For transposed convolution, we need a kernel to compute patch sums
        # The weight shape for transpose conv is (in_channels, out_channels_per_group, *kernel_size)
        # So we need ones kernel of shape (in_channels // groups, 1, *kernel_size)
        in_channels_per_group = self.in_channels // self.groups
        ones_kernel_shape = (in_channels_per_group, 1) + self.kernel_size
        ones_kernel = torch.ones(ones_kernel_shape, device=input.device, dtype=input.dtype)

        # Compute patch squared sum using same transposed convolution function
        if hasattr(self, '_output_padding') and output_size is not None:
            patch_sq_sum_map_raw = conv_transpose_fn(
                input_squared, ones_kernel, None, self.stride, self.padding,
                output_padding, self.groups, self.dilation
            )
        else:
            if hasattr(self, 'output_padding'):
                patch_sq_sum_map_raw = conv_transpose_fn(
                    input_squared, ones_kernel, None, self.stride, self.padding,
                    self.output_padding, self.groups, self.dilation
                )
            else:
                patch_sq_sum_map_raw = conv_transpose_fn(
                    input_squared, ones_kernel, None, self.stride, self.padding,
                    self.dilation, self.groups
                )

        # Handle grouping for output channels
        # For transpose conv, we need to repeat across all output channels
        if self.out_channels > 1:
            patch_sq_sum_map = patch_sq_sum_map_raw.repeat(1, self.out_channels, *([1] * (patch_sq_sum_map_raw.dim() - 2)))
        else:
            patch_sq_sum_map = patch_sq_sum_map_raw

        # Compute kernel squared sum per filter
        # For transpose conv, weight shape is (in_channels, out_channels_per_group, *kernel_size)
        # We want to sum over spatial dimensions and input channels, keeping output channels
        if self.transposed:
            # Sum over dimensions (0, 2, 3, ...) keeping dimension 1 (out_channels per group)
            reduce_axes_for_kernel_sq = (0,) + tuple(range(2, weight.dim()))
        else:
            # Regular convolution case
            reduce_axes_for_kernel_sq = tuple(range(weight.dim() - 1))
        
        kernel_sq_sum_per_filter = torch.sum(weight**2, dim=reduce_axes_for_kernel_sq)

        # Reshape for broadcasting
        view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
        kernel_sq_sum_reshaped = kernel_sq_sum_per_filter.view(*view_shape)

        # YAT computation: distance_squared = ||patch||^2 + ||kernel||^2 - 2 * dot_product
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        y = dot_prod_map**2 / (distance_sq_map + self.epsilon)

        # Add bias if present
        if self.bias is not None:
            y = y + self.bias.view(*view_shape)

        # Apply alpha scaling if enabled
        if self.use_alpha and self.alpha is not None:
            scale = (math.sqrt(self.out_channels) / math.log(1.0 + self.out_channels)) ** self.alpha
            y = y * scale

        return y


class YatConvTranspose1d(YatConvTransposeNd, ConvTranspose1d):
    """1D YAT transposed convolutional layer."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = _single(padding)
        dilation_ = _single(dilation)
        output_padding_ = _single(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            use_alpha,
            use_dropconnect,
            mask,
            epsilon,
            drop_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None, *, deterministic: bool = False) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for YatConvTranspose1d"
            )
        return self._yat_transpose_forward(input, F.conv_transpose1d, deterministic, output_size)


class YatConvTranspose2d(YatConvTransposeNd, ConvTranspose2d):
    """2D YAT transposed convolutional layer."""
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
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        output_padding_ = _pair(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            use_alpha,
            use_dropconnect,
            mask,
            epsilon,
            drop_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None, *, deterministic: bool = False) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for YatConvTranspose2d"
            )
        return self._yat_transpose_forward(input, F.conv_transpose2d, deterministic, output_size)


class YatConvTranspose3d(YatConvTransposeNd, ConvTranspose3d):
    """3D YAT transposed convolutional layer."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilation_ = _triple(dilation)
        output_padding_ = _triple(output_padding)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            True,
            output_padding_,
            groups,
            bias,
            padding_mode,
            use_alpha,
            use_dropconnect,
            mask,
            epsilon,
            drop_rate,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None, *, deterministic: bool = False) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for YatConvTranspose3d"
            )
        return self._yat_transpose_forward(input, F.conv_transpose3d, deterministic, output_size)
