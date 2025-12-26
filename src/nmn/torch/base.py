# mypy: allow-untyped-defs
import math
from typing import Optional, Union
from typing_extensions import deprecated

import torch
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import functional as F, init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "YatConv1d",
    "YatConv2d",
    "YatConv3d",
    "YatConvTranspose1d",
    "YatConvTranspose2d",
    "YatConvTranspose3d",
]

convolution_notes = {
    "groups_note": r"""* :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).""",
    "depthwise_separable_note": r"""When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.""",
}  # noqa: B950



__all__ = ["_ConvNd", "_ConvTransposeNd", "_ConvTransposeMixin", "YatConvNd", "YatConvTransposeNd", "_LazyConvXdMixin", "convolution_notes", "reproducibility_notes"]

class _ConvNd(Module):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(  # type: ignore[empty-body]
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor: ...

    in_channels: int
    _reversed_padding_repeated_twice: list[int]
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: Union[str, tuple[int, ...]]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

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
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"




class _ConvTransposeNd(_ConvNd):
    def __init__(
        self,
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
        device=None,
        dtype=None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )

        factory_kwargs = {"device": device, "dtype": dtype}
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
            **factory_kwargs,
        )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[list[int]],
        stride: list[int],
        padding: list[int],
        kernel_size: list[int],
        num_spatial_dims: int,
        dilation: Optional[list[int]] = None,
    ) -> list[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} "
                    f"or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})"
                )

            min_sizes = torch.jit.annotate(list[int], [])
            max_sizes = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})"
                    )

            res = torch.jit.annotate(list[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret




class _ConvTransposeMixin(_ConvTransposeNd):
    @deprecated(
        "`_ConvTransposeMixin` is a deprecated internal class. "
        "Please consider using public APIs.",
        category=FutureWarning,
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




class YatConvNd(_ConvNd):
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
        # Remove device and dtype from kwargs to avoid duplicate keyword arguments
        # when they are passed both explicitly and within **kwargs
        kwargs.pop('device', None)
        kwargs.pop('dtype', None)
        
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
        
        # For grouped convolution, we need one kernel per group
        # Each kernel sums over the input channels in that group
        # PyTorch conv expects: (out_channels, in_channels_per_group, *kernel_size)
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


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap




class _LazyConvXdMixin(LazyModuleMixin):
    groups: int
    transposed: bool
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    weight: UninitializedParameter
    bias: UninitializedParameter

    def reset_parameters(self) -> None:
        # has_uninitialized_params is defined in parent class and it is using a protocol on self
        if not self.has_uninitialized_params() and self.in_channels != 0:  # type: ignore[misc]
            # "type:ignore[..]" is required because mypy thinks that "reset_parameters" is undefined
            # in super class. Turns out that it is defined in _ConvND which is inherited by any class
            # that also inherits _LazyConvXdMixin
            super().reset_parameters()  # type: ignore[misc]

    # Signature of "initialize_parameters" is incompatible with the definition in supertype LazyModuleMixin
    def initialize_parameters(self, input: Tensor, *args, **kwargs) -> None:  # type: ignore[override]
        # defined by parent class but using a protocol
        if self.has_uninitialized_params():  # type: ignore[misc]
            self.in_channels = self._get_in_channels(input)
            if self.in_channels % self.groups != 0:
                raise ValueError("in_channels must be divisible by groups")
            assert isinstance(self.weight, UninitializedParameter)
            if self.transposed:
                self.weight.materialize(
                    (
                        self.in_channels,
                        self.out_channels // self.groups,
                        *self.kernel_size,
                    )
                )
            else:
                self.weight.materialize(
                    (
                        self.out_channels,
                        self.in_channels // self.groups,
                        *self.kernel_size,
                    )
                )
            if self.bias is not None:
                assert isinstance(self.bias, UninitializedParameter)
                self.bias.materialize((self.out_channels,))
            self.reset_parameters()

    # Function to extract in_channels from first input.
    def _get_in_channels(self, input: Tensor) -> int:
        num_spatial_dims = self._get_num_spatial_dims()
        num_dims_no_batch = num_spatial_dims + 1  # +1 for channels dim
        num_dims_batch = num_dims_no_batch + 1
        if input.dim() not in (num_dims_no_batch, num_dims_batch):
            raise RuntimeError(
                f"Expected {num_dims_no_batch}D (unbatched) or {num_dims_batch}D (batched) input "
                f"to {self.__class__.__name__}, but "
                f"got input of size: {input.shape}"
            )
        return input.shape[1] if input.dim() == num_dims_batch else input.shape[0]

    # Function to return the number of spatial dims expected for inputs to the module.
    # This is expected to be implemented by subclasses.
    def _get_num_spatial_dims(self) -> int:
        raise NotImplementedError


# LazyConv1d defines weight as a Tensor but derived class defines it as UnitializeParameter


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




