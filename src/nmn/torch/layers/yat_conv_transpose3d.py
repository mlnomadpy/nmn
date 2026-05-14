# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _triple

from torch.nn import ConvTranspose3d

from ._yat_conv_core import setup_yat_attrs, yat_conv_transpose_forward

__all__ = ["YatConvTranspose3D"]

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatConvTranspose3D(ConvTranspose3d):
    """3D YAT transposed convolution layer implementing the YAT algorithm.

    Computes: y = (x * W + b)² / (||x - W||² + ε), with optional alpha scaling.

    Args:
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
            use that value. If None (default), use learnable alpha when
            use_alpha=True.
        param_dtype: dtype for parameter initialization (default: None, uses
            PyTorch ConvTranspose3d default). Separate from computation dtype.
    """

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
        constant_bias: Optional[float] = None,
        softplus_bias: bool = False,
        scalar_bias: bool = False,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        drop_rate: float = 0.0,
        device=None,
        dtype=None,
        param_dtype=None,
    ) -> None:
        storage_dtype = param_dtype if param_dtype is not None else dtype

        # If constant_bias or scalar_bias is set, don't allocate a per-channel
        # learnable bias in the parent — we handle bias ourselves.
        parent_bias = False if (constant_bias is not None or scalar_bias) else bias

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            parent_bias,
            dilation,
            padding_mode,
            device,
            storage_dtype,
        )

        setup_yat_attrs(
            self,
            bias=bias, constant_bias=constant_bias,
            softplus_bias=softplus_bias, scalar_bias=scalar_bias,
            use_alpha=use_alpha, constant_alpha=constant_alpha,
            use_dropconnect=use_dropconnect, drop_rate=drop_rate, mask=mask,
            epsilon=epsilon, learnable_epsilon=learnable_epsilon,
            storage_dtype=storage_dtype, compute_dtype=dtype, device=device,
        )

    def forward(self, input: Tensor, output_size: Optional[list[int]] = None, *, deterministic: bool = False) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for YatConvTranspose3D")
        if output_size is not None:
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding,
                self.kernel_size, num_spatial_dims=3, dilation=self.dilation,
            )
        else:
            output_padding = self.output_padding
        return yat_conv_transpose_forward(
            self, input, F.conv_transpose3d, output_padding,
            deterministic=deterministic,
        )
