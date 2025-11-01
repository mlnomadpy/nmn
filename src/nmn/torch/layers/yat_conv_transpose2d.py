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
from .conv_transpose2d import ConvTranspose2d

__all__ = ["YatConvTranspose2d"]


class YatConvTranspose2d(YatConvTransposeNd, ConvTranspose2d):
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


