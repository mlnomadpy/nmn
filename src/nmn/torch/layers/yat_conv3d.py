# mypy: allow-untyped-defs
import logging
import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.common_types import _size_3_t
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _triple

from torch.nn import Conv3d

from ._yat_conv_core import setup_yat_attrs, yat_conv_forward

logger = logging.getLogger(__name__)

__all__ = ["YatConv3D"]

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatConv3D(Conv3d):
    """3D YAT convolution layer implementing the YAT algorithm.

    Computes: y = (x * W + b)² / (||x - W||² + ε), with optional alpha scaling.

    Args:
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
            use that value. If None (default), use learnable alpha when
            use_alpha=True.        weight_normalized: If True, normalize each kernel filter to have norm 1.
            This optimization avoids recomputing kernel norms in YAT distance
            calculation since they are guaranteed to be 1.0.        tie_kernel_bank: If True, reuse shared kernels across compatible layers.
        kernel_bank_size: Optional explicit size for shared bank (auto-expands if needed).
        kernel_bank_id: Namespace for shared banks (allows multiple independent banks).
        param_dtype: dtype for parameter initialization (default: None, uses
            PyTorch Conv3d default). Separate from computation dtype.
    """
    
    # Class-level shared kernel banks
    _KERNEL_BANKS = {}

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
        constant_bias: Optional[float] = None,
        softplus_bias: bool = False,
        scalar_bias: bool = False,
        padding_mode: str = "zeros",
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_dropconnect: bool = False,
        mask: Optional[Tensor] = None,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        drop_rate: float = 0.0,
        weight_normalized: bool = False,
        tie_kernel_bank: bool = False,
        kernel_bank_size: Optional[int] = None,
        kernel_bank_id: str = 'default',
        device=None,
        dtype=None,
        param_dtype=None,
    ) -> None:
        storage_dtype = param_dtype if param_dtype is not None else dtype
        
        # Handle shared kernel bank
        if tie_kernel_bank:
            bank_out_channels = kernel_bank_size or out_channels
        else:
            bank_out_channels = out_channels

        # If constant_bias or scalar_bias is set, don't allocate a per-channel
        # learnable bias in the parent — we handle bias ourselves.
        parent_bias = False if (constant_bias is not None or scalar_bias) else bias

        super().__init__(
            in_channels,
            bank_out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            parent_bias,
            padding_mode,
            device,
            storage_dtype,
        )

        # Kernel-bank-specific attrs (used in forward() and bank registration below).
        self.weight_normalized = weight_normalized
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None, out_channels)
        self._actual_out_channels = out_channels
        
        # Normalize kernel if requested
        if self.weight_normalized:
            reduce_dims = tuple(range(1, self.weight.dim()))
            kernel_norm = torch.sqrt(torch.sum(self.weight**2, dim=reduce_dims, keepdim=True))
            self.weight.data = self.weight.data / (kernel_norm + 1e-8)
        
        # Handle auto-expanding shared kernel bank
        if tie_kernel_bank:
            bank_key = (kernel_bank_id, in_channels, kernel_size, groups, storage_dtype)
            shared_weight = YatConv3D._KERNEL_BANKS.get(bank_key)

            if shared_weight is None:
                # First layer: register the weight as shared
                YatConv3D._KERNEL_BANKS[bank_key] = self.weight
            else:
                # Bank exists: auto-expand if needed
                existing_channels = shared_weight.shape[0]
                if bank_out_channels > existing_channels:
                    # Auto-expand: pad with new random initialization
                    logger.info("Auto-expanding kernel bank '%s': %d -> %d filters",
                                kernel_bank_id, existing_channels, bank_out_channels)
                    old_weight = shared_weight.data
                    # Create new weight with expanded size
                    new_weight = torch.empty(
                        (bank_out_channels,) + old_weight.shape[1:],
                        dtype=storage_dtype,
                        device=old_weight.device
                    )
                    nn.init.kaiming_uniform_(new_weight, nonlinearity='relu')
                    # Copy old weights
                    new_weight[:existing_channels].copy_(old_weight)
                    shared_weight.data = new_weight
                    
                self.weight = shared_weight

        setup_yat_attrs(
            self,
            bias=bias, constant_bias=constant_bias,
            softplus_bias=softplus_bias, scalar_bias=scalar_bias,
            use_alpha=use_alpha, constant_alpha=constant_alpha,
            use_dropconnect=use_dropconnect, drop_rate=drop_rate, mask=mask,
            epsilon=epsilon, learnable_epsilon=learnable_epsilon,
            storage_dtype=storage_dtype, compute_dtype=dtype, device=device,
        )

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        out_channels = self._actual_out_channels
        if self.tie_kernel_bank:
            original_weight = self.weight
            self.weight = nn.Parameter(original_weight[self._kernel_slice])
            try:
                return yat_conv_forward(
                    self, input, F.conv3d,
                    out_channels=out_channels,
                    deterministic=deterministic,
                )
            finally:
                self.weight = original_weight
        return yat_conv_forward(
            self, input, F.conv3d,
            out_channels=out_channels,
            deterministic=deterministic,
        )
