# mypy: allow-untyped-defs
import math
import threading
from typing import ClassVar, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.parameter import Parameter

from ._yat_conv_core import (
    apply_preserving_epsilon_dtype,
    setup_yat_attrs,
    yat_conv_forward,
)

__all__ = ["YatConv2D"]

# Default constant alpha value (sqrt(2))
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatConv2D(Conv2d):
    """2D YAT convolution layer implementing the YAT algorithm.

    Computes: y = (x * W + b)² / (||x - W||² + ε), with optional alpha scaling.

    Args:
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float,
            use that value. If None (default), use learnable alpha when
            use_alpha=True.
        weight_normalized: If True, normalize each kernel filter to have norm 1.
            This optimization avoids recomputing kernel norms in YAT distance
            calculation since they are guaranteed to be 1.0.
        tie_kernel_bank: If True, reuse shared kernels across compatible layers.
        kernel_bank_size: Optional explicit capacity. The bank auto-expands only
            during construction, before any tied consumer executes.
        kernel_bank_id: Namespace for shared banks (allows multiple independent banks).
        param_dtype: dtype for parameter initialization (default: None, uses
            PyTorch Conv2d default). Separate from computation dtype.
    """

    # Class-level shared kernel banks (guarded by a lock for thread safety)
    weight: Parameter
    _KERNEL_BANKS: ClassVar[dict[tuple[object, ...], Parameter]] = {}
    _KERNEL_BANK_USED: ClassVar[dict[tuple[object, ...], bool]] = {}
    _KERNEL_BANKS_LOCK = threading.Lock()

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
        kernel_bank_id: str = "default",
        device=None,
        dtype=None,
        param_dtype=None,
    ) -> None:
        storage_dtype = param_dtype if param_dtype is not None else dtype

        # Validate groups upfront so errors surface at construction time
        if in_channels % groups != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by groups ({groups})"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by groups ({groups})"
            )

        # Handle shared kernel bank - create with auto-sized out_channels
        if tie_kernel_bank:
            bank_out_channels = kernel_bank_size or out_channels
            if bank_out_channels < out_channels:
                raise ValueError(
                    f"kernel_bank_size ({bank_out_channels}) must be at least "
                    f"out_channels ({out_channels})"
                )
        else:
            bank_out_channels = out_channels

        # If constant_bias or scalar_bias is set, don't allocate a per-channel
        # learnable bias in the parent — we handle bias ourselves.
        parent_bias = (
            not (tie_kernel_bank or constant_bias is not None or scalar_bias) and bias
        )

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
            kernel_norm = torch.sqrt(
                torch.sum(self.weight**2, dim=reduce_dims, keepdim=True)
            )
            self.weight.data = self.weight.data / (kernel_norm + 1e-8)

        # Handle auto-expanding shared kernel bank
        if tie_kernel_bank:
            bank_key = (
                kernel_bank_id,
                in_channels,
                tuple(self.kernel_size),
                groups,
                self.weight.dtype,
                self.weight.device,
            )
            with YatConv2D._KERNEL_BANKS_LOCK:
                shared_weight = YatConv2D._KERNEL_BANKS.get(bank_key)

                if shared_weight is None:
                    # First layer: register the weight as shared
                    YatConv2D._KERNEL_BANKS[bank_key] = self.weight
                    YatConv2D._KERNEL_BANK_USED[bank_key] = False
                else:
                    if (
                        shared_weight.device != self.weight.device
                        or shared_weight.dtype != self.weight.dtype
                    ):
                        raise RuntimeError(
                            "shared kernel bank device/dtype registry is stale"
                        )
                    existing_channels = shared_weight.shape[0]
                    if bank_out_channels > existing_channels:
                        if YatConv2D._KERNEL_BANK_USED.get(bank_key, False):
                            raise ValueError(
                                f"kernel bank '{kernel_bank_id}' capacity is frozen "
                                f"at {existing_channels} after first use; requested "
                                f"{bank_out_channels}"
                            )
                        old_weight = shared_weight.data
                        new_weight = torch.empty(
                            (bank_out_channels,) + old_weight.shape[1:],
                            dtype=old_weight.dtype,
                            device=old_weight.device,
                        )
                        nn.init.kaiming_uniform_(new_weight, nonlinearity="relu")
                        new_weight[:existing_channels].copy_(old_weight)
                        shared_weight.data = new_weight

                    self.weight = shared_weight

                self._kernel_bank_key = bank_key

            if bias and constant_bias is None and not scalar_bias:
                self.bias = Parameter(
                    torch.empty(out_channels, device=device, dtype=storage_dtype)
                )
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        self.out_channels = out_channels

        setup_yat_attrs(
            self,
            bias=bias,
            constant_bias=constant_bias,
            softplus_bias=softplus_bias,
            scalar_bias=scalar_bias,
            use_alpha=use_alpha,
            constant_alpha=constant_alpha,
            use_dropconnect=use_dropconnect,
            drop_rate=drop_rate,
            mask=mask,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            storage_dtype=storage_dtype,
            compute_dtype=dtype,
            device=device,
        )

    def forward(self, input: Tensor, *, deterministic: bool = False) -> Tensor:
        out_channels = self._actual_out_channels
        if self.tie_kernel_bank:
            with YatConv2D._KERNEL_BANKS_LOCK:
                YatConv2D._KERNEL_BANK_USED[self._kernel_bank_key] = True
            return yat_conv_forward(
                self,
                input,
                F.conv2d,
                out_channels=out_channels,
                deterministic=deterministic,
                weight_override=self.weight[self._kernel_slice],
            )
        return yat_conv_forward(
            self,
            input,
            F.conv2d,
            out_channels=out_channels,
            deterministic=deterministic,
        )

    def _apply(self, fn, recurse=True):
        if getattr(self, "tie_kernel_bank", False):
            raise RuntimeError(
                "device/dtype migration is unsupported for tied kernel-bank "
                "consumers; construct them with the target device and dtype"
            )
        return apply_preserving_epsilon_dtype(self, fn, super()._apply, recurse=recurse)
