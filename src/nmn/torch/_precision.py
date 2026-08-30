"""Autograd-safe precision helpers shared by PyTorch YAT layers."""

from __future__ import annotations

import torch
from torch import Tensor


def _saturate_gradient(grad_output: Tensor, input_dtype: torch.dtype) -> Tensor:
    limits = torch.finfo(input_dtype)
    return grad_output.clamp(min=limits.min, max=limits.max).to(input_dtype)


if hasattr(torch.autograd.Function, "setup_context"):

    class _SaturatingUpcast(torch.autograd.Function):
        """Modern autograd boundary with torch.func transform support."""

        generate_vmap_rule = True

        @staticmethod
        def forward(tensor: Tensor) -> Tensor:
            return tensor.float()

        @staticmethod
        def setup_context(ctx, inputs, output) -> None:
            del output
            (tensor,) = inputs
            ctx.input_dtype = tensor.dtype

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
            return (_saturate_gradient(grad_output, ctx.input_dtype),)

        @staticmethod
        def jvp(ctx, grad_tensor: Tensor) -> Tensor:
            del ctx
            return grad_tensor.float()

else:  # pragma: no cover - compatibility path for torch 1.11-1.x

    class _SaturatingUpcast(torch.autograd.Function):
        """Legacy autograd boundary for PyTorch versions before torch.func."""

        @staticmethod
        def forward(ctx, tensor: Tensor) -> Tensor:
            ctx.input_dtype = tensor.dtype
            return tensor.float()

        @staticmethod
        def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
            return (_saturate_gradient(grad_output, ctx.input_dtype),)


def saturating_upcast(tensor: Tensor) -> Tensor:
    """Return fp32 ``tensor`` with a finite low-precision gradient boundary."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return _SaturatingUpcast.apply(tensor)
    return tensor.float()
