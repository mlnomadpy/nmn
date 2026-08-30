"""Shared YAT forward bodies for the six torch conv layers.

Each ``YatConv{1,2,3}D`` / ``YatConvTranspose{1,2,3}D`` used to inline
~100 lines of identical forward-pass logic, parameterized only by which
``F.conv*`` / ``F.conv_transpose*`` to call. This module holds the single
source of truth so per-dimension classes can be thin (just ``__init__``
+ a short ``forward`` that delegates here).

The helpers are duck-typed on the ``layer`` argument — they read the
same instance attributes (``self.weight``, ``self.bias``,
``self._constant_alpha_value``, ``self.use_dropconnect``, …) that the
torch conv classes already expose.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _single
from torch.nn.parameter import Parameter

from nmn._epsilon import (
    inverse_softplus,
    validate_epsilon,
    validate_epsilon_for_dtype,
)

from .._precision import saturating_downcast, saturating_upcast


__all__ = [
    "DEFAULT_CONSTANT_ALPHA",
    "apply_preserving_epsilon_dtype",
    "promote_to_compute_dtype",
    "setup_yat_attrs",
    "yat_conv_forward",
    "yat_conv_transpose_forward",
]


# Default constant alpha value (sqrt(2)).
DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def apply_preserving_epsilon_dtype(layer, fn, super_apply, *, recurse=True):
    """Apply migration while retaining epsilon's protected storage dtype."""
    epsilon_param = layer._parameters.get("epsilon_param")
    if epsilon_param is None:
        return super_apply(fn, recurse=recurse)

    # Keep the same Parameter object out of Module._apply's dtype conversion.
    layer._parameters["epsilon_param"] = None
    try:
        result = super_apply(fn, recurse=recurse)
    finally:
        layer._parameters["epsilon_param"] = epsilon_param

    # Probe only the destination device.  Applying ``fn`` to the raw inverse
    # softplus value could overflow/underflow before converting it back.
    probe = fn(torch.empty(0, device=epsilon_param.device, dtype=epsilon_param.dtype))
    with torch.no_grad():
        epsilon_param.data = epsilon_param.data.to(device=probe.device)
        if epsilon_param.grad is not None:
            epsilon_param.grad.data = epsilon_param.grad.data.to(device=probe.device)
    return result


def setup_yat_attrs(
    layer,
    *,
    bias: bool,
    constant_bias: Optional[float],
    softplus_bias: bool,
    scalar_bias: bool,
    use_alpha: bool,
    constant_alpha,
    use_dropconnect: bool,
    drop_rate: float,
    mask: Optional[Tensor],
    epsilon: float,
    learnable_epsilon: bool,
    storage_dtype,
    compute_dtype,
    device,
):
    """Attach YAT-specific attributes to a freshly super().__init__()'d
    torch conv / transpose-conv layer.

    Sets:
      _constant_bias_value, constant_bias, bias (may rebuild as scalar),
      softplus_bias, scalar_bias, compute_dtype, param_dtype,
      use_dropconnect, epsilon, learnable_epsilon, epsilon_param,
      drop_rate, _constant_alpha_value, alpha, use_alpha,
      constant_alpha, mask buffer.
    """
    # Constant bias.
    layer._constant_bias_value = None
    if constant_bias is not None and constant_bias is not False:
        layer._constant_bias_value = float(constant_bias)
        bias = True
    layer.constant_bias = constant_bias

    # Scalar bias: one shared learnable parameter that broadcasts across
    # all output channels.
    if scalar_bias and constant_bias is None and bias:
        bias_param_dtype = storage_dtype if storage_dtype is not None else torch.float32
        layer.bias = nn.Parameter(torch.zeros((1,), dtype=bias_param_dtype, device=device))
    layer.softplus_bias = softplus_bias and layer.bias is not None
    layer.scalar_bias = scalar_bias and layer.bias is not None

    layer.compute_dtype = compute_dtype
    layer.param_dtype = storage_dtype
    layer.use_dropconnect = use_dropconnect

    layer.epsilon = validate_epsilon(epsilon)
    layer.learnable_epsilon = learnable_epsilon
    if learnable_epsilon:
        epsilon_dtype = storage_dtype or torch.float32
        if epsilon_dtype in (torch.float16, torch.bfloat16):
            epsilon_dtype = torch.float32
        validate_epsilon_for_dtype(layer.epsilon, epsilon_dtype)
        raw_eps = inverse_softplus(layer.epsilon)
        layer.epsilon_param = nn.Parameter(
            torch.full(
                (1,), raw_eps,
                dtype=epsilon_dtype,
                device=device,
            )
        )
    else:
        layer.register_parameter("epsilon_param", None)
    layer.drop_rate = drop_rate

    # Alpha. Priority: constant_alpha > use_alpha.
    layer._constant_alpha_value = None
    if constant_alpha is not None and constant_alpha is not False:
        layer._constant_alpha_value = (
            DEFAULT_CONSTANT_ALPHA if constant_alpha is True else float(constant_alpha)
        )
        layer.register_parameter("alpha", None)
        use_alpha = True
    elif use_alpha:
        layer.alpha = Parameter(torch.ones(1, device=device, dtype=storage_dtype))
    else:
        layer.register_parameter("alpha", None)
    layer.use_alpha = use_alpha
    layer.constant_alpha = constant_alpha

    if mask is not None:
        layer.register_buffer("mask", mask)
    else:
        layer.register_buffer("mask", None)


def promote_to_compute_dtype(layer, *tensors: Optional[Tensor]):
    """Cast a bag of tensors to ``layer.compute_dtype`` (or first non-None dtype)."""
    target = getattr(layer, "compute_dtype", None)
    if target is None:
        for t in tensors:
            if t is not None:
                target = t.dtype
                break
        if target is None:
            target = torch.float32
    return tuple(
        (
            saturating_upcast(t).to(target)
            if t is not None
            and t.dtype in (torch.float16, torch.bfloat16)
            and target != t.dtype
            else t.to(target) if t is not None else None
        )
        for t in tensors
    )


def _resolve_bias_val(layer, weight: Tensor, out_channels: int) -> Optional[Tensor]:
    if layer._constant_bias_value is not None:
        return torch.full(
            (out_channels,),
            layer._constant_bias_value,
            dtype=layer.param_dtype if layer.param_dtype is not None else weight.dtype,
            device=weight.device,
        )
    bias = layer.bias
    if bias is not None and layer.softplus_bias:
        bias = F.softplus(bias)
    return bias


def _resolve_alpha(layer, input: Tensor) -> Optional[Tensor]:
    if layer._constant_alpha_value is not None:
        return torch.tensor(layer._constant_alpha_value, device=input.device)
    return getattr(layer, "alpha", None)


def _resolve_eps(layer, dtype: torch.dtype):
    if layer.learnable_epsilon and layer.epsilon_param is not None:
        epsilon_param = layer.epsilon_param
        dtype = torch.promote_types(dtype, epsilon_param.dtype)
        epsilon_param = epsilon_param.to(dtype)
        return F.softplus(epsilon_param)
    return layer.epsilon


def _yat_score(layer, dot_prod_map: Tensor, distance_sq_map: Tensor,
               bias_val: Optional[Tensor], alpha: Optional[Tensor]) -> Tensor:
    """(dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha — shared tail."""
    view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
    if bias_val is not None:
        dot_prod_map = dot_prod_map + bias_val.view(*view_shape)
    eps = _resolve_eps(layer, distance_sq_map.dtype)
    if isinstance(eps, Tensor):
        dot_prod_map = dot_prod_map.to(eps.dtype)
        distance_sq_map = distance_sq_map.to(eps.dtype)
    y = dot_prod_map ** 2 / (distance_sq_map + eps)
    if layer._constant_alpha_value is not None:
        y = y * layer._constant_alpha_value
    elif alpha is not None:
        y = y * alpha
    return y


# ----------------------------------------------------------------------
# Standard (non-transposed) conv
# ----------------------------------------------------------------------


def yat_conv_forward(
    layer,
    input: Tensor,
    conv_fn: Callable,
    *,
    out_channels: int,
    deterministic: bool = False,
    weight_override: Optional[Tensor] = None,
) -> Tensor:
    """Shared forward body for YatConv{1,2,3}D.

    ``layer`` provides: weight, bias, in_channels, groups, kernel_size, stride,
    padding, dilation, padding_mode, _reversed_padding_repeated_twice,
    _conv_forward, plus all the YAT-specific attributes
    (use_bias, _constant_bias_value, use_alpha, _constant_alpha_value,
    use_dropconnect, drop_rate, training, mask, weight_normalized,
    learnable_epsilon, epsilon_param, epsilon).
    """
    weight = layer.weight if weight_override is None else weight_override
    bias_val = _resolve_bias_val(layer, weight, out_channels)
    alpha = _resolve_alpha(layer, input)

    input, weight, bias_val, alpha = promote_to_compute_dtype(
        layer, input, weight, bias_val, alpha
    )
    output_dtype = input.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        # The distance subtraction and reciprocal backward are particularly
        # cancellation-prone in low precision. Accumulate YAT math in fp32,
        # while preserving parameter storage and the public output dtype.
        input, weight, bias_val, alpha = tuple(
            saturating_upcast(tensor) if tensor is not None else None
            for tensor in (input, weight, bias_val, alpha)
        )

    # DropConnect (inverted-dropout pattern).
    if layer.use_dropconnect and not deterministic and layer.drop_rate > 0.0 and layer.training:
        keep_prob = 1.0 - layer.drop_rate
        drop_mask = torch.bernoulli(torch.full_like(weight, keep_prob))
        weight = (weight * drop_mask) / keep_prob

    if layer.mask is not None:
        weight = weight * layer.mask

    if layer.weight_normalized:
        reduce_dims = tuple(range(1, weight.dim()))
        kernel_norm = torch.sqrt(torch.sum(weight ** 2, dim=reduce_dims, keepdim=True))
        weight = weight / (kernel_norm + 1e-8)

    # Dot product (delegates to the layer's _conv_forward which already
    # handles padding_mode != "zeros").
    dot_prod_map = layer._conv_forward(input, weight, None)

    # ||patch||^2 via convolution with ones kernel.
    input_squared = input * input
    in_channels_per_group = layer.in_channels // layer.groups
    ones_kernel = torch.ones(
        (layer.groups, in_channels_per_group) + tuple(layer.kernel_size),
        device=input.device, dtype=input.dtype,
    )
    if layer.padding_mode != "zeros":
        patch_sq_sum_raw = conv_fn(
            F.pad(input_squared, layer._reversed_padding_repeated_twice, mode=layer.padding_mode),
            ones_kernel, None, layer.stride, _single(0), layer.dilation, layer.groups,
        )
    else:
        patch_sq_sum_raw = conv_fn(
            input_squared, ones_kernel, None, layer.stride, layer.padding,
            layer.dilation, layer.groups,
        )
    if layer.groups > 1:
        num_out_channels_per_group = out_channels // layer.groups
        patch_sq_sum = patch_sq_sum_raw.repeat_interleave(num_out_channels_per_group, dim=1)
    else:
        patch_sq_sum = patch_sq_sum_raw.repeat(1, out_channels, *([1] * (patch_sq_sum_raw.dim() - 2)))

    # ||kernel||^2 per output filter.
    if layer.weight_normalized:
        kernel_sq_sum = torch.ones(weight.shape[0], device=weight.device, dtype=weight.dtype)
    else:
        reduce_dims = tuple(range(1, weight.dim()))
        kernel_sq_sum = torch.sum(weight ** 2, dim=reduce_dims)

    view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
    distance_sq = (
        patch_sq_sum + kernel_sq_sum.view(*view_shape) - 2 * dot_prod_map
    ).clamp(min=0.0)

    output = _yat_score(layer, dot_prod_map, distance_sq, bias_val, alpha)
    return saturating_downcast(output, output_dtype)


# ----------------------------------------------------------------------
# Transposed conv
# ----------------------------------------------------------------------


def yat_conv_transpose_forward(
    layer,
    input: Tensor,
    conv_transpose_fn: Callable,
    output_padding: Iterable[int],
    *,
    deterministic: bool = False,
) -> Tensor:
    """Shared forward body for YatConvTranspose{1,2,3}D.

    Caller passes the resolved ``output_padding`` (either the layer attribute
    or what ``_output_padding`` returned for an explicit ``output_size``).
    """
    weight = layer.weight
    bias_val = _resolve_bias_val(layer, weight, layer.out_channels)
    alpha = _resolve_alpha(layer, input)

    input, weight, bias_val, alpha = promote_to_compute_dtype(
        layer, input, weight, bias_val, alpha
    )
    output_dtype = input.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        input, weight, bias_val, alpha = tuple(
            saturating_upcast(tensor) if tensor is not None else None
            for tensor in (input, weight, bias_val, alpha)
        )

    if layer.use_dropconnect and not deterministic and layer.drop_rate > 0.0 and layer.training:
        keep_prob = 1.0 - layer.drop_rate
        drop_mask = torch.bernoulli(torch.full_like(weight, keep_prob))
        weight = (weight * drop_mask) / keep_prob

    if layer.mask is not None:
        weight = weight * layer.mask

    # Dot product (transposed conv).
    dot_prod_map = conv_transpose_fn(
        input, weight, None, layer.stride, layer.padding,
        output_padding, layer.groups, layer.dilation,
    )

    # ||patch||^2 via conv_transpose with ones kernel in the (in_channels,
    # out_channels // groups, *k) layout that the transposed-conv weight uses.
    gin = layer.in_channels // layer.groups
    gout = layer.out_channels // layer.groups
    ones_kernel = torch.ones(
        (layer.in_channels, gout) + tuple(layer.kernel_size),
        device=input.device, dtype=input.dtype,
    )
    input_squared = input * input
    patch_sq_sum = conv_transpose_fn(
        input_squared, ones_kernel, None, layer.stride, layer.padding,
        output_padding, layer.groups, layer.dilation,
    )

    # ||kernel||^2 per output filter. For groups > 1 each absolute output
    # channel sees only its group's input channels — reshape to (groups, gin,
    # gout, *k), reduce over (gin, spatial), flatten to (out_channels,).
    w_grouped = weight.view(layer.groups, gin, gout, *layer.kernel_size)
    reduce_dims = (1,) + tuple(range(3, w_grouped.dim()))
    kernel_sq_sum = (w_grouped ** 2).sum(dim=reduce_dims).reshape(-1)

    view_shape = (1, -1) + (1,) * (dot_prod_map.dim() - 2)
    distance_sq = (
        patch_sq_sum + kernel_sq_sum.view(*view_shape) - 2 * dot_prod_map
    ).clamp(min=0.0)

    output = _yat_score(layer, dot_prod_map, distance_sq, bias_val, alpha)
    return saturating_downcast(output, output_dtype)
