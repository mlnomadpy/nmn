# mypy: allow-untyped-defs
"""YatNMN - Yet Another Transformation Neural Matter Network."""
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["YatNMN"]


class YatNMN(nn.Module):
    """
    A PyTorch implementation of the Yat neuron with squared Euclidean distance transformation.

    y = (x · W + b)² / (||x - W||² + ε)

  With optional scaling:
    y = y * alpha (learnable) or y = y * sqrt(2) (constant)

    Attributes:
        in_features (int): Size of each input sample
        out_features (int): Size of each output sample
        bias (bool): Whether to add a bias to the output
        alpha (bool): Whether to multiply with alpha. Ignored if constant_alpha is set.
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float, use that value.
            If None (default), use learnable alpha when alpha=True.
        dtype (torch.dtype): Data type for computation (default: infer from input and params).
        param_dtype (torch.dtype): Data type for parameter initialization (default: float32).
        epsilon (float): Small constant to avoid division by zero
        spherical (bool): Whether to use spherical mode (normalize inputs and kernel)
        kernel_init (callable): Initializer for the weight matrix
        bias_init (callable): Initializer for the bias
        alpha_init (callable): Initializer for the scaling parameter (only used if constant_alpha is None)
    """

    # Default constant alpha value (sqrt(2))
    DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        dtype: Optional[torch.dtype] = None,
        param_dtype: torch.dtype = torch.float32,
        epsilon: float = 1e-5,
        spherical: bool = False,
        positive_init: bool = False,
        kernel_init: callable = None,
        bias_init: callable = None,
        alpha_init: callable = None
    ):
        super().__init__()

        # Store attributes
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.epsilon = epsilon
        self.spherical = spherical
        self.positive_init = positive_init

        # Weight initialization
        if kernel_init is None:
            kernel_init = nn.init.xavier_normal_

        # Create weight parameter
        self.weight = nn.Parameter(torch.empty(
            (out_features, in_features),
            dtype=param_dtype
        ))

        # Handle alpha configuration
        # Priority: constant_alpha > alpha
        #
        # Options:
        #   1. constant_alpha=True -> use sqrt(2) as constant
        #   2. constant_alpha=<float> -> use that value as constant
        #   3. alpha=True (default) -> learnable alpha parameter
        #   4. alpha=False -> no alpha scaling
        self._constant_alpha_value = None
        if constant_alpha is not None:
            # Use constant alpha (no learnable parameter)
            if constant_alpha is True:
                self._constant_alpha_value = self.DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.register_parameter('alpha', None)
            alpha = True  # Alpha scaling is enabled (but constant)
        elif alpha:
            self.alpha = nn.Parameter(torch.ones(
                (1,),
                dtype=param_dtype
            ))
        else:
            self.register_parameter('alpha', None)
        self.use_alpha = alpha
        self.constant_alpha = constant_alpha

        # Bias parameter
        if bias:
            self.bias = nn.Parameter(torch.empty(
                (out_features,),
                dtype=param_dtype
            ))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters(kernel_init, bias_init, alpha_init)

    def reset_parameters(
        self,
        kernel_init: callable = None,
        bias_init: callable = None,
        alpha_init: callable = None
    ):
        """
        Initialize network parameters with specified or default initializers.
        """
        # Kernel (weight) initialization
        if kernel_init is None:
            kernel_init = nn.init.xavier_normal_
        kernel_init(self.weight)
        if self.positive_init:
            with torch.no_grad():
                self.weight.abs_()

        # Bias initialization
        if self.bias is not None:
            if bias_init is None:
                # Default: uniform initialization
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                bias_init(self.bias)

        # Alpha initialization (default to 1.0)
        if self.alpha is not None:
            if alpha_init is None:
                self.alpha.data.fill_(1.0)
            else:
                alpha_init(self.alpha)

    def _promote_dtype(self, *tensors):
        """
        Promote tensors to the computation dtype.

        If self.dtype is set, cast all tensors to that dtype.
        Otherwise, infer the dtype from the first non-None tensor.
        """
        if self.dtype is not None:
            target = self.dtype
        else:
            # Infer from first non-None tensor
            target = None
            for t in tensors:
                if t is not None:
                    target = t.dtype
                    break
            if target is None:
                target = self.param_dtype

        return tuple(
            t.to(target) if t is not None else None
            for t in tensors
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with squared Euclidean distance transformation.

        Computes: y = (x · W + b)² / (||x - W||² + ε), with optional alpha scaling.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed output
        """
        kernel = self.weight
        bias = self.bias
        alpha_param = self.alpha if self.alpha is not None else None

        # Promote all tensors to computation dtype
        x, kernel, bias, alpha_param = self._promote_dtype(x, kernel, bias, alpha_param)

        # Spherical mode: normalize inputs and kernel
        if self.spherical:
            x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
            kernel = kernel / (torch.norm(kernel, dim=-1, keepdim=True) + 1e-8)

        # Compute dot product: x · W
        y = torch.matmul(x, kernel.t())

        # Add bias before squaring (matches NNX: (x·W + b)² / (dist + ε))
        if bias is not None:
            y = y + bias

        # Compute squared distances
        if self.spherical:
            # Spherical: inputs and kernel are normalized
            # ||x - W||² = ||x||² + ||W||² - 2(x·W) = 1 + 1 - 2(x·W) = 2 - 2(x·W)
            # Reuse y (before bias) since it equals x · W^T
            distances = 2 - 2 * (y - bias if bias is not None else y)
        else:
            inputs_squared_sum = torch.sum(x**2, dim=-1, keepdim=True)
            kernel_squared_sum = torch.sum(kernel**2, dim=-1)
            # Reuse dot product for distance: ||x||² + ||W||² - 2(x·W)
            dot_for_dist = y - bias if bias is not None else y
            distances = inputs_squared_sum + kernel_squared_sum - 2 * dot_for_dist

        # Apply squared Euclidean distance transformation: (x·W + b)² / (||x - W||² + ε)
        y = y ** 2 / (distances + self.epsilon)

        # Dynamic scaling
        if self._constant_alpha_value is not None:
            # Constant alpha: use directly as the scale factor (e.g. sqrt(2))
            y = y * self._constant_alpha_value
        elif alpha_param is not None:
            # Simple learnable alpha scaling
            y = y * alpha_param

        return y

    def extra_repr(self) -> str:
        """
        Extra representation of the module for print formatting.
        """
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"alpha={self.alpha is not None}, "
                f"constant_alpha={self.constant_alpha}, "
                f"spherical={self.spherical}, "
                f"dtype={self.dtype}, "
                f"param_dtype={self.param_dtype}")
