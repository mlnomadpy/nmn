# mypy: allow-untyped-defs
"""YatNMN - Yet Another Transformation Neural Matter Network."""
import logging
import math
import threading
from typing import Optional, Union

logger = logging.getLogger(__name__)

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
        constant_bias: If a float, use that value as a fixed (non-learnable) bias
            constant. If None (default), use learnable bias when bias=True.
        softplus_bias (bool): If True, the learnable bias parameter is passed
            through softplus in the forward pass to guarantee strict positivity
            (default: False). Ignored when constant_bias is set or bias=False.
        scalar_bias (bool): If True, the learnable bias is a single scalar
            (shape ``(1,)``) shared and broadcast across all ``out_features``
            neurons (default: False). Ignored when constant_bias is set or
            bias=False.
        alpha (bool): Whether to multiply with alpha. Ignored if constant_alpha is set.
        constant_alpha: If True, use sqrt(2) as constant alpha. If a float, use that value.
            If None (default), use learnable alpha when alpha=True.
        dtype (torch.dtype): Data type for computation (default: infer from input and params).
        param_dtype (torch.dtype): Data type for parameter initialization (default: float32).
        epsilon (float): Small constant to avoid division by zero
        learnable_epsilon (bool): If True, epsilon becomes a learnable parameter passed
            through softplus to guarantee strict positivity (default: False).
        spherical (bool): Whether to use spherical mode (normalize inputs and kernel)
        weight_normalized (bool): If True, normalize each neuron (row) of the kernel
            to have norm 1. (PyTorch kernels use shape ``(out_features, in_features)``,
            so each row is one neuron.) This optimization avoids recomputing kernel
            norms in YAT distance calculation since they are guaranteed to be 1.0.
        tie_kernel_bank (bool): If True, reuse shared kernels across compatible layers.
        kernel_bank_size (int): Optional explicit size for shared bank (auto-expands if needed).
        kernel_bank_id (str): Namespace for shared banks (allows multiple independent banks).
        kernel_init (callable): Initializer for the weight matrix
        bias_init (callable): Initializer for the bias
        alpha_init (callable): Initializer for the scaling parameter (only used if constant_alpha is None)
    """

    # Default constant alpha value (sqrt(2))
    DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)
    # Class-level shared kernel banks (guarded by a lock for thread safety)
    _KERNEL_BANKS = {}
    _KERNEL_BANKS_LOCK = threading.Lock()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        constant_bias: Optional[float] = None,
        softplus_bias: bool = False,
        scalar_bias: bool = False,
        alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        dtype: Optional[torch.dtype] = None,
        param_dtype: torch.dtype = torch.float32,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        spherical: bool = False,
        positive_init: bool = False,
        weight_normalized: bool = False,
        tie_kernel_bank: bool = False,
        kernel_bank_size: Optional[int] = None,
        kernel_bank_id: str = 'default',
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
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        if learnable_epsilon:
            # Initialize so that softplus(raw) ≈ epsilon: raw = log(exp(eps) - 1)
            raw_eps = math.log(math.exp(epsilon) - 1.0)
            self.epsilon_param = nn.Parameter(
                torch.full((1,), raw_eps, dtype=param_dtype)
            )
        else:
            self.register_parameter('epsilon_param', None)
        self.spherical = spherical
        self.positive_init = positive_init
        self.weight_normalized = weight_normalized
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id
        self._kernel_slice = slice(None)

        # Weight initialization
        if kernel_init is None:
            kernel_init = nn.init.xavier_normal_

        # Handle shared kernel bank
        if tie_kernel_bank:
            # Auto-calculate bank size
            bank_out_features = kernel_bank_size or out_features
            bank_key = (kernel_bank_id, in_features, param_dtype, id(kernel_init), positive_init)

            with YatNMN._KERNEL_BANKS_LOCK:
                shared_weight = YatNMN._KERNEL_BANKS.get(bank_key)
                if shared_weight is None:
                    # First layer: create bank
                    self.weight = nn.Parameter(torch.empty(
                        (bank_out_features, in_features),
                        dtype=param_dtype
                    ))
                    YatNMN._KERNEL_BANKS[bank_key] = self.weight
                else:
                    # Bank exists: auto-expand if needed
                    existing_size = shared_weight.shape[0]
                    if bank_out_features > existing_size:
                        # Auto-expand: pad with new random initialization
                        logger.info("Auto-expanding kernel bank '%s': %d -> %d neurons",
                                    kernel_bank_id, existing_size, bank_out_features)
                        old_weight = shared_weight.data
                        new_weight = torch.empty(
                            (bank_out_features, in_features),
                            dtype=param_dtype,
                            device=old_weight.device
                        )
                        kernel_init(new_weight)
                        if positive_init:
                            new_weight.abs_()
                        # Copy old weights
                        new_weight[:existing_size].copy_(old_weight)
                        shared_weight.data = new_weight
                        YatNMN._KERNEL_BANKS[bank_key] = shared_weight
                    self.weight = shared_weight

            self._kernel_slice = slice(0, out_features)
        else:
            # Create weight parameter (non-shared)
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
        if constant_alpha is not None and constant_alpha is not False:
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

        # Bias parameter (learnable, constant, or none)
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            self.register_parameter('bias', None)
            bias = True  # Bias is applied (but constant)
        elif bias:
            bias_shape = (1,) if scalar_bias else (out_features,)
            self.bias = nn.Parameter(torch.empty(
                bias_shape,
                dtype=param_dtype
            ))
        else:
            self.register_parameter('bias', None)
        self.use_bias = bias
        self.constant_bias = constant_bias
        self.softplus_bias = softplus_bias and self.bias is not None
        self.scalar_bias = scalar_bias and self.bias is not None

        # Initialize parameters
        self.reset_parameters(kernel_init, bias_init, alpha_init)

        # Normalize kernel if requested: normalize each neuron (column) to have norm 1
        if self.weight_normalized:
            weight_norm = torch.sqrt(torch.sum(self.weight**2, dim=-1, keepdim=True))
            self.weight.data = self.weight.data / (weight_norm + 1e-8)

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

        # Bias initialization (only for learnable bias)
        if self.bias is not None:
            if bias_init is None:
                # Default: uniform initialization
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                bias_init(self.bias)
        # Note: when constant_bias is set, self.bias is None and the constant
        # value is held in self._constant_bias_value (used at forward time)

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
        # Slice shared kernel if tying is enabled
        if self.tie_kernel_bank:
            kernel = kernel[self._kernel_slice]

        # Resolve bias (learnable or constant)
        if self._constant_bias_value is not None:
            bias = torch.full(
                (self.out_features,),
                self._constant_bias_value,
                dtype=self.param_dtype,
                device=kernel.device,
            )
        else:
            bias = self.bias
            if bias is not None and self.softplus_bias:
                bias = F.softplus(bias)
        alpha_param = self.alpha if self.alpha is not None else None

        # Promote all tensors to computation dtype
        x, kernel, bias, alpha_param = self._promote_dtype(x, kernel, bias, alpha_param)

        # Spherical mode: normalize inputs and kernel
        if self.spherical:
            x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
            kernel = kernel / (torch.norm(kernel, dim=-1, keepdim=True) + 1e-8)

        # Normalize kernel if weight normalization is enabled
        if self.weight_normalized:
            kernel = kernel / (torch.sqrt(torch.sum(kernel**2, dim=-1, keepdim=True)) + 1e-8)

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
            # Clamp to zero: bf16 cancellation can make distance negative when x ≈ W
            distances = (2 - 2 * (y - bias if bias is not None else y)).clamp(min=0.0)
        else:
            inputs_squared_sum = torch.sum(x**2, dim=-1, keepdim=True)

            # Optimization: if weights are normalized, ||W||² = 1 for each neuron
            if self.weight_normalized:
                kernel_squared_sum = torch.ones(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
            else:
                kernel_squared_sum = torch.sum(kernel**2, dim=-1)

            # Reuse dot product for distance: ||x||² + ||W||² - 2(x·W)
            dot_for_dist = y - bias if bias is not None else y
            # Clamp to zero: bf16 cancellation can make distance negative when x ≈ W
            distances = (inputs_squared_sum + kernel_squared_sum - 2 * dot_for_dist).clamp(min=0.0)

        # Resolve effective epsilon (learnable via softplus, or constant)
        if self.learnable_epsilon and self.epsilon_param is not None:
            eps = F.softplus(self.epsilon_param.to(distances.dtype))
        else:
            eps = self.epsilon

        # Apply squared Euclidean distance transformation: (x·W + b)² / (||x - W||² + ε)
        y = y ** 2 / (distances + eps)

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
