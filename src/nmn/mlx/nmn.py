"""YAT dense layer for MLX (``mlx.nn.Module``-based).

Mirrors the surface of ``nmn.tf.nmn.YatNMN`` (Tier-0 feature set):

* learnable / constant / off bias inside the numerator square
* learnable / constant / off alpha output scaling
* learnable epsilon (softplus-reparameterized) or constant
* spherical mode (unit-norm inputs + kernel rows)
* weight normalization (unit-norm kernel rows at forward time)
* ``positive_init`` (``abs()`` of initial kernel)
* lazy build on first call so the input dimension does not need to be
  specified up front, matching the TF/Keras ergonomics
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


class YatNMN(nn.Module):
    """Dense layer implementing the ⵟ-Product (YAT) transformation.

    Computes ``y = (x · Wᵀ + b)² / (‖x − W_row‖² + ε)`` over the last
    dimension of the input. No separate activation function is needed —
    the non-linearity is intrinsic.

    Args:
        features: Number of output features.
        use_bias: If ``True``, add a bias inside the squared numerator.
        constant_bias: If a ``float``, use that value as a fixed (non-learnable)
            bias constant. If ``None``, use a learnable bias when
            ``use_bias=True``.
        use_alpha: Whether to apply output scaling via an alpha parameter.
            Has no effect when ``constant_alpha`` is set.
        constant_alpha: If ``True``, applies a fixed √2 scale factor. If a
            ``float``, applies that value as a fixed scale. If ``None``,
            a learnable scalar ``alpha`` is created when ``use_alpha=True``.
        positive_init: Initialise kernel weights with ``abs(.)`` so all
            entries are non-negative (default: ``False``).
        dtype: MLX dtype used for parameters and computation (default
            ``mx.float32``).
        epsilon: Small constant added to the denominator to avoid division
            by zero (default ``1e-5``).
        learnable_epsilon: If ``True``, epsilon becomes a learnable parameter
            passed through softplus to guarantee strict positivity.
        spherical: If ``True``, normalize inputs and each kernel row to unit
            norm before computing the YAT formula.
        weight_normalized: If ``True``, normalize each kernel row to unit
            norm at forward time. Skips the ``‖W‖²`` recomputation in the
            distance term as it is known to be 1.
        return_weights: If ``True``, ``__call__`` returns
            ``(output, kernel)`` instead of just ``output``.
        lazy: If ``True``, freeze ONLY the ``kernel`` (the feature directions)
            so it is excluded from :meth:`trainable_parameters` and the
            optimizer never updates it. The ``bias``, ``alpha`` and learnable
            ``epsilon`` remain fully trainable. Implemented via mlx's
            ``Module.freeze(keys=["kernel"])`` (the idiomatic per-variable
            freeze), not stop-gradient. ``freeze_kernel`` is an accepted alias.
            Default ``False`` (fully backward compatible).
        freeze_kernel: Alias for ``lazy``.

    Example::

        >>> import mlx.core as mx
        >>> from nmn.mlx import YatNMN
        >>> layer = YatNMN(features=10)
        >>> x = mx.zeros((4, 8))
        >>> layer(x).shape
        (4, 10)
    """

    def __init__(
        self,
        features: int,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        positive_init: bool = False,
        use_dropconnect: bool = False,
        drop_rate: float = 0.0,
        fused: bool = False,
        dtype: mx.Dtype = mx.float32,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        spherical: bool = False,
        weight_normalized: bool = False,
        return_weights: bool = False,
        lazy: bool = False,
        freeze_kernel: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

        # ``freeze_kernel`` is an alias for ``lazy``; either enables lazy mode.
        self.lazy = bool(lazy) or bool(freeze_kernel)

        self.features = features
        self.dtype = dtype
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.spherical = spherical
        self.weight_normalized = weight_normalized
        self.return_weights = return_weights
        self.positive_init = positive_init
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.fused = fused

        # ── Bias config ────────────────────────────────────────────────
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # constant bias is still "applied"
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # ── Alpha config ───────────────────────────────────────────────
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        # Parameters created lazily on first call so users don't have to
        # supply in_features up front (matches the TF/Keras ergonomics).
        self.is_built = False
        self.input_dim: Optional[int] = None

    # -----------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------

    def build(self, last_dim: int) -> None:
        """Allocate parameters once the input feature dimension is known."""
        if self.is_built:
            return
        self.input_dim = int(last_dim)

        # Xavier-normal initialisation (consistent with the TF backend).
        fan_in = last_dim
        fan_out = self.features
        std = math.sqrt(2.0 / (fan_in + fan_out))
        kernel = mx.random.normal(shape=(self.features, last_dim)) * std
        if self.positive_init:
            kernel = mx.abs(kernel)
        self.kernel = kernel.astype(self.dtype)

        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = mx.ones((1,), dtype=self.dtype)

        if self.use_bias and self._constant_bias_value is None:
            self.bias = mx.zeros((self.features,), dtype=self.dtype)

        if self.learnable_epsilon:
            # softplus(raw) = epsilon  ->  raw = log(exp(epsilon) - 1)
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = mx.array([raw_eps], dtype=self.dtype)

        self.is_built = True

        # Lazy mode: freeze ONLY the kernel so it is excluded from
        # ``trainable_parameters()`` (the optimizer never updates it). The
        # bias, alpha and learnable epsilon stay trainable.
        if self.lazy:
            self.freeze(keys=["kernel"], recurse=False)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def __call__(
        self,
        inputs: mx.array,
        *,
        deterministic: bool = True,
    ) -> Union[mx.array, Tuple[mx.array, mx.array]]:
        if inputs.dtype != self.dtype:
            inputs = inputs.astype(self.dtype)

        last_dim = inputs.shape[-1]
        if not self.is_built:
            self.build(last_dim)
        elif self.input_dim != last_dim:
            raise ValueError(
                f"Input shape changed: expected last dimension "
                f"{self.input_dim}, got {last_dim}"
            )

        kernel = self.kernel

        # DropConnect: randomly drop weights at the kernel level.
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            dc_mask = mx.random.bernoulli(p=keep_prob, shape=kernel.shape)
            kernel = (kernel * dc_mask.astype(kernel.dtype)) / keep_prob

        # Fused path: skip when an option that isn't supported by the
        # kernel is active (spherical, weight_normalized — those alter
        # the kernel before the YAT formula, and we don't want to drift
        # from the eager path).
        if (
            self.fused
            and not self.spherical
            and not self.weight_normalized
            and not self.return_weights
        ):
            from .fused import fused_yat_score

            if self.use_bias:
                if self._constant_bias_value is not None:
                    bias = mx.full(
                        (self.features,), self._constant_bias_value, dtype=self.dtype
                    )
                else:
                    bias = self.bias
                    if self.softplus_bias if hasattr(self, "softplus_bias") else False:
                        pass  # not supported in fused path
            else:
                bias = mx.zeros((self.features,), dtype=self.dtype)
            if self._constant_alpha_value is not None:
                alpha_arr = mx.array(
                    [self._constant_alpha_value], dtype=self.dtype
                )
            elif self.use_alpha and getattr(self, "alpha", None) is not None:
                alpha_arr = self.alpha
            else:
                alpha_arr = mx.ones((1,), dtype=self.dtype)
            if self.learnable_epsilon and getattr(self, "epsilon_param", None) is not None:
                eps_val = float(nn.softplus(self.epsilon_param)[0])
            else:
                eps_val = self.epsilon
            return fused_yat_score(inputs, kernel, bias=bias, alpha=alpha_arr,
                                   epsilon=eps_val)

        # Spherical: normalize inputs and each kernel row to unit norm.
        if self.spherical:
            inputs = inputs / (
                mx.sqrt(mx.sum(inputs * inputs, axis=-1, keepdims=True)) + 1e-8
            )
            kernel = kernel / (
                mx.sqrt(mx.sum(kernel * kernel, axis=-1, keepdims=True)) + 1e-8
            )

        # Weight normalization: normalize each kernel row at forward time.
        if self.weight_normalized:
            kernel = kernel / (
                mx.sqrt(mx.sum(kernel * kernel, axis=-1, keepdims=True)) + 1e-8
            )

        # y = x @ Wᵀ
        y = inputs @ kernel.T

        # Squared distance ||x − W_row||² broadcast across the (batch, feat)
        # axes.  For spherical inputs we use the closed-form 2 − 2·dot.
        if self.spherical:
            distances = mx.maximum(2.0 - 2.0 * y, 0.0)
        else:
            inputs_sq_sum = mx.sum(inputs * inputs, axis=-1, keepdims=True)
            if self.weight_normalized:
                kernel_sq_sum = mx.ones((self.features,), dtype=kernel.dtype)
            else:
                kernel_sq_sum = mx.sum(kernel * kernel, axis=-1)
            kernel_sq_sum = mx.reshape(
                kernel_sq_sum, [1] * (y.ndim - 1) + [self.features]
            )
            distances = mx.maximum(inputs_sq_sum + kernel_sq_sum - 2.0 * y, 0.0)

        # Bias inside the squared numerator (learnable or constant).
        if self.use_bias:
            bias_shape = [1] * (y.ndim - 1) + [self.features]
            if self._constant_bias_value is not None:
                bias_const = mx.full(
                    (self.features,),
                    self._constant_bias_value,
                    dtype=self.dtype,
                )
                y = y + mx.reshape(bias_const, bias_shape)
            else:
                y = y + mx.reshape(self.bias, bias_shape)

        # Effective epsilon.
        if self.learnable_epsilon and getattr(self, "epsilon_param", None) is not None:
            eps = nn.softplus(self.epsilon_param)
        else:
            eps = self.epsilon

        y = (y * y) / (distances + eps)

        # Alpha scaling.
        if self._constant_alpha_value is not None:
            y = y * mx.array(self._constant_alpha_value, dtype=self.dtype)
        elif self.use_alpha and getattr(self, "alpha", None) is not None:
            y = y * self.alpha

        if self.return_weights:
            return y, self.kernel
        return y

    # -----------------------------------------------------------------
    # Weight accessors — kept symmetric with nmn.tf.nmn.YatNMN.
    # -----------------------------------------------------------------

    def get_weights(self) -> List[mx.array]:
        if not self.is_built:
            raise ValueError("Layer must be built before weights can be retrieved.")
        weights: List[mx.array] = [self.kernel]
        if self.use_bias and self._constant_bias_value is None:
            weights.append(self.bias)
        if self.use_alpha and self._constant_alpha_value is None:
            weights.append(self.alpha)
        if self.learnable_epsilon:
            weights.append(self.epsilon_param)
        return weights

    def set_weights(self, weights: List[mx.array]) -> None:
        if not self.is_built:
            raise ValueError("Layer must be built before weights can be set.")
        expected = self.get_weights()
        if len(weights) != len(expected):
            raise ValueError(
                f"Expected {len(expected)} weight tensors, got {len(weights)}"
            )
        self.kernel = weights[0].astype(self.dtype)
        idx = 1
        if self.use_bias and self._constant_bias_value is None:
            self.bias = weights[idx].astype(self.dtype)
            idx += 1
        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = weights[idx].astype(self.dtype)
            idx += 1
        if self.learnable_epsilon:
            self.epsilon_param = weights[idx].astype(self.dtype)


# Backward-compat alias used by the other backends.
YatDense = YatNMN
