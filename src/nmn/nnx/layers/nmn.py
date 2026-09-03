from __future__ import annotations

import functools
import math
import threading
import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import lax

from ._numerics import fp32_if_low_precision, inverse_softplus

Array: tp.TypeAlias = jax.Array
Axis = int
Size = int


default_kernel_init = initializers.xavier_normal()
default_bias_init = initializers.zeros_init()
default_alpha_init = initializers.ones_init()


class FrozenParam(nnx.Variable):
    """A frozen (non-trainable) parameter variable.

    Used by :class:`YatNMN` in *lazy* mode (issue #37) to hold the kernel so that
    it is naturally excluded from ``nnx.state(model, nnx.Param)`` — and therefore
    from the optimizer/gradient — while ``bias``, ``alpha`` and the learnable
    ``epsilon`` remain ordinary :class:`flax.nnx.Param` and stay trainable.

    This is the idiomatic NNX mechanism (preferred over ``jax.lax.stop_gradient``):
    the kernel is excluded by *variable type* from the trainable state, not merely
    zero-gradient. ``nnx.state(model, FrozenParam)`` recovers the frozen kernel.
    """

    pass


class YatNMN(Module):
    """A YAT linear transformation applied over the last dimension of the input.

    The YAT  operation computes:
      y = (x · W)² / (||x - W||² + ε)

    With optional scaling:
      y = y * alpha (learnable) or y = y * sqrt(2) (constant)

    Example usage::

      >>> from flax import nnx
      >>> from nmn.nnx.nmn import YatNMN
      >>> import jax.numpy as jnp

      >>> # Learnable alpha (default)
      >>> layer = YatNMN(in_features=3, out_features=4, rngs=nnx.Rngs(0))

      >>> # Constant alpha = sqrt(2) (recommended default)
      >>> layer = YatNMN(in_features=3, out_features=4, constant_alpha=True, rngs=nnx.Rngs(0))

      >>> # Custom constant alpha value
      >>> layer = YatNMN(in_features=3, out_features=4, constant_alpha=1.5, rngs=nnx.Rngs(0))

      >>> # No alpha scaling
      >>> layer = YatNMN(in_features=3, out_features=4, use_alpha=False, rngs=nnx.Rngs(0))

    Args:
      in_features: the number of input features.
      out_features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      constant_bias: if a float, use that value as a fixed (non-learnable) bias constant.
        If None (default), use learnable bias when use_bias=True.
      softplus_bias: if True, the learnable bias parameter is passed through
        softplus in the forward pass to guarantee strict positivity (default: False).
        Ignored when constant_bias is set or use_bias=False.
      scalar_bias: if True, the learnable bias is a single scalar (shape ``(1,)``)
        shared and broadcast across all ``out_features`` neurons (default: False).
        Ignored when constant_bias is set or use_bias=False.
      use_alpha: whether to use alpha scaling (default: True). Ignored if constant_alpha is set.
      constant_alpha: if True, use sqrt(2) as constant alpha. If a float, use that value.
        If None (default) or False, use learnable alpha when use_alpha=True.
        Note: False is treated as None — pass a float (e.g. 0.0) to actually freeze
        alpha at a numeric value.
      use_dropconnect: whether to use DropConnect (default: False).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      compute_mode: numerical mode for the YAT score. ``"fp32"`` preserves the
        reference implementation, ``"mixed"`` keeps operands in ``dtype`` while
        accumulating dot products and reductions in float32, and ``"bf16"`` uses
        a dimension-scaled strict-BF16 formulation (default: ``"fp32"``).
      distance_floor: non-negative lower bound applied to the squared distance
        before epsilon is added. This is especially useful for near-collisions in
        ``"bf16"`` mode (default: 0.0).
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      alpha_init: initializer function for the learnable alpha (only used if constant_alpha is None).
      dot_general: dot product function.
      promote_dtype: function to promote the dtype of the arrays to the desired
        dtype. The function should accept a tuple of ``(inputs, kernel, bias)``
        and a ``dtype`` keyword argument, and return a tuple of arrays with the
        promoted dtype.
      epsilon: A small float added to the denominator to prevent division by zero.
      learnable_epsilon: if True, epsilon becomes a learnable parameter passed
        through softplus to guarantee strict positivity (default: False).
      drop_rate: dropout rate for DropConnect (default: 0.0).
      weight_normalized: if True, normalize each neuron (column) of the kernel to
        have norm 1. This optimization avoids recomputing kernel norms in YAT
        distance calculation since they are guaranteed to be 1.0.
      tie_kernel_bank: if True, reuse a shared kernel bank across compatible
        YatNMN instances and slice the first ``out_features`` neurons.
      kernel_bank_size: total neurons in the shared bank. If None, defaults to
        ``out_features`` for the creating instance. Bank capacity is immutable;
        declare the largest required size on the first consumer.
      kernel_bank_id: optional bank namespace to control sharing groups.
      rngs: rng key.
    """

    __data__ = ("kernel", "bias", "alpha", "epsilon_param", "dropconnect_key")
    _KERNEL_BANKS: dict[tuple[tp.Any, ...], nnx.Param] = {}
    _KERNEL_BANKS_LOCK = threading.Lock()

    # Default constant alpha value (sqrt(2))
    DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)  # jnp.sqrt(2.0)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        constant_bias: tp.Optional[float] = None,
        softplus_bias: bool = False,
        scalar_bias: bool = False,
        use_alpha: bool = True,
        constant_alpha: tp.Optional[tp.Union[bool, float]] = None,
        positive_init: bool = False,
        use_dropconnect: bool = False,
        fused: bool = False,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        compute_mode: str = "fp32",
        distance_floor: float = 0.0,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        alpha_init: Initializer = default_alpha_init,
        dot_general: DotGeneralT = lax.dot_general,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        spherical: bool = False,
        drop_rate: float = 0.0,
        weight_normalized: bool = False,
        tie_kernel_bank: bool = False,
        kernel_bank_size: tp.Optional[int] = None,
        kernel_bank_id: str = "default",
        lazy: bool = False,
        freeze_kernel: tp.Optional[bool] = None,
        rngs: rnglib.Rngs,
    ):

        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(
                "drop_rate must be in the half-open interval [0, 1), "
                f"got {drop_rate}"
            )

        # ── Lazy mode (issue #37): freeze ONLY the kernel ───────────────────────
        # `freeze_kernel` is an alias for `lazy`. When enabled, the kernel is stored
        # under FrozenParam (a non-Param nnx.Variable) so it is excluded from
        # nnx.state(model, nnx.Param) — and hence from the optimizer/grad — while
        # bias, alpha and the learnable epsilon stay ordinary nnx.Param (trainable).
        if freeze_kernel is not None:
            lazy = bool(freeze_kernel)
        self.lazy = lazy
        self.freeze_kernel = lazy
        if lazy and tie_kernel_bank:
            raise ValueError(
                "lazy/freeze_kernel is not supported together with tie_kernel_bank: "
                "a shared kernel bank cannot be frozen per-instance."
            )
        # Pick the variable type used to wrap the kernel.
        _kernel_var = FrozenParam if lazy else nnx.Param

        self._tie_kernel_bank = tie_kernel_bank
        self._kernel_slice = slice(None)
        self.kernel_shape = (in_features, out_features)

        if tie_kernel_bank:
            bank_out_features = (
                out_features if kernel_bank_size is None else kernel_bank_size
            )
            if bank_out_features < out_features:
                raise ValueError(
                    "kernel_bank_size must be at least out_features, "
                    f"got {bank_out_features} < {out_features}"
                )

            bank_shape = (in_features, bank_out_features)
            bank_key = (
                kernel_bank_id,
                in_features,
                param_dtype,
                kernel_init,
                positive_init,
            )

            with YatNMN._KERNEL_BANKS_LOCK:
                shared_kernel = YatNMN._KERNEL_BANKS.get(bank_key)
                if shared_kernel is None:
                    # The first consumer fixes capacity. Resizing a live Param can
                    # invalidate gradients and optimizer moments that already hold
                    # its original shape, and NNX state extraction is not observable
                    # here. Requiring up-front capacity is therefore the only safe
                    # class-global sharing contract.
                    kernel_key = rngs.params()
                    kernel_val = kernel_init(kernel_key, bank_shape, param_dtype)
                    if positive_init:
                        kernel_val = jnp.abs(kernel_val)
                    shared_kernel = nnx.Param(kernel_val)
                    YatNMN._KERNEL_BANKS[bank_key] = shared_kernel
                else:
                    existing_shape = shared_kernel[...].shape
                    existing_bank_size = existing_shape[-1]

                    if bank_out_features > existing_bank_size:
                        raise ValueError(
                            f"Kernel bank {kernel_bank_id!r} has fixed capacity "
                            f"{existing_bank_size}; requested {bank_out_features}. "
                            "Set kernel_bank_size to the maximum required capacity "
                            "when constructing the first consumer."
                        )

            self.kernel = shared_kernel
            self._kernel_slice = slice(0, out_features)
        else:
            kernel_key = rngs.params()
            kernel_val = kernel_init(
                kernel_key, (in_features, out_features), param_dtype
            )
            if positive_init:
                kernel_val = jnp.abs(kernel_val)
            # In lazy mode the kernel is a FrozenParam (excluded from trainable state);
            # otherwise a normal trainable nnx.Param.
            self.kernel = _kernel_var(kernel_val)
        self.bias: nnx.Param[jax.Array] | None
        self._constant_bias_value: tp.Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            self.bias = None
            use_bias = True  # Bias is applied (but constant)
        elif use_bias:
            bias_key = rngs.params()
            bias_shape = (1,) if scalar_bias else (out_features,)
            self.bias = nnx.Param(bias_init(bias_key, bias_shape, param_dtype))
        else:
            self.bias = None

        # Handle alpha configuration
        # Priority: constant_alpha > use_alpha
        #
        # Options:
        #   1. constant_alpha=True -> use sqrt(2) as constant
        #   2. constant_alpha=<float> -> use that value as constant
        #   3. use_alpha=True (default) -> learnable alpha parameter
        #   4. use_alpha=False -> no alpha scaling

        self.alpha: nnx.Param[jax.Array] | None

        if constant_alpha is not None and constant_alpha is not False:
            # Use constant alpha (no learnable parameter)
            if constant_alpha is True:
                self._constant_alpha_value = self.DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            self.alpha = None
            use_alpha = True  # Alpha scaling is enabled (but constant)
        else:
            self._constant_alpha_value = None
            if use_alpha:
                # Use learnable alpha
                alpha_key = rngs.params()
                self.alpha = nnx.Param(alpha_init(alpha_key, (1,), param_dtype))
            else:
                # No alpha scaling
                self.alpha = None

        self.use_alpha = use_alpha

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.constant_bias = constant_bias
        self.softplus_bias = softplus_bias and self.bias is not None
        self.scalar_bias = scalar_bias and self.bias is not None
        self.constant_alpha = constant_alpha
        self.use_dropconnect = use_dropconnect
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        if compute_mode not in ("fp32", "mixed", "bf16"):
            raise ValueError(
                "compute_mode must be one of 'fp32', 'mixed', or 'bf16', "
                f"got {compute_mode!r}"
            )
        distance_floor = float(distance_floor)
        if not math.isfinite(distance_floor) or distance_floor < 0:
            raise ValueError(
                f"distance_floor must be finite and non-negative, got {distance_floor}"
            )
        self.compute_mode = compute_mode
        self.distance_floor = distance_floor
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general
        self.promote_dtype = promote_dtype
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.epsilon_param: nnx.Param[jax.Array] | None
        if learnable_epsilon:
            # Initialize so that softplus(raw) ≈ epsilon: raw = log(exp(eps) - 1)
            # Compute this in Python float precision before casting. Evaluating the
            # inverse in bf16/fp16 rounds exp(1e-5) to 1 and produces ``-inf``.
            self.epsilon_param = nnx.Param(inverse_softplus(epsilon, param_dtype))
        else:
            self.epsilon_param = None
        self.spherical = spherical
        self.drop_rate = drop_rate
        self.fused = fused
        self.weight_normalized = weight_normalized
        self.tie_kernel_bank = tie_kernel_bank
        self.kernel_bank_size = kernel_bank_size
        self.kernel_bank_id = kernel_bank_id

        # Normalize kernel if requested: normalize each neuron (column) to have norm 1
        if self.weight_normalized:
            kernel_val = self.kernel[...]
            kernel_norm = jnp.sqrt(jnp.sum(kernel_val**2, axis=0, keepdims=True))
            self.kernel[...] = kernel_val / (kernel_norm + 1e-8)

        if use_dropconnect:
            self.dropconnect_key = rngs.dropout.fork()
        else:
            self.dropconnect_key = None

    def __call__(self, inputs: Array, *, deterministic: bool = False) -> Array:
        """Applies a YAT linear transformation to the inputs along the last dimension.

        Computes: y = (x · W)² / (||x - W||² + ε), with optional alpha scaling.

        Args:
          inputs: The nd-array to be transformed.
          deterministic: If true, DropConnect is not applied (e.g., during inference).

        Returns:
          The transformed input.
        """
        kernel = self.kernel[...]
        if self._tie_kernel_bank:
            kernel = kernel[:, self._kernel_slice]

        # Get bias value (either learnable or constant)
        if self._constant_bias_value is not None:
            bias = jnp.full(
                (self.out_features,), self._constant_bias_value, dtype=self.param_dtype
            )
        elif self.bias is not None:
            bias = self.bias[...]
            if self.softplus_bias:
                bias = jax.nn.softplus(bias)
        else:
            bias = None

        # Get alpha value (either learnable or constant)
        if self._constant_alpha_value is not None:
            alpha = jnp.array(self._constant_alpha_value, dtype=self.param_dtype)
        elif self.alpha is not None:
            alpha = self.alpha[...]
        else:
            alpha = None

        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            mask = jax.random.bernoulli(
                self.dropconnect_key(), p=keep_prob, shape=kernel.shape
            )
            kernel = (kernel * mask) / keep_prob

        # Normalize kernel if weight normalization is enabled
        if self.weight_normalized:
            kernel = kernel / (
                jnp.sqrt(jnp.sum(kernel**2, axis=0, keepdims=True)) + 1e-8
            )

        if self.spherical:
            inputs = inputs / (jnp.linalg.norm(inputs, axis=-1, keepdims=True) + 1e-8)
            kernel = kernel / (jnp.linalg.norm(kernel, axis=0, keepdims=True) + 1e-8)

        inputs, kernel, bias, alpha = self.promote_dtype(
            (inputs, kernel, bias, alpha), dtype=self.dtype
        )

        # Resolve effective epsilon (learnable via softplus, or constant)
        if self.learnable_epsilon and self.epsilon_param is not None:
            (raw_epsilon,) = fp32_if_low_precision(self.epsilon_param[...])
            eps = jax.nn.softplus(raw_epsilon)
        else:
            eps = self.epsilon

        # ── Fused path: optimized/reference or exact mode-aware custom VJP ──
        if self.fused and not self.spherical:
            return _fused_yat_call(
                inputs,
                kernel,
                alpha,
                bias,
                eps,
                self._constant_alpha_value,
                self.compute_mode,
                self.distance_floor,
                self.precision,
                self.weight_normalized,
            )

        # ── Standard path ──────────────────────────────────────────────────
        if self._constant_alpha_value is not None:
            alpha_value = jnp.asarray(self._constant_alpha_value)
        elif alpha is not None:
            alpha_value = alpha
        else:
            alpha_value = jnp.ones((), dtype=inputs.dtype)
        if bias is None:
            bias_value = jnp.zeros((1,), dtype=inputs.dtype)
        else:
            bias_value = bias
        eps_value = jnp.asarray(eps)

        return _yat_value(
            inputs,
            kernel,
            alpha_value,
            bias_value,
            eps_value,
            bias is not None,
            self.compute_mode,
            self.distance_floor,
            self.precision,
            self.spherical,
            self.weight_normalized,
            self.dot_general,
        )


def _yat_value(
    x,
    kernel,
    alpha,
    bias,
    eps,
    has_bias,
    compute_mode,
    distance_floor,
    precision,
    spherical=False,
    weight_normalized=False,
    dot_general=lax.dot_general,
):
    """Evaluate a YAT score using the requested numerical mode."""
    output_dtype = x.dtype
    dimension_numbers = (((x.ndim - 1,), (0,)), ((), ()))

    if compute_mode == "fp32":
        x_compute = x.astype(jnp.float32)
        kernel_compute = kernel.astype(jnp.float32)
        accumulation_dtype = jnp.float32
        dot = dot_general(
            x_compute,
            kernel_compute,
            dimension_numbers,
            precision=precision,
        )
    elif compute_mode == "mixed":
        x_compute = x
        kernel_compute = kernel
        accumulation_dtype = jnp.float32
        dot = dot_general(
            x_compute,
            kernel_compute,
            dimension_numbers,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
    else:
        # The strict-BF16 mode intentionally rounds products and reductions to BF16.
        x_compute = x.astype(jnp.bfloat16)
        kernel_compute = kernel.astype(jnp.bfloat16)
        accumulation_dtype = jnp.bfloat16
        dot = dot_general(
            x_compute,
            kernel_compute,
            dimension_numbers,
            precision=precision,
        )

    alpha_compute = alpha.astype(accumulation_dtype)
    eps_compute = eps.astype(accumulation_dtype)
    floor = jnp.asarray(distance_floor, dtype=accumulation_dtype)
    bias_shape = (1,) * (dot.ndim - 1) + (-1,)
    bias_compute = jnp.reshape(bias.astype(accumulation_dtype), bias_shape)

    if spherical:
        distances = jnp.maximum(
            jnp.asarray(2, accumulation_dtype) - 2 * dot,
            floor,
        )
        numerator = dot + bias_compute if has_bias else dot
        result = alpha_compute * numerator**2 / (distances + eps_compute)
    elif compute_mode == "bf16":
        # Dividing all reductions by d keeps BF16 intermediates near unit scale:
        # d * (mean(xw) + b/d)^2 / (mean((x-w)^2) + epsilon/d).
        width = jnp.asarray(x.shape[-1], dtype=accumulation_dtype)
        mean_dot = dot / width
        input_mean_square = (
            jnp.sum(
                x_compute * x_compute,
                axis=-1,
                keepdims=True,
                dtype=accumulation_dtype,
            )
            / width
        )
        if weight_normalized:
            kernel_mean_square = (
                jnp.ones((1, kernel_compute.shape[-1]), dtype=accumulation_dtype)
                / width
            )
        else:
            kernel_mean_square = (
                jnp.sum(
                    kernel_compute * kernel_compute,
                    axis=0,
                    keepdims=True,
                    dtype=accumulation_dtype,
                )
                / width
            )
        mean_distances = jnp.maximum(
            input_mean_square + kernel_mean_square - 2 * mean_dot,
            floor / width,
        )
        numerator = mean_dot + bias_compute / width if has_bias else mean_dot
        result = (
            alpha_compute
            * width
            * numerator**2
            / (mean_distances + eps_compute / width)
        )
    else:
        input_squared_sum = jnp.sum(
            x_compute * x_compute,
            axis=-1,
            keepdims=True,
            dtype=accumulation_dtype,
        )
        if weight_normalized:
            kernel_squared_sum = jnp.ones(
                (1, kernel_compute.shape[-1]), dtype=accumulation_dtype
            )
        else:
            kernel_squared_sum = jnp.sum(
                kernel_compute * kernel_compute,
                axis=0,
                keepdims=True,
                dtype=accumulation_dtype,
            )
        distances = jnp.maximum(
            input_squared_sum + kernel_squared_sum - 2 * dot,
            floor,
        )
        numerator = dot + bias_compute if has_bias else dot
        result = alpha_compute * numerator**2 / (distances + eps_compute)

    return result.astype(output_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Fused YatNMN kernel — unified custom_vjp for reduced activation memory
#
# Standard autodiff saves the score intermediates. The fused version saves only
# its five array operands and recomputes the exact mode-specific forward graph in
# backward. This keeps the clamp derivative and BF16 rounding identical to the
# standard implementation.
#
# Supports all combinations of: bias / no-bias, learnable / constant epsilon,
# learnable / constant / no alpha.  Boolean flags via nondiff_argnums let JAX
# eliminate dead branches at trace time — zero runtime overhead.
# ══════════════════════════════════════════════════════════════════════════════


def _fused_yat_call(
    inputs,
    kernel,
    alpha,
    bias,
    eps,
    constant_alpha_value,
    compute_mode,
    distance_floor,
    precision,
    weight_normalized,
):
    """Dispatch to unified fused forward."""
    # Alpha array + grad flag
    if constant_alpha_value is not None:
        a = jnp.array(constant_alpha_value, dtype=jnp.float32)
        has_alpha_grad = False
    elif alpha is not None:
        a = alpha
        has_alpha_grad = True
    else:
        a = jnp.array(1.0, dtype=jnp.float32)
        has_alpha_grad = False

    # Bias array + flag
    if bias is not None:
        b = bias
        has_bias = True
    else:
        b = jnp.zeros(1, dtype=jnp.float32)
        has_bias = False

    # Epsilon — if it's a JAX array (from softplus), it's learnable
    if isinstance(eps, jnp.ndarray):
        e = eps
        has_eps_grad = True
    else:
        e = jnp.array(eps, dtype=jnp.float32)
        has_eps_grad = False

    # Preserve the existing optimized analytical VJP for the default/reference
    # configuration. Mode-aware or clamped execution uses the exact recomputing
    # VJP below so its BF16 rounding and clamp derivative match standard autodiff.
    if compute_mode == "fp32" and distance_floor == 0.0:
        return _fused_yat_fp32(
            inputs, kernel, a, b, e, has_alpha_grad, has_bias, has_eps_grad
        )

    return _fused_yat(
        inputs,
        kernel,
        a,
        b,
        e,
        has_alpha_grad,
        has_bias,
        has_eps_grad,
        compute_mode,
        distance_floor,
        precision,
        weight_normalized,
    )


# ── Unified custom_vjp ──


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _fused_yat_fp32(
    x, kernel, alpha, bias, eps, has_alpha_grad, has_bias, has_eps_grad
):
    x_f32 = x.astype(jnp.float32)
    kernel_f32 = kernel.astype(jnp.float32)
    alpha_f32 = alpha.astype(jnp.float32)
    dot = x_f32 @ kernel_f32
    input_squared_sum = jnp.sum(x_f32**2, axis=-1, keepdims=True)
    kernel_squared_sum = jnp.sum(kernel_f32**2, axis=0, keepdims=True)
    distance = jnp.maximum(
        input_squared_sum + kernel_squared_sum - 2 * dot, 0.0
    ) + eps.astype(jnp.float32)
    numerator = dot + bias.astype(jnp.float32) if has_bias else dot
    return (alpha_f32 * numerator**2 / distance).astype(x.dtype)


def _fused_yat_fp32_fwd(
    x, kernel, alpha, bias, eps, has_alpha_grad, has_bias, has_eps_grad
):
    x_f32 = x.astype(jnp.float32)
    kernel_f32 = kernel.astype(jnp.float32)
    alpha_f32 = alpha.astype(jnp.float32)
    eps_f32 = eps.astype(jnp.float32)
    dot = x_f32 @ kernel_f32
    input_squared_sum = jnp.sum(x_f32**2, axis=-1, keepdims=True)
    kernel_squared_sum = jnp.sum(kernel_f32**2, axis=0, keepdims=True)
    raw_distance = input_squared_sum + kernel_squared_sum - 2 * dot
    distance = jnp.maximum(raw_distance, 0.0) + eps_f32
    numerator = dot + bias.astype(jnp.float32) if has_bias else dot
    out = (alpha_f32 * numerator**2 / distance).astype(x.dtype)
    return out, (x, kernel, alpha, bias, eps, dot, raw_distance, distance)


def _fused_yat_fp32_bwd(has_alpha_grad, has_bias, has_eps_grad, res, g):
    x, kernel, alpha, bias, eps, dot, raw_distance, distance = res
    alpha_f32 = alpha.astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)
    numerator = dot + bias.astype(jnp.float32) if has_bias else dot

    # Match jnp.maximum(raw_distance, 0)'s subgradient exactly: zero below
    # the floor, one above it, and one half at equality.
    distance_clamp_grad = jnp.where(
        raw_distance > 0.0,
        1.0,
        jnp.where(raw_distance < 0.0, 0.0, 0.5),
    )

    g_x, g_kernel = _fused_yat_fp32_grad_xw(
        x, kernel, numerator, distance, distance_clamp_grad, g, alpha_f32
    )

    if has_alpha_grad:
        raw_yat = numerator**2 / distance
        g_alpha = jnp.sum(g_f32 * raw_yat).reshape(alpha.shape).astype(alpha.dtype)
    else:
        g_alpha = jnp.zeros_like(alpha)

    if has_bias:
        g_bias = jnp.sum(
            g_f32 * alpha_f32 * 2 * numerator / distance,
            axis=tuple(range(g.ndim - 1)),
        )
        if bias.shape != g_bias.shape:
            g_bias = jnp.sum(g_bias, keepdims=True)
        g_bias = g_bias.astype(bias.dtype)
    else:
        g_bias = jnp.zeros_like(bias)

    if has_eps_grad:
        g_eps = jnp.sum(g_f32 * (-alpha_f32 * numerator**2 / distance**2))
        g_eps = g_eps.reshape(eps.shape).astype(eps.dtype)
    else:
        g_eps = jnp.zeros_like(eps)

    return g_x, g_kernel, g_alpha, g_bias, g_eps


def _fused_yat_fp32_grad_xw(
    x, kernel, numerator, distance, distance_clamp_grad, g, alpha
):
    """Compute the optimized reference-mode input and kernel gradients."""
    g_f32 = g.astype(jnp.float32)
    x_f32 = x.astype(jnp.float32)
    kernel_f32 = kernel.astype(jnp.float32)
    inverse_distance = 1.0 / distance
    g_numerator = g_f32 * alpha * 2 * numerator * inverse_distance
    g_distance = g_f32 * -alpha * numerator**2 * inverse_distance**2
    g_distance = g_distance * distance_clamp_grad
    g_dot = g_numerator - 2 * g_distance
    g_distance_sum = jnp.sum(g_distance, axis=-1, keepdims=True)
    g_x = g_dot @ kernel_f32.T + 2 * x_f32 * g_distance_sum

    x_flat = x_f32.reshape(-1, x_f32.shape[-1])
    g_dot_flat = g_dot.reshape(-1, g_dot.shape[-1])
    g_distance_flat = g_distance.reshape(-1, g_distance.shape[-1])
    g_kernel = x_flat.T @ g_dot_flat
    g_kernel += 2 * kernel_f32 * jnp.sum(g_distance_flat, axis=0, keepdims=True)
    return g_x.astype(x.dtype), g_kernel.astype(kernel.dtype)


_fused_yat_fp32.defvjp(_fused_yat_fp32_fwd, _fused_yat_fp32_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11))
def _fused_yat(
    x,
    kernel,
    alpha,
    bias,
    eps,
    has_alpha_grad,
    has_bias,
    has_eps_grad,
    compute_mode,
    distance_floor,
    precision,
    weight_normalized,
):
    return _yat_value(
        x,
        kernel,
        alpha,
        bias,
        eps,
        has_bias,
        compute_mode,
        distance_floor,
        precision,
        False,
        weight_normalized,
    )


def _fused_yat_fwd(
    x,
    kernel,
    alpha,
    bias,
    eps,
    has_alpha_grad,
    has_bias,
    has_eps_grad,
    compute_mode,
    distance_floor,
    precision,
    weight_normalized,
):
    out = _yat_value(
        x,
        kernel,
        alpha,
        bias,
        eps,
        has_bias,
        compute_mode,
        distance_floor,
        precision,
        False,
        weight_normalized,
    )
    return out, (x, kernel, alpha, bias, eps)


def _fused_yat_bwd(
    has_alpha_grad,
    has_bias,
    has_eps_grad,
    compute_mode,
    distance_floor,
    precision,
    weight_normalized,
    res,
    g,
):
    x, kernel, alpha, bias, eps = res

    def forward(x, kernel, alpha, bias, eps):
        return _yat_value(
            x,
            kernel,
            alpha,
            bias,
            eps,
            has_bias,
            compute_mode,
            distance_floor,
            precision,
            False,
            weight_normalized,
        )

    _, pullback = jax.vjp(forward, x, kernel, alpha, bias, eps)
    g_x, g_kernel, g_alpha, g_bias, g_eps = pullback(g)
    if not has_alpha_grad:
        g_alpha = jnp.zeros_like(alpha)
    if not has_bias:
        g_bias = jnp.zeros_like(bias)
    if not has_eps_grad:
        g_eps = jnp.zeros_like(eps)
    return g_x, g_kernel, g_alpha, g_bias, g_eps


_fused_yat.defvjp(_fused_yat_fwd, _fused_yat_bwd)
