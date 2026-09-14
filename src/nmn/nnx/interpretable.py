"""Strict, trainable NNX kernel expansions with observable native edits.

This research path uses direct squared differences in the parameter dtype,
not the general YatNMN layer's expanded-distance FP32 execution policy.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from .layers.nmn import YatNMN


class YatExpansion(nnx.Module):
    """Fixed positive epsilon, unbiased kernel sections and signed coefficients."""

    def __init__(
        self,
        in_features,
        out_features,
        num_centers=1,
        *,
        epsilon=1.0,
        dtype=jnp.float32,
        rngs,
    ):
        for size in (in_features, out_features, num_centers):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError("dimensions must be positive integers")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        dtype = jnp.dtype(dtype)
        if dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
            raise ValueError("research execution requires float32 or float64")
        if dtype == jnp.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError("enable JAX x64 before constructing a float64 model")
        if not 0 < float(jnp.asarray(epsilon, dtype=dtype)) < math.inf:
            raise ValueError("epsilon must remain finite and positive in model dtype")
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.epsilon = float(epsilon)
        self.kernel = YatNMN(
            in_features,
            num_centers,
            use_bias=False,
            use_alpha=False,
            epsilon=epsilon,
            param_dtype=dtype,
            dtype=dtype,
            rngs=rngs,
        )
        self.coefficients = nnx.Param(
            jnp.full((out_features, num_centers), 1.0 / num_centers, dtype=dtype)
        )

    @property
    def centers(self):
        """Read-only array view; mutate kernel.kernel[...] to update parameters."""
        return self.kernel.kernel[...].T

    def features(self, x):
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("input has incorrect feature dimension")
        if x.dtype != self.centers.dtype:
            raise ValueError("input dtype must equal model dtype")
        distance = jnp.sum((x[..., None, :] - self.centers) ** 2, axis=-1)
        return (x @ self.centers.T) ** 2 / (distance + self.epsilon)

    def contributions(self, x):
        return self.features(x)[..., None, :] * self.coefficients[...]

    def __call__(self, x):
        return jnp.sum(self.contributions(x), axis=-1)


class ThreeNeuronYat(nnx.Module):
    """h=H(u), p=P(v), y=Y(h,v); outputs are (target, protected).

    Controls map state names to dictionaries with gate and/or replacement.
    Replacement wins; controls must broadcast without expanding output shape.
    Trace values remain live JAX arrays suitable for JIT and autodiff.
    """

    state_names = ("h", "p", "y")
    output_names = ("target", "protected")

    def __init__(self, num_centers=1, *, epsilon=1.0, dtype=jnp.float32, rngs):
        options = dict(num_centers=num_centers, epsilon=epsilon, dtype=dtype, rngs=rngs)
        self.h = YatExpansion(1, 1, **options)
        self.p = YatExpansion(1, 1, **options)
        self.y = YatExpansion(2, 1, **options)

    @classmethod
    def reference(cls, *, dtype=jnp.float32):
        model = cls(dtype=dtype, rngs=nnx.Rngs(0))
        for name in model.state_names:
            module = getattr(model, name)
            module.kernel.kernel[...] = jnp.ones_like(module.kernel.kernel[...])
            module.coefficients[...] = jnp.ones_like(module.coefficients[...])
        return model

    def forward_with_trace(self, x, interventions=None):
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != 2:
            raise ValueError("expected (..., 2) inputs")
        controls = {} if interventions is None else interventions
        if set(controls) - set(self.state_names):
            raise ValueError("unknown intervention state")
        trace = {}

        def evaluate(name, inputs):
            control = controls.get(name, {})
            if not isinstance(control, dict) or set(control) - {"gate", "replacement"}:
                raise ValueError("control must contain only gate/replacement")
            parts = getattr(self, name).contributions(inputs)
            raw = jnp.sum(parts, axis=-1)
            replacement = control.get("replacement")
            value = control.get("gate", 1.0) if replacement is None else replacement
            value = jnp.broadcast_to(jnp.asarray(value, dtype=raw.dtype), raw.shape)
            effective = raw * value if replacement is None else value
            trace.update(
                {
                    name + ".input": inputs,
                    name + ".contributions": parts,
                    name + ".raw": raw,
                    name: effective,
                }
            )
            return effective

        h = evaluate("h", x[..., :1])
        p = evaluate("p", x[..., 1:])
        y = evaluate("y", jnp.concatenate((h, x[..., 1:]), axis=-1))
        return jnp.concatenate((y, p), axis=-1), trace

    def __call__(self, x, interventions=None):
        return self.forward_with_trace(x, interventions)[0]
