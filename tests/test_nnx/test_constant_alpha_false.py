"""Regression tests for issue #3: ``constant_alpha=False`` (and
``constant_bias=False``) must not be silently coerced to ``0.0``.

False should be treated equivalently to None — i.e., the user is opting out
of the constant path and asking for the default (learnable α when
``use_alpha=True``; no bias when ``use_bias`` is also False; etc.).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("flax")

from flax import nnx

from nmn.nnx.layers import YatNMN


def _new_layer(**kwargs):
    base = dict(in_features=8, out_features=16, rngs=nnx.Rngs(0))
    base.update(kwargs)
    return YatNMN(**base)


class TestConstantAlphaFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer_false = _new_layer(use_alpha=True, constant_alpha=False)
        layer_none = _new_layer(use_alpha=True, constant_alpha=None)
        assert layer_false._constant_alpha_value is None
        assert layer_none._constant_alpha_value is None

    def test_false_keeps_learnable_alpha(self) -> None:
        layer = _new_layer(use_alpha=True, constant_alpha=False)
        assert layer.alpha is not None, "alpha should be a learnable Param"
        params = nnx.state(layer, nnx.Param)
        leaves = jax.tree_util.tree_leaves(params)
        assert leaves, "expected at least one learnable Param"

    def test_false_does_not_zero_output(self) -> None:
        layer = _new_layer(use_alpha=True, constant_alpha=False)
        x = jax.random.normal(jax.random.key(0), (4, 8))
        y = layer(x)
        assert jnp.isfinite(y).all(), "non-finite output"
        assert float(jnp.linalg.norm(y)) > 0.0, (
            "output collapsed to zero — constant_alpha=False was silently "
            "coerced to 0.0 (issue #3 regression)"
        )

    def test_zero_float_is_still_respected(self) -> None:
        """Numeric 0.0 is a legitimate user choice — keep the explicit value."""
        layer = _new_layer(use_alpha=True, constant_alpha=0.0)
        assert layer._constant_alpha_value == 0.0
        assert layer.alpha is None

    def test_true_uses_default_constant(self) -> None:
        layer = _new_layer(use_alpha=True, constant_alpha=True)
        assert layer._constant_alpha_value == pytest.approx(
            YatNMN.DEFAULT_CONSTANT_ALPHA
        )
        assert layer.alpha is None

    def test_float_is_used_verbatim(self) -> None:
        layer = _new_layer(use_alpha=True, constant_alpha=1.5)
        assert layer._constant_alpha_value == pytest.approx(1.5)
        assert layer.alpha is None


class TestConstantBiasFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(use_bias=False, constant_bias=False)
        assert layer._constant_bias_value is None

    def test_false_with_use_bias_true_keeps_learnable_bias(self) -> None:
        layer = _new_layer(use_bias=True, constant_bias=False)
        assert layer._constant_bias_value is None
        assert layer.bias is not None

    def test_zero_float_is_still_respected(self) -> None:
        layer = _new_layer(use_bias=True, constant_bias=0.0)
        assert layer._constant_bias_value == 0.0
        assert layer.bias is None
