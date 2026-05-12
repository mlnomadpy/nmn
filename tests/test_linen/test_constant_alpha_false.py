"""Regression tests for issue #3 (Flax Linen backend): ``constant_alpha=False``
and ``constant_bias=False`` must not be silently coerced to ``0.0``.

Linen uses dataclass field access (``self.constant_alpha``) rather than a
local variable, which is why the initial sweep missed it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flax")

import jax
import jax.numpy as jnp
import flax.linen as nn

from nmn.linen import YatNMN


class TestConstantAlphaFalseTrap:
    def test_false_does_not_zero_output(self) -> None:
        model = YatNMN(features=16, use_alpha=True, constant_alpha=False)
        key = jax.random.key(0)
        x = jax.random.normal(jax.random.key(1), (4, 8))
        params = model.init(key, x)
        y = model.apply(params, x)
        assert jnp.isfinite(y).all()
        assert float(jnp.linalg.norm(y)) > 0.0, (
            "output collapsed to zero — constant_alpha=False silently coerced "
            "to 0.0 (issue #3 regression)"
        )

    def test_false_keeps_learnable_alpha(self) -> None:
        model = YatNMN(features=16, use_alpha=True, constant_alpha=False)
        key = jax.random.key(0)
        x = jnp.ones((4, 8))
        params = model.init(key, x)
        # Learnable alpha is registered as a param with key 'alpha'
        flat = jax.tree_util.tree_leaves_with_path(params)
        keys = {"/".join(str(k).strip("'") for k in path) for path, _ in flat}
        assert any("alpha" in k for k in keys), f"alpha param missing: {keys}"


class TestConstantBiasFalseTrap:
    def test_false_does_not_zero_output(self) -> None:
        model = YatNMN(features=16, use_bias=True, constant_bias=False)
        key = jax.random.key(0)
        x = jax.random.normal(jax.random.key(1), (4, 8))
        params = model.init(key, x)
        y = model.apply(params, x)
        assert jnp.isfinite(y).all()
        assert float(jnp.linalg.norm(y)) > 0.0
