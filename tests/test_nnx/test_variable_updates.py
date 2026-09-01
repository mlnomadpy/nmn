"""Regressions for modern Flax NNX variable mutation semantics."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx import Embed, YatConv, YatNMN

ROOT = Path(__file__).resolve().parents[2]
DEPRECATED_SETTER = "'.value' setter is now deprecated"


def _without_deprecated_setter_warning(factory):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = factory()

    deprecated = [
        warning for warning in caught if DEPRECATED_SETTER in str(warning.message)
    ]
    assert not deprecated
    return result


@pytest.mark.parametrize("kind", ["dense", "conv", "embed"])
def test_weight_normalization_uses_same_shape_variable_update(kind):
    if kind == "dense":
        layer = _without_deprecated_setter_warning(
            lambda: YatNMN(3, 4, weight_normalized=True, rngs=nnx.Rngs(0))
        )
        values = np.asarray(layer.kernel[...])
        norms = np.linalg.norm(values, axis=0)
    elif kind == "conv":
        layer = _without_deprecated_setter_warning(
            lambda: YatConv(2, 4, 3, weight_normalized=True, rngs=nnx.Rngs(0))
        )
        values = np.asarray(layer.kernel[...])
        norms = np.linalg.norm(values, axis=(0, 1))
    else:
        layer = _without_deprecated_setter_warning(
            lambda: Embed(5, 3, weight_normalized=True, rngs=nnx.Rngs(0))
        )
        values = np.asarray(layer.embedding[...])
        norms = np.linalg.norm(values, axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_shared_bank_shape_expansion_uses_replacement_update(kind):
    bank_id = f"issue-111-{kind}"
    if kind == "dense":
        narrow = YatNMN(
            2,
            2,
            tie_kernel_bank=True,
            kernel_bank_id=bank_id,
            rngs=nnx.Rngs(0),
        )
        old_values = np.asarray(narrow.kernel[...]).copy()
        wide = _without_deprecated_setter_warning(
            lambda: YatNMN(
                2,
                4,
                tie_kernel_bank=True,
                kernel_bank_id=bank_id,
                rngs=nnx.Rngs(1),
            )
        )
        inputs = jnp.ones((3, 2), dtype=jnp.float32)
    else:
        narrow = YatConv(
            2,
            2,
            1,
            tie_kernel_bank=True,
            kernel_bank_id=bank_id,
            rngs=nnx.Rngs(0),
        )
        old_values = np.asarray(narrow.kernel[...]).copy()
        wide = _without_deprecated_setter_warning(
            lambda: YatConv(
                2,
                4,
                1,
                tie_kernel_bank=True,
                kernel_bank_id=bank_id,
                rngs=nnx.Rngs(1),
            )
        )
        inputs = jnp.ones((3, 5, 2), dtype=jnp.float32)

    assert narrow.kernel is wide.kernel
    assert wide.kernel[...].shape[-1] == 4
    np.testing.assert_array_equal(np.asarray(wide.kernel[..., :2]), old_values)

    grads = nnx.grad(lambda module, x: jnp.sum(module(x)))(wide, inputs)
    assert grads.kernel[...].shape == wide.kernel[...].shape
    assert np.isfinite(np.asarray(grads.kernel[...])).all()


def test_no_direct_nnx_variable_value_assignments_remain():
    pattern = re.compile(r"(?:self\.(?:kernel|embedding)|shared_kernel)\.value\s*=")
    paths = (
        ROOT / "src/nmn/nnx/layers/nmn.py",
        ROOT / "src/nmn/nnx/layers/conv/yat_conv.py",
        ROOT / "src/nmn/nnx/layers/embed.py",
    )
    for path in paths:
        assert pattern.search(path.read_text(encoding="utf-8")) is None
