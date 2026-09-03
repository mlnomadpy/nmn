"""Regressions for modern Flax NNX variable mutation semantics."""

from __future__ import annotations

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx, serialization

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


def _bank_layer(kind, out_features, bank_id, *, bank_size=None, seed=0):
    if kind == "dense":
        return YatNMN(
            2,
            out_features,
            tie_kernel_bank=True,
            kernel_bank_size=bank_size,
            kernel_bank_id=bank_id,
            rngs=nnx.Rngs(seed),
        )
    return YatConv(
        2,
        out_features,
        1,
        tie_kernel_bank=True,
        kernel_bank_size=bank_size,
        kernel_bank_id=bank_id,
        rngs=nnx.Rngs(seed),
    )


def _bank_inputs(kind):
    shape = (3, 2) if kind == "dense" else (3, 5, 2)
    return jnp.ones(shape, dtype=jnp.float32)


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_shared_bank_capacity_is_fixed_and_can_be_preallocated(kind):
    bank_id = f"issue-140-capacity-{kind}"
    narrow = _bank_layer(kind, 2, bank_id, bank_size=4)
    wide = _bank_layer(kind, 4, bank_id, seed=1)

    assert narrow.kernel is wide.kernel
    assert wide.kernel[...].shape[-1] == 4
    assert narrow(_bank_inputs(kind)).shape[-1] == 2
    assert wide(_bank_inputs(kind)).shape[-1] == 4


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_shared_bank_rejects_expansion_without_mutating_live_state(kind):
    bank_id = f"issue-140-optimizer-{kind}"
    layer = _bank_layer(kind, 2, bank_id)
    inputs = _bank_inputs(kind)
    optimizer = nnx.Optimizer(layer, optax.adam(1e-3), wrt=nnx.Param)

    def loss_fn(model):
        return jnp.sum(model(inputs))

    _, grads = nnx.value_and_grad(loss_fn)(layer)
    optimizer.update(layer, grads)
    kernel_before = np.asarray(layer.kernel[...]).copy()
    grad_shape = grads.kernel[...].shape

    with pytest.raises(ValueError, match="fixed capacity 2; requested 3"):
        _bank_layer(kind, 3, bank_id, seed=1)

    assert layer.kernel[...].shape == kernel_before.shape
    assert grads.kernel[...].shape == grad_shape
    np.testing.assert_array_equal(np.asarray(layer.kernel[...]), kernel_before)

    # Existing optimizer moments remain compatible after the rejected request.
    _, next_grads = nnx.value_and_grad(loss_fn)(layer)
    optimizer.update(layer, next_grads)
    assert layer.kernel[...].shape == kernel_before.shape


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_shared_bank_rejection_is_jit_and_serialization_safe(kind):
    bank_id = f"issue-140-jit-serialization-{kind}"
    layer = _bank_layer(kind, 2, bank_id)
    inputs = _bank_inputs(kind)
    optimizer = nnx.Optimizer(layer, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt):
        _, grads = nnx.value_and_grad(lambda m: jnp.sum(m(inputs)))(model)
        opt.update(model, grads)

    train_step(layer, optimizer)
    state = nnx.state(layer)
    pure_state = nnx.to_pure_dict(state)
    encoded = serialization.to_bytes(pure_state)

    with pytest.raises(ValueError, match="fixed capacity"):
        _bank_layer(kind, 3, bank_id, seed=1)

    restored = serialization.from_bytes(pure_state, encoded)
    assert restored["kernel"].shape == layer.kernel[...].shape
    train_step(layer, optimizer)


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_shared_bank_rejects_concurrent_expansion_atomically(kind):
    bank_id = f"issue-140-concurrent-{kind}"
    layer = _bank_layer(kind, 2, bank_id)
    kernel_before = np.asarray(layer.kernel[...]).copy()

    def construct_wider(seed):
        with pytest.raises(ValueError, match="fixed capacity"):
            _bank_layer(kind, 3, bank_id, seed=seed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(construct_wider, range(1, 9)))

    assert layer.kernel[...].shape[-1] == 2
    np.testing.assert_array_equal(np.asarray(layer.kernel[...]), kernel_before)


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_kernel_bank_size_cannot_be_smaller_than_consumer(kind):
    with pytest.raises(ValueError, match="must be at least out_features"):
        _bank_layer(
            kind,
            3,
            f"issue-140-invalid-capacity-{kind}",
            bank_size=2,
        )


def test_no_direct_nnx_variable_value_assignments_remain():
    pattern = re.compile(r"(?:self\.(?:kernel|embedding)|shared_kernel)\.value\s*=")
    paths = (
        ROOT / "src/nmn/nnx/layers/nmn.py",
        ROOT / "src/nmn/nnx/layers/conv/yat_conv.py",
        ROOT / "src/nmn/nnx/layers/embed.py",
    )
    for path in paths:
        assert pattern.search(path.read_text(encoding="utf-8")) is None
