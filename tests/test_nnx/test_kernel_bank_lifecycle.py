"""Lifecycle and model-ownership contracts for Flax NNX tied kernel banks."""

from __future__ import annotations

import copy
import gc
import weakref

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx, serialization

from nmn.nnx import KernelBank, YatConv, YatNMN


class _TiedModel(nnx.Module):
    def __init__(self, bank: KernelBank, *, dtype=jnp.float32) -> None:
        common = {
            "tie_kernel_bank": True,
            "kernel_bank": bank,
            "kernel_bank_id": "model-owned",
            "use_bias": False,
            "use_alpha": False,
            "param_dtype": dtype,
        }
        self.first = YatNMN(3, 2, rngs=nnx.Rngs(0), **common)
        self.second = YatNMN(3, 2, rngs=nnx.Rngs(1), **common)

    def __call__(self, inputs):
        return self.first(inputs) + self.second(inputs)


@pytest.mark.parametrize(
    ("layer_type", "factory"),
    [
        (
            YatNMN,
            lambda bank_id: YatNMN(
                2,
                2,
                tie_kernel_bank=True,
                kernel_bank_id=bank_id,
                rngs=nnx.Rngs(0),
            ),
        ),
        (
            YatConv,
            lambda bank_id: YatConv(
                2,
                2,
                1,
                tie_kernel_bank=True,
                kernel_bank_id=bank_id,
                rngs=nnx.Rngs(0),
            ),
        ),
    ],
)
def test_legacy_bank_registry_releases_unowned_parameters(layer_type, factory):
    layer = factory("legacy-gc")
    parameter = weakref.ref(layer.kernel)

    assert len(layer_type._KERNEL_BANKS) == 1
    del layer
    gc.collect()

    assert parameter() is None
    assert len(layer_type._KERNEL_BANKS) == 0


def test_thousand_transient_models_do_not_grow_legacy_registry():
    YatNMN._KERNEL_BANKS.clear()
    for index in range(1_000):
        layer = YatNMN(
            2,
            2,
            tie_kernel_bank=True,
            kernel_bank_id=f"transient-{index}",
            rngs=nnx.Rngs(index),
        )
        del layer
    gc.collect()

    assert len(YatNMN._KERNEL_BANKS) == 0


def test_live_legacy_id_sharing_remains_a_compatibility_path():
    first = YatNMN(
        2,
        2,
        tie_kernel_bank=True,
        kernel_bank_id="legacy-sharing",
        rngs=nnx.Rngs(0),
    )
    second = YatNMN(
        2,
        2,
        tie_kernel_bank=True,
        kernel_bank_id="legacy-sharing",
        rngs=nnx.Rngs(1),
    )

    assert first.kernel is second.kernel
    assert first.kernel_bank is second.kernel_bank is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda bank, seed: YatNMN(
            2,
            2,
            tie_kernel_bank=True,
            kernel_bank=bank,
            rngs=nnx.Rngs(seed),
        ),
        lambda bank, seed: YatConv(
            2,
            2,
            1,
            tie_kernel_bank=True,
            kernel_bank=bank,
            rngs=nnx.Rngs(seed),
        ),
    ],
)
def test_explicit_bank_shares_only_within_the_same_owner(factory):
    first_bank = KernelBank()
    second_bank = KernelBank()
    first = factory(first_bank, 0)
    peer = factory(first_bank, 1)
    isolated = factory(second_bank, 2)

    assert first.kernel is peer.kernel
    assert isolated.kernel is not first.kernel


def test_explicit_bank_owns_parameter_until_owner_is_released():
    bank = KernelBank()
    first = _TiedModel(bank)
    second = _TiedModel(bank)
    parameter = weakref.ref(first.first.kernel)

    assert first.first.kernel is first.second.kernel
    assert first.first.kernel is second.first.kernel
    del first, second
    gc.collect()
    assert parameter() is not None

    del bank
    gc.collect()
    assert parameter() is None


def test_separate_models_restore_without_cross_mutation_and_preserve_sharing():
    first = _TiedModel(KernelBank())
    second = _TiedModel(KernelBank())
    assert first.first.kernel is first.second.kernel
    assert second.first.kernel is second.second.kernel
    assert first.first.kernel is not second.first.kernel

    first_before = np.asarray(first.first.kernel[...]).copy()
    source = nnx.to_pure_dict(nnx.state(first, nnx.Param))
    source["first"]["kernel"] = source["first"]["kernel"] + 1
    encoded = serialization.to_bytes(source)
    target = nnx.to_pure_dict(nnx.state(second, nnx.Param))
    restored = serialization.from_bytes(target, encoded)
    nnx.update(second, nnx.State(restored))

    np.testing.assert_array_equal(np.asarray(first.first.kernel[...]), first_before)
    assert first.first.kernel is not second.first.kernel
    assert second.first.kernel is second.second.kernel
    np.testing.assert_allclose(
        np.asarray(second.first.kernel[...]), source["first"]["kernel"]
    )

    cloned = copy.deepcopy(first)
    assert cloned.first.kernel is cloned.second.kernel
    assert cloned.first.kernel is not first.first.kernel
    assert cloned.first.kernel is cloned.first.kernel_bank._entries[0].parameter


def test_optimizer_deduplicates_shared_parameter_and_jit_preserves_dtype():
    model = _TiedModel(KernelBank(), dtype=jnp.float16)
    state = nnx.state(model, nnx.Param)
    assert set(state) == {"first"}
    assert model.first.kernel is model.second.kernel
    assert model.first.kernel[...].dtype == jnp.float16

    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    @nnx.jit
    def step(current_model, current_optimizer, inputs):
        _, grads = nnx.value_and_grad(lambda candidate: jnp.sum(candidate(inputs)))(
            current_model
        )
        current_optimizer.update(current_model, grads)

    step(model, optimizer, jnp.ones((2, 3), dtype=jnp.float16))
    assert model.first.kernel is model.second.kernel
    assert model.first.kernel[...].dtype == jnp.float16
