"""Lifecycle and model-ownership contracts for PyTorch tied kernel banks."""

from __future__ import annotations

import copy
import gc
import io
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from nmn.torch import KernelBank, YatConv1D, YatConv2D, YatConv3D, YatNMN

ROOT = Path(__file__).resolve().parents[2]


class _TiedModel(nn.Module):
    def __init__(self, bank: KernelBank, *, dtype=torch.float32) -> None:
        super().__init__()
        common = {
            "tie_kernel_bank": True,
            "kernel_bank": bank,
            "kernel_bank_id": "model-owned",
            "bias": False,
            "alpha": False,
            "param_dtype": dtype,
        }
        self.first = YatNMN(3, 2, **common)
        self.second = YatNMN(3, 2, **common)

    def forward(self, inputs):
        return self.first(inputs) + self.second(inputs)


class _ConstantInitializer:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, tensor):
        return nn.init.constant_(tensor, self.value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda bank_id: YatNMN(2, 2, tie_kernel_bank=True, kernel_bank_id=bank_id),
        lambda bank_id: YatConv1D(
            2, 2, 1, tie_kernel_bank=True, kernel_bank_id=bank_id
        ),
        lambda bank_id: YatConv2D(
            2, 2, 1, tie_kernel_bank=True, kernel_bank_id=bank_id
        ),
        lambda bank_id: YatConv3D(
            2, 2, 1, tie_kernel_bank=True, kernel_bank_id=bank_id
        ),
    ],
)
def test_legacy_bank_registry_releases_unowned_parameters(factory):
    layer = factory("legacy-gc")
    registry = type(layer)._KERNEL_BANKS
    parameter = weakref.ref(layer.weight)

    assert len(registry) == 1
    del layer
    gc.collect()

    assert parameter() is None
    assert len(registry) == 0


def test_thousand_transient_models_do_not_grow_legacy_registry():
    YatNMN._KERNEL_BANKS.clear()
    for index in range(1_000):
        layer = YatNMN(
            2,
            2,
            tie_kernel_bank=True,
            kernel_bank_id=f"transient-{index}",
        )
        del layer
    gc.collect()

    assert len(YatNMN._KERNEL_BANKS) == 0
    assert len(YatNMN._KERNEL_BANK_USED) == 0


def test_thousand_legacy_deepcopies_do_not_grow_used_registry():
    YatNMN._KERNEL_BANKS.clear()
    YatNMN._KERNEL_BANK_USED.clear()
    original = YatNMN(2, 2, tie_kernel_bank=True, kernel_bank_id="deepcopy-gc")
    copies = [copy.deepcopy(original) for _ in range(1_000)]
    inputs = torch.ones((1, 2))
    for layer in copies:
        layer(inputs)
    del layer, copies
    gc.collect()

    assert len(YatNMN._KERNEL_BANKS) == 1
    assert len(YatNMN._KERNEL_BANK_USED) == 1
    del original
    gc.collect()
    assert len(YatNMN._KERNEL_BANKS) == 0
    assert len(YatNMN._KERNEL_BANK_USED) == 0


def test_live_legacy_id_sharing_remains_a_compatibility_path():
    first = YatNMN(2, 2, tie_kernel_bank=True, kernel_bank_id="legacy-sharing")
    second = YatNMN(2, 2, tie_kernel_bank=True, kernel_bank_id="legacy-sharing")

    assert first.weight is second.weight
    assert first.kernel_bank is second.kernel_bank is None


@pytest.mark.parametrize(
    "factory",
    [
        lambda bank: YatNMN(2, 2, tie_kernel_bank=True, kernel_bank=bank),
        lambda bank: YatConv1D(2, 2, 1, tie_kernel_bank=True, kernel_bank=bank),
        lambda bank: YatConv2D(2, 2, 1, tie_kernel_bank=True, kernel_bank=bank),
        lambda bank: YatConv3D(2, 2, 1, tie_kernel_bank=True, kernel_bank=bank),
    ],
)
def test_explicit_bank_shares_only_within_the_same_owner(factory):
    first_bank = KernelBank()
    second_bank = KernelBank()
    first = factory(first_bank)
    peer = factory(first_bank)
    isolated = factory(second_bank)

    assert first.weight is peer.weight
    assert isolated.weight is not first.weight


def test_callable_initializer_instances_keep_distinct_bank_signatures():
    bank = KernelBank()
    first = YatNMN(
        2,
        2,
        tie_kernel_bank=True,
        kernel_bank=bank,
        kernel_init=_ConstantInitializer(1.0),
    )
    second = YatNMN(
        2,
        2,
        tie_kernel_bank=True,
        kernel_bank=bank,
        kernel_init=_ConstantInitializer(2.0),
    )

    assert first.weight is not second.weight
    assert len(bank) == 2
    torch.testing.assert_close(first.weight, torch.ones_like(first.weight))
    torch.testing.assert_close(second.weight, torch.full_like(second.weight, 2.0))


def test_explicit_bank_owns_parameter_until_owner_is_released():
    bank = KernelBank()
    first = _TiedModel(bank)
    second = _TiedModel(bank)
    parameter = weakref.ref(first.first.weight)

    assert first.first.weight is first.second.weight
    assert first.first.weight is second.first.weight
    del first, second
    gc.collect()
    assert parameter() is not None

    del bank
    gc.collect()
    assert parameter() is None


def test_separate_models_load_without_cross_mutation_and_preserve_sharing():
    first = _TiedModel(KernelBank())
    second = _TiedModel(KernelBank())
    assert first.first.weight is first.second.weight
    assert second.first.weight is second.second.weight
    assert first.first.weight is not second.first.weight

    first_before = first.first.weight.detach().clone()
    state = {name: value + 1 for name, value in first.state_dict().items()}
    second.load_state_dict(state)

    torch.testing.assert_close(first.first.weight, first_before)
    assert first.first.weight is not second.first.weight
    torch.testing.assert_close(second.first.weight, state["first.weight"])


def test_optimizer_deduplicates_shared_parameter_and_round_trip_isolated():
    model = _TiedModel(KernelBank(), dtype=torch.float64)
    parameters = list(model.parameters())
    assert parameters == [model.first.weight]
    assert model.first.weight is model.second.weight
    assert model.first.weight.dtype == torch.float64
    assert model.first.weight.device.type == "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.ones((2, 3), dtype=torch.float64)).sum().backward()
    optimizer.step()
    assert list(optimizer.state) == [model.first.weight]

    cloned = copy.deepcopy(model)
    assert cloned.first.weight is cloned.second.weight
    assert cloned.first.weight is not model.first.weight
    assert cloned.first.weight is next(
        iter(cloned.first.kernel_bank._parameters.values())
    )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = _TiedModel(KernelBank(), dtype=torch.float64)
    restored.load_state_dict(torch.load(buffer, weights_only=True))

    assert restored.first.weight is restored.second.weight
    assert restored.first.weight is not model.first.weight
    torch.testing.assert_close(restored.first.weight, model.first.weight)


def test_restored_owner_accepts_compatible_consumer_in_fresh_process(tmp_path):
    bank = KernelBank()
    layer = YatNMN(3, 2, tie_kernel_bank=True, kernel_bank=bank)
    copied_bank = copy.deepcopy(bank)
    wider = YatNMN(3, 3, tie_kernel_bank=True, kernel_bank=copied_bank)
    assert len(copied_bank) == 1
    assert wider.weight.shape == (3, 3)
    path = tmp_path / "kernel-bank.pt"
    torch.save(bank, path)

    program = """
import sys
import torch
from nmn.torch import YatNMN

bank = torch.load(sys.argv[1], weights_only=False)
original = next(iter(bank._parameters.values()))
layer = YatNMN(3, 3, tie_kernel_bank=True, kernel_bank=bank)
assert len(bank) == 1
assert layer.weight is original
assert layer.weight.shape == (3, 3)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "-c", program, str(path)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert len(bank) == 1
    assert layer.weight is next(iter(bank._parameters.values()))
