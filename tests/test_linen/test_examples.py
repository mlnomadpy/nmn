"""Clean-backend smoke tests for the documented Linen example model."""

from __future__ import annotations

import builtins
import importlib
import sys

import jax
import jax.numpy as jnp
import pytest


def _block_example_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"optax", "torchvision"}:
            raise AssertionError(f"example dependency imported eagerly: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_mnist_model_imports_and_runs_without_example_dependencies(monkeypatch):
    sys.modules.pop("nmn.linen.examples.mnist", None)
    _block_example_imports(monkeypatch)
    module = importlib.import_module("nmn.linen.examples.mnist")
    model = module.YatMLP(hidden1=8, hidden2=4, num_classes=3)
    inputs = jnp.ones((2, 28, 28, 1))
    variables = model.init(jax.random.key(0), inputs)
    assert model.apply(variables, inputs).shape == (2, 3)


def test_mnist_loader_reports_exact_example_install(monkeypatch):
    module = importlib.import_module("nmn.linen.examples.mnist")
    monkeypatch.setattr(module.datasets, "_module", None)

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"nmn\[linen,examples\]"):
        module.load_mnist("unused")
