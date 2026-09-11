"""Clean-backend smoke tests for the documented MLX example model."""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

mx = pytest.importorskip("mlx.core")


def test_mnist_model_imports_and_runs_without_torchvision(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] == "torchvision":
            raise AssertionError(f"example dependency imported eagerly: {name}")
        return original(name, *args, **kwargs)

    sys.modules.pop("nmn.mlx.examples.mnist", None)
    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.import_module("nmn.mlx.examples.mnist")
    model = module.YatMLP(hidden1=8, hidden2=4, num_classes=3)
    output = model(mx.ones((2, 28, 28, 1)))
    mx.eval(output)
    assert output.shape == (2, 3)


def test_mnist_loader_reports_exact_example_install(monkeypatch):
    module = importlib.import_module("nmn.mlx.examples.mnist")
    monkeypatch.setattr(module.datasets, "_module", None)

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"nmn\[mlx,examples\]"):
        module.load_mnist("unused")
