"""Smoke tests for documented PyTorch examples."""

import builtins
import importlib
import sys

import pytest
import torch


def test_quick_example_basic_layers_run(capsys):
    from nmn.torch.examples.quick_example import example_1_basic_yat_layers

    example_1_basic_yat_layers()

    output = capsys.readouterr().out
    assert "Conv forward pass" in output
    assert "Linear forward pass" in output


def test_mnist_model_imports_and_runs_without_torchvision(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] == "torchvision":
            raise AssertionError(f"example dependency imported eagerly: {name}")
        return original(name, *args, **kwargs)

    sys.modules.pop("nmn.torch.examples.vision.mnist", None)
    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.import_module("nmn.torch.examples.vision.mnist")
    model = module.YatMLP(hidden1=8, hidden2=4, num_classes=3)
    assert model(torch.ones((2, 28, 28))).shape == (2, 3)


def test_mnist_loader_reports_exact_example_install(monkeypatch):
    module = importlib.import_module("nmn.torch.examples.vision.mnist")
    monkeypatch.setattr(module.datasets, "_module", None)

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"nmn\[torch,examples\]"):
        module.load_mnist("unused")


def test_resnet_models_import_without_example_dependencies(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "datasets",
            "matplotlib",
            "PIL",
            "seaborn",
            "sklearn",
            "torchvision",
            "wandb",
        }:
            raise AssertionError(f"example dependency imported eagerly: {name}")
        return original(name, *args, **kwargs)

    module_name = "nmn.torch.examples.vision.resnet_training"
    sys.modules.pop(module_name, None)
    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.import_module(module_name)
    standard = module.StandardConvNet(module.BasicStandardBlock, [1, 1, 1, 1])
    yat = module.YATConvNet(module.BasicYATBlock, [1, 1, 1, 1])
    assert standard is not None
    assert yat is not None
