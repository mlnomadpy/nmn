"""Clean-backend smoke tests for the documented NNX example model."""

from __future__ import annotations

import builtins
import importlib
import sys

import jax.numpy as jnp
import pytest
from flax import nnx


def _block_example_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "datasets",
            "grain",
            "mteb",
            "optax",
            "orbax",
            "tokenizers",
            "torchvision",
            "wandb",
        }:
            raise AssertionError(f"example dependency imported eagerly: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_mnist_model_imports_and_runs_without_example_dependencies(monkeypatch):
    sys.modules.pop("nmn.nnx.examples.vision.mnist", None)
    _block_example_imports(monkeypatch)
    module = importlib.import_module("nmn.nnx.examples.vision.mnist")
    model = module.YatMLP(rngs=nnx.Rngs(0), hidden1=8, hidden2=4, num_classes=3)
    assert model(jnp.ones((2, 28, 28, 1))).shape == (2, 3)


def test_mnist_loader_reports_exact_example_install(monkeypatch):
    module = importlib.import_module("nmn.nnx.examples.vision.mnist")
    monkeypatch.setattr(module.datasets, "_module", None)

    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=r"nmn\[nnx,examples\]"):
        module.load_mnist("unused")


@pytest.mark.parametrize(
    ("module_name", "factory"),
    [
        (
            "nmn.nnx.examples.language.m3za",
            lambda module: module.MiniBERT(
                maxlen=8,
                vocab_size=16,
                embed_dim=8,
                num_heads=2,
                feed_forward_dim=12,
                num_transformer_blocks=1,
                rngs=nnx.Rngs(0),
            ),
        ),
        (
            "nmn.nnx.examples.language.m3za_perf",
            lambda module: module.MiniBERT(
                maxlen=8,
                vocab_size=16,
                embed_dim=8,
                num_heads=2,
                feed_forward_dim=12,
                num_transformer_blocks=1,
                rngs=nnx.Rngs(0),
            ),
        ),
        (
            "nmn.nnx.examples.vision.aether_resnet50_tpu",
            lambda module: module.ResNet(
                module.BasicBlock,
                [1, 1, 1, 1],
                num_classes=3,
                rngs=nnx.Rngs(0),
            ),
        ),
    ],
)
def test_advanced_models_import_without_example_dependencies(
    monkeypatch, module_name, factory
):
    sys.modules.pop(module_name, None)
    _block_example_imports(monkeypatch)
    module = importlib.import_module(module_name)
    assert factory(module) is not None
