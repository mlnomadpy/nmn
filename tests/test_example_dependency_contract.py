"""Packaging and lazy-import contract for runnable examples."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from nmn._example_dependencies import lazy_example_dependency


def _extras():
    project = Path(__file__).parents[1] / "pyproject.toml"
    return tomllib.loads(project.read_text())["project"]["optional-dependencies"]


def test_core_backend_extras_exclude_example_and_data_dependencies():
    extras = _extras()
    forbidden = {
        "datasets",
        "grain",
        "matplotlib",
        "mteb",
        "optax",
        "orbax-checkpoint",
        "pillow",
        "scikit-learn",
        "seaborn",
        "tensorflow-datasets",
        "tokenizers",
        "torchvision",
        "tqdm",
        "wandb",
    }
    for extra in ("nnx", "torch", "keras", "tf", "linen", "mlx"):
        names = {
            requirement.split("[")[0].split("=")[0].split(">")[0]
            for requirement in extras[extra]
        }
        assert names.isdisjoint(forbidden), (extra, names & forbidden)


def test_data_and_example_extras_publish_the_documented_dependencies():
    extras = _extras()
    assert extras["torch"] == ["torch>=1.11.0"]
    assert {item.split(">")[0] for item in extras["data"]} == {
        "torchvision",
        "tensorflow-datasets",
        "datasets",
    }
    example_text = " ".join(extras["examples"])
    for dependency in (
        "nmn[data]",
        "optax",
        "orbax-checkpoint",
        "wandb",
        "grain",
    ):
        assert dependency in example_text


def test_lazy_dependency_does_not_import_until_first_use(monkeypatch):
    calls = []

    def missing(name):
        calls.append(name)
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing)
    dependency = lazy_example_dependency(
        "optional_data", install="nmn[torch,examples]", purpose="This example"
    )
    assert calls == []
    with pytest.raises(
        ModuleNotFoundError, match=r'pip install "nmn\[torch,examples\]"'
    ):
        dependency.load_data
    assert calls == ["optional_data"]
