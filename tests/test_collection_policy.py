"""Regression tests for optional-backend collection isolation."""

from pathlib import Path

import tests.conftest as shared_config


def test_mlx_availability_uses_native_safe_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(shared_config, "_module_available", lambda name: name == "mlx")
    monkeypatch.setattr(
        shared_config, "mlx_is_usable", lambda: calls.append("probe") or False
    )
    shared_config._mlx_backend_available.cache_clear()

    assert shared_config._mlx_backend_available() is False
    assert shared_config._mlx_backend_available() is False
    assert calls == ["probe"]

    shared_config._mlx_backend_available.cache_clear()


def test_unusable_mlx_tree_is_ignored_before_import(monkeypatch):
    monkeypatch.setattr(shared_config, "_mlx_backend_available", lambda: False)
    path = Path("/checkout/tests/test_mlx/test_all_layers.py")

    assert shared_config.pytest_ignore_collect(path, object()) is True


def test_available_backend_tree_is_collected(monkeypatch):
    monkeypatch.setattr(shared_config, "_module_available", lambda name: name == "torch")
    path = Path("/checkout/tests/test_torch/test_yat_nmn.py")

    assert shared_config.pytest_ignore_collect(path, object()) is False


def test_missing_keras_runtime_is_ignored(monkeypatch):
    monkeypatch.setattr(shared_config, "_keras_backend_available", lambda: False)
    path = Path("/checkout/tests/test_keras/test_attention.py")

    assert shared_config.pytest_ignore_collect(path, object()) is True
