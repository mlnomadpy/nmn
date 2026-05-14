"""Test config for the MLX backend.

Pins the default device to CPU for parity tests: Apple Silicon GPU
matmuls accumulate at a lower precision than numpy CPU fp32, so the
< 1e-5 element-wise checks against the numpy reference are only
meaningful on the CPU device. End-users on GPU should expect ~1e-3
drift — that is a property of Metal's matmul kernel, not the layer.
"""
from __future__ import annotations

import pytest

mlx_core = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _force_cpu():
    """Pin every MLX test to the CPU device for deterministic parity."""
    prev = mlx_core.default_device()
    mlx_core.set_default_device(mlx_core.cpu)
    yield
    mlx_core.set_default_device(prev)
