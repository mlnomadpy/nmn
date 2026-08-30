"""Test config for the MLX backend.

Pins the default device to CPU for parity tests: Apple Silicon GPU
matmuls accumulate at a lower precision than numpy CPU fp32, so the
< 1e-5 element-wise checks against the numpy reference are only
meaningful on the CPU device. End-users on GPU should expect ~1e-3
drift — that is a property of Metal's matmul kernel, not the layer.
"""
from __future__ import annotations

import pytest

from tests._isolated_backend import mlx_is_usable


# A plain ``pytest.importorskip`` cannot catch a native abort.  Ignore the MLX
# modules before collection unless a child process can both import MLX and
# initialize its device runtime successfully.
_MLX_USABLE = mlx_is_usable()
collect_ignore_glob = [] if _MLX_USABLE else ["test_*.py"]

if _MLX_USABLE:
    import mlx.core as mlx_core
else:
    mlx_core = None


@pytest.fixture(autouse=True)
def _force_cpu():
    """Pin every MLX test to the CPU device for deterministic parity."""
    assert mlx_core is not None
    prev = mlx_core.default_device()
    mlx_core.set_default_device(mlx_core.cpu)
    yield
    mlx_core.set_default_device(prev)


@pytest.fixture
def mlx_gpu(_force_cpu):
    """Override the CPU fixture for tests that must execute Metal kernels."""
    assert mlx_core is not None
    previous = mlx_core.default_device()
    mlx_core.set_default_device(mlx_core.gpu)
    assert str(mlx_core.default_device()) == "Device(gpu, 0)"
    try:
        yield mlx_core.gpu
        # Catch tests that accidentally switch back to the CPU before teardown.
        assert str(mlx_core.default_device()) == "Device(gpu, 0)"
    finally:
        mlx_core.set_default_device(previous)
