"""Shared test fixtures and optional-backend collection policy."""

import importlib.util
import os
from functools import lru_cache

import numpy as np
import pytest

from tests._isolated_backend import mlx_is_usable

_OPTIONAL_TEST_DEPENDENCIES = {
    "test_torch": ("torch",),
    "test_nnx": ("jax", "flax"),
    "test_linen": ("jax", "flax"),
    "test_tf": ("tensorflow",),
    "scripts": ("jax", "flax"),
    "benchmarks": ("jax", "flax"),
}


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _keras_backend_available():
    if not _module_available("keras"):
        return False
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    backend_module = {"tensorflow": "tensorflow", "torch": "torch", "jax": "jax"}.get(
        backend, backend
    )
    return _module_available(backend_module)


@lru_cache(maxsize=1)
def _mlx_backend_available():
    """Check MLX in a child because native Metal initialization can abort."""
    return _module_available("mlx") and mlx_is_usable()


def pytest_ignore_collect(collection_path, config):
    """Skip an optional backend's tests before importing that backend.

    Individual test modules may still use ``pytest.importorskip`` when useful,
    but collection safety is enforced once here so a plain ``pytest`` works in
    a base NMN installation with no ML framework installed.
    """
    del config
    parts = collection_path.parts
    try:
        test_root = parts[parts.index("tests") + 1]
    except (ValueError, IndexError):
        return False

    if test_root == "test_keras":
        return not _keras_backend_available()
    if test_root == "test_mlx":
        return not _mlx_backend_available()

    required_modules = _OPTIONAL_TEST_DEPENDENCIES.get(test_root)
    if required_modules is None:
        return False
    return not all(_module_available(module) for module in required_modules)


# ============================================================================
# Common Test Data Generators
# ============================================================================


def generate_random_input(shape, seed=42):
    """Generate deterministic random input for testing."""
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


def generate_batch_input_1d(batch_size=2, length=16, channels=3, seed=42):
    """Generate 1D input: (batch, length, channels)."""
    return generate_random_input((batch_size, length, channels), seed)


def generate_batch_input_2d(batch_size=2, height=8, width=8, channels=3, seed=42):
    """Generate 2D input: (batch, height, width, channels)."""
    return generate_random_input((batch_size, height, width, channels), seed)


def generate_batch_input_3d(
    batch_size=2, depth=4, height=4, width=4, channels=3, seed=42
):
    """Generate 3D input: (batch, depth, height, width, channels)."""
    return generate_random_input((batch_size, depth, height, width, channels), seed)


def generate_dense_input(batch_size=4, features=8, seed=42):
    """Generate dense input: (batch, features)."""
    return generate_random_input((batch_size, features), seed)


# ============================================================================
# YAT Math Reference Implementation (for validation)
# ============================================================================


def yat_reference_dense(inputs, weights, bias=None, alpha=None, epsilon=1e-6):
    """
    Reference implementation of YAT dense layer for validation.

    YAT formula: y = (dot_product)^2 / (distance_squared + epsilon)
    where distance_squared = ||input||^2 + ||weight||^2 - 2 * dot_product

    Args:
        inputs: (batch, in_features) numpy array
        weights: (in_features, out_features) numpy array
        bias: (out_features,) numpy array or None
        alpha: scalar or None
        epsilon: small constant for numerical stability

    Returns:
        output: (batch, out_features) numpy array
    """
    # Compute dot product
    dot_prod = np.matmul(inputs, weights)  # (batch, out_features)

    # Compute squared norms
    inputs_sq_sum = np.sum(inputs**2, axis=-1, keepdims=True)  # (batch, 1)
    weights_sq_sum = np.sum(weights**2, axis=0, keepdims=True)  # (1, out_features)

    # Compute squared Euclidean distance
    distance_sq = inputs_sq_sum + weights_sq_sum - 2 * dot_prod

    # YAT transformation: bias goes INSIDE the square
    if bias is not None:
        y = (dot_prod + bias) ** 2 / (distance_sq + epsilon)
    else:
        y = dot_prod**2 / (distance_sq + epsilon)

    # Apply alpha scaling (simple multiply)
    if alpha is not None:
        y = y * alpha

    return y


def yat_reference_conv2d(
    inputs, weights, bias=None, alpha=None, epsilon=1e-6, stride=1, padding=0
):
    """
    Reference implementation of YAT 2D convolution for validation.
    Uses simple sliding window approach (not optimized).

    Args:
        inputs: (batch, height, width, channels) numpy array (NHWC format)
        weights: (kernel_h, kernel_w, in_channels, out_channels) numpy array
        bias: (out_channels,) numpy array or None
        alpha: scalar or None
        epsilon: small constant
        stride: int or tuple
        padding: int or tuple

    Returns:
        output: (batch, out_h, out_w, out_channels) numpy array
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    batch, h, w, in_channels = inputs.shape
    kh, kw, _, out_channels = weights.shape

    # Apply padding
    if padding[0] > 0 or padding[1] > 0:
        inputs = np.pad(
            inputs,
            ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        h = inputs.shape[1]
        w = inputs.shape[2]

    # Output dimensions
    out_h = (h - kh) // stride[0] + 1
    out_w = (w - kw) // stride[1] + 1

    # Initialize output
    output = np.zeros((batch, out_h, out_w, out_channels), dtype=np.float32)

    # Flatten weights for distance computation
    weights_flat = weights.reshape(-1, out_channels)  # (kh*kw*in_c, out_c)
    weights_sq_sum = np.sum(weights_flat**2, axis=0)  # (out_c,)

    for b in range(batch):
        for i in range(out_h):
            for j in range(out_w):
                # Extract patch
                y_start = i * stride[0]
                x_start = j * stride[1]
                patch = inputs[b, y_start : y_start + kh, x_start : x_start + kw, :]
                patch_flat = patch.reshape(-1)  # (kh*kw*in_c,)

                # Compute dot product
                dot_prod = np.matmul(patch_flat, weights_flat)  # (out_c,)

                # Compute distance squared
                patch_sq_sum = np.sum(patch_flat**2)
                distance_sq = patch_sq_sum + weights_sq_sum - 2 * dot_prod

                # YAT transformation: bias goes INSIDE the square
                if bias is not None:
                    output[b, i, j, :] = (dot_prod + bias) ** 2 / (
                        distance_sq + epsilon
                    )
                else:
                    output[b, i, j, :] = dot_prod**2 / (distance_sq + epsilon)

    # Apply alpha scaling (simple multiply)
    if alpha is not None:
        output = output * alpha

    return output


# ============================================================================
# Test Tolerances
# ============================================================================

ATOL = 1e-4  # Absolute tolerance for floating point comparisons
RTOL = 1e-4  # Relative tolerance for floating point comparisons


# ============================================================================
# Framework Availability Checks
# ============================================================================


def is_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


def is_tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf

        return True
    except ImportError:
        return False


def is_jax_available():
    """Check if JAX is available."""
    try:
        import flax
        import jax

        return True
    except ImportError:
        return False


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture
def dense_test_data():
    """Fixture providing test data for dense layer tests."""
    return {
        "batch_size": 4,
        "in_features": 8,
        "out_features": 16,
        "input": generate_dense_input(4, 8),
    }


@pytest.fixture
def conv2d_test_data():
    """Fixture providing test data for 2D convolution tests."""
    return {
        "batch_size": 2,
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "input_size": 16,
        "input": generate_batch_input_2d(2, 16, 16, 3),
    }


@pytest.fixture
def conv1d_test_data():
    """Fixture providing test data for 1D convolution tests."""
    return {
        "batch_size": 2,
        "in_channels": 3,
        "out_channels": 8,
        "kernel_size": 3,
        "input_length": 16,
        "input": generate_batch_input_1d(2, 16, 3),
    }
