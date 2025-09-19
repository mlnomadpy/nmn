"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def dummy_input_2d():
    """Create a dummy 2D input for testing dense layers."""
    return np.random.randn(4, 8).astype(np.float32)


@pytest.fixture  
def dummy_input_4d():
    """Create a dummy 4D input for testing convolutional layers."""
    return np.random.randn(2, 32, 32, 3).astype(np.float32)


@pytest.fixture
def small_conv_input():
    """Create a small 4D input for fast convolutional tests."""
    return np.random.randn(1, 8, 8, 3).astype(np.float32)


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42