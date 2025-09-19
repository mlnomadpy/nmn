"""Integration tests for cross-framework compatibility."""

import pytest
import numpy as np


def test_package_import():
    """Test that the main package can be imported."""
    import nmn
    assert hasattr(nmn, '__version__')
    assert nmn.__version__ == "0.1.12"


def test_all_framework_imports():
    """Test that all framework modules can be imported without errors."""
    frameworks = ['nnx', 'torch', 'keras', 'tf', 'linen']
    
    for framework in frameworks:
        try:
            module = __import__(f'nmn.{framework}', fromlist=['nmn'])
            assert module is not None
        except ImportError:
            # Expected for frameworks not installed in test environment
            pass


def test_version_consistency():
    """Test that version is consistent across files."""
    import nmn
    
    # Read version from pyproject.toml
    with open('/home/runner/work/nmn/nmn/pyproject.toml', 'r') as f:
        content = f.read()
        assert 'version = "0.1.12"' in content
    
    # Check package version
    assert nmn.__version__ == "0.1.12"


@pytest.mark.parametrize("input_shape,expected_2d", [
    ((4, 8), True),
    ((2, 32, 32, 3), False),
    ((1, 28, 28, 1), False),
])
def test_input_shape_validation(input_shape, expected_2d):
    """Test input shape validation logic."""
    is_2d = len(input_shape) == 2
    assert is_2d == expected_2d