"""Integration tests for cross-framework compatibility."""

import pytest
import numpy as np
import os


def test_package_import():
    """Test that the main package can be imported."""
    import nmn
    assert hasattr(nmn, '__version__')


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
    
    # Find pyproject.toml relative to this file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(test_dir))
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    
    # Read version from pyproject.toml
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    # Extract version from pyproject.toml
    import re
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    assert match is not None, "Could not find version in pyproject.toml"
    pyproject_version = match.group(1)
    
    # Check package version matches pyproject.toml
    assert nmn.__version__ == pyproject_version, \
        f"Package version {nmn.__version__} doesn't match pyproject.toml version {pyproject_version}"


@pytest.mark.parametrize("input_shape,expected_2d", [
    ((4, 8), True),
    ((2, 32, 32, 3), False),
    ((1, 28, 28, 1), False),
])
def test_input_shape_validation(input_shape, expected_2d):
    """Test input shape validation logic."""
    is_2d = len(input_shape) == 2
    assert is_2d == expected_2d