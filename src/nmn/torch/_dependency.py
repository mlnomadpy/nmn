"""Validate the optional dependency for the PyTorch backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("torch", backend="PyTorch", extra="torch")
