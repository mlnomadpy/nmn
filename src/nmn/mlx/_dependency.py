"""Validate the optional dependency for the MLX backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("mlx", backend="MLX", extra="mlx")
