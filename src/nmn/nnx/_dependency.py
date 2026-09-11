"""Validate the optional dependencies for the Flax NNX backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("jax", backend="Flax NNX", extra="nnx")
require_optional_dependency("flax", backend="Flax NNX", extra="nnx")
