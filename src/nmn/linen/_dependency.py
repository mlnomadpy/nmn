"""Validate the optional dependencies for the Flax Linen backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("jax", backend="Flax Linen", extra="linen")
require_optional_dependency("flax", backend="Flax Linen", extra="linen")
