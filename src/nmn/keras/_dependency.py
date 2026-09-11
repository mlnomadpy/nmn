"""Validate the optional dependency for the Keras backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("keras", backend="Keras", extra="keras")
