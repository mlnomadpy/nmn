"""Validate the optional dependency for the TensorFlow backend."""

from nmn._optional import require_optional_dependency

require_optional_dependency("tensorflow", backend="TensorFlow", extra="tf")
