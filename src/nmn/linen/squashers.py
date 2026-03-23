"""Squashing functions for Flax Linen.

Re-exports from the NNX squashers since both use JAX.
"""

from nmn.nnx.layers.squashers import softermax, softer_sigmoid, soft_tanh

__all__ = ["softermax", "softer_sigmoid", "soft_tanh"]
