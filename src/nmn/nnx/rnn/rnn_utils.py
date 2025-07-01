from __future__ import annotations

from typing import Any, TypeVar
from collections.abc import Mapping
from collections.abc import Callable
from functools import partial
from typing_extensions import Protocol
from absl import logging

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import filterlib, rnglib
from flax.nnx.module import Module, first_from
from flax.nnx.nn import initializers
from flax.nnx.transforms import iteration
from flax.typing import (
    Dtype,
    Initializer,
    Shape
)

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any


default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()


class RNNCellBase(Module):
    """RNN cell base class."""

    def initialize_carry(
        self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
    ) -> Carry:
        """Initialize the RNN cell carry.
        Args:
          rngs: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.
        Returns:
          An initialized carry for the given RNN cell.
        """
        raise NotImplementedError

    def __call__(
        self,
        carry: Carry,
        inputs: Array,
        *,
        deterministic: bool = False,
    ) -> tuple[Carry, Array]:
        """Run the RNN cell.
        Args:
          carry: the hidden state of the RNN cell.
          inputs: an ndarray with the input for the current time step.
            All dimensions except the final are considered batch dimensions.
          deterministic: If true, DropConnect is not applied (e.g., during inference).
        Returns:
          A tuple with the new carry and the output.
        """
        raise NotImplementedError

    @property
    def num_feature_axes(self) -> int:
        """Returns the number of feature axes of the RNN cell."""
        raise NotImplementedError


def modified_orthogonal(key: Array, shape: Shape, dtype: Dtype = jnp.float32) -> Array:
    """Modified orthogonal initializer for compatibility with half precision."""
    initializer = initializers.orthogonal()
    return initializer(key, shape).astype(dtype)


class RNN(Module):
  """The ``RNN`` module takes any :class:`RNNCellBase` instance and applies it over a sequence
  using :func:`flax.nnx.scan`.
  """

  state_axes: dict[str, int | type[iteration.Carry] | None]

  __data__ = ('cell', 'rngs')

  def __init__(
    self,
    cell: RNNCellBase,
    *,
    time_major: bool = False,
    return_carry: bool = False,
    reverse: bool = False,
    keep_order: bool = False,
    unroll: int = 1,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
    state_axes: Mapping[str, int | type[iteration.Carry] | None] | None = None,
    broadcast_rngs: filterlib.Filter = None,
  ):
    self.cell = cell
    self.time_major = time_major
    self.return_carry = return_carry
    self.reverse = reverse
    self.keep_order = keep_order
    self.unroll = unroll
    self.deterministic = deterministic
    if rngs is None:
      rngs = rnglib.Rngs(0)
    self.rngs = rngs
    self.state_axes = state_axes or {...: iteration.Carry}  # type: ignore
    self.broadcast_rngs = broadcast_rngs

  def __call__(
    self,
    inputs: Array,
    *,
    initial_carry: Carry | None = None,
    seq_lengths: Array | None = None,
    return_carry: bool | None = None,
    time_major: bool | None = None,
    reverse: bool | None = None,
    keep_order: bool | None = None,
    deterministic: bool | None = None,
    rngs: rnglib.Rngs | None = None,
  ):
    if return_carry is None:
      return_carry = self.return_carry
    if time_major is None:
      time_major = self.time_major
    if reverse is None:
      reverse = self.reverse
    if keep_order is None:
      keep_order = self.keep_order

    is_deterministic = first_from(
        deterministic,
        self.deterministic,
        error_msg="""No `deterministic` argument was provided to RNN
          as either a __call__ argument, class attribute, or nnx.flag.""",
      )

    time_axis = 0 if time_major else inputs.ndim - (self.cell.num_feature_axes + 1)

    if time_axis < 0:
      time_axis += inputs.ndim

    if time_major:
      batch_dims = inputs.shape[1 : -self.cell.num_feature_axes]
    else:
      batch_dims = inputs.shape[:time_axis]

    if reverse:
      inputs = jax.tree_util.tree_map(
                lambda x: flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                inputs,
            )
    if rngs is None:
      rngs = self.rngs
    carry: Carry = (
            self.cell.initialize_carry(
                inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :], rngs
            )
            if initial_carry is None
            else initial_carry
        )

    slice_carry = seq_lengths is not None and return_carry
    broadcast_rngs = nnx.All(nnx.RngState, self.broadcast_rngs)
    state_axes = iteration.StateAxes({broadcast_rngs: None, **self.state_axes})  # type: ignore[misc]

    @nnx.split_rngs(splits=1, only=self.broadcast_rngs, squeeze=True)
    @nnx.scan(
      in_axes=(state_axes, iteration.Carry, time_axis),
      out_axes=(iteration.Carry, (0, time_axis))
      if slice_carry
      else (iteration.Carry, time_axis),
      unroll=self.unroll,
    )
    def scan_fn(
      cell: RNNCellBase, carry: Carry, x: Array
    ) -> tuple[Carry, Array] | tuple[Carry, tuple[Carry, Array]]:
      carry, y = cell(carry, x, deterministic=is_deterministic)
      if slice_carry:
        return carry, (carry, y)
      return carry, y

    scan_output = scan_fn(self.cell, carry, inputs)

    if slice_carry:
      assert seq_lengths is not None
      _, (carries, outputs) = scan_output
      carry = _select_last_carry(carries, seq_lengths)
    else:
      carry, outputs = scan_output

    if reverse and keep_order:
      outputs = jax.tree_util.tree_map(
                lambda x: flip_sequences(
                    x,
                    seq_lengths,
                    num_batch_dims=len(batch_dims),
                    time_major=time_major,  # type: ignore
                ),
                outputs,
            )

    if return_carry:
      return carry, outputs
    else:
      return outputs


def _select_last_carry(sequence: A, seq_lengths: jnp.ndarray) -> A:
    last_idx = seq_lengths - 1

    def _slice_array(x: jnp.ndarray):
        return x[last_idx, jnp.arange(x.shape[1])]

    return jax.tree_util.tree_map(_slice_array, sequence)


def _expand_dims_like(x, target):
    """Expands the shape of `x` to match `target`'s shape by adding singleton dimensions."""
    return x.reshape(list(x.shape) + [1] * (target.ndim - x.ndim))


def flip_sequences(
    inputs: Array,
    seq_lengths: Array | None,
    num_batch_dims: int,
    time_major: bool,
) -> Array:
    time_axis = 0 if time_major else num_batch_dims
    max_steps = inputs.shape[time_axis]

    if seq_lengths is None:
        inputs = jnp.flip(inputs, axis=time_axis)
        return inputs

    seq_lengths = jnp.expand_dims(seq_lengths, axis=time_axis)

    idxs = jnp.arange(max_steps - 1, -1, -1)
    if time_major:
        idxs = jnp.reshape(idxs, [max_steps] + [1] * num_batch_dims)
    else:
        idxs = jnp.reshape(
            idxs, [1] * num_batch_dims + [max_steps]
        )
    idxs = (idxs + seq_lengths) % max_steps
    idxs = _expand_dims_like(idxs, target=inputs)
    outputs = jnp.take_along_axis(inputs, idxs, axis=time_axis)

    return outputs


def _concatenate(a: Array, b: Array) -> Array:
    """Concatenates two arrays along the last dimension."""
    return jnp.concatenate([a, b], axis=-1)


class RNNBase(Protocol):
    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: Carry | None = None,
        rngs: rnglib.Rngs | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
        deterministic: bool | None = None,
    ) -> Output | tuple[Carry, Output]: ...


class Bidirectional(Module):
    forward_rnn: RNNBase
    backward_rnn: RNNBase
    merge_fn: Callable[[Array, Array], Array] = _concatenate
    time_major: bool = False
    return_carry: bool = False

    def __init__(
        self,
        forward_rnn: RNNBase,
        backward_rnn: RNNBase,
        *,
        merge_fn: Callable[[Array, Array], Array] = _concatenate,
        time_major: bool = False,
        return_carry: bool = False,
        rngs: rnglib.Rngs | None = None,
    ):
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.merge_fn = merge_fn
        self.time_major = time_major
        self.return_carry = return_carry
        if rngs is None:
            rngs = rnglib.Rngs(0)
        self.rngs = rngs

    def __call__(
        self,
        inputs: Array,
        *,
        initial_carry: tuple[Carry, Carry] | None = None,
        rngs: rnglib.Rngs | None = None,
        seq_lengths: Array | None = None,
        return_carry: bool | None = None,
        time_major: bool | None = None,
        deterministic: bool | None = None,
    ) -> Output | tuple[tuple[Carry, Carry], Output]:
        if time_major is None:
            time_major = self.time_major
        if return_carry is None:
            return_carry = self.return_carry
        if rngs is None:
            rngs = self.rngs
        if initial_carry is not None:
            initial_carry_forward, initial_carry_backward = initial_carry
        else:
            initial_carry_forward = None
            initial_carry_backward = None

        if self.forward_rnn is self.backward_rnn:
            logging.warning(
                "forward_rnn and backward_rnn is the same object, so "
                "they will share parameters."
            )

        carry_forward, outputs_forward = self.forward_rnn(
            inputs,
            initial_carry=initial_carry_forward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=False,
            deterministic=deterministic,
        )

        carry_backward, outputs_backward = self.backward_rnn(
            inputs,
            initial_carry=initial_carry_backward,
            rngs=rngs,
            seq_lengths=seq_lengths,
            return_carry=True,
            time_major=time_major,
            reverse=True,
            keep_order=True,
            deterministic=deterministic,
        )

        carry = (carry_forward, carry_backward) if return_carry else None
        outputs = jax.tree_util.tree_map(
            self.merge_fn, outputs_forward, outputs_backward
        )

        if return_carry:
            return carry, outputs
        else:
            return outputs 