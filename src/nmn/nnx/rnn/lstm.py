from functools import partial
import jax.numpy as jnp
from jax import Array
from nmn.nnx.squashers import softer_sigmoid, soft_tanh
from flax.nnx import rnglib
from flax.nnx.nn import initializers
from flax.typing import (
    Dtype,
    Initializer,
)
import typing as tp
from nmn.nnx.nmn import YatNMN
from nmn.nnx.rnn.rnn_utils import RNNCellBase, default_kernel_init, modified_orthogonal, default_bias_init
from typing import Any

class YatLSTMCell(RNNCellBase):
  r"""Yat LSTM cell.
  The mathematical definition of the cell is as follows
  .. math::
      \begin{array}{ll}
      i = \sigma(W_{ii} x + W_{hi} h + b_{hi}) \\
      f = \sigma(W_{if} x + W_{hf} h + b_{hf}) \\
      g = \tanh(W_{ig} x + W_{hg} h + b_{hg}) \\
      o = \sigma(W_{io} x + W_{ho} h + b_{ho}) \\
      c' = f * c + i * g \\
      h' = o * \tanh(c') \\
      \end{array}
  where x is the input, h is the output of the previous time step, and c is
  the memory.
  """

  def __init__(
    self,
    in_features: int,
    hidden_features: int,
    *,
    gate_fn: tp.Callable[..., Any] = softer_sigmoid,
    activation_fn: tp.Callable[..., Any] = soft_tanh,
    kernel_init: Initializer = default_kernel_init,
    recurrent_kernel_init: Initializer = modified_orthogonal,
    bias_init: Initializer = initializers.zeros_init(),
    dtype: Dtype | None = None,
    param_dtype: Dtype = jnp.float32,
    carry_init: Initializer = initializers.zeros_init(),
    use_bias: bool = True,
    use_alpha: bool = True,
    use_dropconnect: bool = False,
    drop_rate: float = 0.0,
    epsilon: float = 1e-5,
    rngs: rnglib.Rngs,
  ):
    self.in_features = in_features
    self.hidden_features = hidden_features
    self.gate_fn = gate_fn
    self.activation_fn = activation_fn
    self.kernel_init = kernel_init
    self.recurrent_kernel_init = recurrent_kernel_init
    self.bias_init = bias_init
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.carry_init = carry_init
    self.rngs = rngs

    self.dense_i = YatNMN(
      in_features=in_features,
      out_features=4 * hidden_features,
      use_bias=use_bias,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_alpha=use_alpha,
      use_dropconnect=use_dropconnect,
      drop_rate=drop_rate,
      epsilon=epsilon,
      rngs=rngs,
    )

    self.dense_h = YatNMN(
      in_features=hidden_features,
      out_features=4 * hidden_features,
      use_bias=False,
      kernel_init=self.recurrent_kernel_init,
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_alpha=use_alpha,
      use_dropconnect=use_dropconnect,
      drop_rate=drop_rate,
      epsilon=epsilon,
      rngs=rngs,
    )

  def __call__(
    self, carry: tuple[Array, Array], inputs: Array, *, deterministic: bool = False
  ) -> tuple[tuple[Array, Array], Array]:
    c, h = carry
    y = self.dense_i(inputs, deterministic=deterministic) + self.dense_h(h, deterministic=deterministic)
    i, f, g, o = jnp.split(y, indices_or_sections=4, axis=-1)
    i = self.gate_fn(i)
    f = self.gate_fn(f)
    g = self.activation_fn(g)
    o = self.gate_fn(o)
    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  def initialize_carry(
    self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None
  ) -> tuple[Array, Array]:
    batch_dims = input_shape[:-1]
    if rngs is None:
      rngs = self.rngs
    mem_shape = batch_dims + (self.hidden_features,)
    c = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
    h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
    return (c, h)

  @property
  def num_feature_axes(self) -> int:
    return 1 