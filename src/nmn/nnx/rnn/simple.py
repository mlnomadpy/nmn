import jax.numpy as jnp
from jax import Array
from nmn.nnx.squashers import soft_tanh
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


class YatSimpleCell(RNNCellBase):
    r"""Yat Simple cell.
    The mathematical definition of the cell is as follows
    .. math::
        \begin{array}{ll}
        h' = \tanh(W_i x + b_i + W_h h)
        \end{array}
    where x is the input and h is the output of the previous time step.
    If `residual` is `True`,
    .. math::
        \begin{array}{ll}
        h' = \tanh(W_i x + b_i + W_h h + h)
        \end{array}
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        residual: bool = False,
        activation_fn: tp.Callable[..., Any] = soft_tanh,
        kernel_init: Initializer = default_kernel_init,
        recurrent_kernel_init: Initializer = modified_orthogonal,
        bias_init: Initializer = default_bias_init,
        use_bias: bool = True,
        use_alpha: bool = True,
        use_dropconnect: bool = False,
        drop_rate: float = 0.0,
        epsilon: float = 1e-5,
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = residual
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        self.dense_h = YatNMN(
            in_features=self.hidden_features,
            out_features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            use_alpha=use_alpha,
            use_dropconnect=use_dropconnect,
            drop_rate=drop_rate,
            epsilon=epsilon,
            rngs=rngs,
        )
        self.dense_i = YatNMN(
            in_features=self.in_features,
            out_features=self.hidden_features,
            use_bias=use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_alpha=use_alpha,
            use_dropconnect=use_dropconnect,
            drop_rate=drop_rate,
            epsilon=epsilon,
            rngs=rngs,
        )

    def __call__(self, carry: Array, inputs: Array, *, deterministic: bool = False) -> tuple[Array, Array]:
        new_carry = self.dense_i(inputs, deterministic=deterministic) + self.dense_h(carry, deterministic=deterministic)
        if self.residual:
            new_carry += carry
        new_carry = self.activation_fn(new_carry)
        return new_carry, new_carry

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array:
        if rngs is None:
            rngs = self.rngs
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(rngs.carry(), mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1 