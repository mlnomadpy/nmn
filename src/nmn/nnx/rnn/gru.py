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


class YatGRUCell(RNNCellBase):
    r"""Yat GRU cell.
    The mathematical definition of the cell is as follows
    .. math::
        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h \\
        \end{array}
    where x is the input and h is the output of the previous time step.
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
        bias_init: Initializer = default_bias_init,
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
            out_features=3 * hidden_features,  # r, z, n
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
            out_features=3 * hidden_features,  # r, z, n
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

    def __call__(self, carry: Array, inputs: Array, *, deterministic: bool = False) -> tuple[Array, Array]:
        h = carry
        x_transformed = self.dense_i(inputs, deterministic=deterministic)
        h_transformed = self.dense_h(h, deterministic=deterministic)

        xi_r, xi_z, xi_n = jnp.split(x_transformed, 3, axis=-1)
        hh_r, hh_z, hh_n = jnp.split(h_transformed, 3, axis=-1)

        r = self.gate_fn(xi_r + hh_r)
        z = self.gate_fn(xi_z + hh_z)
        n = self.activation_fn(xi_n + r * hh_n)
        new_h = (1.0 - z) * n + z * h
        return new_h, new_h

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array:
        batch_dims = input_shape[:-1]
        if rngs is None:
            rngs = self.rngs
        mem_shape = batch_dims + (self.hidden_features,)
        h = self.carry_init(rngs.carry(), mem_shape, self.param_dtype)
        return h

    @property
    def num_feature_axes(self) -> int:
        return 1 