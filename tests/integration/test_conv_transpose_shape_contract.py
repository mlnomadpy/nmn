"""Runnable cross-framework tests for canonical ConvTranspose sizing."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nnx = pytest.importorskip("flax.nnx")

from nmn.linen import YatConvTranspose1D as LinenTranspose1D
from nmn.linen import YatConvTranspose2D as LinenTranspose2D
from nmn.linen import YatConvTranspose3D as LinenTranspose3D
from nmn.nnx import YatConvTranspose as NnxTranspose
from nmn.torch import YatConvTranspose1D as TorchTranspose1D
from nmn.torch import YatConvTranspose2D as TorchTranspose2D
from nmn.torch import YatConvTranspose3D as TorchTranspose3D

CASES = [
    (
        LinenTranspose1D,
        TorchTranspose1D,
        (1, 3, 1),
        (2,),
        (3,),
        (1,),
        (0,),
        (1, 8, 1),
    ),
    (
        LinenTranspose2D,
        TorchTranspose2D,
        (1, 2, 3, 1),
        (2, 3),
        (3, 2),
        (1, 2),
        (1, 0),
        (1, 6, 9, 1),
    ),
    (
        LinenTranspose3D,
        TorchTranspose3D,
        (1, 2, 2, 2, 1),
        (2, 3, 2),
        (3, 2, 4),
        (1, 2, 1),
        (0, 1, 2),
        (1, 5, 8, 8, 1),
    ),
]


def _channels_first(value):
    return np.transpose(value, (0, value.ndim - 1, *range(1, value.ndim - 1)))


def _channels_last(value):
    return np.transpose(value, (0, *range(2, value.ndim), 1))


def test_jax_implicit_legacy_shape_is_preserved_until_output_padding_is_explicit():
    inputs = jnp.ones((1, 3, 1), dtype=jnp.float32)
    linen_legacy = LinenTranspose1D(
        1, (2,), strides=(3,), padding="VALID", use_bias=False, use_alpha=False
    )
    linen_canonical = LinenTranspose1D(
        1,
        (2,),
        strides=(3,),
        padding="VALID",
        output_padding=0,
        use_bias=False,
        use_alpha=False,
    )
    legacy_variables = linen_legacy.init(jax.random.key(0), inputs)
    canonical_variables = linen_canonical.init(jax.random.key(0), inputs)
    assert linen_legacy.apply(legacy_variables, inputs).shape == (1, 9, 1)
    assert linen_canonical.apply(canonical_variables, inputs).shape == (1, 8, 1)

    nnx_legacy = NnxTranspose(
        1,
        1,
        (2,),
        (3,),
        padding="VALID",
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    nnx_canonical = NnxTranspose(
        1,
        1,
        (2,),
        (3,),
        padding="VALID",
        output_padding=0,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    assert nnx_legacy(inputs).shape == (1, 9, 1)
    assert nnx_canonical(inputs).shape == (1, 8, 1)


def test_linen_nnx_canonical_same_output_and_gradient_parity_under_jit():
    inputs = jnp.asarray([[[-0.5], [0.25], [0.75]]], dtype=jnp.float32)
    kernel = jnp.asarray([[[-0.4]], [[0.6]]], dtype=jnp.float32)
    cotangent = jnp.linspace(0.2, 1.1, 10, dtype=jnp.float32).reshape(1, 10, 1)

    linen = LinenTranspose1D(
        features=1,
        kernel_size=(2,),
        strides=(3,),
        padding="SAME",
        output_padding=1,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    linen_apply = jax.jit(lambda x, w: linen.apply({"params": {"kernel": w}}, x))

    nnx_layer = NnxTranspose(
        1,
        1,
        (2,),
        (3,),
        padding="SAME",
        output_padding=1,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
        rngs=nnx.Rngs(0),
    )
    nnx_layer.kernel[...] = kernel
    nnx_apply = jax.jit(lambda x: nnx_layer(x))

    linen_output = linen_apply(inputs, kernel)
    nnx_output = nnx_apply(inputs)
    linen_input_grad, linen_kernel_grad = jax.grad(
        lambda x, w: jnp.sum(linen_apply(x, w) * cotangent), argnums=(0, 1)
    )(inputs, kernel)
    nnx_input_grad = jax.grad(lambda x: jnp.sum(nnx_apply(x) * cotangent))(inputs)
    _, nnx_grads = nnx.value_and_grad(lambda model: jnp.sum(model(inputs) * cotangent))(
        nnx_layer
    )

    assert linen_output.shape == nnx_output.shape == (1, 10, 1)
    np.testing.assert_allclose(linen_output, nnx_output, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(linen_input_grad, nnx_input_grad, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        linen_kernel_grad, nnx_grads.kernel[...], rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize(
    "linen_cls,torch_cls,input_shape,kernel_shape,strides,dilation,"
    "output_padding,expected",
    CASES,
)
def test_torch_linen_nnx_canonical_valid_output_and_gradient_parity(
    linen_cls,
    torch_cls,
    input_shape,
    kernel_shape,
    strides,
    dilation,
    output_padding,
    expected,
):
    rank = len(kernel_shape)
    input_values = np.linspace(
        -0.7, 0.9, np.prod(input_shape), dtype=np.float32
    ).reshape(input_shape)
    kernel = np.linspace(-0.4, 0.6, np.prod(kernel_shape), dtype=np.float32).reshape(
        (*kernel_shape, 1, 1)
    )

    linen = linen_cls(
        features=1,
        kernel_size=kernel_shape,
        strides=strides,
        padding="VALID",
        kernel_dilation=dilation,
        output_padding=output_padding,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    variables = linen.init(jax.random.key(0), jnp.asarray(input_values))
    variables["params"]["kernel"] = jnp.asarray(kernel)
    linen_apply = jax.jit(lambda x, w: linen.apply({"params": {"kernel": w}}, x))

    nnx_layer = NnxTranspose(
        1,
        1,
        kernel_shape,
        strides,
        padding="VALID",
        kernel_dilation=dilation,
        output_padding=output_padding,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
        rngs=nnx.Rngs(0),
    )
    nnx_layer.kernel[...] = jnp.asarray(kernel)

    torch_layer = torch_cls(
        1,
        1,
        kernel_shape,
        stride=strides,
        padding=0,
        output_padding=output_padding,
        dilation=dilation,
        bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    # JAX's ``conv_transpose`` uses the forward-correlation kernel convention;
    # PyTorch stores the mathematically transposed (spatially reversed) kernel.
    torch_kernel = np.transpose(
        np.flip(kernel, axis=tuple(range(rank))),
        (rank, rank + 1, *range(rank)),
    ).copy()
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(torch_kernel))

    linen_x = jnp.asarray(input_values)
    linen_w = jnp.asarray(kernel)
    linen_output = linen_apply(linen_x, linen_w)
    nnx_output = jax.jit(lambda value: nnx_layer(value))(linen_x)
    torch_x = torch.tensor(_channels_first(input_values), requires_grad=True)
    torch_output_cf = torch_layer(torch_x)
    torch_output = _channels_last(torch_output_cf.detach().numpy())
    assert linen_output.shape == nnx_output.shape == torch_output.shape == expected

    cotangent = np.linspace(0.2, 1.0, np.prod(expected), dtype=np.float32).reshape(
        expected
    )
    linen_input_grad, linen_kernel_grad = jax.grad(
        lambda x, w: jnp.sum(linen_apply(x, w) * cotangent), argnums=(0, 1)
    )(linen_x, linen_w)
    nnx_input_grad = jax.grad(lambda value: jnp.sum(nnx_layer(value) * cotangent))(
        linen_x
    )
    _, nnx_grads = nnx.value_and_grad(
        lambda model: jnp.sum(model(linen_x) * cotangent)
    )(nnx_layer)
    (torch_output_cf * torch.tensor(_channels_first(cotangent))).sum().backward()

    np.testing.assert_allclose(
        np.asarray(linen_output), torch_output, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(nnx_output), torch_output, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        np.asarray(linen_input_grad),
        _channels_last(torch_x.grad.numpy()),
        rtol=3e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx_input_grad),
        np.asarray(linen_input_grad),
        rtol=3e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(nnx_grads.kernel[...]),
        np.asarray(linen_kernel_grad),
        rtol=3e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        np.asarray(linen_kernel_grad),
        np.flip(
            np.transpose(torch_layer.weight.grad.numpy(), (*range(2, rank + 2), 0, 1)),
            axis=tuple(range(rank)),
        ),
        rtol=3e-4,
        atol=3e-5,
    )
