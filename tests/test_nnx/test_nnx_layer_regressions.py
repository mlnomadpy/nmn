"""Regression tests for NNX convolution, embedding, epsilon and DropConnect."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx import (
    Embed,
    MultiHeadAttention,
    RotaryYatAttention,
    YatConv,
    YatConvTranspose,
    YatNMN,
)


def _dropconnect_cases(seed):
    rngs = nnx.Rngs(params=0, dropout=seed)
    return [
        (
            YatNMN(
                16,
                12,
                use_bias=False,
                use_alpha=False,
                use_dropconnect=True,
                drop_rate=0.5,
                rngs=rngs,
            ),
            jnp.arange(32, dtype=jnp.float32).reshape(2, 16) / 32,
        ),
        (
            YatConv(
                4,
                8,
                3,
                use_bias=False,
                use_alpha=False,
                use_dropconnect=True,
                drop_rate=0.5,
                rngs=rngs,
            ),
            jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) / 40,
        ),
        (
            YatConvTranspose(
                4,
                8,
                3,
                use_bias=False,
                use_alpha=False,
                use_dropconnect=True,
                drop_rate=0.5,
                rngs=rngs,
            ),
            jnp.arange(40, dtype=jnp.float32).reshape(2, 5, 4) / 40,
        ),
        (
            MultiHeadAttention(
                4,
                16,
                use_bias=False,
                use_alpha=False,
                use_dropconnect=True,
                dropconnect_rate=0.5,
                deterministic=False,
                rngs=rngs,
            ),
            jnp.arange(96, dtype=jnp.float32).reshape(2, 3, 16) / 96,
        ),
    ]


@pytest.mark.parametrize("case", range(4))
def test_dropconnect_is_stochastic_deterministic_and_uses_dropout_stream(case):
    model, x = _dropconnect_cases(1)[case]
    first = model(x, deterministic=False)
    second = model(x, deterministic=False)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(
        model(x, deterministic=True), model(x, deterministic=True)
    )

    same_params_other_stream, _ = _dropconnect_cases(2)[case]
    assert not np.array_equal(first, same_params_other_stream(x, deterministic=False))


@pytest.mark.parametrize("case", range(4))
def test_dropconnect_mutable_stream_works_under_nnx_jit(case):
    model, x = _dropconnect_cases(7)[case]
    call = nnx.jit(lambda module, value: module(value, deterministic=False))
    first = call(model, x)
    second = call(model, x)
    assert not np.array_equal(first, second)


def test_decode_validation_does_not_advance_dropconnect_or_cache():
    module = MultiHeadAttention(
        2,
        8,
        decode=True,
        deterministic=False,
        use_alpha=False,
        use_dropconnect=True,
        dropconnect_rate=0.5,
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    module.init_cache((1, 3, 8))
    rng_count = int(module.dropconnect_rng.count[...])
    cache = (
        np.asarray(module.cached_key[...]).copy(),
        np.asarray(module.cached_value[...]).copy(),
        int(module.cache_index[...]),
    )

    invalid_mask = jnp.ones((1, 1, 2, 3), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="[Mm]ask shape"):
        module(
            jnp.ones((1, 1, 8)),
            mask=invalid_mask,
            deterministic=False,
        )

    assert int(module.dropconnect_rng.count[...]) == rng_count
    np.testing.assert_array_equal(module.cached_key[...], cache[0])
    np.testing.assert_array_equal(module.cached_value[...], cache[1])
    assert int(module.cache_index[...]) == cache[2]


def test_attention_failure_does_not_commit_dropconnect_or_cache():
    def failing_attention(*args, **kwargs):
        raise RuntimeError("attention failed")

    module = MultiHeadAttention(
        2,
        8,
        decode=True,
        deterministic=False,
        use_alpha=False,
        use_dropconnect=True,
        dropconnect_rate=0.5,
        attention_fn=failing_attention,
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    module.init_cache((1, 3, 8))
    rng_count = int(module.dropconnect_rng.count[...])
    cache = (
        np.asarray(module.cached_key[...]).copy(),
        np.asarray(module.cached_value[...]).copy(),
        int(module.cache_index[...]),
    )

    with pytest.raises(RuntimeError, match="attention failed"):
        module(jnp.ones((1, 1, 8)), deterministic=False)

    assert int(module.dropconnect_rng.count[...]) == rng_count
    np.testing.assert_array_equal(module.cached_key[...], cache[0])
    np.testing.assert_array_equal(module.cached_value[...], cache[1])
    assert int(module.cache_index[...]) == cache[2]


@pytest.mark.parametrize("drop_rate", [-0.1, 1.0, np.nan, np.inf])
@pytest.mark.parametrize("kind", ["dense", "conv", "transpose"])
def test_invalid_dropconnect_rates_are_rejected(kind, drop_rate):
    kwargs = dict(
        use_dropconnect=True,
        drop_rate=drop_rate,
        rngs=nnx.Rngs(0),
    )
    with pytest.raises(ValueError, match="drop_rate"):
        if kind == "dense":
            YatNMN(2, 2, **kwargs)
        elif kind == "conv":
            YatConv(2, 2, 1, **kwargs)
        else:
            YatConvTranspose(2, 2, 1, **kwargs)


def _epsilon_modules(dtype):
    kwargs = dict(
        epsilon=1e-5,
        learnable_epsilon=True,
        param_dtype=dtype,
        rngs=nnx.Rngs(0),
    )
    return [
        YatNMN(2, 2, **kwargs),
        Embed(2, 2, **kwargs),
        YatConv(2, 2, 1, **kwargs),
        YatConvTranspose(2, 2, 1, **kwargs),
        MultiHeadAttention(1, 2, **kwargs),
        RotaryYatAttention(2, 1, max_seq_len=2, **kwargs),
    ]


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_learnable_epsilon_is_positive_and_close_to_requested_value(dtype):
    for module in _epsilon_modules(dtype):
        raw = module.epsilon_param[...]
        effective = jax.nn.softplus(raw.astype(jnp.float32))
        assert jnp.isfinite(raw).all()
        assert float(effective[0]) > 0.0
        np.testing.assert_allclose(float(effective[0]), 1e-5, rtol=0.03)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_learnable_epsilon_has_finite_aggregate_gradient(dtype):
    layer = YatConv(
        2,
        1,
        1,
        dtype=dtype,
        param_dtype=dtype,
        epsilon=1e-3,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.full((1, 128, 2), 0.25, dtype=dtype)
    layer.kernel[...] = jnp.full_like(layer.kernel[...], 0.25)

    def summed_score(module, value):
        return jnp.sum(module(value).astype(jnp.float32))

    _, grads = jax.value_and_grad(summed_score)(layer, x)
    assert jnp.isfinite(grads.epsilon_param[...]).all()


def _loss(module, x):
    return jnp.mean(module(x).astype(jnp.float32) ** 2)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("kind", ["conv", "transpose"])
def test_low_precision_conv_collision_matches_synchronized_fp32(kind, dtype):
    cls = YatConv if kind == "conv" else YatConvTranspose
    low = cls(
        2,
        1,
        1,
        dtype=dtype,
        param_dtype=dtype,
        epsilon=1e-3,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    ref = cls(
        2,
        1,
        1,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        epsilon=1e-3,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(1),
    )
    x = jnp.asarray([[[0.125, -0.25]]], dtype=dtype)
    kernel = x[0].reshape(low.kernel[...].shape)
    low.kernel[...] = kernel
    ref.kernel[...] = kernel.astype(jnp.float32)

    low_out = low(x)
    ref_out = ref(x.astype(jnp.float32))
    _, (low_grads, low_dx) = jax.value_and_grad(_loss, argnums=(0, 1))(low, x)
    _, (ref_grads, ref_dx) = jax.value_and_grad(_loss, argnums=(0, 1))(
        ref, x.astype(jnp.float32)
    )

    assert low_out.dtype == dtype
    assert jnp.isfinite(low_out).all()
    assert jnp.isfinite(low_dx).all()
    assert jnp.isfinite(low_grads.kernel[...]).all()
    np.testing.assert_allclose(
        np.asarray(low_out, dtype=np.float32), np.asarray(ref_out), rtol=0.03, atol=0.03
    )
    np.testing.assert_allclose(
        np.asarray(low_dx, dtype=np.float32), np.asarray(ref_dx), rtol=0.08, atol=0.08
    )
    np.testing.assert_allclose(
        np.asarray(low_grads.kernel[...], dtype=np.float32),
        np.asarray(ref_grads.kernel[...]),
        rtol=0.08,
        atol=0.08,
    )


@pytest.mark.parametrize("kind", ["conv", "transpose"])
def test_fp16_large_exact_collision_saturates_with_finite_gradients(kind):
    cls = YatConv if kind == "conv" else YatConvTranspose
    layer = cls(
        2,
        1,
        1,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        epsilon=1e-5,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    x = jnp.asarray([[[0.75, 0.75]]], dtype=jnp.float16)
    layer.kernel[...] = x[0].reshape(layer.kernel[...].shape)
    output = layer(x)
    _, (grads, dx) = jax.value_and_grad(_loss, argnums=(0, 1))(layer, x)
    assert output[0, 0, 0] == jnp.finfo(jnp.float16).max
    assert all(jnp.isfinite(value).all() for value in (output, dx, grads.kernel[...]))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_embed_collision_and_gradients_match_fp32(dtype):
    low = Embed(
        2,
        3,
        dtype=dtype,
        param_dtype=dtype,
        epsilon=1e-3,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    ref = Embed(
        2,
        3,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        epsilon=1e-3,
        use_alpha=False,
        rngs=nnx.Rngs(1),
    )
    values = jnp.asarray([[0.125, -0.25, 0.375], [-0.25, 0.5, 0.125]], dtype=dtype)
    low.embedding[...] = values
    ref.embedding[...] = values.astype(jnp.float32)
    query = values[:1]

    def loss(module, value):
        return jnp.mean(module.attend(value).astype(jnp.float32) ** 2)

    low_out = low.attend(query)
    ref_out = ref.attend(query.astype(jnp.float32))
    _, (low_grads, low_dq) = jax.value_and_grad(loss, argnums=(0, 1))(low, query)
    _, (ref_grads, ref_dq) = jax.value_and_grad(loss, argnums=(0, 1))(
        ref, query.astype(jnp.float32)
    )
    assert low_out.dtype == dtype
    assert all(
        jnp.isfinite(x).all() for x in (low_out, low_dq, low_grads.embedding[...])
    )
    np.testing.assert_allclose(
        np.asarray(low_out, dtype=np.float32), np.asarray(ref_out), rtol=0.03, atol=0.03
    )
    np.testing.assert_allclose(
        np.asarray(low_dq, dtype=np.float32), np.asarray(ref_dq), rtol=0.08, atol=0.08
    )
    np.testing.assert_allclose(
        np.asarray(low_grads.embedding[...], dtype=np.float32),
        np.asarray(ref_grads.embedding[...]),
        rtol=0.08,
        atol=0.08,
    )


def test_fp16_embed_large_collision_saturates_with_finite_gradients():
    layer = Embed(
        1,
        2,
        dtype=jnp.float16,
        param_dtype=jnp.float16,
        epsilon=1e-5,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    query = jnp.asarray([[0.75, 0.75]], dtype=jnp.float16)
    layer.embedding[...] = query

    def loss(module, value):
        return jnp.mean(module.attend(value).astype(jnp.float32) ** 2)

    output = layer.attend(query)
    _, (grads, dq) = jax.value_and_grad(loss, argnums=(0, 1))(layer, query)
    assert output[0, 0] == jnp.finfo(jnp.float16).max
    assert all(
        jnp.isfinite(value).all() for value in (output, dq, grads.embedding[...])
    )


@pytest.mark.parametrize("kind", ["conv", "transpose", "embed"])
def test_low_precision_nan_forward_and_cotangent_are_preserved(kind):
    if kind == "embed":
        module = Embed(
            2,
            2,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
            use_alpha=False,
            rngs=nnx.Rngs(0),
        )
        call = lambda value: module.attend(value)
        clean = jnp.ones((1, 2), dtype=jnp.float16)
    else:
        cls = YatConv if kind == "conv" else YatConvTranspose
        module = cls(
            2,
            2,
            1,
            dtype=jnp.float16,
            param_dtype=jnp.float16,
            use_bias=False,
            use_alpha=False,
            rngs=nnx.Rngs(0),
        )
        call = lambda value: module(value)
        clean = jnp.ones((1, 2, 2), dtype=jnp.float16)

    with_nan = clean.at[(0,) * (clean.ndim - 1) + (0,)].set(jnp.nan)
    assert jnp.isnan(call(with_nan)).any()

    output, pullback = jax.vjp(call, clean)
    (cotangent,) = pullback(jnp.full_like(output, jnp.nan))
    assert jnp.isnan(cotangent).any()


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_spherical_embed_zero_vectors_are_finite(dtype):
    layer = Embed(
        3, 4, dtype=dtype, param_dtype=dtype, spherical=True, rngs=nnx.Rngs(0)
    )
    layer.embedding[...] = jnp.zeros_like(layer.embedding[...])
    output = layer.attend(jnp.zeros((2, 4), dtype=dtype))
    assert output.dtype == dtype
    assert jnp.isfinite(output).all()


def _explicit_grouped_reference(x, kernel, strides, padding, dilation, epsilon):
    spatial_ndim = kernel.ndim - 2
    groups = x.shape[-1] // kernel.shape[-2]
    filters_per_group = kernel.shape[-1] // groups
    pad_width = ((0, 0), *padding, (0, 0))
    padded = jnp.pad(x, pad_width)
    effective = tuple((k - 1) * d + 1 for k, d in zip(kernel.shape[:-2], dilation))
    output_spatial = tuple(
        (padded.shape[axis + 1] - effective[axis]) // strides[axis] + 1
        for axis in range(spatial_ndim)
    )
    batches = []
    for batch in range(x.shape[0]):
        positions = []
        for output_index in itertools.product(
            *(range(size) for size in output_spatial)
        ):
            spatial_indices = [
                output_index[axis] * strides[axis]
                + jnp.arange(kernel.shape[axis]) * dilation[axis]
                for axis in range(spatial_ndim)
            ]
            patch = padded[(batch, *jnp.ix_(*spatial_indices))]
            scores = []
            for group in range(groups):
                patch_group = patch[
                    ..., group * kernel.shape[-2] : (group + 1) * kernel.shape[-2]
                ]
                for offset in range(filters_per_group):
                    filter_index = group * filters_per_group + offset
                    filter_value = kernel[..., filter_index]
                    dot = jnp.sum(patch_group * filter_value)
                    distance = jnp.maximum(
                        jnp.sum(patch_group**2) + jnp.sum(filter_value**2) - 2 * dot,
                        0.0,
                    )
                    scores.append(dot**2 / (distance + epsilon))
            positions.append(jnp.stack(scores))
        batches.append(
            jnp.stack(positions).reshape((*output_spatial, kernel.shape[-1]))
        )
    return jnp.stack(batches)


@pytest.mark.parametrize(
    "spatial,kernel_size,strides,padding,dilation",
    [
        ((6,), (2,), (2,), ((1, 0),), (2,)),
        ((5, 6), (2, 2), (2, 1), ((1, 0), (0, 1)), (1, 2)),
        ((4, 5, 4), (2, 2, 2), (1, 2, 1), ((0, 1), (1, 0), (1, 1)), (1, 1, 2)),
    ],
)
def test_grouped_conv_forward_and_gradients_match_explicit_patches(
    spatial, kernel_size, strides, padding, dilation
):
    layer = YatConv(
        4,
        4,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_dilation=dilation,
        feature_group_count=2,
        epsilon=1e-3,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    x = (
        jnp.arange(np.prod((1, *spatial, 4)), dtype=jnp.float32).reshape(
            (1, *spatial, 4)
        )
        / 50
    )
    kernel = (
        jnp.arange(layer.kernel[...].size, dtype=jnp.float32).reshape(
            layer.kernel[...].shape
        )
        / 30
    )
    layer.kernel[...] = kernel
    reference = lambda value, weights: _explicit_grouped_reference(
        value, weights, strides, padding, dilation, 1e-3
    )
    actual = layer(x)
    expected = reference(x, kernel)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    _, (actual_grads, actual_dx) = jax.value_and_grad(_loss, argnums=(0, 1))(layer, x)
    ref_loss = lambda value, weights: jnp.mean(reference(value, weights) ** 2)
    _, (expected_dx, expected_dw) = jax.value_and_grad(ref_loss, argnums=(0, 1))(
        x, kernel
    )
    np.testing.assert_allclose(actual_dx, expected_dx, rtol=5e-4, atol=5e-4)
    np.testing.assert_allclose(
        actual_grads.kernel[...], expected_dw, rtol=5e-4, atol=5e-4
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_transpose_kernel_layout_forward_and_gradient_parity(ndim):
    kernel_size = (2,) * ndim
    shape = (1, *((3,) * ndim), 2)
    normal = YatConvTranspose(
        2,
        3,
        kernel_size,
        padding="VALID",
        epsilon=1e-3,
        transpose_kernel=False,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(0),
    )
    transposed = YatConvTranspose(
        2,
        3,
        kernel_size,
        padding="VALID",
        epsilon=1e-3,
        transpose_kernel=True,
        use_bias=False,
        use_alpha=False,
        rngs=nnx.Rngs(1),
    )
    kernel = (
        jnp.arange(normal.kernel[...].size, dtype=jnp.float32).reshape(
            normal.kernel[...].shape
        )
        / 20
    )
    transform = lambda value: jnp.swapaxes(
        jnp.flip(value, axis=tuple(range(ndim))), -1, -2
    )
    normal.kernel[...] = kernel
    transposed.kernel[...] = transform(kernel)
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 20

    np.testing.assert_allclose(normal(x), transposed(x), rtol=2e-5, atol=2e-5)
    _, (normal_grads, normal_dx) = jax.value_and_grad(_loss, argnums=(0, 1))(normal, x)
    _, (transposed_grads, transposed_dx) = jax.value_and_grad(_loss, argnums=(0, 1))(
        transposed, x
    )
    np.testing.assert_allclose(normal_dx, transposed_dx, rtol=1e-4, atol=1e-2)
    np.testing.assert_allclose(
        transform(normal_grads.kernel[...]),
        transposed_grads.kernel[...],
        rtol=1e-4,
        atol=1e-2,
    )
