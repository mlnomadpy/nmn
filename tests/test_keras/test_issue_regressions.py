"""Regression coverage for Keras issues #55, #56, #73-#76 and #78."""

from __future__ import annotations

import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest

from nmn.keras import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
)


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (2, 7, 4)),
        (YatConv2D, (2, 7, 6, 4)),
        (YatConv3D, (2, 6, 5, 4, 4)),
    ],
)
def test_grouped_convolutions_have_per_group_patch_norms_and_gradients(
    layer_cls, input_shape
):
    x = jax.random.normal(jax.random.key(len(input_shape)), input_shape)
    layer = layer_cls(4, 3, groups=2, padding="same", use_bias=False)

    y = layer(x)
    dx = jax.grad(lambda value: jnp.sum(layer(value)))(x)

    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == 4
    assert np.all(np.isfinite(np.asarray(y)))
    assert np.all(np.isfinite(np.asarray(dx)))


def test_causal_conv1d_is_causal_and_uses_effective_dilated_padding():
    layer = YatConv1D(
        1,
        3,
        padding="causal",
        dilation_rate=2,
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    x = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 8, 1)
    changed_future = x.at[:, 5:, :].set(10_000.0)

    y = layer(x)
    changed_y = layer(changed_future)

    assert y.shape == (1, 8, 1)
    np.testing.assert_allclose(np.asarray(y[:, :5]), np.asarray(changed_y[:, :5]))
    assert layer.compute_output_shape((None, None, 1)) == (None, None, 1)


def test_conv1d_rejects_stride_and_dilation_combination():
    with pytest.raises(ValueError, match="strides > 1"):
        YatConv1D(2, 3, strides=2, dilation_rate=2, padding="causal")


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (2, 9, 2)),
        (YatConv2D, (2, 9, 8, 2)),
        (YatConv3D, (2, 8, 7, 6, 2)),
        (YatConvTranspose1D, (2, 5, 2)),
        (YatConvTranspose2D, (2, 5, 4, 2)),
        (YatConvTranspose3D, (2, 4, 4, 3, 2)),
    ],
)
@pytest.mark.parametrize(
    ("padding", "stride_value", "dilation_value"),
    [("valid", 1, 2), ("same", 2, 1)],
)
def test_dilation_aware_output_shape_matches_runtime(
    layer_cls, input_shape, padding, stride_value, dilation_value
):
    rank = len(input_shape) - 2
    layer = layer_cls(
        3,
        (3,) * rank,
        padding=padding,
        strides=(stride_value,) * rank,
        dilation_rate=(dilation_value,) * rank,
        use_bias=False,
    )
    x = jnp.ones(input_shape, dtype=jnp.float32)

    assert tuple(layer(x).shape) == tuple(layer.compute_output_shape(input_shape))

    unknown = (None,) + (None,) * rank + (input_shape[-1],)
    expected = (None,) + (None,) * rank + (3,)
    assert tuple(layer.compute_output_shape(unknown)) == expected


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (1, 5, 2)),
        (YatConv2D, (1, 5, 5, 2)),
        (YatConv3D, (1, 5, 5, 5, 2)),
        (YatConvTranspose1D, (1, 5, 2)),
        (YatConvTranspose2D, (1, 5, 5, 2)),
        (YatConvTranspose3D, (1, 5, 5, 5, 2)),
    ],
)
def test_kernel_bank_expansion_is_rejected_without_mutation(layer_cls, input_shape):
    layer_cls._KERNEL_BANKS.clear()
    rank = len(input_shape) - 2
    kwargs = dict(
        filters=2,
        kernel_size=(1,) * rank,
        tie_kernel_bank=True,
        kernel_bank_size=3,
        kernel_bank_id=f"regression-{layer_cls.__name__}",
        kernel_initializer="ones",
    )
    first = layer_cls(**kwargs)
    first(jnp.ones(input_shape))
    before = np.asarray(first.kernel)

    compatible = layer_cls(**{**kwargs, "filters": 1})
    compatible(jnp.ones(input_shape))
    assert compatible.kernel is first.kernel
    assert any(variable is first.kernel for variable in compatible.trainable_weights)

    too_large = layer_cls(**{**kwargs, "filters": 4, "kernel_bank_size": 4})
    with pytest.raises(ValueError, match="cannot be expanded in place"):
        too_large(jnp.ones(input_shape))

    np.testing.assert_array_equal(np.asarray(first.kernel), before)
    assert first.kernel.shape == before.shape


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_exact_matches_are_finite_with_finite_gradients(dtype):
    numeric_dtype = getattr(jnp, dtype)
    query = jnp.asarray([[[0.5], [-0.25], [0.75]]], dtype=numeric_dtype)
    kernel = jnp.reshape(query, (3, 1, 1))
    conv = YatConv1D(
        1,
        3,
        use_bias=False,
        use_alpha=False,
        dtype=dtype,
        kernel_initializer="zeros",
    )
    conv(query)
    conv.kernel.assign(kernel)

    conv_output = conv(query)
    conv_grad = jax.grad(lambda value: jnp.sum(conv(value)))(query)

    embed = YatEmbed(
        1,
        3,
        use_alpha=False,
        dtype=dtype,
        embedding_initializer="zeros",
    )
    embed(jnp.array([0]))
    embed.embedding.assign(jnp.reshape(query, (1, 3)))
    embed_query = jnp.reshape(query, (1, 3))
    embed_output = embed.attend(embed_query)
    embed_grad = jax.grad(lambda value: jnp.sum(embed.attend(value)))(embed_query)

    for value in (conv_output, conv_grad, embed_output, embed_grad):
        array = np.asarray(value)
        assert np.all(np.isfinite(array))
    assert np.all(np.asarray(conv_output) >= 0)
    assert np.all(np.asarray(embed_output) >= 0)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [("float16", 3e-2, 3e-2), ("bfloat16", 8e-2, 8e-2)],
)
def test_low_precision_conv_and_embedding_track_fp32_off_collision(dtype, rtol, atol):
    x32 = jnp.asarray([[[0.2], [-0.3], [0.4], [0.1]]], dtype=jnp.float32)
    kernel32 = jnp.asarray([[[0.35]], [[-0.1]]], dtype=jnp.float32)

    def make_conv(policy):
        layer = YatConv1D(
            1,
            2,
            use_bias=False,
            use_alpha=False,
            dtype=policy,
            kernel_initializer="zeros",
        )
        layer(ops_cast(x32, policy))
        layer.kernel.assign(ops_cast(kernel32, policy))
        return layer

    conv32 = make_conv("float32")
    conv_low = make_conv(dtype)
    np.testing.assert_allclose(
        np.asarray(conv_low(ops_cast(x32, dtype))),
        np.asarray(conv32(x32)),
        rtol=rtol,
        atol=atol,
    )

    embedding32 = jnp.asarray([[0.2, -0.1, 0.3], [-0.4, 0.25, 0.1]])
    query32 = jnp.asarray([[0.15, 0.35, -0.2]])

    def make_embed(policy):
        layer = YatEmbed(
            2,
            3,
            use_alpha=False,
            dtype=policy,
            embedding_initializer="zeros",
        )
        layer(jnp.array([0]))
        layer.embedding.assign(ops_cast(embedding32, policy))
        return layer

    embed32 = make_embed("float32")
    embed_low = make_embed(dtype)
    np.testing.assert_allclose(
        np.asarray(embed_low.attend(ops_cast(query32, dtype))),
        np.asarray(embed32.attend(query32)),
        rtol=rtol,
        atol=atol,
    )


def ops_cast(value, dtype):
    """Cast through JAX without importing backend-specific Keras internals."""
    return jnp.asarray(value, dtype=getattr(jnp, dtype))


@pytest.mark.parametrize("layer", [YatEmbed(8, 4), MultiHeadYatAttention(4, 2)])
def test_registered_layers_round_trip_without_custom_objects(layer):
    serialized = keras.saving.serialize_keras_object(layer)
    restored = keras.saving.deserialize_keras_object(serialized)

    assert type(restored) is type(layer)
    assert restored.get_config() == layer.get_config()
    assert serialized["registered_name"].startswith("nmn>")


def test_registered_layers_clone_and_full_model_round_trip(tmp_path):
    inputs = keras.Input((3,), dtype="int32")
    embedded = YatEmbed(
        8,
        4,
        constant_alpha=True,
        dtype="float32",
        name="yat_embed",
    )(inputs)
    outputs = MultiHeadYatAttention(
        4,
        2,
        constant_alpha=1.25,
        normalize_qk=True,
        name="yat_attention",
    )(embedded)
    model = keras.Model(inputs, outputs)
    sample = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    reference = np.asarray(model(sample))

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    np.testing.assert_allclose(np.asarray(clone(sample)), reference, rtol=1e-6)

    path = tmp_path / "registered.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_allclose(np.asarray(restored(sample)), reference, rtol=1e-6)
    assert restored.get_layer("yat_embed").constant_alpha is True
    assert restored.get_layer("yat_embed").dtype_policy.name == "float32"
    assert restored.get_layer("yat_attention").constant_alpha == 1.25


def test_keras_extra_declares_keras3_without_tensorflow():
    metadata = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text()
    )["project"]["optional-dependencies"]

    assert metadata["keras"] == ["keras>=3.0.0"]
    assert all("tensorflow" not in requirement.lower() for requirement in metadata["keras"])
    assert any("tensorflow" in requirement.lower() for requirement in metadata["tf"])
