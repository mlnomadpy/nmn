"""Regression coverage for Keras issues #55, #56, #73-#76 and #78."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

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
    yat_attention,
    yat_attention_weights,
)
from nmn.keras._yat_core import stable_yat_ratio

BACKEND = keras.backend.backend()


def _attention_value_and_gradients(query, key, value, mask):
    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        apply = jax.jit(lambda q, k, v: yat_attention(q, k, v, mask=mask))
        output = apply(query, key, value)
        grads = jax.grad(lambda q, k, v: jnp.sum(apply(q, k, v)), (0, 1, 2))(
            query, key, value
        )
        return output, grads
    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        apply = torch.compile(
            lambda q, k, v: yat_attention(q, k, v, mask=mask), backend="eager"
        )
        output = apply(query, key, value)
        return output, torch.autograd.grad(output.sum(), (query, key, value))
    tf = pytest.importorskip("tensorflow")
    apply = tf.function(lambda q, k, v: yat_attention(q, k, v, mask=mask))
    with tf.GradientTape() as tape:
        tape.watch((query, key, value))
        output = apply(query, key, value)
        loss = tf.reduce_sum(output)
    return output, tape.gradient(loss, (query, key, value))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_fully_masked_attention_is_zero_differentiable_and_backend_stable(dtype):
    rng = np.random.default_rng(70)
    query = tensor(rng.normal(size=(1, 2, 2, 4)).astype(np.float32), dtype=dtype)
    key = tensor(rng.normal(size=(1, 3, 2, 4)).astype(np.float32), dtype=dtype)
    value = tensor(rng.normal(size=(1, 3, 2, 5)).astype(np.float32), dtype=dtype)
    mask_np = np.array([[[[False, False, False], [True, False, True]]]])
    mask = tensor(mask_np, dtype="bool")

    output, grads = _attention_value_and_gradients(query, key, value, mask)
    weights = to_numpy(yat_attention_weights(query, key, mask=mask))

    np.testing.assert_array_equal(to_numpy(output)[:, 0], 0.0)
    np.testing.assert_array_equal(weights[..., 0, :], 0.0)
    assert all(np.all(np.isfinite(to_numpy(grad))) for grad in grads)

    # Independent NumPy oracle for the partially masked row.  Fixed operands
    # make this a parity contract shared by the JAX/Torch/TF Keras CI jobs.
    q = to_numpy(query)
    k = to_numpy(key)
    dot = np.einsum("bqhd,bkhd->bhqk", q, k)
    q_sq = np.sum(q * q, axis=-1).transpose(0, 2, 1)[..., None]
    k_sq = np.sum(k * k, axis=-1).transpose(0, 2, 1)[:, :, None, :]
    scores = dot**2 / ((np.maximum(q_sq + k_sq - 2.0 * dot, 0.0) + 1e-5) * 2.0)
    scores = np.where(mask_np, scores, -np.inf)
    row = scores[..., 1, :]
    expected = np.exp(row - np.max(row, axis=-1, keepdims=True))
    expected = np.where(mask_np[..., 1, :], expected, 0.0)
    expected /= np.sum(expected, axis=-1, keepdims=True)
    tolerance = 5e-3 if dtype == "float16" else 2e-5
    np.testing.assert_allclose(
        weights[..., 1, :], expected, rtol=tolerance, atol=tolerance
    )


def test_negative_scale_cannot_make_masked_key_win_softmax():
    query = tensor(np.ones((1, 1, 1, 4), dtype=np.float32))
    key = tensor(np.ones((1, 2, 1, 4), dtype=np.float32))
    mask = tensor([True, False], dtype="bool")
    weights = to_numpy(yat_attention_weights(query, key, mask=mask, scale=-1.0))
    np.testing.assert_array_equal(weights, [[[[1.0, 0.0]]]])


@pytest.mark.parametrize("mask_rank", [2, 4])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_attention_layer_zeroes_fully_masked_rows_after_projection(
    cross_attention, mask_rank
):
    layer = MultiHeadYatAttention(embed_dim=8, num_heads=2)
    query = keras.random.normal((1, 2, 8), seed=73)
    context = keras.random.normal((1, 3, 8), seed=74)
    _ = layer(query)
    layer.out_bias.assign(np.full(layer.out_bias.shape, 3.0, dtype=np.float32))
    kv_length = 3 if cross_attention else 2
    shape = (2, kv_length) if mask_rank == 2 else (1, 1, 2, kv_length)
    mask_np = np.ones(shape, dtype=bool)
    mask_np[..., 0, :] = False
    mask = tensor(mask_np, dtype="bool")
    output = (
        layer(query, key=context, value=context, attention_mask=mask)
        if cross_attention
        else layer(query, attention_mask=mask)
    )
    np.testing.assert_array_equal(to_numpy(output)[:, 0], 0.0)
    assert np.all(np.isfinite(to_numpy(output)))


def tensor(value, dtype="float32"):
    return keras.ops.convert_to_tensor(np.asarray(value), dtype=dtype)


def to_numpy(value, dtype=None):
    array = keras.ops.convert_to_numpy(value)
    return np.asarray(array, dtype=dtype) if dtype is not None else array


def input_gradient(layer, value):
    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        return jax.grad(lambda x: jnp.sum(layer(x)))(value)
    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        with tf.GradientTape() as tape:
            tape.watch(value)
            loss = tf.reduce_sum(layer(value))
        return tape.gradient(loss, value)
    pytest.skip("gradient assertion is implemented for JAX and TensorFlow backends")


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
    x = keras.random.normal(input_shape, seed=len(input_shape))
    layer = layer_cls(4, 3, groups=2, padding="same", use_bias=False)

    y = layer(x)
    dx = input_gradient(layer, x)

    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == 4
    assert np.all(np.isfinite(to_numpy(y)))
    assert np.all(np.isfinite(to_numpy(dx)))


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
    x_array = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1)
    changed_array = x_array.copy()
    changed_array[:, 5:, :] = 10_000.0
    x = tensor(x_array)
    changed_future = tensor(changed_array)

    y = layer(x)
    changed_y = layer(changed_future)

    assert y.shape == (1, 8, 1)
    np.testing.assert_allclose(to_numpy(y[:, :5]), to_numpy(changed_y[:, :5]))
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
    x = keras.ops.ones(input_shape, dtype="float32")

    is_transpose = layer_cls in (
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    )
    effective_kernel = dilation_value * 2 + 1
    if is_transpose:
        expected_spatial = tuple(
            (
                size * stride_value
                if padding == "same"
                else (size - 1) * stride_value + effective_kernel
            )
            for size in input_shape[1:-1]
        )
    else:
        expected_spatial = tuple(
            (
                (size + stride_value - 1) // stride_value
                if padding == "same"
                else (size - effective_kernel) // stride_value + 1
            )
            for size in input_shape[1:-1]
        )
    computed = tuple(layer.compute_output_shape(input_shape))
    assert computed == (input_shape[0], *expected_spatial, 3)

    # TensorFlow CPU does not implement dilated transposed convolution.  The
    # canonical shape formula above remains backend-neutral; runtime parity is
    # exercised for this case by the JAX and Torch clean-environment jobs.
    tensorflow_cpu_limitation = (
        BACKEND == "tensorflow" and is_transpose and dilation_value > 1
    )
    if not tensorflow_cpu_limitation:
        assert tuple(layer(x).shape) == computed

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
    first(keras.ops.ones(input_shape))
    before = to_numpy(first.kernel)

    compatible = layer_cls(**{**kwargs, "filters": 1})
    compatible(keras.ops.ones(input_shape))
    assert compatible.kernel is first.kernel
    assert any(variable is first.kernel for variable in compatible.trainable_weights)

    too_large = layer_cls(**{**kwargs, "filters": 4, "kernel_bank_size": 4})
    with pytest.raises(ValueError, match="cannot be expanded in place"):
        too_large(keras.ops.ones(input_shape))

    np.testing.assert_array_equal(to_numpy(first.kernel), before)
    assert first.kernel.shape == before.shape


@pytest.mark.parametrize(
    "layer_cls",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
def test_kernel_bank_capacity_smaller_than_filters_is_rejected_before_state(layer_cls):
    layer_cls._KERNEL_BANKS.clear()
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3

    with pytest.raises(ValueError, match="must be greater than or equal to filters"):
        layer_cls(
            3,
            (1,) * rank,
            tie_kernel_bank=True,
            kernel_bank_size=2,
            kernel_bank_id="invalid-capacity",
        )

    assert not layer_cls._KERNEL_BANKS


def test_tied_kernel_bank_functional_save_load_preserves_sharing_and_optimizer(
    tmp_path,
):
    YatConv1D._KERNEL_BANKS.clear()
    inputs = keras.Input((5, 1))
    common = dict(
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=3,
        kernel_bank_id="functional-round-trip",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    first_layer = YatConv1D(2, name="bank_first", **common)
    second_layer = YatConv1D(1, name="bank_second", **common)
    outputs = keras.layers.Concatenate()([first_layer(inputs), second_layer(inputs)])
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    sample = tensor(np.arange(5, dtype=np.float32).reshape(1, 5, 1))
    target = keras.ops.zeros((1, 5, 3))
    model.train_on_batch(sample, target)
    reference = to_numpy(model(sample))

    assert first_layer.kernel is second_layer.kernel
    assert len(first_layer.trainable_weights) == 1
    assert len(second_layer.trainable_weights) == 1
    assert len(model.trainable_variables) == 1
    iterations = int(to_numpy(model.optimizer.iterations))

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    assert clone.get_layer("bank_first").kernel is clone.get_layer("bank_second").kernel
    np.testing.assert_allclose(to_numpy(clone(sample)), reference, rtol=1e-6)

    path = tmp_path / "tied-bank.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    restored_first = restored.get_layer("bank_first")
    restored_second = restored.get_layer("bank_second")

    assert restored_first.kernel is restored_second.kernel
    assert len(restored_first.trainable_weights) == 1
    assert len(restored_second.trainable_weights) == 1
    assert len(restored.trainable_variables) == 1
    assert int(to_numpy(restored.optimizer.iterations)) == iterations
    np.testing.assert_allclose(to_numpy(restored(sample)), reference, rtol=1e-6)
    np.testing.assert_allclose(
        to_numpy(restored_first.kernel), to_numpy(first_layer.kernel), rtol=1e-6
    )


def test_tied_kernel_bank_creation_is_atomic_across_threads():
    YatConv1D._KERNEL_BANKS.clear()
    start = threading.Barrier(2)
    common = dict(
        filters=2,
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=2,
        kernel_bank_id="threaded-first-creation",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    layers = [YatConv1D(**common), YatConv1D(**common)]

    def build(layer):
        start.wait(timeout=5)
        layer.build((None, 4, 1))
        return layer.kernel

    with ThreadPoolExecutor(max_workers=2) as executor:
        kernels = list(executor.map(build, layers))

    assert kernels[0] is kernels[1]
    assert layers[0]._kernel_bank_ref is layers[1]._kernel_bank_ref
    assert all(len(layer.trainable_weights) == 1 for layer in layers)


@pytest.mark.parametrize("dtype", ["float16", "mixed_float16"])
def test_tied_kernel_banks_are_separated_by_effective_dtype_policy(dtype):
    YatConv1D._KERNEL_BANKS.clear()
    common = dict(
        filters=1,
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=1,
        kernel_bank_id=f"dtype-policy-{dtype}",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    float32_layer = YatConv1D(dtype="float32", **common)
    low_precision_layer = YatConv1D(dtype=dtype, **common)

    float32_output = float32_layer(tensor([[[0.25]]], "float32"))
    low_precision_output = low_precision_layer(tensor([[[0.25]]], "float16"))

    assert float32_layer.kernel is not low_precision_layer.kernel
    assert float32_layer._kernel_bank_ref is not low_precision_layer._kernel_bank_ref
    assert keras.backend.standardize_dtype(float32_output.dtype) == "float32"
    assert keras.backend.standardize_dtype(low_precision_output.dtype) == "float16"
    assert keras.backend.standardize_dtype(float32_layer.kernel.dtype) == "float32"
    expected_variable_dtype = "float32" if dtype == "mixed_float16" else "float16"
    assert (
        keras.backend.standardize_dtype(low_precision_layer.kernel.dtype)
        == expected_variable_dtype
    )


@pytest.mark.parametrize(
    "layer_cls",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_exact_matches_are_finite_for_every_conv_family(layer_cls, dtype):
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3
    input_shape = (1,) + (1,) * rank + (1,)
    value = tensor(np.full(input_shape, 0.5), dtype)
    layer = layer_cls(
        1,
        (1,) * rank,
        use_bias=False,
        use_alpha=False,
        dtype=dtype,
    )

    # The default orthogonal initializer must build on JAX low-precision
    # policies without dispatching unsupported float16/bfloat16 LAPACK.
    layer(value)
    layer.kernel.assign(tensor(np.full(layer.kernel.shape, 0.5), dtype))
    output = layer(value)
    gradient = input_gradient(layer, value)

    assert keras.backend.standardize_dtype(output.dtype) == dtype
    assert np.all(np.isfinite(to_numpy(output)))
    assert np.all(to_numpy(output) >= 0)
    assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_embedding_exact_match_preserves_policy_and_gradients(dtype):
    query = tensor([[0.5, -0.25, 0.75]], dtype)
    embed = YatEmbed(
        1,
        3,
        use_alpha=False,
        dtype=dtype,
        embedding_initializer="zeros",
    )
    embed(keras.ops.convert_to_tensor([0], dtype="int32"))
    embed.embedding.assign(query)
    output = embed.attend(query)

    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        gradient = jax.grad(lambda x: jnp.sum(embed.attend(x)))(query)
    elif BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        with tf.GradientTape() as tape:
            tape.watch(query)
            loss = tf.reduce_sum(embed.attend(query))
        gradient = tape.gradient(loss, query)
    else:
        pytest.skip("gradient assertion is implemented for JAX and TensorFlow")

    assert keras.backend.standardize_dtype(output.dtype) == dtype
    assert np.all(np.isfinite(to_numpy(output)))
    assert np.all(to_numpy(output) >= 0)
    assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.skipif(BACKEND != "jax", reason="JAX cotangent regression")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_embed_shaped_exact_collision_reduces_epsilon_gradient_before_clipping(dtype):
    import jax
    import jax.numpy as jnp

    dot = jnp.full((2, 3), 0.5, dtype=getattr(jnp, dtype))
    distance = jnp.zeros_like(dot)
    # YatEmbed passes its configured epsilon as a scalar into this helper.
    epsilon = jnp.asarray(1e-5, dtype=getattr(jnp, dtype))

    gradient = jax.grad(lambda eps: jnp.sum(stable_yat_ratio(dot, distance, eps)))(
        epsilon
    )
    gradient32 = to_numpy(gradient, dtype=np.float32)

    assert gradient.shape == epsilon.shape
    assert np.all(np.isfinite(gradient32))
    assert np.all(gradient32 < 0)
    if dtype == "float16":
        np.testing.assert_array_equal(gradient32, np.asarray(-65504.0))
    else:
        epsilon32 = to_numpy(epsilon, dtype=np.float32)
        expected = -dot.size * 0.25 / np.square(epsilon32)
        np.testing.assert_allclose(gradient32, expected, rtol=2e-2)


@pytest.mark.skipif(BACKEND != "jax", reason="JAX cotangent regression")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_conv_exact_collision_has_finite_learnable_epsilon_gradient(dtype):
    import jax
    import jax.numpy as jnp

    value = jnp.full((1, 4, 1), 0.5, dtype=getattr(jnp, dtype))
    layer = YatConv1D(
        1,
        1,
        use_bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        kernel_initializer="zeros",
        dtype=dtype,
    )
    layer(value)
    layer.kernel.assign(jnp.full(layer.kernel.shape, 0.5, dtype=getattr(jnp, dtype)))
    trainable_values = [variable.value for variable in layer.trainable_variables]
    epsilon_index = next(
        index
        for index, variable in enumerate(layer.trainable_variables)
        if variable is layer.epsilon_param
    )

    def loss(epsilon_param):
        values = list(trainable_values)
        values[epsilon_index] = epsilon_param
        output, _ = layer.stateless_call(values, layer.non_trainable_variables, value)
        return jnp.sum(output)

    gradient = jax.grad(loss)(trainable_values[epsilon_index])

    assert gradient.shape == layer.epsilon_param.shape
    assert np.all(np.isfinite(to_numpy(gradient, dtype=np.float32)))
    assert np.all(to_numpy(gradient, dtype=np.float32) < 0)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [("float16", 3e-2, 3e-2), ("bfloat16", 8e-2, 8e-2)],
)
@pytest.mark.parametrize(
    "layer_cls",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
def test_low_precision_conv_families_track_fp32_off_collision(
    layer_cls, dtype, rtol, atol
):
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3
    input_shape = (1,) + (2,) * rank + (1,)
    x32 = tensor(np.full(input_shape, 0.2))
    kernel_shape = (1,) * rank + (1, 1)
    kernel32 = tensor(np.full(kernel_shape, 0.35))

    def make_conv(policy):
        layer = layer_cls(
            1,
            (1,) * rank,
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
        to_numpy(conv_low(ops_cast(x32, dtype))),
        to_numpy(conv32(x32)),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [("float16", 3e-2, 3e-2), ("bfloat16", 8e-2, 8e-2)],
)
def test_low_precision_embedding_tracks_fp32_off_collision(dtype, rtol, atol):
    embedding32 = tensor([[0.2, -0.1, 0.3], [-0.4, 0.25, 0.1]])
    query32 = tensor([[0.15, 0.35, -0.2]])

    def make_embed(policy):
        layer = YatEmbed(
            2,
            3,
            use_alpha=False,
            dtype=policy,
            embedding_initializer="zeros",
        )
        layer(keras.ops.convert_to_tensor([0], dtype="int32"))
        layer.embedding.assign(ops_cast(embedding32, policy))
        return layer

    embed32 = make_embed("float32")
    embed_low = make_embed(dtype)
    np.testing.assert_allclose(
        to_numpy(embed_low.attend(ops_cast(query32, dtype))),
        to_numpy(embed32.attend(query32)),
        rtol=rtol,
        atol=atol,
    )


def ops_cast(value, dtype):
    return keras.ops.cast(value, dtype)


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
    sample = keras.ops.convert_to_tensor([[0, 1, 2]], dtype="int32")
    reference = to_numpy(model(sample))

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    np.testing.assert_allclose(to_numpy(clone(sample)), reference, rtol=1e-6)

    path = tmp_path / "registered.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_allclose(to_numpy(restored(sample)), reference, rtol=1e-6)
    assert restored.get_layer("yat_embed").constant_alpha is True
    assert restored.get_layer("yat_embed").dtype_policy.name == "float32"
    assert restored.get_layer("yat_attention").constant_alpha == 1.25


def test_keras_extra_declares_keras3_without_tensorflow():
    metadata = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text()
    )["project"]["optional-dependencies"]

    assert metadata["keras"] == ["keras>=3.0.0"]
    assert all(
        "tensorflow" not in requirement.lower() for requirement in metadata["keras"]
    )
    assert any("tensorflow" in requirement.lower() for requirement in metadata["tf"])
