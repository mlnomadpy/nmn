"""Regression tests for low-precision Keras reduction boundaries."""

from __future__ import annotations

import keras
from keras import ops
import numpy as np
import pytest

from nmn.keras import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    YatNMN,
    yat_attention,
)
from nmn.keras._yat_core import stable_yat_ratio


BACKEND = keras.backend.backend()
SUPPORTED_GRADIENT_BACKENDS = {"jax", "torch", "tensorflow"}
CONV_CLASSES = (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)


def to_numpy(value):
    return ops.convert_to_numpy(value)


def rank_for(layer_cls):
    if "1D" in layer_cls.__name__:
        return 1
    if "2D" in layer_cls.__name__:
        return 2
    return 3


def is_transpose(layer_cls):
    return "Transpose" in layer_cls.__name__


def make_conv(layer_cls, dtype, epsilon, value, kernel_value):
    rank = rank_for(layer_cls)
    shape = (2,) + (3,) * rank + (2,)
    inputs = ops.full(shape, value, dtype=dtype)
    layer = layer_cls(
        3,
        (1,) * rank,
        use_bias=False,
        use_alpha=False,
        epsilon=epsilon,
        learnable_epsilon=True,
        kernel_initializer="zeros",
        dtype=dtype,
    )
    layer(inputs)
    if is_transpose(layer_cls):
        kernel_shape = (1,) * rank + (3, 2)
    else:
        kernel_shape = (1,) * rank + (2, 3)
    layer.kernel.assign(ops.full(kernel_shape, kernel_value, dtype=dtype))
    return layer, inputs


def keras_layer_value_and_gradients(layer, inputs):
    """Return output plus input/kernel/epsilon gradients on JAX or Torch."""
    if BACKEND == "jax":
        import jax

        variables = list(layer.trainable_variables)
        values = [variable.value for variable in variables]
        kernel_index = next(
            i for i, variable in enumerate(variables) if variable is layer.kernel
        )
        epsilon_index = next(
            i for i, variable in enumerate(variables) if variable is layer.epsilon_param
        )

        def evaluate(input_value, kernel_value, epsilon_value):
            current = list(values)
            current[kernel_index] = kernel_value
            current[epsilon_index] = epsilon_value
            output, _ = layer.stateless_call(
                current, layer.non_trainable_variables, input_value
            )
            return ops.sum(output), output

        (_, output), gradients = jax.value_and_grad(
            evaluate, argnums=(0, 1, 2), has_aux=True
        )(inputs, values[kernel_index], values[epsilon_index])
        return output, gradients

    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        input_value = inputs.detach().clone().requires_grad_(True)
        output = layer(input_value)
        gradients = torch.autograd.grad(
            output.sum(),
            (input_value, layer.kernel.value, layer.epsilon_param.value),
        )
        return output, gradients

    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        input_value = tf.Variable(inputs)
        with tf.GradientTape() as tape:
            output = layer(input_value)
            loss = ops.sum(output)
        gradients = tape.gradient(
            loss, (input_value, layer.kernel.value, layer.epsilon_param.value)
        )
        return output, gradients

    pytest.skip("gradient regression requires a supported Keras autodiff backend")


def keras_embed_value_and_gradients(layer, query):
    if BACKEND == "jax":
        import jax

        variables = list(layer.trainable_variables)
        values = [variable.value for variable in variables]
        embedding_index = next(
            i for i, variable in enumerate(variables) if variable is layer.embedding
        )

        def evaluate(query_value, embedding_value):
            current = list(values)
            current[embedding_index] = embedding_value
            state_mapping = list(zip(variables, current))
            with keras.StatelessScope(state_mapping):
                output = layer.attend(query_value)
            return ops.sum(output), output

        (_, output), gradients = jax.value_and_grad(
            evaluate, argnums=(0, 1), has_aux=True
        )(query, values[embedding_index])
        return output, gradients

    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        query_value = query.detach().clone().requires_grad_(True)
        output = layer.attend(query_value)
        gradients = torch.autograd.grad(
            output.sum(), (query_value, layer.embedding.value)
        )
        return output, gradients

    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        query_value = tf.Variable(query)
        with tf.GradientTape() as tape:
            output = layer.attend(query_value)
            loss = ops.sum(output)
        gradients = tape.gradient(loss, (query_value, layer.embedding.value))
        return output, gradients

    pytest.skip("gradient regression requires a supported Keras autodiff backend")


def keras_dense_value_and_gradients(layer, inputs):
    if BACKEND == "jax":
        import jax

        variables = list(layer.trainable_variables)
        values = [variable.value for variable in variables]
        kernel_index = next(
            i for i, variable in enumerate(variables) if variable is layer.kernel
        )

        def evaluate(input_value, kernel_value):
            current = list(values)
            current[kernel_index] = kernel_value
            output, _ = layer.stateless_call(
                current, layer.non_trainable_variables, input_value
            )
            return ops.sum(ops.cast(output, "float32")), output

        (_, output), gradients = jax.value_and_grad(
            evaluate, argnums=(0, 1), has_aux=True
        )(inputs, values[kernel_index])
        return output, gradients

    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        input_value = inputs.detach().clone().requires_grad_(True)
        output = layer(input_value)
        gradients = torch.autograd.grad(
            ops.sum(ops.cast(output, "float32")),
            (input_value, layer.kernel.value),
        )
        return output, gradients

    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        input_value = tf.Variable(inputs)
        with tf.GradientTape() as tape:
            output = layer(input_value)
            loss = tf.reduce_sum(tf.cast(output, tf.float32))
        gradients = tape.gradient(loss, (input_value, layer.kernel.value))
        return output, gradients

    pytest.skip("gradient regression requires a supported Keras autodiff backend")


def keras_attention_value_and_gradients(dtype):
    query = ops.full((1, 1, 1, 2), 100.0, dtype=dtype)
    key = ops.full((1, 2, 1, 2), 100.0, dtype=dtype)
    value = ops.convert_to_tensor([[[[1.0]], [[2.0]]]], dtype=dtype)

    def attend(q, k, v):
        return yat_attention(q, k, v, training=False, epsilon=1.0)

    if BACKEND == "jax":
        import jax

        def loss(q, k, v):
            output = attend(q, k, v)
            return ops.sum(ops.cast(output, "float32")) * 0.015625

        output = attend(query, key, value)
        gradients = jax.grad(loss, argnums=(0, 1, 2))(query, key, value)
        return output, gradients

    if BACKEND == "torch":
        query = query.detach().clone().requires_grad_(True)
        key = key.detach().clone().requires_grad_(True)
        value = value.detach().clone().requires_grad_(True)
        output = attend(query, key, value)
        gradients = pytest.importorskip("torch").autograd.grad(
            ops.sum(ops.cast(output, "float32")) * 0.015625,
            (query, key, value),
        )
        return output, gradients

    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        query, key, value = tf.Variable(query), tf.Variable(key), tf.Variable(value)
        with tf.GradientTape() as tape:
            output = attend(query, key, value)
            loss = tf.reduce_sum(tf.cast(output, tf.float32)) * 0.015625
        return output, tape.gradient(loss, (query, key, value))

    pytest.skip("gradient regression requires a supported Keras autodiff backend")


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
@pytest.mark.parametrize("layer_cls", CONV_CLASSES)
def test_fp16_multi_output_exact_collision_conv_gradients_are_finite(layer_cls):
    layer, inputs = make_conv(layer_cls, "float16", 1e-5, 0.5, 0.5)
    output, gradients = keras_layer_value_and_gradients(layer, inputs)

    assert keras.backend.standardize_dtype(output.dtype) == "float16"
    assert to_numpy(output).shape[-1] == 3
    assert np.all(np.isfinite(to_numpy(output)))
    for gradient in gradients:
        assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
def test_fp16_multi_row_multi_embedding_exact_collision_gradients_are_finite():
    query = ops.full((3, 4), 0.25, dtype="float16")
    layer = YatEmbed(
        5,
        4,
        use_alpha=False,
        epsilon=1e-5,
        embedding_initializer="zeros",
        dtype="float16",
    )
    layer(ops.convert_to_tensor([0], dtype="int32"))
    layer.embedding.assign(ops.full((5, 4), 0.25, dtype="float16"))

    output, gradients = keras_embed_value_and_gradients(layer, query)

    assert keras.backend.standardize_dtype(output.dtype) == "float16"
    assert output.shape == (3, 5)
    assert np.all(np.isfinite(to_numpy(output)))
    for gradient in gradients:
        assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
@pytest.mark.parametrize("layer_cls", CONV_CLASSES)
def test_fp16_conv_off_collision_matches_fp32_forward_and_gradients(layer_cls):
    reference, reference_inputs = make_conv(layer_cls, "float32", 0.1, 0.2, 0.35)
    lowp, lowp_inputs = make_conv(layer_cls, "float16", 0.1, 0.2, 0.35)

    reference_output, reference_gradients = keras_layer_value_and_gradients(
        reference, reference_inputs
    )
    lowp_output, lowp_gradients = keras_layer_value_and_gradients(lowp, lowp_inputs)

    np.testing.assert_allclose(
        to_numpy(lowp_output), to_numpy(reference_output), rtol=3e-2, atol=3e-2
    )
    for lowp_gradient, reference_gradient in zip(lowp_gradients, reference_gradients):
        np.testing.assert_allclose(
            to_numpy(lowp_gradient),
            to_numpy(reference_gradient),
            rtol=5e-2,
            atol=5e-2,
        )


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
def test_fp16_embedding_off_collision_matches_fp32_forward_and_gradients():
    def make(dtype):
        query = ops.full((3, 4), 0.2, dtype=dtype)
        layer = YatEmbed(
            5,
            4,
            use_alpha=False,
            epsilon=0.1,
            embedding_initializer="zeros",
            dtype=dtype,
        )
        layer(ops.convert_to_tensor([0], dtype="int32"))
        layer.embedding.assign(ops.full((5, 4), 0.35, dtype=dtype))
        return layer, query

    reference, reference_query = make("float32")
    lowp, lowp_query = make("float16")
    reference_output, reference_gradients = keras_embed_value_and_gradients(
        reference, reference_query
    )
    lowp_output, lowp_gradients = keras_embed_value_and_gradients(lowp, lowp_query)

    np.testing.assert_allclose(
        to_numpy(lowp_output), to_numpy(reference_output), rtol=3e-2, atol=3e-2
    )
    for lowp_gradient, reference_gradient in zip(lowp_gradients, reference_gradients):
        np.testing.assert_allclose(
            to_numpy(lowp_gradient),
            to_numpy(reference_gradient),
            rtol=5e-2,
            atol=5e-2,
        )


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
def test_low_precision_ratio_preserves_nan():
    initial = ops.convert_to_tensor([[np.nan, 0.5]], dtype="float16")

    if BACKEND == "jax":
        import jax

        def evaluate(dot):
            return ops.sum(stable_yat_ratio(dot, ops.zeros_like(dot), 1e-5))

        output = stable_yat_ratio(initial, ops.zeros_like(initial), 1e-5)
        gradient = jax.grad(evaluate)(initial)
    elif BACKEND == "torch":
        torch = pytest.importorskip("torch")
        dot = initial.detach().clone().requires_grad_(True)
        output = stable_yat_ratio(dot, ops.zeros_like(dot), 1e-5)
        (gradient,) = torch.autograd.grad(output.sum(), (dot,))
    else:
        tf = pytest.importorskip("tensorflow")
        dot = tf.Variable(initial)
        with tf.GradientTape() as tape:
            output = stable_yat_ratio(dot, ops.zeros_like(dot), 1e-5)
            loss = ops.sum(output)
        gradient = tape.gradient(loss, dot)

    assert np.isnan(to_numpy(output)[0, 0])
    assert np.isfinite(to_numpy(output)[0, 1])
    assert np.isnan(to_numpy(gradient)[0, 0])


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_magnitude_dense_matches_fp32_forward_and_gradients(dtype):
    def make(layer_dtype):
        inputs = ops.full((1, 2), 100.0, dtype=layer_dtype)
        layer = YatNMN(
            1, use_bias=False, use_alpha=False, epsilon=1.0,
            kernel_initializer="zeros", dtype=layer_dtype,
        )
        layer(inputs)
        layer.kernel.assign(
            ops.convert_to_tensor([[-100.0], [-99.0]], dtype=layer_dtype)
        )
        return layer, inputs

    reference, reference_inputs = make("float32")
    lowp, lowp_inputs = make(dtype)
    reference_output, reference_gradients = keras_dense_value_and_gradients(
        reference, reference_inputs
    )
    output, gradients = keras_dense_value_and_gradients(lowp, lowp_inputs)
    np.testing.assert_allclose(
        to_numpy(ops.cast(output, "float32")), to_numpy(reference_output),
        rtol=5e-3, atol=0.15,
    )
    for actual, expected in zip(gradients, reference_gradients):
        np.testing.assert_allclose(
            to_numpy(ops.cast(actual, "float32")), to_numpy(expected),
            rtol=5e-3, atol=0.15,
        )


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
def test_fp16_dense_aggregate_cotangents_match_saturated_fp32_reference():
    def make(layer_dtype):
        inputs = ops.full((4096, 2), 100.0, dtype=layer_dtype)
        layer = YatNMN(
            1, use_bias=False, use_alpha=False, epsilon=1.0,
            kernel_initializer="zeros", dtype=layer_dtype,
        )
        layer(inputs)
        layer.kernel.assign(
            ops.convert_to_tensor([[-100.0], [-99.0]], dtype=layer_dtype)
        )
        return layer, inputs

    reference, reference_inputs = make("float32")
    lowp, lowp_inputs = make("float16")
    reference_output, reference_gradients = keras_dense_value_and_gradients(
        reference, reference_inputs
    )
    output, gradients = keras_dense_value_and_gradients(lowp, lowp_inputs)
    np.testing.assert_allclose(
        to_numpy(ops.cast(output, "float32")), to_numpy(reference_output),
        rtol=5e-3, atol=2.0,
    )
    limit = np.finfo(np.float16)
    for actual, expected in zip(gradients, reference_gradients):
        clipped = np.clip(to_numpy(expected), limit.min, limit.max).astype(np.float16)
        assert np.all(np.isfinite(to_numpy(actual)))
        np.testing.assert_allclose(to_numpy(actual), clipped, rtol=5e-3, atol=8.0)


@pytest.mark.skipif(
    BACKEND not in SUPPORTED_GRADIENT_BACKENDS,
    reason="requires a supported Keras autodiff backend",
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_magnitude_attention_matches_fp32_forward_and_gradients(dtype):
    ref_output, ref_gradients = keras_attention_value_and_gradients("float32")
    output, gradients = keras_attention_value_and_gradients(dtype)
    np.testing.assert_allclose(
        to_numpy(ops.cast(output, "float32")), to_numpy(ref_output), atol=2e-3
    )
    for actual, expected in zip(gradients, ref_gradients):
        np.testing.assert_allclose(
            to_numpy(ops.cast(actual, "float32")), to_numpy(expected),
            rtol=7e-3, atol=8.0,
        )


def test_dense_and_attention_preserve_genuine_nan():
    inputs = ops.convert_to_tensor([[np.nan, 1.0]], dtype="float16")
    layer = YatNMN(
        1, use_bias=False, use_alpha=False, kernel_initializer="ones",
        dtype="float16",
    )
    assert np.isnan(to_numpy(layer(inputs))).all()

    query = ops.convert_to_tensor([[[[np.nan, 100.0]]]], dtype="float16")
    key = ops.full((1, 2, 1, 2), 100.0, dtype="float16")
    value = ops.ones((1, 2, 1, 1), dtype="float16")
    assert np.isnan(to_numpy(yat_attention(query, key, value))).all()
