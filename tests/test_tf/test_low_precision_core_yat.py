"""Large-magnitude low-precision regressions for TensorFlow YAT arithmetic."""

import numpy as np
import pytest
import tensorflow as tf

from nmn.tf import (
    YatConv1D, YatConv2D, YatConv3D,
    YatConvTranspose1D, YatConvTranspose2D, YatConvTranspose3D,
    YatNMN, yat_attention,
)
from nmn.tf._precision import reduction_safe_upcast


LOWP_DTYPES = (tf.float16, tf.bfloat16)
CONVS = (
    (YatConv1D, (1, 1, 1), 1),
    (YatConv2D, (1, 1, 1, 1), (1, 1)),
    (YatConv3D, (1, 1, 1, 1, 1), (1, 1, 1)),
    (YatConvTranspose1D, (1, 1, 1), 1),
    (YatConvTranspose2D, (1, 1, 1, 1), (1, 1)),
    (YatConvTranspose3D, (1, 1, 1, 1, 1), (1, 1, 1)),
)


def _as_f32(value):
    return tf.cast(value, tf.float32).numpy()


def _dense_value_and_grads(dtype):
    layer = YatNMN(
        features=1, use_bias=False, use_alpha=False, epsilon=1.0, dtype=dtype
    )
    x = tf.Variable([[100.0, 100.0]], dtype=dtype)
    layer(x)
    layer.kernel.assign(tf.constant([[-100.0, -99.0]], dtype=dtype))
    with tf.GradientTape() as tape:
        output = layer(x)
        loss = tf.reduce_sum(tf.cast(output, tf.float32))
    gradients = tape.gradient(loss, (x, layer.kernel))
    return output, gradients


@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_dense_matches_fp32_forward_and_gradients(dtype):
    ref_output, ref_grads = _dense_value_and_grads(tf.float32)
    output, grads = _dense_value_and_grads(dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), rtol=5e-3, atol=0.15)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=5e-3, atol=0.15)


def _aggregate_dense_grads(dtype, compiled=False):
    layer = YatNMN(features=1, use_bias=True, use_alpha=True, epsilon=1.0, dtype=dtype)
    x = tf.Variable(tf.fill((4096, 2), tf.cast(100.0, dtype)))
    layer(x)
    layer.kernel.assign(tf.constant([[-100.0, -99.0]], dtype=dtype))
    layer.bias.assign(tf.constant([0.5], dtype=dtype))
    layer.alpha.assign(tf.constant([1.25], dtype=dtype))
    def evaluate():
        with tf.GradientTape() as tape:
            output = layer(x)
            loss = tf.reduce_sum(tf.cast(output, tf.float32))
        return output, tape.gradient(loss, (x, layer.kernel, layer.bias, layer.alpha))

    return tf.function(evaluate)() if compiled else evaluate()


@pytest.mark.parametrize("compiled", [False, True])
def test_fp16_dense_aggregate_cotangents_match_saturated_fp32_reference(compiled):
    reference_output, reference_grads = _aggregate_dense_grads(tf.float32, compiled)
    output, grads = _aggregate_dense_grads(tf.float16, compiled)
    limit = np.finfo(np.float16)
    np.testing.assert_allclose(_as_f32(output), _as_f32(reference_output), rtol=5e-3, atol=2.0)
    for actual, expected in zip(grads, reference_grads):
        clipped = np.clip(expected.numpy(), limit.min, limit.max).astype(np.float16)
        assert np.all(np.isfinite(actual.numpy()))
        np.testing.assert_allclose(actual.numpy(), clipped, rtol=5e-3, atol=8.0)


def _conv_value_and_grads(layer_cls, shape, kernel_size, dtype):
    layer = layer_cls(
        filters=1, kernel_size=kernel_size, use_bias=False, use_alpha=False,
        epsilon=1.0, dtype=dtype,
    )
    x = tf.Variable(tf.fill(shape, tf.cast(100.0, dtype)))
    layer(x)
    layer.kernel.assign(tf.fill(layer.kernel.shape, tf.cast(-100.0, dtype)))
    with tf.GradientTape() as tape:
        output = layer(x)
        loss = tf.reduce_sum(tf.cast(output, tf.float32))
    gradients = tape.gradient(loss, (x, layer.kernel))
    return output, gradients


@pytest.mark.parametrize("layer_cls,shape,kernel_size", CONVS)
@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_conv_families_match_fp32(layer_cls, shape, kernel_size, dtype):
    ref_output, ref_grads = _conv_value_and_grads(
        layer_cls, shape, kernel_size, tf.float32
    )
    output, grads = _conv_value_and_grads(layer_cls, shape, kernel_size, dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), rtol=5e-3, atol=0.15)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=5e-3, atol=0.15)


def _attention_value_and_grads(dtype):
    query = tf.Variable(tf.fill((1, 1, 1, 2), tf.cast(100.0, dtype)))
    key = tf.Variable(tf.fill((1, 2, 1, 2), tf.cast(100.0, dtype)))
    value = tf.Variable(tf.constant([[[[1.0]], [[2.0]]]], dtype=dtype))
    with tf.GradientTape() as tape:
        output = yat_attention(query, key, value, training=False, epsilon=1.0)
        loss = tf.reduce_sum(tf.cast(output, tf.float32)) * 0.015625
    return output, tape.gradient(loss, (query, key, value))


@pytest.mark.parametrize("dtype", LOWP_DTYPES)
def test_large_magnitude_attention_matches_fp32_forward_and_gradients(dtype):
    ref_output, ref_grads = _attention_value_and_grads(tf.float32)
    output, grads = _attention_value_and_grads(dtype)
    np.testing.assert_allclose(_as_f32(output), _as_f32(ref_output), atol=2e-3)
    for actual, expected in zip(grads, ref_grads):
        np.testing.assert_allclose(_as_f32(actual), _as_f32(expected), rtol=7e-3, atol=8.0)


def _aggregate_attention_grads(dtype, compiled=False):
    query = tf.Variable(tf.fill((1, 1, 1, 2), tf.cast(100.0, dtype)))
    key = tf.Variable(tf.fill((1, 2, 1, 2), tf.cast(99.0, dtype)))
    value = tf.Variable(tf.constant([[[[0.0]], [[1.0]]]], dtype=dtype))
    def evaluate():
        with tf.GradientTape() as tape:
            output = yat_attention(query, key, value, training=False, epsilon=1.0)
            loss = tf.reduce_sum(tf.cast(output, tf.float32))
        return output, tape.gradient(loss, (query, key, value))

    return tf.function(evaluate)() if compiled else evaluate()


@pytest.mark.parametrize("compiled", [False, True])
def test_fp16_attention_aggregate_cotangents_match_saturated_fp32_reference(compiled):
    reference_output, reference_grads = _aggregate_attention_grads(tf.float32, compiled)
    output, grads = _aggregate_attention_grads(tf.float16, compiled)
    limit = np.finfo(np.float16)
    np.testing.assert_allclose(_as_f32(output), _as_f32(reference_output), atol=2e-3)
    for actual, expected in zip(grads, reference_grads):
        clipped = np.clip(expected.numpy(), limit.min, limit.max).astype(np.float16)
        assert np.all(np.isfinite(actual.numpy()))
        np.testing.assert_allclose(actual.numpy(), clipped, rtol=7e-3, atol=8.0)


def test_low_precision_core_preserves_genuine_nan():
    layer = YatNMN(
        features=1, use_bias=False, use_alpha=False, dtype=tf.float16
    )
    inputs = tf.constant([[np.nan, 1.0]], dtype=tf.float16)
    layer(inputs)
    layer.kernel.assign(tf.ones_like(layer.kernel))
    assert np.isnan(layer(inputs).numpy()).all()

    conv = YatConv1D(
        filters=1, kernel_size=1, use_bias=False, use_alpha=False,
        dtype=tf.float16,
    )
    conv_inputs = tf.constant([[[np.nan]]], dtype=tf.float16)
    conv(conv_inputs)
    conv.kernel.assign(tf.ones_like(conv.kernel))
    assert np.isnan(conv(conv_inputs).numpy()).all()

    query = tf.constant([[[[np.nan, 100.0]]]], dtype=tf.float16)
    key = tf.fill((1, 2, 1, 2), tf.constant(100.0, tf.float16))
    value = tf.ones((1, 2, 1, 1), dtype=tf.float16)
    assert np.isnan(yat_attention(query, key, value)).all()

    variable = tf.Variable([1.0], dtype=tf.float16)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(reduction_safe_upcast(variable) * np.nan)
    assert np.isnan(tape.gradient(loss, variable).numpy()).all()
