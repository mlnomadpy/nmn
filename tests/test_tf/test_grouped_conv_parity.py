"""Parity regressions for native TensorFlow grouped YAT convolutions."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from nmn.tf.conv import YatConv1D, YatConv2D, YatConv3D

CASES = [
    (YatConv1D, 2, (1, 6)),
    (YatConv2D, (2, 2), (1, 4, 4)),
    (YatConv3D, (2, 2, 2), (1, 3, 3, 3)),
]


def _conv(layer, inputs, kernel):
    if inputs.shape.rank == 3:
        return tf.nn.conv1d(
            inputs,
            kernel,
            stride=layer.strides,
            padding=layer.padding,
            dilations=layer.dilation_rate,
        )
    if inputs.shape.rank == 4:
        return tf.nn.conv2d(
            inputs,
            kernel,
            strides=[1, *layer.strides, 1],
            padding=layer.padding,
            dilations=[1, *layer.dilation_rate, 1],
        )
    return tf.nn.conv3d(
        inputs,
        kernel,
        strides=[1, *layer.strides, 1],
        padding=layer.padding,
        dilations=[1, *layer.dilation_rate, 1],
    )


def _split_group_reference(layer, inputs):
    """Apply each convolution group independently, then concatenate."""
    input_groups = tf.split(inputs, layer.groups, axis=-1)
    kernel_groups = tf.split(layer.kernel, layer.groups, axis=-1)
    spatial_axes = list(range(layer.kernel.shape.rank - 2))
    reduce_axes = spatial_axes + [layer.kernel.shape.rank - 2]
    outputs = []

    for group_inputs, group_kernel in zip(input_groups, kernel_groups):
        dot = _conv(layer, group_inputs, group_kernel)
        norm_kernel = tf.ones(tuple(group_kernel.shape[:-1]) + (1,), dtype=layer.dtype)
        patch_norm = _conv(layer, tf.square(group_inputs), norm_kernel)
        kernel_norm = tf.reduce_sum(tf.square(group_kernel), axis=reduce_axes)
        broadcast_shape = [1] * (inputs.shape.rank - 1) + [-1]
        distance = patch_norm + tf.reshape(kernel_norm, broadcast_shape) - 2.0 * dot
        outputs.append(tf.square(dot) / (distance + layer.epsilon))

    return tf.concat(outputs, axis=-1)


def _value_and_gradients(forward, inputs, kernel):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        output = forward(inputs)
        loss = tf.reduce_sum(output)
    input_grad, kernel_grad = tape.gradient(loss, (inputs, kernel))
    return output, input_grad, kernel_grad


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("layer_cls,kernel_size,input_prefix", CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "tf_function"])
def test_grouped_conv_matches_split_reference_forward_and_gradients(
    layer_cls, kernel_size, input_prefix, groups, compiled
):
    channels = groups * 2
    filters = groups * 2
    shape = input_prefix + (channels,)
    inputs = tf.constant(
        np.linspace(-0.7, 0.9, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    )
    layer = layer_cls(
        filters=filters,
        kernel_size=kernel_size,
        groups=groups,
        use_bias=False,
        use_alpha=False,
        epsilon=0.2,
    )
    layer(inputs)
    kernel_values = np.linspace(
        -0.35, 0.45, num=int(np.prod(layer.kernel.shape)), dtype=np.float32
    ).reshape(layer.kernel.shape)
    layer.kernel.assign(kernel_values)

    actual = lambda x: layer(x)
    reference = lambda x: _split_group_reference(layer, x)
    if compiled:
        actual = tf.function(actual)
        reference = tf.function(reference)

    actual_result = _value_and_gradients(actual, inputs, layer.kernel)
    reference_result = _value_and_gradients(reference, inputs, layer.kernel)

    for actual_value, reference_value in zip(actual_result, reference_result):
        np.testing.assert_allclose(
            actual_value.numpy(), reference_value.numpy(), rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("layer_cls,kernel_size,input_prefix", CASES)
def test_group_validation_errors_are_clear(layer_cls, kernel_size, input_prefix):
    del input_prefix
    with pytest.raises(ValueError, match="groups must be a positive integer"):
        layer_cls(filters=4, kernel_size=kernel_size, groups=0)

    with pytest.raises(ValueError, match=r"Filters \(5\).*groups \(2\)"):
        layer_cls(filters=5, kernel_size=kernel_size, groups=2)

    layer = layer_cls(filters=4, kernel_size=kernel_size, groups=2)
    rank = len(kernel_size) if isinstance(kernel_size, tuple) else 1
    bad_shape = (1,) + (4,) * rank + (3,)
    with pytest.raises(ValueError, match=r"Input channels \(3\).*groups \(2\)"):
        layer(tf.zeros(bad_shape))
