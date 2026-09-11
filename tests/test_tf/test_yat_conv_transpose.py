"""Tests for TensorFlow YatConvTranspose layers - TDD: Write tests first."""

import numpy as np
import pytest


def test_tf_yat_conv_transpose2d_import():
    """Test that YatConvTranspose2D can be imported."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        assert YatConvTranspose2D is not None
    except ImportError as e:
        pytest.skip(f"TensorFlow dependencies not available: {e}")


def test_tf_yat_conv_transpose2d_instantiation():
    """Test YatConvTranspose2D can be instantiated."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        assert layer is not None
        assert layer.filters == 8

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_build():
    """Test YatConvTranspose2D builds correctly."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))
        # Build with input [batch, height, width, channels]
        layer.build([None, 16, 16, 16])

        assert layer.kernel is not None

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_forward():
    """Test YatConvTranspose2D forward pass (upsamples by 2x)."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))

        # Create input [batch, height, width, channels]
        dummy_input = tf.constant(np.random.randn(2, 16, 16, 16).astype(np.float32))
        output = layer(dummy_input)

        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 32, 32, 8)

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose1d_forward():
    """Test YatConvTranspose1D forward pass."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose1D

        layer = YatConvTranspose1D(filters=8, kernel_size=2, strides=2)

        # Create input [batch, length, channels]
        dummy_input = tf.constant(np.random.randn(2, 16, 16).astype(np.float32))
        output = layer(dummy_input)

        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 32, 8)

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose3d_forward():
    """Test YatConvTranspose3D forward pass."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose3D

        layer = YatConvTranspose3D(filters=8, kernel_size=(2, 2, 2), strides=(2, 2, 2))

        # Create input [batch, depth, height, width, channels]
        dummy_input = tf.constant(np.random.randn(2, 8, 8, 8, 16).astype(np.float32))
        output = layer(dummy_input)

        # For transpose conv with stride=2, kernel=2, output should be 2x input
        assert output.shape == (2, 16, 16, 16, 8)

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_gradient():
    """Test that YatConvTranspose2D can compute gradients."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        layer = YatConvTranspose2D(filters=8, kernel_size=(2, 2), strides=(2, 2))

        dummy_input = tf.constant(np.random.randn(2, 16, 16, 16).astype(np.float32))

        with tf.GradientTape() as tape:
            output = layer(dummy_input)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


def test_tf_yat_conv_transpose2d_no_bias():
    """Test YatConvTranspose2D without bias."""
    try:
        import tensorflow as tf

        from nmn.tf.conv import YatConvTranspose2D

        layer = YatConvTranspose2D(
            filters=8, kernel_size=(2, 2), strides=(2, 2), use_bias=False
        )
        layer.build([None, 16, 16, 16])

        assert layer.bias is None

    except ImportError:
        pytest.skip("TensorFlow dependencies not available")


@pytest.mark.parametrize(
    "class_name,input_shape,kernel,strides,expected",
    [
        ("YatConvTranspose1D", (1, 3, 1), 2, 3, (1, 8, 1)),
        ("YatConvTranspose2D", (1, 2, 3, 1), (2, 3), (3, 2), (1, 5, 7, 1)),
        (
            "YatConvTranspose3D",
            (1, 2, 2, 2, 1),
            (2, 3, 2),
            (3, 2, 4),
            (1, 5, 5, 6, 1),
        ),
    ],
)
def test_canonical_valid_stride_gap_output_and_gradients(
    class_name, input_shape, kernel, strides, expected
):
    tf = pytest.importorskip("tensorflow")
    from nmn.tf import (
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    )

    layer_type = {
        "YatConvTranspose1D": YatConvTranspose1D,
        "YatConvTranspose2D": YatConvTranspose2D,
        "YatConvTranspose3D": YatConvTranspose3D,
    }[class_name]
    layer = layer_type(
        1,
        kernel,
        strides=strides,
        padding="valid",
        output_padding=0,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    inputs = tf.Variable(
        tf.reshape(tf.linspace(-0.7, 0.9, np.prod(input_shape)), input_shape)
    )

    @tf.function
    def loss_and_output():
        with tf.GradientTape() as tape:
            output = layer(inputs)
            loss = tf.reduce_sum(output)
        gradients = tape.gradient(loss, (inputs, layer.kernel))
        return output, gradients

    output, gradients = loss_and_output()
    assert tuple(output.shape) == expected
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients
    )


def test_canonical_dilation_and_same_output_padding():
    tf = pytest.importorskip("tensorflow")
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("TensorFlow CPU Conv2DBackpropInput does not support dilation > 1")
    from nmn.tf import YatConvTranspose1D

    valid = YatConvTranspose1D(
        1,
        3,
        strides=1,
        padding="valid",
        dilation_rate=2,
        output_padding=0,
        use_bias=False,
        use_alpha=False,
    )
    same = YatConvTranspose1D(
        1,
        2,
        strides=3,
        padding="same",
        output_padding=1,
        use_bias=False,
        use_alpha=False,
    )
    assert tuple(valid(tf.ones((1, 3, 1))).shape) == (1, 7, 1)
    assert tuple(same(tf.ones((1, 3, 1))).shape) == (1, 10, 1)


def test_canonical_same_output_padding_uncrops_valid_kernel_contributions():
    tf = pytest.importorskip("tensorflow")
    from nmn.tf import YatConvTranspose1D

    inputs = tf.constant([[[0.5], [1.0], [1.5]]])
    same = YatConvTranspose1D(
        1,
        3,
        strides=2,
        padding="same",
        output_padding=1,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    valid = YatConvTranspose1D(
        1,
        3,
        strides=2,
        padding="valid",
        output_padding=0,
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    same(inputs)
    valid(inputs)
    kernel = tf.reshape(tf.range(1, 4, dtype=tf.float32), (3, 1, 1))
    same.kernel.assign(kernel)
    valid.kernel.assign(kernel)

    same_output = same(inputs)
    valid_output = valid(inputs)
    assert tuple(same_output.shape) == tuple(valid_output.shape) == (1, 7, 1)
    tf.debugging.assert_near(same_output, valid_output)


def test_implicit_tensorflow_stride_gap_shape_remains_backward_compatible():
    tf = pytest.importorskip("tensorflow")
    from nmn.tf import YatConvTranspose1D

    inputs = tf.ones((1, 3, 1))
    legacy = YatConvTranspose1D(
        1, 2, strides=3, padding="valid", use_bias=False, use_alpha=False
    )
    canonical = YatConvTranspose1D(
        1,
        2,
        strides=3,
        padding="valid",
        output_padding=0,
        use_bias=False,
        use_alpha=False,
    )
    assert tuple(legacy(inputs).shape) == (1, 9, 1)
    assert tuple(canonical(inputs).shape) == (1, 8, 1)
