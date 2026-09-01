"""TensorFlow CPU parity for Keras channels-first YAT convolutions (#92)."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
import keras  # noqa: E402

from nmn.keras import (  # noqa: E402
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
)

pytestmark = pytest.mark.skipif(
    keras.backend.backend() != "tensorflow",
    reason="TensorFlow CPU layout fallback is backend-specific coverage",
)


CASES = (
    pytest.param(
        YatConv1D,
        (2, 11, 4),
        dict(filters=4, kernel_size=3, padding="same", dilation_rate=2, groups=2),
        id="conv1d-dilated-grouped",
    ),
    pytest.param(
        YatConv1D,
        (2, 11, 4),
        dict(filters=4, kernel_size=3, padding="causal", groups=2),
        id="conv1d-causal-grouped",
    ),
    pytest.param(
        YatConv2D,
        (2, 8, 7, 4),
        dict(filters=4, kernel_size=(3, 2), padding="same", strides=(2, 1), groups=2),
        id="conv2d-strided-grouped",
    ),
    pytest.param(
        YatConv3D,
        (2, 5, 6, 4, 4),
        dict(
            filters=4,
            kernel_size=(2, 2, 2),
            padding="same",
            dilation_rate=(1, 2, 1),
            groups=2,
        ),
        id="conv3d-dilated-grouped",
    ),
    pytest.param(
        YatConvTranspose1D,
        (2, 6, 3),
        dict(filters=4, kernel_size=3, strides=2, output_padding=1),
        id="transpose1d-output-padding",
    ),
    pytest.param(
        YatConvTranspose2D,
        (2, 4, 5, 3),
        dict(
            filters=4,
            kernel_size=(3, 2),
            strides=(2, 1),
            output_padding=(1, 0),
        ),
        id="transpose2d-output-padding",
    ),
    pytest.param(
        YatConvTranspose3D,
        (2, 3, 4, 3, 3),
        dict(
            filters=4,
            kernel_size=(2, 2, 2),
            strides=(2, 1, 2),
            output_padding=(1, 0, 1),
        ),
        id="transpose3d-output-padding",
    ),
)


def _channels_last_to_first(value):
    rank = len(value.shape)
    return tf.transpose(value, (0, rank - 1) + tuple(range(1, rank - 1)))


def _channels_first_to_last(value):
    rank = len(value.shape)
    return tf.transpose(value, (0,) + tuple(range(2, rank)) + (1,))


def _make_evaluator(layer, compiled):
    def evaluate(inputs, cotangent):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output = layer(inputs)
            loss = tf.reduce_sum(output * cotangent)
        gradients = tape.gradient(loss, (inputs, *layer.trainable_variables))
        return output, gradients

    return tf.function(evaluate) if compiled else evaluate


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "tf-function"])
@pytest.mark.parametrize("layer_cls,input_shape,kwargs", CASES)
def test_channels_first_matches_channels_last_forward_and_all_gradients(
    layer_cls, input_shape, kwargs, compiled
):
    rng = np.random.default_rng(92)
    input_last = tf.convert_to_tensor(rng.normal(size=input_shape).astype(np.float32))
    input_first = _channels_last_to_first(input_last)

    common = dict(
        use_bias=True,
        use_alpha=True,
        learnable_epsilon=True,
        epsilon=0.25,
        kernel_initializer="glorot_uniform",
    )
    with tf.device("/CPU:0"):
        channels_last = layer_cls(data_format="channels_last", **common, **kwargs)
        channels_first = layer_cls(data_format="channels_first", **common, **kwargs)
        output_last = channels_last(input_last)
        output_first = channels_first(input_first)
        channels_first.set_weights(channels_last.get_weights())

        # A non-uniform cotangent catches layout mistakes that a scalar sum can
        # hide while exercising input, kernel, bias, alpha, and epsilon grads.
        output_last = channels_last(input_last)
        output_first = channels_first(input_first)
        assert output_first.shape == _channels_last_to_first(output_last).shape
        cotangent_last = tf.convert_to_tensor(
            rng.normal(size=output_last.shape).astype(np.float32)
        )
        cotangent_first = _channels_last_to_first(cotangent_last)

        eval_last = _make_evaluator(channels_last, compiled)
        eval_first = _make_evaluator(channels_first, compiled)
        output_last, gradients_last = eval_last(input_last, cotangent_last)
        output_first, gradients_first = eval_first(input_first, cotangent_first)

    np.testing.assert_allclose(
        _channels_first_to_last(output_first).numpy(),
        output_last.numpy(),
        rtol=2e-5,
        atol=2e-5,
    )
    assert all(gradient is not None for gradient in gradients_last)
    assert all(gradient is not None for gradient in gradients_first)
    np.testing.assert_allclose(
        _channels_first_to_last(gradients_first[0]).numpy(),
        gradients_last[0].numpy(),
        rtol=4e-5,
        atol=4e-5,
    )
    for gradient_first, gradient_last in zip(gradients_first[1:], gradients_last[1:]):
        np.testing.assert_allclose(
            gradient_first.numpy(),
            gradient_last.numpy(),
            rtol=4e-5,
            atol=4e-5,
        )


@pytest.mark.parametrize(
    "layer_cls,output_padding",
    [
        (YatConvTranspose1D, 1),
        (YatConvTranspose2D, (1, 0)),
        (YatConvTranspose3D, (1, 0, 1)),
    ],
)
def test_output_padding_config_roundtrip(layer_cls, output_padding):
    layer = layer_cls(4, 3, strides=2, output_padding=output_padding)
    config = layer.get_config()
    restored = layer_cls.from_config(config)
    assert restored.output_padding == layer.output_padding
