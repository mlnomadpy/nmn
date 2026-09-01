"""Smoke tests for documented Keras examples."""

import keras


def test_mnist_model_builds_and_runs():
    from nmn.keras.examples.mnist import build_model

    model = build_model(hidden1=8, hidden2=4, num_classes=3)
    output = model(keras.ops.ones((2, 28, 28)))

    assert output.shape == (2, 3)
