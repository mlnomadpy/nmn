"""Smoke tests for documented native TensorFlow examples."""

import pytest

tf = pytest.importorskip("tensorflow")


def test_mnist_model_imports_without_tensorflow_datasets_and_runs():
    from nmn.tf.examples.mnist import YatMLP

    model = YatMLP(hidden1=8, hidden2=4, num_classes=3)
    output = model(tf.ones((2, 28, 28)))

    assert output.shape == (2, 3)
