"""Tests for TensorFlow squasher functions."""

import pytest
import numpy as np

tf = pytest.importorskip("tensorflow")

from nmn.tf.squashers import softermax, softer_sigmoid, soft_tanh


class TestSoftermax:
    def test_output_shape(self):
        x = tf.random.uniform((4, 8))
        out = softermax(x)
        assert out.shape == (4, 8)

    def test_sums_to_one(self):
        x = tf.random.uniform((4, 8))
        out = softermax(x)
        sums = tf.reduce_sum(out, axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_no_nan(self):
        x = tf.random.uniform((2, 16))
        out = softermax(x, n=2.0)
        assert not np.any(np.isnan(out.numpy()))

    def test_power_parameter(self):
        x = tf.random.uniform((4, 8))
        out1 = softermax(x, n=1.0)
        out2 = softermax(x, n=2.0)
        assert not np.allclose(out1.numpy(), out2.numpy())


class TestSofterSigmoid:
    def test_output_range(self):
        x = tf.random.uniform((4, 8))
        out = softer_sigmoid(x).numpy()
        assert np.all(out >= 0)
        assert np.all(out < 1)

    def test_zero_input(self):
        x = tf.zeros((4,))
        out = softer_sigmoid(x).numpy()
        np.testing.assert_allclose(out, np.zeros(4))


class TestSoftTanh:
    def test_output_range(self):
        x = tf.random.uniform((4, 8))
        out = soft_tanh(x).numpy()
        assert np.all(out >= -1)
        assert np.all(out < 1)

    def test_zero_gives_minus_one(self):
        x = tf.zeros((4,))
        out = soft_tanh(x).numpy()
        np.testing.assert_allclose(out, -np.ones(4))

    def test_one_gives_zero(self):
        x = tf.ones((4,))
        out = soft_tanh(x).numpy()
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-6)
