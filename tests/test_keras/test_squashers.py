"""Tests for Keras squasher functions."""

import pytest
import numpy as np

keras = pytest.importorskip("keras")

from nmn.keras.squashers import softermax, softer_sigmoid, soft_tanh


class TestSoftermax:
    def test_output_shape(self):
        x = np.random.rand(4, 8).astype(np.float32)
        out = softermax(x)
        assert out.shape == (4, 8)

    def test_sums_to_one(self):
        x = np.random.rand(4, 8).astype(np.float32)
        out = np.array(softermax(x))
        sums = out.sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

    def test_no_nan(self):
        x = np.random.rand(2, 16).astype(np.float32)
        out = np.array(softermax(x, n=2.0))
        assert not np.any(np.isnan(out))

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            softermax(np.random.rand(4, 8).astype(np.float32), n=-1.0)


class TestSofterSigmoid:
    def test_output_range(self):
        x = np.random.rand(4, 8).astype(np.float32)
        out = np.array(softer_sigmoid(x))
        assert np.all(out >= 0)
        assert np.all(out < 1)

    def test_zero_input(self):
        x = np.zeros(4, dtype=np.float32)
        out = np.array(softer_sigmoid(x))
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-6)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            softer_sigmoid(np.random.rand(4).astype(np.float32), n=0)


class TestSoftTanh:
    def test_output_range(self):
        x = np.random.rand(4, 8).astype(np.float32)
        out = np.array(soft_tanh(x))
        assert np.all(out >= -1)
        assert np.all(out < 1)

    def test_zero_gives_minus_one(self):
        x = np.zeros(4, dtype=np.float32)
        out = np.array(soft_tanh(x))
        np.testing.assert_allclose(out, -np.ones(4))

    def test_one_gives_zero(self):
        x = np.ones(4, dtype=np.float32)
        out = np.array(soft_tanh(x))
        np.testing.assert_allclose(out, np.zeros(4), atol=1e-6)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            soft_tanh(np.random.rand(4).astype(np.float32), n=-0.5)
