"""Tests for PyTorch squasher functions."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch.squashers import softermax, softer_sigmoid, soft_tanh


class TestSoftermax:
    def test_output_shape(self):
        x = torch.rand(4, 8)
        out = softermax(x)
        assert out.shape == x.shape

    def test_sums_to_one(self):
        x = torch.rand(4, 8)
        out = softermax(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_no_nan(self):
        x = torch.rand(2, 16)
        out = softermax(x, n=2.0)
        assert not torch.isnan(out).any()

    def test_power_parameter(self):
        x = torch.rand(4, 8)
        out1 = softermax(x, n=1.0)
        out2 = softermax(x, n=2.0)
        assert not torch.allclose(out1, out2)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            softermax(torch.rand(4, 8), n=-1.0)

    def test_custom_dim(self):
        x = torch.rand(3, 4, 5)
        out = softermax(x, dim=1)
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(3, 5), atol=1e-5)


class TestSofterSigmoid:
    def test_output_range(self):
        x = torch.rand(4, 8)
        out = softer_sigmoid(x)
        assert (out >= 0).all()
        assert (out < 1).all()

    def test_zero_input(self):
        x = torch.zeros(4)
        out = softer_sigmoid(x)
        assert torch.allclose(out, torch.zeros(4))

    def test_no_nan(self):
        x = torch.rand(2, 16) * 10
        out = softer_sigmoid(x, n=2.0)
        assert not torch.isnan(out).any()

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            softer_sigmoid(torch.rand(4), n=0)


class TestSoftTanh:
    def test_output_range(self):
        x = torch.rand(4, 8)
        out = soft_tanh(x)
        assert (out >= -1).all()
        assert (out < 1).all()

    def test_zero_gives_minus_one(self):
        x = torch.zeros(4)
        out = soft_tanh(x)
        assert torch.allclose(out, -torch.ones(4))

    def test_one_gives_zero(self):
        x = torch.ones(4)
        out = soft_tanh(x)
        assert torch.allclose(out, torch.zeros(4))

    def test_no_nan(self):
        x = torch.rand(2, 16) * 10
        out = soft_tanh(x, n=2.0)
        assert not torch.isnan(out).any()

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="positive"):
            soft_tanh(torch.rand(4), n=-0.5)
