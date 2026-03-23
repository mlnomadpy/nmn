"""Tests for PyTorch YatEmbed."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch.embed import YatEmbed


class TestYatEmbed:
    def test_forward_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        tokens = torch.tensor([0, 5, 10, 99])
        out = embed(tokens)
        assert out.shape == (4, 32)

    def test_forward_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
        out = embed(tokens)
        assert out.shape == (2, 3, 16)

    def test_attend_shape(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = torch.randn(4, 32)
        out = embed.attend(query)
        assert out.shape == (4, 100)

    def test_attend_batch(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        query = torch.randn(2, 10, 16)
        out = embed.attend(query)
        assert out.shape == (2, 10, 50)

    def test_attend_no_nan(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = torch.randn(4, 32)
        out = embed.attend(query)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_attend_positive_output(self):
        embed = YatEmbed(num_embeddings=100, features=32)
        query = torch.randn(4, 32)
        out = embed.attend(query)
        assert (out >= 0).all()

    def test_constant_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=True)
        assert embed._constant_alpha_value == pytest.approx(1.4142135, abs=1e-5)
        query = torch.randn(4, 16)
        out = embed.attend(query)
        assert not torch.isnan(out).any()

    def test_constant_alpha_float(self):
        embed = YatEmbed(num_embeddings=50, features=16, constant_alpha=2.5)
        assert embed._constant_alpha_value == 2.5

    def test_no_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=False)
        assert embed.alpha is None
        query = torch.randn(4, 16)
        out = embed.attend(query)
        assert not torch.isnan(out).any()

    def test_learnable_alpha(self):
        embed = YatEmbed(num_embeddings=50, features=16, use_alpha=True)
        assert embed.alpha is not None
        assert embed.alpha.requires_grad

    def test_spherical_mode(self):
        embed = YatEmbed(num_embeddings=50, features=16, spherical=True)
        query = torch.randn(4, 16)
        out = embed.attend(query)
        assert out.shape == (4, 50)
        assert not torch.isnan(out).any()

    def test_weight_normalized(self):
        embed = YatEmbed(num_embeddings=50, features=16, weight_normalized=True)
        norms = embed.embedding.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones(50), atol=1e-5)

    def test_gradient_flow(self):
        embed = YatEmbed(num_embeddings=50, features=16)
        query = torch.randn(4, 16, requires_grad=True)
        out = embed.attend(query)
        loss = out.sum()
        loss.backward()
        assert query.grad is not None
        assert not torch.isnan(query.grad).any()
