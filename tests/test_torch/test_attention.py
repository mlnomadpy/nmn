"""Tests for PyTorch YAT attention."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch.attention.multi_head import MultiHeadYatAttention
from nmn.torch.attention.yat_attention import (
    normalize_qk,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
)


class TestAttentionFunctions:
    @pytest.mark.parametrize(
        ("dtype", "rtol", "atol"),
        [
            pytest.param(torch.float16, 5e-3, 5e-4, id="float16"),
            pytest.param(torch.bfloat16, 2e-2, 3e-3, id="bfloat16"),
        ],
    )
    @pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
    def test_normalized_low_precision_matches_fp32_output_and_gradients(
        self, dtype, rtol, atol, compiled
    ):
        if compiled and not hasattr(torch, "compile"):
            pytest.skip("torch.compile requires PyTorch 2")

        # Build the reference from values quantized to the caller dtype so this
        # measures arithmetic precision rather than input representation error.
        query_data = torch.tensor(
            [[[[300.0, 300.0, 300.0, 300.0]], [[300.0, -300.0, 200.0, -200.0]]]],
            dtype=dtype,
        )
        key_data = torch.tensor(
            [
                [
                    [[300.0, 300.0, 300.0, -300.0]],
                    [[300.0, -300.0, -300.0, 300.0]],
                    [[-300.0, 250.0, 200.0, 100.0]],
                ]
            ],
            dtype=dtype,
        )
        value_data = torch.tensor(
            [[[[1.0, -1.0]], [[3.0, 2.0]], [[-2.0, 4.0]]]], dtype=dtype
        )
        mask = torch.tensor([[[[False, False, False], [True, True, False]]]])
        cotangent = torch.tensor([[[[0.5, -0.25]], [[1.5, 0.75]]]])

        def evaluate(compute_dtype, use_compile, normalized):
            query = query_data.to(compute_dtype).detach().requires_grad_()
            key = key_data.to(compute_dtype).detach().requires_grad_()
            value = value_data.to(compute_dtype).detach().requires_grad_()
            alpha = torch.tensor(1.25, dtype=dtype).to(compute_dtype).requires_grad_()

            def apply(q, k, v, a):
                if normalized:
                    return yat_attention_normalized(
                        q, k, v, mask=mask, training=False, alpha=a
                    )
                return yat_attention(
                    q, k, v, mask=mask, training=False, alpha=a, spherical=True
                )

            if use_compile:
                apply = torch.compile(apply, backend="eager")
            output = apply(query, key, value, alpha)
            gradients = torch.autograd.grad(
                (output.float() * cotangent).sum(), (query, key, value, alpha)
            )
            return output, gradients

        reference_output, reference_gradients = evaluate(torch.float32, False, False)
        output, gradients = evaluate(dtype, compiled, True)

        assert output.dtype == dtype
        assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        torch.testing.assert_close(
            output.float(), reference_output, rtol=rtol, atol=atol
        )
        for gradient, reference_gradient in zip(gradients, reference_gradients):
            torch.testing.assert_close(
                gradient.float(), reference_gradient, rtol=rtol, atol=atol
            )

    def test_negative_scale_cannot_make_masked_key_win_softmax(self):
        q = torch.ones(1, 1, 1, 4)
        k = torch.ones(1, 2, 1, 4)
        weights = yat_attention_weights(
            q, k, mask=torch.tensor([True, False]), scale=-1.0
        )
        assert torch.equal(weights, torch.tensor([[[[1.0, 0.0]]]]))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("spherical", [False, True])
    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2"
    )
    def test_fully_masked_rows_are_zero_with_finite_compiled_gradients(
        self, spherical, dtype
    ):
        q = torch.randn(1, 2, 2, 4, dtype=dtype, requires_grad=True)
        k = torch.randn(1, 3, 2, 4, dtype=dtype, requires_grad=True)
        v = torch.randn(1, 3, 2, 5, dtype=dtype, requires_grad=True)
        mask = torch.tensor([[[[False, False, False], [True, False, True]]]])

        def apply(q, k, v):
            return yat_attention(q, k, v, mask=mask, spherical=spherical)

        eager = apply(q, k, v)
        compiled = torch.compile(apply, backend="eager")(q, k, v)
        weights = yat_attention_weights(q, k, mask=mask, spherical=spherical)
        grads = torch.autograd.grad(compiled.sum(), (q, k, v))

        assert torch.equal(weights[..., 0, :], torch.zeros_like(weights[..., 0, :]))
        assert torch.equal(eager[:, 0], torch.zeros_like(eager[:, 0]))
        torch.testing.assert_close(compiled, eager)
        assert all(torch.isfinite(grad).all() for grad in grads)

    def test_normalize_qk(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 5, 4, 8)
        q_n, k_n = normalize_qk(q, k)
        q_norms = q_n.norm(dim=-1)
        assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)

    def test_attention_weights_shape(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        weights = yat_attention_weights(q, k)
        assert weights.shape == (2, 4, 5, 7)

    def test_attention_weights_sum_to_one(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        weights = yat_attention_weights(q, k)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_attention_output_shape(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 16)
        out = yat_attention(q, k, v)
        assert out.shape == (2, 5, 4, 16)

    def test_attention_no_nan(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        out = yat_attention(q, k, v)
        assert not torch.isnan(out).any()

    def test_attention_normalized_shape(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        out = yat_attention_normalized(q, k, v)
        assert out.shape == (2, 5, 4, 8)

    def test_attention_with_alpha(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        alpha = torch.tensor([1.5])
        out = yat_attention(q, k, v, alpha=alpha)
        assert not torch.isnan(out).any()

    def test_attention_with_scale(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        out = yat_attention(q, k, v, scale=1.414)
        assert not torch.isnan(out).any()

    def test_attention_with_mask(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        mask = torch.ones(2, 4, 5, 7, dtype=torch.bool)
        out = yat_attention(q, k, v, mask=mask)
        assert out.shape == (2, 5, 4, 8)

    def test_spherical_mode(self):
        q = torch.randn(2, 5, 4, 8)
        k = torch.randn(2, 7, 4, 8)
        v = torch.randn(2, 7, 4, 8)
        out = yat_attention(q, k, v, spherical=True)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        q = torch.randn(2, 5, 4, 8, requires_grad=True)
        k = torch.randn(2, 7, 4, 8, requires_grad=True)
        v = torch.randn(2, 7, 4, 8, requires_grad=True)
        out = yat_attention(q, k, v)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()


class TestMultiHeadYatAttention:
    @pytest.mark.parametrize("mask_rank", [2, 4])
    @pytest.mark.parametrize("cross_attention", [False, True])
    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile requires PyTorch 2"
    )
    def test_fully_masked_rows_stay_zero_after_biased_output_projection(
        self, cross_attention, mask_rank
    ):
        attn = MultiHeadYatAttention(embed_dim=8, num_heads=2)
        with torch.no_grad():
            attn.out_proj.bias.fill_(3.0)
        query = torch.randn(1, 2, 8, requires_grad=True)
        context = torch.randn(1, 3, 8, requires_grad=True)
        kv_length = 3 if cross_attention else 2
        if mask_rank == 2:
            mask = torch.ones(2, kv_length, dtype=torch.bool)
            mask[0, :] = False
        else:
            mask = torch.ones(1, 1, 2, kv_length, dtype=torch.bool)
            mask[..., 0, :] = False
        output = (
            torch.compile(attn, backend="eager")(
                query, key=context, value=context, mask=mask
            )
            if cross_attention
            else torch.compile(attn, backend="eager")(query, mask=mask)
        )
        assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
        assert torch.isfinite(output).all()
        grads = torch.autograd.grad(
            output.square().sum(),
            (query, context) if cross_attention else (query,),
        )
        assert all(torch.isfinite(grad).all() for grad in grads)

    def test_self_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_cross_attention(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        q = torch.randn(2, 5, 32)
        kv = torch.randn(2, 10, 32)
        out = attn(q, key=kv)
        assert out.shape == (2, 5, 32)

    def test_no_nan(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_constant_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, constant_alpha=True)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_no_alpha(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_alpha=False)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_no_bias(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, bias=False)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_no_out_proj(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, use_out_proj=False)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert out.shape == (2, 10, 32)

    def test_spherical(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4, spherical=True)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        assert not torch.isnan(out).any()

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadYatAttention(embed_dim=33, num_heads=4)

    def test_with_mask(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        mask = torch.ones(2, 4, 10, 10, dtype=torch.bool)
        out = attn(x, mask=mask)
        assert out.shape == (2, 10, 32)

    def test_gradient_flow(self):
        attn = MultiHeadYatAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
