"""Cross-normalization regressions for fully masked attention rows (#70)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx.layers.attention import (
    MultiHeadAttention,
    RotaryYatAttention,
    yat_attention,
    yat_attention_normalized,
    yat_attention_weights,
)


def _inputs():
    q = jax.random.normal(jax.random.key(70), (1, 2, 2, 4))
    k = jax.random.normal(jax.random.key(71), (1, 3, 2, 4))
    v = jax.random.normal(jax.random.key(72), (1, 3, 2, 5))
    mask = jnp.array([[[[False, False, False], [True, False, True]]]], dtype=jnp.bool_)
    return q, k, v, mask


def _broadcast_mask(rank, batch, heads, q_length, kv_length):
    if rank == 1:
        mask = jnp.zeros((kv_length,), dtype=jnp.bool_)
    elif rank == 2:
        mask = jnp.ones((q_length, kv_length), dtype=jnp.bool_)
        mask = mask.at[0, :].set(False)
    elif rank == 3:
        mask = jnp.ones((heads, q_length, kv_length), dtype=jnp.bool_)
        mask = mask.at[:, 0, :].set(False)
    else:
        mask = jnp.ones((batch, heads, q_length, kv_length), dtype=jnp.bool_)
        mask = mask.at[0, :, 0, :].set(False)
        mask = mask.at[1, :, 1, :].set(False)
    effective = jnp.broadcast_to(mask, (batch, heads, q_length, kv_length))
    return mask, jnp.any(effective, axis=(1, 3))


@pytest.mark.parametrize("normalization", ["softmax", "l1", "softermax"])
def test_boolean_mask_zero_policy_is_jitted_differentiable_and_exact(normalization):
    q, k, v, mask = _inputs()

    def apply(q, k, v):
        return yat_attention(
            q,
            k,
            v,
            mask=mask,
            deterministic=True,
            normalization=normalization,
        )

    eager = apply(q, k, v)
    compiled = jax.jit(apply)(q, k, v)
    weights = yat_attention_weights(
        q, k, mask=mask, deterministic=True, normalization=normalization
    )
    grads = jax.jit(jax.grad(lambda q, k, v: jnp.sum(apply(q, k, v)), (0, 1, 2)))(
        q, k, v
    )

    np.testing.assert_array_equal(np.asarray(weights[..., 0, :]), 0.0)
    np.testing.assert_array_equal(np.asarray(eager[:, 0]), 0.0)
    np.testing.assert_allclose(compiled, eager, rtol=2e-7, atol=1e-7)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad)))


def test_negative_infinity_additive_mask_matches_boolean_mask_and_gradients():
    q, k, v, mask = _inputs()
    bias = jnp.where(mask, 0.0, -jnp.inf)

    def boolean_apply(q, k, v):
        return yat_attention(q, k, v, mask=mask, deterministic=True)

    def additive_apply(q, k, v, bias):
        return yat_attention(q, k, v, bias=bias, deterministic=True)

    expected = jax.jit(boolean_apply)(q, k, v)
    actual = jax.jit(additive_apply)(q, k, v, bias)
    grads = jax.grad(
        lambda q, k, v, b: jnp.sum(additive_apply(q, k, v, b)), (0, 1, 2, 3)
    )(q, k, v, bias)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("use_softermax", [False, True])
def test_normalized_qk_implementation_has_the_same_zero_policy(use_softermax):
    q, k, v, mask = _inputs()

    def apply(q, k, v):
        return yat_attention_normalized(
            q,
            k,
            v,
            mask=mask,
            use_softermax=use_softermax,
            deterministic=True,
        )

    output = jax.jit(apply)(q, k, v)
    grads = jax.grad(lambda q, k, v: jnp.sum(apply(q, k, v)), (0, 1, 2))(q, k, v)
    np.testing.assert_array_equal(np.asarray(output[:, 0]), 0.0)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_fully_masked_softmax_is_finite(dtype):
    q, k, v, mask = _inputs()
    q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)

    def apply(q, k, v):
        return yat_attention(q, k, v, mask=mask, deterministic=True)

    output = jax.jit(apply)(q, k, v)
    grads = jax.grad(
        lambda q, k, v: jnp.sum(apply(q, k, v).astype(jnp.float32)), (0, 1, 2)
    )(q, k, v)
    np.testing.assert_array_equal(np.asarray(output[:, 0], dtype=np.float32), 0.0)
    for grad in grads:
        assert np.all(np.isfinite(np.asarray(grad, dtype=np.float32)))


def test_negative_scale_cannot_make_a_masked_key_win_softmax():
    q = jnp.ones((1, 1, 1, 4), dtype=jnp.float32)
    k = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    mask = jnp.array([True, False])
    weights = yat_attention_weights(
        q, k, mask=mask, alpha=jnp.array(-1.0), deterministic=True
    )
    np.testing.assert_array_equal(np.asarray(weights), [[[[1.0, 0.0]]]])


def test_fully_masked_output_and_gradient_parity_with_torch():
    torch = pytest.importorskip("torch")
    from nmn.torch.attention import yat_attention as torch_yat_attention
    from nmn.torch.attention import yat_attention_weights as torch_yat_attention_weights

    q, k, v, mask = _inputs()
    tq = torch.tensor(np.asarray(q), requires_grad=True)
    tk = torch.tensor(np.asarray(k), requires_grad=True)
    tv = torch.tensor(np.asarray(v), requires_grad=True)
    tm = torch.tensor(np.asarray(mask))

    def jax_loss(q, k, v):
        return jnp.sum(yat_attention(q, k, v, mask=mask, deterministic=True) ** 2)

    jax_output = jax.jit(
        lambda q, k, v: yat_attention(q, k, v, mask=mask, deterministic=True)
    )(q, k, v)
    jax_weights = yat_attention_weights(q, k, mask=mask, deterministic=True)
    jax_grads = jax.grad(jax_loss, (0, 1, 2))(q, k, v)

    torch_output = torch_yat_attention(tq, tk, tv, mask=tm)
    torch_weights = torch_yat_attention_weights(tq, tk, mask=tm)
    torch_grads = torch.autograd.grad((torch_output**2).sum(), (tq, tk, tv))

    np.testing.assert_allclose(torch_output.detach(), jax_output, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        torch_weights.detach(), jax_weights, rtol=2e-6, atol=2e-6
    )
    for torch_grad, jax_grad in zip(torch_grads, jax_grads):
        np.testing.assert_allclose(torch_grad.detach(), jax_grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("mask_rank", [1, 2, 3, 4])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_multi_head_module_zeroes_fully_masked_rows_after_projection(
    cross_attention, mask_rank
):
    module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        out_bias_init=nnx.initializers.constant(3.0),
        rngs=nnx.Rngs(70),
    )
    query = jax.random.normal(jax.random.key(73), (2, 2, 8))
    context = jax.random.normal(jax.random.key(74), (2, 3, 8))
    kv_length = 3 if cross_attention else 2
    mask, expected_valid = _broadcast_mask(mask_rank, 2, 2, 2, kv_length)

    @nnx.jit
    def apply(module, query, context, mask):
        if cross_attention:
            return module(query, context, context, mask=mask, deterministic=True)
        return module(query, mask=mask, deterministic=True)

    output = apply(module, query, context, mask)
    np.testing.assert_array_equal(np.asarray(output)[~np.asarray(expected_valid)], 0.0)
    assert np.all(np.isfinite(np.asarray(output)))
    if cross_attention:
        grads = jax.grad(
            lambda q, c: jnp.sum(module(q, c, c, mask=mask, deterministic=True)),
            (0, 1),
        )(query, context)
    else:
        grads = (
            jax.grad(lambda q: jnp.sum(module(q, mask=mask, deterministic=True)))(
                query
            ),
        )
    assert all(np.all(np.isfinite(np.asarray(grad))) for grad in grads)


@pytest.mark.parametrize("mask_rank", [1, 2, 3, 4])
@pytest.mark.parametrize("normalization", ["softmax", "l1", "softermax"])
def test_rotary_module_zeroes_fully_masked_rows_for_every_normalization(
    normalization, mask_rank
):
    module = RotaryYatAttention(
        embed_dim=8,
        num_heads=2,
        max_seq_len=4,
        normalization=normalization,
        use_bias=True,
        rngs=nnx.Rngs(75),
    )
    # Make the projection-bias regression observable.
    module.o_proj.bias[...] = 2.0
    x = jax.random.normal(jax.random.key(76), (2, 2, 8))
    mask, expected_valid = _broadcast_mask(mask_rank, 2, 2, 2, 2)
    output = nnx.jit(lambda m, x, mask: m(x, mask=mask, deterministic=True))(
        module, x, mask
    )
    np.testing.assert_array_equal(np.asarray(output)[~np.asarray(expected_valid)], 0.0)
    assert np.all(np.isfinite(np.asarray(output)))
