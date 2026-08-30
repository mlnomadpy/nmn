"""Regression coverage for NNX attention API, gradients, masks, and decode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from nmn.nnx.layers.attention import MultiHeadAttention, RotaryYatAttention
from nmn.nnx.layers.attention.fused_yat_attention import (
    fused_yat_l1_attention,
    fused_yat_l1_self_attention,
)
from nmn.nnx.layers.attention.yat_attention import yat_attention


def _causal_mask(batch_size: int, length: int) -> jax.Array:
    return jnp.broadcast_to(
        jnp.tril(jnp.ones((length, length), dtype=jnp.bool_)),
        (batch_size, 1, length, length),
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_fp16_attention_aggregate_cotangents_match_saturated_fp32_reference(compiled):
    def evaluate(dtype):
        query = jnp.full((1, 1, 1, 2), 100.0, dtype=dtype)
        key = jnp.full((1, 2, 1, 2), 99.0, dtype=dtype)
        value = jnp.array([[[[0.0]], [[1.0]]]], dtype=dtype)

        def loss(q, k, v):
            return yat_attention(q, k, v, deterministic=True, epsilon=1.0).astype(jnp.float32).sum()

        output = yat_attention(query, key, value, deterministic=True, epsilon=1.0)
        gradient_fn = jax.grad(loss, argnums=(0, 1, 2))
        if compiled:
            gradient_fn = jax.jit(gradient_fn)
        return output, gradient_fn(query, key, value)

    reference_output, reference_grads = evaluate(jnp.float32)
    output, grads = evaluate(jnp.float16)
    limit = jnp.finfo(jnp.float16)
    np.testing.assert_allclose(np.asarray(output, np.float32), np.asarray(reference_output), atol=2e-3)
    for actual, expected in zip(grads, reference_grads):
        clipped = jnp.asarray(jnp.clip(expected, limit.min, limit.max), jnp.float16)
        assert jnp.all(jnp.isfinite(actual))
        np.testing.assert_allclose(
            np.asarray(actual, np.float32), np.asarray(clipped, np.float32),
            rtol=7e-3, atol=8.0,
        )


def _reference_l1(query, key, value, bias=None, epsilon=1e-3):
    dot = jnp.einsum("...qhd,...khd->...hqk", query, key)
    q_sq = jnp.sum(query**2, axis=-1, keepdims=True).transpose(0, 2, 1, 3)
    k_sq = jnp.sum(key**2, axis=-1, keepdims=True).transpose(0, 2, 1, 3)
    raw_dist = q_sq + jnp.swapaxes(k_sq, -2, -1) - 2.0 * dot
    dist = jnp.maximum(raw_dist, 0.0) + epsilon
    numerator = dot if bias is None else dot + bias
    scores = numerator**2 / (dist * jnp.sqrt(jnp.float32(query.shape[-1])))
    weights = scores / (jnp.sum(scores, axis=-1, keepdims=True) + 1e-12)
    return jnp.einsum("...hqk,...khd->...qhd", weights, value)


def _reference_l1_self(x, value, bias=None, epsilon=1e-3):
    dot = jnp.einsum("...qhd,...khd->...hqk", x, x)
    x_sq = jnp.sum(x**2, axis=-1, keepdims=True).transpose(0, 2, 1, 3)
    raw_dist = x_sq + jnp.swapaxes(x_sq, -2, -1) - 2.0 * dot
    dist = jnp.maximum(raw_dist, 0.0) + epsilon
    numerator = dot if bias is None else dot + bias
    scores = numerator**2 / (dist * jnp.sqrt(jnp.float32(x.shape[-1])))
    normalizer = jnp.sum(scores, axis=-1, keepdims=True)
    scores = jnp.where(jnp.eye(x.shape[-3], dtype=jnp.bool_), 0.0, scores)
    weights = scores / (normalizer + 1e-12)
    return jnp.einsum("...hqk,...khd->...qhd", weights, value)


@pytest.mark.parametrize("case", ["positive", "zero", "negative"])
def test_fused_l1_clamp_subgradient_matches_autodiff(case):
    if case == "positive":
        query = jnp.array([[[[0.0, 1.0]], [[2.0, -1.0]]]])
        key = jnp.array([[[[1.0, 0.0]], [[-1.0, 2.0]]]])
    elif case == "zero":
        query = jnp.array([[[[1.0, 2.0]], [[-2.0, 1.0]]]])
        key = query
    else:
        # Deterministically contains negative reconstructed self-distances from
        # fp32 cancellation (seed 1, width 16, scale 10).
        query = jax.random.normal(jax.random.key(1), (1, 2, 1, 16)) * 10.0
        key = query
    value = jax.random.normal(jax.random.key(9), query.shape)
    dot = jnp.einsum("...qhd,...khd->...hqk", query, key)
    q_sq = jnp.sum(query**2, axis=-1, keepdims=True).transpose(0, 2, 1, 3)
    k_sq = jnp.sum(key**2, axis=-1, keepdims=True).transpose(0, 2, 1, 3)
    raw_dist = q_sq + jnp.swapaxes(k_sq, -2, -1) - 2.0 * dot
    if case == "positive":
        assert jnp.all(raw_dist > 0.0)
    elif case == "zero":
        assert jnp.any(raw_dist == 0.0)
    else:
        assert jnp.any(raw_dist < 0.0)

    def fused_loss(q, k, v):
        return jnp.sum(fused_yat_l1_attention(q, k, v, epsilon=1e-3) ** 2)

    def reference_loss(q, k, v):
        return jnp.sum(_reference_l1(q, k, v) ** 2)

    actual = jax.grad(fused_loss, argnums=(0, 1, 2))(query, key, value)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(query, key, value)
    for got, want in zip(actual, expected):
        assert jnp.allclose(got, want, rtol=2e-5, atol=2e-5)


def test_fused_l1_self_negative_clamp_subgradient_matches_autodiff():
    x = jax.random.normal(jax.random.key(1), (1, 2, 1, 16)) * 10.0
    value = jax.random.normal(jax.random.key(12), x.shape)

    def fused_loss(x, value):
        return jnp.sum(fused_yat_l1_self_attention(x, value, epsilon=1e-3) ** 2)

    def reference_loss(x, value):
        return jnp.sum(_reference_l1_self(x, value) ** 2)

    actual = jax.grad(fused_loss, argnums=(0, 1))(x, value)
    expected = jax.grad(reference_loss, argnums=(0, 1))(x, value)
    for got, want in zip(actual, expected):
        assert jnp.allclose(got, want, rtol=2e-5, atol=2e-5)


def _bias_for_case(case, batch, heads, q_length, kv_length):
    shapes = {
        "scalar": (),
        "per_head": (heads, 1, 1),
        "batched": (batch, heads, 1, kv_length),
        "full": (batch, heads, q_length, kv_length),
    }
    return jax.random.normal(jax.random.key(50), shapes[case]) * 0.05


@pytest.mark.parametrize("bias_case", ["scalar", "per_head", "batched", "full"])
def test_fused_l1_all_operand_gradients_for_broadcast_bias(bias_case):
    query = jax.random.normal(jax.random.key(51), (2, 3, 2, 4))
    key = jax.random.normal(jax.random.key(52), (2, 4, 2, 4))
    value = jax.random.normal(jax.random.key(53), (2, 4, 2, 5))
    bias = _bias_for_case(bias_case, 2, 2, 3, 4)
    epsilon = jnp.array(0.1)

    def fused_loss(q, k, v, b, e):
        return jnp.sum(
            fused_yat_l1_attention(q, k, v, bias=b, epsilon=e) ** 2
        )

    def reference_loss(q, k, v, b, e):
        return jnp.sum(_reference_l1(q, k, v, bias=b, epsilon=e) ** 2)

    actual = jax.grad(fused_loss, argnums=(0, 1, 2, 3, 4))(
        query, key, value, bias, epsilon
    )
    expected = jax.grad(reference_loss, argnums=(0, 1, 2, 3, 4))(
        query, key, value, bias, epsilon
    )
    for got, want in zip(actual, expected):
        assert jnp.allclose(got, want, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("bias_case", ["scalar", "per_head", "batched", "full"])
def test_fused_l1_self_all_operand_gradients_for_broadcast_bias(bias_case):
    x = jax.random.normal(jax.random.key(54), (2, 3, 2, 4))
    value = jax.random.normal(jax.random.key(55), (2, 3, 2, 5))
    bias = _bias_for_case(bias_case, 2, 2, 3, 3)
    epsilon = jnp.array(0.1)

    def fused_loss(x, v, b, e):
        return jnp.sum(
            fused_yat_l1_self_attention(x, v, bias=b, epsilon=e) ** 2
        )

    def reference_loss(x, v, b, e):
        return jnp.sum(_reference_l1_self(x, v, bias=b, epsilon=e) ** 2)

    actual = jax.grad(fused_loss, argnums=(0, 1, 2, 3))(
        x, value, bias, epsilon
    )
    expected = jax.grad(reference_loss, argnums=(0, 1, 2, 3))(
        x, value, bias, epsilon
    )
    for got, want in zip(actual, expected):
        assert jnp.allclose(got, want, rtol=3e-5, atol=3e-5)


def test_multi_head_honors_output_projection_initializers_and_default_decode():
    module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        qkv_features=4,
        out_features=3,
        out_kernel_init=nnx.initializers.zeros_init(),
        out_bias_init=nnx.initializers.constant(2.5),
        rngs=nnx.Rngs(0),
    )
    output = module(jnp.ones((2, 5, 8)), deterministic=True)
    assert output.shape == (2, 5, 3)
    assert jnp.all(output == 2.5)


def test_multi_head_normalize_qk_is_parameter_free_per_head_l2():
    captured = {}

    def capture_attention(query, key, value, **_):
        captured["query"] = query
        captured["key"] = key
        return value

    module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        normalize_qk=True,
        attention_fn=capture_attention,
        decode=False,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(3), (2, 4, 8))
    projected_q = module.query(x).reshape(2, 4, 2, 4)
    projected_k = module.key(x).reshape(2, 4, 2, 4)
    expected_q = projected_q / jnp.maximum(
        jnp.linalg.norm(projected_q, axis=-1, keepdims=True), 1e-12
    )
    expected_k = projected_k / jnp.maximum(
        jnp.linalg.norm(projected_k, axis=-1, keepdims=True), 1e-12
    )
    module(x, deterministic=True)

    assert module.query_ln is None and module.key_ln is None
    assert jnp.allclose(captured["query"], expected_q, atol=1e-6)
    assert jnp.allclose(captured["key"], expected_k, atol=1e-6)


def test_multi_head_normalize_qk_matches_torch_for_zero_and_tiny_vectors():
    torch = pytest.importorskip("torch")
    captured = {}

    def capture_attention(query, key, value, **_):
        captured["query"] = query
        captured["key"] = key
        return value

    module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        normalize_qk=True,
        attention_fn=capture_attention,
        use_alpha=False,
        decode=False,
        rngs=nnx.Rngs(0),
    )
    identity = jnp.eye(8, dtype=jnp.float32)
    module.query.kernel[...] = identity
    module.key.kernel[...] = identity
    module.query.bias[...] = 0.0
    module.key.bias[...] = 0.0
    inputs = jnp.array(
        [[
            [0.0] * 8,
            [1e-14, -1e-14, 2e-14, -2e-14, 0.0, 0.0, 1e-15, -1e-15],
            [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.0, -2.0],
        ]],
        dtype=jnp.float32,
    )
    module(inputs, deterministic=True)
    projected = inputs.reshape(1, 3, 2, 4)
    expected = torch.nn.functional.normalize(
        torch.from_numpy(np.array(projected, copy=True)), p=2, dim=-1
    ).numpy()
    assert np.allclose(np.asarray(captured["query"]), expected, atol=1e-7)
    assert np.allclose(np.asarray(captured["key"]), expected, atol=1e-7)


def test_multi_head_normalize_qk_matches_torch_with_synchronized_weights():
    torch = pytest.importorskip("torch")
    from nmn.torch import MultiHeadYatAttention

    nnx_module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        normalize_qk=True,
        use_alpha=False,
        decode=False,
        rngs=nnx.Rngs(0),
    )
    torch_module = MultiHeadYatAttention(
        embed_dim=8,
        num_heads=2,
        normalize_qk=True,
        use_alpha=False,
    )
    generator = np.random.default_rng(0)

    with torch.no_grad():
        for nnx_layer, torch_layer in (
            (nnx_module.query, torch_module.q_proj),
            (nnx_module.key, torch_module.k_proj),
            (nnx_module.value, torch_module.v_proj),
        ):
            kernel = generator.normal(size=(8, 8)).astype(np.float32) * 0.1
            bias = generator.normal(size=(8,)).astype(np.float32) * 0.1
            nnx_layer.kernel[...] = jnp.asarray(kernel)
            nnx_layer.bias[...] = jnp.asarray(bias)
            torch_layer.weight.copy_(torch.from_numpy(kernel.T))
            torch_layer.bias.copy_(torch.from_numpy(bias))

        out_kernel = generator.normal(size=(8, 8)).astype(np.float32) * 0.1
        out_bias = generator.normal(size=(8,)).astype(np.float32) * 0.1
        nnx_module.out.kernel[...] = jnp.asarray(out_kernel.reshape(2, 4, 8))
        nnx_module.out.bias[...] = jnp.asarray(out_bias)
        torch_module.out_proj.weight.copy_(torch.from_numpy(out_kernel.T))
        torch_module.out_proj.bias.copy_(torch.from_numpy(out_bias))

    inputs = generator.normal(size=(2, 5, 8)).astype(np.float32)
    nnx_output = np.asarray(nnx_module(jnp.asarray(inputs), deterministic=True))
    torch_output = (
        torch_module(torch.from_numpy(inputs), deterministic=True).detach().numpy()
    )
    assert np.allclose(nnx_output, torch_output, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_multi_head_full_causal_matches_incremental_decode(batch_size):
    length, features = 5, 8
    module = MultiHeadAttention(
        num_heads=2,
        in_features=features,
        decode=False,
        use_alpha=False,
        rngs=nnx.Rngs(10),
    )
    x = jax.random.normal(jax.random.key(11), (batch_size, length, features))
    full = module(x, mask=_causal_mask(batch_size, length), deterministic=True)

    module.init_cache(x.shape)
    incremental = jnp.concatenate(
        [
            module(x[:, i : i + 1], decode=True, deterministic=True)
            for i in range(length)
        ],
        axis=1,
    )
    assert jnp.allclose(incremental, full, rtol=2e-5, atol=2e-5)
    with pytest.raises(ValueError, match="cache is full"):
        module(x[:, :1], decode=True, deterministic=True)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotary_full_causal_matches_incremental_decode(batch_size):
    length, features = 5, 8
    module = RotaryYatAttention(
        embed_dim=features,
        num_heads=2,
        max_seq_len=length,
        use_alpha=False,
        rngs=nnx.Rngs(20),
    )
    x = jax.random.normal(jax.random.key(21), (batch_size, length, features))
    full = module(x, mask=_causal_mask(batch_size, length), deterministic=True)

    module.init_cache(batch_size, length)
    incremental = jnp.concatenate(
        [
            module(x[:, i : i + 1], decode=True, deterministic=True)
            for i in range(length)
        ],
        axis=1,
    )
    assert jnp.allclose(incremental, full, rtol=3e-5, atol=3e-5)
    with pytest.raises(ValueError, match="cache is full"):
        module(x[:, :1], decode=True, deterministic=True)


def test_multi_head_decode_cache_mutates_under_nnx_jit():
    module = MultiHeadAttention(
        num_heads=2, in_features=8, decode=True, use_alpha=False, rngs=nnx.Rngs(30)
    )
    x = jax.random.normal(jax.random.key(32), (2, 4, 8))
    full = module(x, mask=_causal_mask(2, 4), decode=False, deterministic=True)
    module.init_cache(x.shape)

    @nnx.jit
    def step(attention, token):
        return attention(token, deterministic=True)

    incremental = jnp.concatenate(
        [step(module, x[:, i : i + 1]) for i in range(x.shape[1])], axis=1
    )
    assert jnp.allclose(incremental, full, rtol=2e-5, atol=2e-5)
    assert int(module.cache_index[...]) == 4


def test_rotary_decode_cache_mutates_under_nnx_jit():
    module = RotaryYatAttention(
        embed_dim=8, num_heads=2, max_seq_len=4, use_alpha=False, rngs=nnx.Rngs(31)
    )
    x = jax.random.normal(jax.random.key(33), (2, 4, 8))
    full = module(x, mask=_causal_mask(2, 4), deterministic=True)
    module.init_cache(2, 4)

    @nnx.jit
    def step(attention, token):
        return attention(token, decode=True, deterministic=True)

    incremental = jnp.concatenate(
        [step(module, x[:, i : i + 1]) for i in range(x.shape[1])], axis=1
    )
    assert jnp.allclose(incremental, full, rtol=3e-5, atol=3e-5)
    assert int(module.cache_index[...]) == 4


def _cache_snapshot(module):
    return (
        np.array(module.cached_key[...], copy=True),
        np.array(module.cached_value[...], copy=True),
        int(module.cache_index[...]),
    )


def _assert_cache_unchanged(module, snapshot):
    key, value, index = snapshot
    assert np.array_equal(np.asarray(module.cached_key[...]), key)
    assert np.array_equal(np.asarray(module.cached_value[...]), value)
    assert int(module.cache_index[...]) == index


@pytest.mark.parametrize("kind", ["multi_head", "rotary"])
def test_decode_jit_overflow_fails_without_overwriting_cache(kind):
    if kind == "multi_head":
        module = MultiHeadAttention(
            num_heads=2,
            in_features=8,
            decode=True,
            use_alpha=False,
            rngs=nnx.Rngs(60),
        )
        module.init_cache((1, 2, 8))

        @nnx.jit
        def step(attention, token):
            return attention(token, deterministic=True)

    else:
        module = RotaryYatAttention(
            embed_dim=8,
            num_heads=2,
            max_seq_len=2,
            use_alpha=False,
            rngs=nnx.Rngs(60),
        )
        module.init_cache(1, 2)

        @nnx.jit
        def step(attention, token):
            return attention(token, decode=True, deterministic=True)

    step(module, jnp.ones((1, 1, 8)))
    step(module, jnp.full((1, 1, 8), 2.0))
    snapshot = _cache_snapshot(module)
    with pytest.raises(Exception, match="cache is full"):
        step(module, jnp.full((1, 1, 8), 3.0))
    _assert_cache_unchanged(module, snapshot)


@pytest.mark.parametrize("kind", ["multi_head", "rotary"])
def test_decode_validation_failures_are_transactional(kind):
    if kind == "multi_head":
        module = MultiHeadAttention(
            num_heads=2,
            in_features=8,
            decode=True,
            use_alpha=False,
            rngs=nnx.Rngs(61),
        )
        module.init_cache((1, 2, 8))
        call = lambda token, mask=None: module(
            token, mask=mask, deterministic=True
        )
    else:
        module = RotaryYatAttention(
            embed_dim=8,
            num_heads=2,
            max_seq_len=2,
            use_alpha=False,
            rngs=nnx.Rngs(61),
        )
        module.init_cache(1, 2)
        call = lambda token, mask=None: module(
            token, mask=mask, decode=True, deterministic=True
        )

    snapshot = _cache_snapshot(module)
    with pytest.raises(ValueError, match="shape|one token"):
        call(jnp.ones((1, 2, 8)))
    _assert_cache_unchanged(module, snapshot)

    invalid_mask = jnp.ones((1, 1, 2, 2), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="[Mm]ask shape"):
        call(jnp.ones((1, 1, 8)), mask=invalid_mask)
    _assert_cache_unchanged(module, snapshot)


def test_multi_head_attention_failure_does_not_commit_cache():
    def failing_attention(*_, **__):
        raise RuntimeError("attention failed")

    module = MultiHeadAttention(
        num_heads=2,
        in_features=8,
        decode=True,
        use_alpha=False,
        attention_fn=failing_attention,
        rngs=nnx.Rngs(62),
    )
    module.init_cache((1, 2, 8))
    snapshot = _cache_snapshot(module)
    with pytest.raises(RuntimeError, match="attention failed"):
        module(jnp.ones((1, 1, 8)), deterministic=True)
    _assert_cache_unchanged(module, snapshot)


@pytest.mark.parametrize("kind", ["slay", "maclaurin", "radial"])
def test_rotary_performer_respects_key_padding_mask(kind):
    module = RotaryYatAttention(
        embed_dim=8,
        num_heads=2,
        max_seq_len=8,
        use_performer=True,
        performer_kind=kind,
        performer_num_features=32,
        performer_sketch_m=2,
        performer_num_radial=2,
        performer_radial_dim=2,
        use_alpha=False,
        rngs=nnx.Rngs(40),
    )
    x = jax.random.normal(jax.random.key(41), (2, 5, 8))
    all_keys = jnp.ones((2, 1, 1, 5), dtype=jnp.bool_)
    padded = all_keys.at[..., -1].set(False)
    out_all = module(x, mask=all_keys, deterministic=True)
    out_padded = module(x, mask=padded, deterministic=True)
    assert not jnp.allclose(out_all, out_padded)

    with pytest.raises(ValueError, match="key-padding masks"):
        module(x, mask=_causal_mask(2, 5), deterministic=True)
    module.init_cache(2, 5)
    with pytest.raises(ValueError, match="Incremental decode"):
        module(x[:, :1], decode=True, deterministic=True)


@pytest.mark.parametrize("failure", ["mask", "decode"])
def test_rotary_performer_rejection_does_not_advance_rng_or_cache(failure):
    module = RotaryYatAttention(
        embed_dim=8,
        num_heads=2,
        max_seq_len=5,
        dropout_rate=0.5,
        use_performer=True,
        performer_kind="maclaurin",
        performer_num_features=16,
        use_alpha=False,
        rngs=nnx.Rngs(63),
    )
    module.init_cache(1, 5)
    call_rngs = nnx.Rngs(dropout=64)
    rng_count = int(call_rngs.dropout.count[...])
    cache = _cache_snapshot(module)
    x = jnp.ones((1, 5, 8))

    if failure == "mask":
        with pytest.raises(ValueError, match="key-padding masks"):
            module(
                x,
                mask=_causal_mask(1, 5),
                deterministic=False,
                rngs=call_rngs,
            )
    else:
        with pytest.raises(ValueError, match="Incremental decode"):
            module(
                x[:, :1],
                decode=True,
                deterministic=False,
                rngs=call_rngs,
            )

    assert int(call_rngs.dropout.count[...]) == rng_count
    _assert_cache_unchanged(module, cache)
