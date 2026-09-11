"""Deterministic conformance formerly mixed into the attention benchmark."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from nmn.nnx.layers.attention import (
    RotaryYatAttention,
    create_yat_projection,
    dot_product_attention,
    yat_attention,
    yat_attention_normalized,
    yat_performer_attention,
)


@pytest.mark.parametrize("normalize_inputs", [False, True])
def test_attention_function_outputs_are_finite_and_shape_correct(normalize_inputs):
    q_key, k_key, v_key, projection_key = jax.random.split(jax.random.key(0), 4)
    shape = (2, 16, 4, 8)
    q = jax.random.normal(q_key, shape)
    k = jax.random.normal(k_key, shape)
    v = jax.random.normal(v_key, shape)
    projection = create_yat_projection(projection_key, 16, shape[-1])
    outputs = (
        dot_product_attention(q, k, v),
        yat_attention(q, k, v),
        yat_attention_normalized(q, k, v),
        yat_performer_attention(q, k, v, projection, normalize_inputs=normalize_inputs),
    )
    for output in outputs:
        assert output.shape == shape
        assert jnp.isfinite(output).all()


@pytest.mark.parametrize(
    "use_performer,performer_normalize",
    [(False, False), (True, False), (True, True)],
)
def test_rotary_attention_outputs_are_finite_and_shape_correct(
    use_performer, performer_normalize
):
    batch, sequence_length, embed_dim, num_heads = 2, 16, 32, 4
    layer = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=sequence_length,
        use_performer=use_performer,
        num_prf_features=16,
        performer_normalize=performer_normalize,
        rngs=nnx.Rngs(0),
    )
    inputs = jax.random.normal(jax.random.key(1), (batch, sequence_length, embed_dim))
    output = jax.jit(lambda value: layer(value, deterministic=True))(inputs)
    assert output.shape == inputs.shape
    assert jnp.isfinite(output).all()


@pytest.mark.slow
def test_long_sequence_performer_shape_and_finiteness():
    sequence_length, embed_dim, num_heads = 2048, 256, 4
    layer = RotaryYatAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=sequence_length,
        use_performer=True,
        num_prf_features=128,
        performer_normalize=True,
        rngs=nnx.Rngs(0),
    )
    inputs = jax.random.normal(jax.random.key(1), (1, sequence_length, embed_dim))
    output = jax.jit(lambda value: layer(value, deterministic=True))(inputs)
    assert output.shape == inputs.shape
    assert jnp.isfinite(output).all()
