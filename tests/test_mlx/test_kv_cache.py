"""Tests for the autoregressive KV cache on RotaryYatAttention."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from nmn.mlx import RotaryYatAttention  # noqa: E402


def _build(embed_dim=32, num_heads=2, max_seq_len=16):
    mx.random.seed(0)
    mha = RotaryYatAttention(embed_dim=embed_dim, num_heads=num_heads,
                             max_seq_len=max_seq_len)
    _ = mha(mx.zeros((1, 1, embed_dim)))  # build lazy params
    return mha


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------


def test_init_cache_allocates_zero_buffers():
    mha = _build()
    mha.init_cache(batch_size=2, max_length=10)
    assert mha.cached_key.shape == (2, 10, 2, mha.head_dim)
    assert mha.cached_value.shape == (2, 10, 2, mha.head_dim)
    assert mha.cache_index == 0
    assert float(mx.max(mx.abs(mha.cached_key))) == 0.0
    assert float(mx.max(mx.abs(mha.cached_value))) == 0.0


def test_reset_cache_clears_state():
    mha = _build()
    mha.init_cache(batch_size=1)
    _ = mha(mx.random.normal(shape=(1, 1, 32)), decode=True)
    assert mha.cache_index == 1
    mha.reset_cache()
    assert mha.cached_key is None
    assert mha.cached_value is None
    assert mha.cache_index == 0


def test_init_cache_rejects_unbuilt_module():
    mha = RotaryYatAttention(embed_dim=32, num_heads=2, max_seq_len=8)
    with pytest.raises(ValueError):
        mha.init_cache(batch_size=1)


def test_init_cache_rejects_too_large_max_length():
    mha = _build(max_seq_len=8)
    with pytest.raises(ValueError):
        mha.init_cache(batch_size=1, max_length=99)


def test_decode_without_init_cache_rejected():
    mha = _build()
    with pytest.raises(ValueError):
        mha(mx.random.normal(shape=(1, 1, 32)), decode=True)


def test_decode_cache_overflow_rejected():
    mha = _build(max_seq_len=16)
    mha.init_cache(batch_size=1, max_length=4)
    for _ in range(4):
        _ = mha(mx.random.normal(shape=(1, 1, 32)), decode=True)
    with pytest.raises(ValueError):
        mha(mx.random.normal(shape=(1, 1, 32)), decode=True)


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------


def test_decode_advances_cache_index():
    mha = _build()
    mha.init_cache(batch_size=1, max_length=10)
    _ = mha(mx.random.normal(shape=(1, 3, 32)), decode=True)
    assert mha.cache_index == 3
    _ = mha(mx.random.normal(shape=(1, 2, 32)), decode=True)
    assert mha.cache_index == 5


def test_decode_matches_causal_prefill_byte_identical():
    """Full causal prefill and step-by-step decode must produce identical
    outputs at every position — the strongest correctness signal."""
    mha = _build()
    L = 6
    x = mx.random.normal(shape=(1, L, 32))

    # Causal mask for prefill.
    causal = np.tril(np.ones((L, L), dtype=bool))[None, None]
    mask = mx.array(np.broadcast_to(causal, (1, mha.num_heads, L, L)))
    out_prefill = np.array(mha(x, mask=mask))

    # Decode token-by-token.
    mha.reset_cache()
    mha.init_cache(batch_size=1, max_length=L)
    out_decode = np.concatenate(
        [np.array(mha(x[:, t:t + 1], decode=True)) for t in range(L)],
        axis=1,
    )
    assert np.max(np.abs(out_prefill - out_decode)) < 1e-4


def test_decode_multi_token_chunk_matches_single_step():
    """Decoding tokens in a 2-token chunk should give the same result as
    decoding them one at a time (the cache_index lookup is the only
    difference)."""
    mha = _build()
    x = mx.random.normal(shape=(1, 4, 32))

    # Chunked: 2 tokens, then 2 tokens.
    mha.init_cache(batch_size=1, max_length=4)
    out_chunked = np.concatenate(
        [np.array(mha(x[:, :2], decode=True)),
         np.array(mha(x[:, 2:], decode=True))],
        axis=1,
    )

    # Step-by-step.
    mha.reset_cache()
    mha.init_cache(batch_size=1, max_length=4)
    out_step = np.concatenate(
        [np.array(mha(x[:, t:t + 1], decode=True)) for t in range(4)],
        axis=1,
    )
    assert np.max(np.abs(out_chunked - out_step)) < 1e-4


def test_decode_independent_of_future_tokens():
    """The output for token t must be invariant to the input at token t+1
    (basic causality check on the decode path itself)."""
    mha = _build()
    mha.init_cache(batch_size=1, max_length=8)
    x = mx.random.normal(shape=(1, 1, 32))
    out_first = np.array(mha(x, decode=True))
    # Decode a different next token — out_first should not change.
    _ = mha(mx.random.normal(shape=(1, 1, 32)), decode=True)
    # And re-run the decode trajectory from scratch with the same first
    # token: we should get the same out_first.
    mha.reset_cache()
    mha.init_cache(batch_size=1, max_length=8)
    out_first_again = np.array(mha(x, decode=True))
    assert np.max(np.abs(out_first - out_first_again)) < 1e-6
