"""Regression coverage for bugs surfaced by the codebase audit (issues #17–#24).

These tests don't replace per-framework suites — they exist so that if any of
the audit-era fixes regress, ``pytest tests/integration/`` flags it immediately
rather than days later under specific user code paths (channels_first models,
grouped transposed conv, long-context rotary attention, etc.).
"""

from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# #19 — NNX YatNMN spherical guard against zero-norm inputs / columns
# =============================================================================


def test_nnx_spherical_yatnmn_zero_input_is_finite() -> None:
    pytest.importorskip("flax")
    import jax.numpy as jnp
    from flax import nnx

    from nmn.nnx import YatNMN

    layer = YatNMN(in_features=4, out_features=2, spherical=True, rngs=nnx.Rngs(0))
    y = layer(jnp.zeros((1, 4)))
    assert bool(jnp.isfinite(y).all()), "spherical YatNMN must not NaN on zero input"


# =============================================================================
# #18 — Linen YatEmbed.attend distance clamp
# =============================================================================


def test_linen_yat_embed_attend_finite_when_query_near_embedding() -> None:
    pytest.importorskip("flax")
    import jax
    import jax.numpy as jnp

    from nmn.linen import YatEmbed

    embed = YatEmbed(num_embeddings=8, features=16)
    # Initialize through `attend` so the alpha param exists in the variable tree.
    query0 = jnp.zeros((16,), dtype=jnp.float32)
    params = embed.init(jax.random.key(0), query0, method=embed.attend)
    embedding = params["params"]["embedding"]
    # Query exactly equal to an embedding row drives the YAT distance toward zero;
    # without the jnp.maximum(..., 0) clamp in attend(), bf16/fp16 cancellation
    # can flip the denominator sign. Verify finiteness for both that pathological
    # case and a random query.
    scores_near = embed.apply(params, embedding[0], method=embed.attend)
    scores_far = embed.apply(params, jax.random.normal(jax.random.key(1), (16,)), method=embed.attend)
    assert bool(jnp.isfinite(scores_near).all()), "attend NaN'd on near-zero distance"
    assert bool(jnp.isfinite(scores_far).all()), "attend NaN'd on random query"


# =============================================================================
# #17 — Keras YatConv* channels_first parity with channels_last
# =============================================================================


CHANNELS_FIRST_CASES = [
    # (factory, channels_last input shape, channels_last→channels_first perm)
    ("YatConv1D", (2, 8, 3), (0, 2, 1)),
    ("YatConvTranspose1D", (2, 8, 3), (0, 2, 1)),
    ("YatConv2D", (2, 8, 8, 3), (0, 3, 1, 2)),
    ("YatConvTranspose2D", (2, 8, 8, 3), (0, 3, 1, 2)),
    ("YatConv3D", (2, 4, 4, 4, 3), (0, 4, 1, 2, 3)),
    ("YatConvTranspose3D", (2, 4, 4, 4, 3), (0, 4, 1, 2, 3)),
]


@pytest.mark.parametrize("cls_name,in_shape_last,perm", CHANNELS_FIRST_CASES)
def test_keras_conv_channels_first_matches_channels_last(cls_name, in_shape_last, perm) -> None:
    keras = pytest.importorskip("keras")
    from nmn.keras import conv as kconv

    cls = getattr(kconv, cls_name)
    x_last = np.random.RandomState(0).randn(*in_shape_last).astype("float32")
    x_first = np.transpose(x_last, perm)

    keras.utils.set_random_seed(42)
    y_last = np.asarray(cls(filters=4, kernel_size=3, data_format="channels_last")(x_last))
    keras.utils.set_random_seed(42)
    y_first = np.asarray(cls(filters=4, kernel_size=3, data_format="channels_first")(x_first))
    inv_perm = [0] + list(range(2, y_first.ndim)) + [1]
    np.testing.assert_allclose(y_last, np.transpose(y_first, inv_perm), atol=1e-5)


# =============================================================================
# #21 — torch YatConv2D DropConnect rescaling
# =============================================================================


def test_torch_yat_conv2d_dropconnect_is_rescaled() -> None:
    torch = pytest.importorskip("torch")
    from nmn.torch.layers.yat_conv2d import YatConv2D

    torch.manual_seed(0)
    layer = YatConv2D(8, 16, 3, use_dropconnect=True, drop_rate=0.5)
    x = torch.randn(8, 8, 16, 16)
    layer.train()
    train_out = layer(x).abs().mean().item()
    layer.eval()
    eval_out = layer(x).abs().mean().item()
    # Without rescaling, training-time forward magnitudes collapse to roughly
    # (1 − drop_rate) of eval. Inverted dropout keeps them in the same ballpark.
    assert train_out > 0.4 * eval_out


# =============================================================================
# #22 — torch YatConvTranspose* accepts groups > 1
# =============================================================================


GROUP_CASES = [
    ("YatConvTranspose1D", (2, 8, 10), 2),
    ("YatConvTranspose1D", (2, 8, 10), 4),
    ("YatConvTranspose2D", (2, 8, 10, 10), 2),
    ("YatConvTranspose2D", (2, 8, 10, 10), 4),
    ("YatConvTranspose3D", (2, 8, 6, 6, 6), 2),
    ("YatConvTranspose3D", (2, 8, 6, 6, 6), 4),
]


@pytest.mark.parametrize("cls_name,in_shape,groups", GROUP_CASES)
def test_torch_yat_conv_transpose_supports_groups(cls_name, in_shape, groups) -> None:
    torch = pytest.importorskip("torch")
    from nmn.torch.layers import yat_conv_transpose1d, yat_conv_transpose2d, yat_conv_transpose3d

    module = {
        "YatConvTranspose1D": yat_conv_transpose1d.YatConvTranspose1D,
        "YatConvTranspose2D": yat_conv_transpose2d.YatConvTranspose2D,
        "YatConvTranspose3D": yat_conv_transpose3d.YatConvTranspose3D,
    }[cls_name]

    layer = module(in_channels=8, out_channels=16, kernel_size=3, groups=groups)
    y = layer(torch.randn(*in_shape))
    assert y.shape[1] == 16


# =============================================================================
# #23 — Rotary YAT explicit bounds error
# =============================================================================


def test_rotary_yat_raises_when_seq_len_exceeds_max() -> None:
    pytest.importorskip("flax")
    import jax.numpy as jnp

    from nmn.nnx.layers.attention.rotary_yat import (
        apply_rotary_emb,
        precompute_freqs_cis,
    )

    freqs_cos, freqs_sin = precompute_freqs_cis(64, 128)
    with pytest.raises(ValueError, match="max_seq_len"):
        apply_rotary_emb(jnp.ones((1, 200, 4, 64)), freqs_cos, freqs_sin)
    with pytest.raises(ValueError, match="max_seq_len"):
        apply_rotary_emb(jnp.ones((1, 100, 4, 64)), freqs_cos, freqs_sin, position_offset=100)


# =============================================================================
# Cross-framework alias surface
# =============================================================================


def test_nnx_exposes_cross_framework_aliases() -> None:
    pytest.importorskip("flax")
    from nmn import nnx as ncnnx

    # Long names should resolve to the canonical NNX classes.
    assert ncnnx.YatEmbed is ncnnx.Embed
    assert ncnnx.YatConv1D is ncnnx.YatConv
    assert ncnnx.YatConv2D is ncnnx.YatConv
    assert ncnnx.YatConv3D is ncnnx.YatConv
    assert ncnnx.YatConvTranspose1D is ncnnx.YatConvTranspose
    assert ncnnx.MultiHeadYatAttention is ncnnx.MultiHeadAttention
