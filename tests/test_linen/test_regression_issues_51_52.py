"""Regression tests for Linen attention alpha and grouped YAT convolution."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
jnp = jax.numpy
from flax.core import unfreeze  # noqa: E402

from nmn.linen import (  # noqa: E402
    MultiHeadAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
)


def _assert_tree_allclose(actual, expected, *, rtol=2e-5, atol=2e-6):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected)
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf), np.asarray(expected_leaf), rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("normalization", ["softmax", "l1"])
def test_attention_constant_and_matched_learnable_alpha_have_same_stage_and_vjp(
    normalization,
):
    """A constant alpha must scale logits, exactly like a matched parameter."""
    alpha = 1.75
    common = dict(
        num_heads=2,
        qkv_features=8,
        out_features=6,
        use_bias=True,
        dropout_rate=0.0,
        normalization=normalization,
    )
    learnable = MultiHeadAttention(**common)
    constant = MultiHeadAttention(**common, constant_alpha=alpha)

    q = jax.random.normal(jax.random.key(1), (2, 3, 6))
    k = jax.random.normal(jax.random.key(2), (2, 4, 6))
    v = jax.random.normal(jax.random.key(3), (2, 4, 6))
    learnable_params = unfreeze(
        learnable.init(jax.random.key(0), q, k, v)["params"]
    )
    learnable_params["alpha"] = jnp.asarray([alpha], dtype=jnp.float32)

    # Synchronize every shared projection parameter.  The constant model must
    # not register an alpha parameter of its own.
    constant_params = dict(learnable_params)
    constant_params.pop("alpha")
    assert "alpha" not in constant.init(jax.random.key(9), q, k, v)["params"]

    learned_out = learnable.apply({"params": learnable_params}, q, k, v)
    constant_out = constant.apply({"params": constant_params}, q, k, v)
    np.testing.assert_allclose(learned_out, constant_out, rtol=1e-6, atol=1e-6)

    cotangent = jnp.linspace(-0.7, 0.9, learned_out.size).reshape(learned_out.shape)

    def learned_loss(params, q_arg, k_arg, v_arg):
        return jnp.vdot(
            learnable.apply({"params": params}, q_arg, k_arg, v_arg), cotangent
        )

    def constant_loss(params, q_arg, k_arg, v_arg):
        return jnp.vdot(
            constant.apply({"params": params}, q_arg, k_arg, v_arg), cotangent
        )

    learned_grads = jax.grad(learned_loss, argnums=(0, 1, 2, 3))(
        learnable_params, q, k, v
    )
    constant_grads = jax.grad(constant_loss, argnums=(0, 1, 2, 3))(
        constant_params, q, k, v
    )
    learned_param_grads = unfreeze(learned_grads[0])
    learned_param_grads.pop("alpha")

    _assert_tree_allclose(learned_param_grads, constant_grads[0])
    _assert_tree_allclose(learned_grads[1:], constant_grads[1:])


_GROUPED_CASES = [
    (YatConv1D, (1, 6, 4), (2,), ("NWC", "WIO", "NWC"), 2, 6),
    (YatConv2D, (1, 4, 5, 4), (2, 2), ("NHWC", "HWIO", "NHWC"), 2, 6),
    (
        YatConv3D,
        (1, 3, 4, 4, 4),
        (2, 2, 2),
        ("NDHWC", "DHWIO", "NDHWC"),
        2,
        6,
    ),
    # Also cover more than two groups, so the channel mapping is not merely a
    # two-way special case.
    (YatConv1D, (1, 5, 8), (2,), ("NWC", "WIO", "NWC"), 4, 8),
]


def _grouped_reference(params, inputs, kernel_size, dimension_spec, groups):
    del dimension_spec  # The explicit reference intentionally does not call lax.conv.
    kernel = params["kernel"]
    output_spatial = tuple(
        input_size - kernel_extent + 1
        for input_size, kernel_extent in zip(inputs.shape[1:-1], kernel_size)
    )
    input_channels_per_group = inputs.shape[-1] // groups
    output_channels_per_group = kernel.shape[-1] // groups
    positions = []

    # Explicitly extract each receptive-field patch.  This reference avoids
    # conv_general_dilated so it independently verifies the group-to-filter
    # mapping in both the forward pass and autodiff.
    for output_index in np.ndindex(output_spatial):
        spatial_slices = tuple(
            slice(index, index + extent)
            for index, extent in zip(output_index, kernel_size)
        )
        channels = []
        for output_channel in range(kernel.shape[-1]):
            group = output_channel // output_channels_per_group
            channel_start = group * input_channels_per_group
            channel_stop = channel_start + input_channels_per_group
            patch = inputs[
                (slice(None),)
                + spatial_slices
                + (slice(channel_start, channel_stop),)
            ]
            filter_kernel = kernel[..., output_channel]
            reduction_axes = tuple(range(1, patch.ndim))
            dot = jnp.sum(patch * filter_kernel, axis=reduction_axes)
            distance = (
                jnp.sum(patch**2, axis=reduction_axes)
                + jnp.sum(filter_kernel**2)
                - 2.0 * dot
            )
            score_dot = dot + params["bias"][output_channel]
            channels.append(score_dot**2 / (distance + 1e-5))
        positions.append(jnp.stack(channels, axis=-1))

    scores = jnp.stack(positions, axis=1)
    scores = scores.reshape(
        (inputs.shape[0],) + output_spatial + (kernel.shape[-1],)
    )
    return scores * params["alpha"]


@pytest.mark.parametrize(
    "layer_type,input_shape,kernel_size,dimension_spec,groups,features",
    _GROUPED_CASES,
)
def test_grouped_convolution_matches_synchronized_reference_forward_and_vjp(
    layer_type, input_shape, kernel_size, dimension_spec, groups, features
):
    layer = layer_type(
        features=features,
        kernel_size=kernel_size,
        feature_group_count=groups,
        padding="VALID",
    )
    inputs = jax.random.normal(jax.random.key(21 + len(input_shape)), input_shape)
    params = layer.init(jax.random.key(20), inputs)["params"]

    actual = layer.apply({"params": params}, inputs)
    expected = _grouped_reference(
        params, inputs, kernel_size, dimension_spec, groups
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    cotangent = jnp.linspace(-0.8, 0.6, actual.size).reshape(actual.shape)

    def module_loss(params_arg, inputs_arg):
        return jnp.vdot(layer.apply({"params": params_arg}, inputs_arg), cotangent)

    def reference_loss(params_arg, inputs_arg):
        return jnp.vdot(
            _grouped_reference(
                params_arg, inputs_arg, kernel_size, dimension_spec, groups
            ),
            cotangent,
        )

    module_grads = jax.grad(module_loss, argnums=(0, 1))(params, inputs)
    reference_grads = jax.grad(reference_loss, argnums=(0, 1))(params, inputs)
    _assert_tree_allclose(module_grads, reference_grads, rtol=4e-5, atol=4e-6)


@pytest.mark.parametrize(
    "layer_type,input_shape,kernel_size",
    [
        (YatConv1D, (1, 5, 4), (2,)),
        (YatConv2D, (1, 4, 4, 4), (2, 2)),
        (YatConv3D, (1, 3, 3, 3, 4), (2, 2, 2)),
    ],
)
@pytest.mark.parametrize(
    "channels,features,match",
    [
        (3, 4, r"Input channels \(3\) must be divisible"),
        (4, 5, r"features \(5\) must be divisible"),
    ],
)
def test_grouped_convolution_rejects_invalid_channel_divisibility(
    layer_type, input_shape, kernel_size, channels, features, match
):
    invalid_shape = input_shape[:-1] + (channels,)
    inputs = jnp.ones(invalid_shape, dtype=jnp.float32)
    layer = layer_type(
        features=features,
        kernel_size=kernel_size,
        feature_group_count=2,
    )
    with pytest.raises(ValueError, match=match):
        layer.init(jax.random.key(30), inputs)
