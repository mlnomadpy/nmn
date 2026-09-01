"""Numerical parity tests for Pallas-fused YAT L1 attention."""

import ast
import importlib
import inspect
import math

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    from nmn.nnx.layers.attention.pallas_yat_attention import (
        pallas_yat_l1_attention,
    )

    pallas_module = importlib.import_module(
        "nmn.nnx.layers.attention.pallas_yat_attention"
    )
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


def _rand(shape, key=0):
    return jax.random.normal(jax.random.key(key), shape)


def _reference(q, k, v, *, epsilon=1e-5, causal=False):
    """Plain-JAX oracle, deliberately independent of either fused custom VJP."""
    q32, k32, v32 = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    dot = jnp.einsum("...qhd,...khd->...hqk", q32, k32)
    q_sq = jnp.einsum("...qhd,...qhd->...hq", q32, q32)[..., :, :, None]
    k_sq = jnp.einsum("...khd,...khd->...hk", k32, k32)[..., :, None, :]
    dist = jnp.maximum(q_sq + k_sq - 2.0 * dot, 0.0) + epsilon
    scores = jnp.square(dot) / (dist * math.sqrt(q.shape[-1]))
    if causal:
        q_pos = jnp.arange(q.shape[-3])[:, None]
        k_pos = jnp.arange(k.shape[-3])[None, :]
        scores = jnp.where(q_pos >= k_pos, scores, 0.0)
    weights = scores / jnp.maximum(scores.sum(axis=-1, keepdims=True), 1e-12)
    return jnp.einsum("...hqk,...khd->...qhd", weights, v32).astype(v.dtype)


def _cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


CASES = [
    pytest.param(2, 2, 4, 4, False, 4, 4, id="tiny-single-tile"),
    pytest.param(7, 11, 4, 3, False, 4, 4, id="ragged-multitile-v-smaller"),
    pytest.param(7, 11, 4, 6, False, 4, 4, id="ragged-multitile-v-larger"),
    pytest.param(5, 9, 4, 3, True, 4, 4, id="causal-q-shorter-than-kv"),
    pytest.param(9, 5, 4, 6, True, 4, 4, id="causal-q-longer-than-kv"),
    pytest.param(8, 13, 4, 5, True, 3, 5, id="unequal-ragged-tiles"),
]


@pytest.mark.parametrize("q_len,kv_len,head_dim,v_dim,causal,block_q,block_k", CASES)
def test_forward_matches_plain_jax(
    q_len,
    kv_len,
    head_dim,
    v_dim,
    causal,
    block_q,
    block_k,
):
    q = _rand((1, q_len, 2, head_dim), 0)
    k = _rand((1, kv_len, 2, head_dim), 1)
    v = _rand((1, kv_len, 2, v_dim), 2)
    actual = pallas_yat_l1_attention(
        q, k, v, causal=causal, block_q=block_q, block_k=block_k, interpret=True
    )
    expected = _reference(q, k, v, causal=causal)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert actual.shape == (1, q_len, 2, v_dim)


@pytest.mark.parametrize("q_len,kv_len,head_dim,v_dim,causal,block_q,block_k", CASES)
def test_gradients_match_plain_jax(
    q_len,
    kv_len,
    head_dim,
    v_dim,
    causal,
    block_q,
    block_k,
):
    q = _rand((1, q_len, 2, head_dim), 3)
    k = _rand((1, kv_len, 2, head_dim), 4)
    v = _rand((1, kv_len, 2, v_dim), 5)
    cotangent = _rand((1, q_len, 2, v_dim), 6)

    def pallas_loss(q, k, v):
        return jnp.vdot(
            pallas_yat_l1_attention(
                q, k, v, causal=causal, block_q=block_q, block_k=block_k, interpret=True
            ),
            cotangent,
        )

    def reference_loss(q, k, v):
        return jnp.vdot(_reference(q, k, v, causal=causal), cotangent)

    actual = jax.grad(pallas_loss, argnums=(0, 1, 2))(q, k, v)
    expected = jax.grad(reference_loss, argnums=(0, 1, 2))(q, k, v)
    for name, got, want in zip(("dQ", "dK", "dV"), actual, expected):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4, err_msg=name)
        assert _cosine(got, want) > 0.99999, name


def test_no_batch_and_multiple_batch_dimensions():
    q = _rand((7, 2, 4), 7)
    k = _rand((11, 2, 4), 8)
    v = _rand((11, 2, 3), 9)
    out = pallas_yat_l1_attention(q, k, v, block_q=4, block_k=4, interpret=True)
    np.testing.assert_allclose(out, _reference(q, k, v), rtol=2e-5, atol=2e-5)

    q2 = jnp.broadcast_to(q, (2, 3) + q.shape)
    k2 = jnp.broadcast_to(k, (2, 3) + k.shape)
    v2 = jnp.broadcast_to(v, (2, 3) + v.shape)
    out2 = pallas_yat_l1_attention(q2, k2, v2, block_q=4, block_k=4, interpret=True)
    np.testing.assert_allclose(out2, _reference(q2, k2, v2), rtol=2e-5, atol=2e-5)


def test_multi_batch_multi_head_ragged_causal_forward_and_gradients():
    q = _rand((2, 3, 7, 3, 4), 10)
    k = _rand((2, 3, 11, 3, 4), 11)
    v = _rand((2, 3, 11, 3, 5), 12)
    cotangent = _rand((2, 3, 7, 3, 5), 13)

    def pallas_loss(q, k, v):
        out = pallas_yat_l1_attention(
            q, k, v, causal=True, block_q=4, block_k=4, interpret=True
        )
        return jnp.vdot(out, cotangent), out

    def reference_loss(q, k, v):
        out = _reference(q, k, v, causal=True)
        return jnp.vdot(out, cotangent), out

    (_, actual), actual_grads = jax.value_and_grad(
        pallas_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (_, expected), expected_grads = jax.value_and_grad(
        reference_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    for name, got, want in zip(("dQ", "dK", "dV"), actual_grads, expected_grads):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4, err_msg=name)
        assert _cosine(got, want) > 0.99999, name


def test_native_block_specs_keep_tpu_minor_axes_legal(monkeypatch):
    """Catch TPU layout failures without requiring TPU hardware.

    Mosaic TPU requires the second-minor/minor BlockSpec dimensions to be
    divisible by 8/128, respectively, unless a block spans that full array
    dimension.  In particular, heads must not be a squeezed minor axis.
    """
    calls = []

    def fake_pallas_call(_kernel, **kwargs):
        def invoke(*args):
            calls.append((args, kwargs))
            return jax.tree.map(
                lambda shape: jnp.zeros(shape.shape, shape.dtype),
                kwargs["out_shape"],
            )

        return invoke

    monkeypatch.setattr(pallas_module.pl, "pallas_call", fake_pallas_call)

    q = jnp.zeros((6, 128, 32), jnp.float32)
    k = jnp.zeros((6, 192, 32), jnp.float32)
    v = jnp.zeros((6, 192, 24), jnp.float32)
    out, l = pallas_module._pallas_yat_l1_fwd_padded(
        q, k, v, 1e-5, False, 64, 64, False
    )
    do = jnp.zeros_like(out)
    pallas_module._pallas_yat_l1_bwd(
        1e-5, False, 64, 64, False, None, (q, k, v, l, out), do
    )

    assert [call[1]["grid"] for call in calls] == [(2, 6), (3, 6), (2, 6)]

    def check(spec, array_shape):
        block_shape = tuple(1 if dim is None else dim for dim in spec.block_shape)
        assert len(block_shape) >= 2
        assert block_shape[-2] == array_shape[-2] or block_shape[-2] % 8 == 0
        assert block_shape[-1] == array_shape[-1] or block_shape[-1] % 128 == 0

    for args, kwargs in calls:
        for spec, array in zip(kwargs["in_specs"], args):
            check(spec, array.shape)
        out_specs = kwargs["out_specs"]
        out_shapes = kwargs["out_shape"]
        if not isinstance(out_specs, (list, tuple)):
            out_specs, out_shapes = [out_specs], [out_shapes]
        for spec, shape in zip(out_specs, out_shapes):
            check(spec, shape.shape)


def test_native_tpu_rejects_only_illegal_multi_tile_block_sizes(monkeypatch):
    q = jnp.ones((1, 128, 2, 32))
    k = jnp.ones((1, 192, 2, 32))
    v = jnp.ones((1, 192, 2, 24))

    monkeypatch.setattr(pallas_module.jax, "default_backend", lambda: "tpu")
    pallas_module._validate_inputs(q, k, v, 64, 64, False)

    with pytest.raises(ValueError, match="block_q must be divisible by 8"):
        pallas_module._validate_inputs(q, k, v, 7, 64, False)
    with pytest.raises(ValueError, match="block_k must be divisible by 8"):
        pallas_module._validate_inputs(q, k, v, 64, 7, False)

    # A single full-sequence block is legal even when its size is not a
    # multiple of eight, and interpret/GPU execution keeps its existing
    # backend-specific flexibility.
    q_short, k_short, v_short = q[:, :7], k[:, :11], v[:, :11]
    pallas_module._validate_inputs(q_short, k_short, v_short, 7, 11, False)
    pallas_module._validate_inputs(q, k, v, 7, 7, True)
    monkeypatch.setattr(pallas_module.jax, "default_backend", lambda: "gpu")
    pallas_module._validate_inputs(q, k, v, 7, 7, False)


def test_all_kernel_dots_honor_the_public_precision_policy():
    """Prevent any kernel contraction from ignoring the precision argument."""
    source = inspect.getsource(pallas_module)
    tree = ast.parse(source)
    direct_pallas_dots = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr == "dot"
    ]
    kernel_dot_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_dot"
    ]

    assert len(direct_pallas_dots) == 1
    precision = next(
        keyword.value
        for keyword in direct_pallas_dots[0].keywords
        if keyword.arg == "precision"
    )
    assert ast.unparse(precision) == "precision"
    assert len(kernel_dot_calls) == 9
    assert all(
        len(call.args) == 3 and ast.unparse(call.args[2]) == "precision"
        for call in kernel_dot_calls
    )


def test_highest_precision_forward_and_gradients_match_reference():
    q = _rand((1, 7, 2, 4), 20)
    k = _rand((1, 11, 2, 4), 21)
    v = _rand((1, 11, 2, 5), 22)
    cotangent = _rand((1, 7, 2, 5), 23)

    def pallas_loss(q, k, v):
        out = pallas_yat_l1_attention(
            q,
            k,
            v,
            block_q=4,
            block_k=4,
            interpret=True,
            precision=jax.lax.Precision.HIGHEST,
        )
        return jnp.vdot(out, cotangent), out

    def reference_loss(q, k, v):
        with jax.default_matmul_precision("float32"):
            out = _reference(q, k, v)
        return jnp.vdot(out, cotangent), out

    (_, actual), actual_grads = jax.value_and_grad(
        pallas_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)
    (_, expected), expected_grads = jax.value_and_grad(
        reference_loss, argnums=(0, 1, 2), has_aux=True
    )(q, k, v)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    for got, want in zip(actual_grads, expected_grads):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    "q_shape,k_shape,v_shape,message",
    [
        ((1, 4, 2, 3), (1, 4, 2, 4), (1, 4, 2, 5), "head_dim"),
        ((1, 4, 2, 3), (1, 5, 2, 3), (1, 4, 2, 5), "sequence"),
        ((1, 4, 2, 3), (1, 4, 3, 3), (1, 4, 2, 5), "heads"),
    ],
)
def test_shape_validation(q_shape, k_shape, v_shape, message):
    with pytest.raises(ValueError, match=message):
        pallas_yat_l1_attention(
            jnp.ones(q_shape), jnp.ones(k_shape), jnp.ones(v_shape), interpret=True
        )
