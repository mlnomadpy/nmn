"""Cross-backend contract tests for functional attention input shapes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

INVALID_QKV_SHAPES = {
    "rank": ((2, 3), (2, 7, 3, 4), (2, 7, 3, 6), "rank"),
    "batch": ((2, 5, 3, 4), (1, 7, 3, 4), (2, 7, 3, 6), "batch dimensions"),
    "heads": ((2, 5, 3, 4), (2, 7, 2, 4), (2, 7, 3, 6), "number of heads"),
    "qk_depth": ((2, 5, 3, 4), (2, 7, 3, 5), (2, 7, 3, 6), "head depth"),
    "kv_length": ((2, 5, 3, 4), (2, 7, 3, 4), (2, 6, 3, 6), "sequence lengths"),
}


def _backend(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from nmn.torch.attention import yat_attention, yat_attention_weights

        return (
            torch.ones,
            yat_attention,
            yat_attention_weights,
            lambda fn: torch.compile(fn, backend="eager"),
        )

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if backend == "nnx":
        from nmn.nnx import yat_attention, yat_attention_weights
    else:
        from nmn.linen import yat_attention, yat_attention_weights
    return jnp.ones, yat_attention, yat_attention_weights, jax.jit


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("case", INVALID_QKV_SHAPES)
@pytest.mark.parametrize("backend", ["torch", "nnx", "linen"])
def test_invalid_qkv_shape_matrix_raises_value_error(backend, case, compiled):
    ones, attention, _, compile_fn = _backend(backend)
    q_shape, k_shape, v_shape, message = INVALID_QKV_SHAPES[case]
    apply = compile_fn(attention) if compiled else attention

    with pytest.raises(ValueError, match=message):
        apply(ones(q_shape), ones(k_shape), ones(v_shape))


@pytest.mark.parametrize("backend", ["nnx", "linen"])
def test_jax_attention_retains_multiple_leading_batch_dimensions(backend):
    ones, attention, _, _ = _backend(backend)
    output = attention(
        ones((2, 3, 5, 2, 4)),
        ones((2, 3, 7, 2, 4)),
        ones((2, 3, 7, 2, 6)),
    )
    assert output.shape == (2, 3, 5, 2, 6)


@pytest.mark.parametrize("backend", ["torch", "nnx", "linen"])
def test_shape_validation_is_active_under_python_optimized_mode(backend):
    # This integration file runs in the JAX-only CI job as well as developer
    # environments that may have every optional backend installed.  Skip in
    # the parent process so a deliberately isolated install does not turn the
    # child import into a false validation failure.
    _backend(backend)
    module = "nmn.torch.attention" if backend == "torch" else f"nmn.{backend}"
    array = "torch.ones" if backend == "torch" else "jnp.ones"
    imports = "import torch" if backend == "torch" else "import jax.numpy as jnp"
    probe = f"""
{imports}
from {module} import yat_attention
try:
    yat_attention(
        {array}((2, 5, 3, 4)),
        {array}((1, 7, 3, 4)),
        {array}((2, 7, 3, 6)),
    )
except ValueError as exc:
    print(type(exc).__name__ + ': ' + str(exc))
else:
    raise RuntimeError('invalid shapes were accepted')
"""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, (str(Path.cwd() / "src"), os.environ.get("PYTHONPATH")))
        ),
    }
    results = [
        subprocess.run(
            [sys.executable, *flags, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        for flags in ([], ["-O"])
    ]
    assert all(result.returncode == 0 for result in results), results
    assert results[0].stdout == results[1].stdout
    assert "ValueError: attention input batch dimensions" in results[0].stdout


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("backend", ["torch", "nnx", "linen"])
def test_attention_weights_reject_mismatched_qk_shapes(backend, compiled):
    ones, _, weights, compile_fn = _backend(backend)
    apply = compile_fn(weights) if compiled else weights

    with pytest.raises(ValueError, match="head depth"):
        apply(ones((2, 5, 3, 4)), ones((2, 7, 3, 5)))


@pytest.mark.parametrize("compiled", [False, True])
def test_nnx_dot_product_attention_uses_shared_shape_contract(compiled):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from nmn.nnx import dot_product_attention

    apply = jax.jit(dot_product_attention) if compiled else dot_product_attention
    with pytest.raises(ValueError, match="sequence lengths"):
        apply(
            jnp.ones((2, 5, 3, 4)),
            jnp.ones((2, 7, 3, 4)),
            jnp.ones((2, 6, 3, 8)),
        )


ADVANCED_ROUTES = [
    pytest.param("torch", "spherical", id="torch-spherical"),
    pytest.param("torch", "may", id="torch-may"),
    pytest.param("torch", "ray", id="torch-ray"),
    pytest.param("torch", "linear-readout", id="torch-linear-readout"),
    pytest.param("linen", "spherical", id="linen-spherical"),
    pytest.param("linen", "may", id="linen-may"),
    pytest.param("linen", "ray", id="linen-ray"),
    pytest.param("linen", "linear-readout", id="linen-linear-readout"),
    pytest.param("nnx", "spherical", id="nnx-spherical"),
    pytest.param("nnx", "dot-product", id="nnx-dot-product"),
    pytest.param("nnx", "dot-product-weights", id="nnx-dot-product-weights"),
    pytest.param("nnx", "fused", id="nnx-fused"),
    pytest.param("nnx", "fused-self", id="nnx-fused-self"),
    pytest.param("nnx", "pallas", id="nnx-pallas"),
    pytest.param("nnx", "may", id="nnx-may"),
    pytest.param("nnx", "ray", id="nnx-ray"),
    pytest.param("nnx", "slay", id="nnx-slay"),
    pytest.param("nnx", "performer", id="nnx-performer"),
    pytest.param("nnx", "rotary", id="nnx-rotary"),
    pytest.param("nnx", "rotary-weights", id="nnx-rotary-weights"),
    pytest.param("nnx", "rotary-performer", id="nnx-rotary-performer"),
]


def _advanced_route(backend, route):
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from nmn.torch.attention import (
            linear_attention_readout,
            maclaurin_yat_attention,
            radial_yat_attention,
            yat_attention_normalized,
        )

        routes = {
            "spherical": yat_attention_normalized,
            "may": lambda q, k, v: maclaurin_yat_attention(q, k, v, None),
            "ray": lambda q, k, v: radial_yat_attention(q, k, v, None),
            "linear-readout": linear_attention_readout,
        }
        return torch.ones, routes[route], lambda fn: torch.compile(fn, backend="eager")

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if backend == "linen":
        from nmn.linen import (
            linear_attention,
            maclaurin_yat_attention,
            radial_yat_attention,
            yat_attention_normalized,
        )

        routes = {
            "spherical": yat_attention_normalized,
            "may": lambda q, k, v: maclaurin_yat_attention(q, k, v, None),
            "ray": lambda q, k, v: radial_yat_attention(q, k, v, None),
            "linear-readout": linear_attention,
        }
        return jnp.ones, routes[route], jax.jit

    from nmn.nnx.layers.attention import (
        dot_product_attention,
        dot_product_attention_weights,
        fused_yat_l1_attention,
        fused_yat_l1_self_attention,
        maclaurin_yat_attention,
        pallas_yat_l1_attention,
        radial_yat_attention,
        rotary_yat_attention,
        rotary_yat_attention_weights,
        rotary_yat_performer_attention,
        yat_attention_normalized,
        yat_performer_attention,
        yat_tp_attention,
    )

    routes = {
        "spherical": yat_attention_normalized,
        "dot-product": dot_product_attention,
        "dot-product-weights": lambda q, k, _v: dot_product_attention_weights(q, k),
        "fused": fused_yat_l1_attention,
        "fused-self": lambda q, _k, v: fused_yat_l1_self_attention(q, v),
        "pallas": lambda q, k, v: pallas_yat_l1_attention(q, k, v, interpret=True),
        "may": lambda q, k, v: maclaurin_yat_attention(q, k, v, None),
        "ray": lambda q, k, v: radial_yat_attention(q, k, v, None),
        "slay": lambda q, k, v: yat_tp_attention(q, k, v, None),
        "performer": lambda q, k, v: yat_performer_attention(q, k, v, None),
        "rotary": lambda q, k, v: rotary_yat_attention(q, k, v, None, None),
        "rotary-weights": lambda q, k, _v: rotary_yat_attention_weights(
            q, k, None, None
        ),
        "rotary-performer": lambda q, k, v: rotary_yat_performer_attention(
            q, k, v, None, None, None
        ),
    }
    return jnp.ones, routes[route], jax.jit


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(("backend", "route"), ADVANCED_ROUTES)
def test_advanced_public_routes_use_shared_shape_contract(backend, route, compiled):
    ones, apply, compile_fn = _advanced_route(backend, route)
    if (
        compiled
        and backend == "torch"
        and not hasattr(pytest.importorskip("torch"), "compile")
    ):
        pytest.skip("torch.compile requires PyTorch 2")
    if compiled:
        apply = compile_fn(apply)

    query = ones((2, 5, 3, 4))
    key = ones((2, 7, 3, 5))
    value_shape = (2, 4, 3, 6) if route == "fused-self" else (2, 7, 3, 6)
    message = "sequence lengths" if route == "fused-self" else "head depth"

    with pytest.raises(ValueError, match=message):
        apply(query, key, ones(value_shape))
