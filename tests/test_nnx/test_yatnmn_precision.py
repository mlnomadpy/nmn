"""Numerical-mode tests for the Flax NNX YatNMN layer."""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from jax import lax

    from nmn.nnx.layers import YatNMN

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


pytestmark = pytest.mark.skipif(not HAS_JAX, reason="JAX/Flax not available")


def _layer(
    width, mode, *, fused=False, epsilon=1e-3, distance_floor=0.0, dot_general=None
):
    if dot_general is None:
        dot_general = lax.dot_general
    return YatNMN(
        width,
        24,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        compute_mode=mode,
        fused=fused,
        epsilon=epsilon,
        distance_floor=distance_floor,
        learnable_epsilon=True,
        dot_general=dot_general,
        rngs=nnx.Rngs(42),
    )


def _loss(model, x):
    return jnp.mean(model(x).astype(jnp.float32) ** 2)


@pytest.mark.parametrize("mode", ["fp32", "mixed", "bf16"])
def test_fused_and_standard_modes_have_exact_forward_and_gradient_parity(mode):
    standard = _layer(32, mode, fused=False, distance_floor=1e-3)
    fused = _layer(32, mode, fused=True, distance_floor=1e-3)
    x = jax.random.normal(jax.random.key(1), (3, 32), dtype=jnp.bfloat16)

    np.testing.assert_array_equal(standard(x), fused(x))

    _, (standard_grads, standard_input_grad) = jax.value_and_grad(
        _loss, argnums=(0, 1)
    )(standard, x)
    _, (fused_grads, fused_input_grad) = jax.value_and_grad(_loss, argnums=(0, 1))(
        fused, x
    )
    np.testing.assert_array_equal(
        standard_input_grad,
        fused_input_grad,
        err_msg=f"input gradient differs in {mode} mode",
    )
    for name in ("kernel", "bias", "alpha", "epsilon_param"):
        np.testing.assert_array_equal(
            getattr(standard_grads, name)[...],
            getattr(fused_grads, name)[...],
            err_msg=f"{name} gradient differs in {mode} mode",
        )


def test_default_fused_gradient_matches_standard_when_distance_clamp_is_active():
    width = 64
    standard = YatNMN(
        width,
        1,
        fused=False,
        use_bias=False,
        use_alpha=False,
        epsilon=1e-5,
        rngs=nnx.Rngs(0),
    )
    fused = YatNMN(
        width,
        1,
        fused=True,
        use_bias=False,
        use_alpha=False,
        epsilon=1e-5,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(2), (1, width)) * 10.0
    standard.kernel[...] = x.T
    fused.kernel[...] = x.T

    raw_distance = (
        jnp.sum(x * x, axis=-1, keepdims=True)
        + jnp.sum(x.T * x.T, axis=0, keepdims=True)
        - 2.0 * (x @ x.T)
    )
    assert float(raw_distance[0, 0]) < 0.0
    np.testing.assert_array_equal(standard(x), fused(x))

    standard_grad = jax.grad(lambda value: jnp.sum(standard(value)))(x)
    fused_grad = jax.grad(lambda value: jnp.sum(fused(value)))(x)
    np.testing.assert_array_equal(standard_grad, fused_grad)
    assert jnp.isfinite(fused_grad).all()


@pytest.mark.parametrize("param_dtype", [jnp.float16, jnp.bfloat16])
def test_low_precision_learnable_epsilon_initializes_finite(param_dtype):
    layer = YatNMN(
        8,
        4,
        param_dtype=param_dtype,
        learnable_epsilon=True,
        epsilon=1e-5,
        rngs=nnx.Rngs(0),
    )

    raw_epsilon = layer.epsilon_param[...]
    effective_epsilon = jax.nn.softplus(raw_epsilon.astype(jnp.float32))
    assert jnp.isfinite(raw_epsilon).all()
    assert jnp.isfinite(effective_epsilon).all()
    np.testing.assert_allclose(effective_epsilon, 1e-5, rtol=0.02)


@pytest.mark.parametrize(
    ("width", "scale", "epsilon"),
    [
        (32, 0.1, 1e-5),
        (128, 1.0, 1e-3),
        (512, 10.0, 1.0),
    ],
)
@pytest.mark.parametrize(
    ("mode", "forward_tolerance", "gradient_tolerance", "min_cosine"),
    [
        ("mixed", 0.01, 0.03, 0.999),
        ("bf16", 0.04, 0.10, 0.995),
    ],
)
def test_reduced_precision_modes_stay_close_to_fp32(
    width,
    scale,
    epsilon,
    mode,
    forward_tolerance,
    gradient_tolerance,
    min_cosine,
):
    x_bf16 = jax.random.normal(
        jax.random.key(width), (3, width), dtype=jnp.bfloat16
    ) * jnp.asarray(scale, dtype=jnp.bfloat16)

    # BF16 parameters and quantized inputs isolate score-computation error from
    # parameter/input quantization when comparing with the FP32 reference.
    reference = YatNMN(
        width,
        24,
        dtype=jnp.float32,
        param_dtype=jnp.bfloat16,
        compute_mode="fp32",
        epsilon=epsilon,
        rngs=nnx.Rngs(42),
    )
    candidate = YatNMN(
        width,
        24,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        compute_mode=mode,
        epsilon=epsilon,
        rngs=nnx.Rngs(42),
    )

    x_fp32 = x_bf16.astype(jnp.float32)
    reference_output = reference(x_fp32)
    candidate_output = candidate(x_bf16).astype(jnp.float32)
    reference_grad = jax.grad(lambda x: _loss(reference, x))(x_fp32)
    candidate_grad = jax.grad(lambda x: _loss(candidate, x))(x_bf16).astype(jnp.float32)

    assert jnp.isfinite(candidate_output).all()
    assert jnp.isfinite(candidate_grad).all()
    forward_error = jnp.linalg.norm(candidate_output - reference_output)
    forward_error /= jnp.maximum(jnp.linalg.norm(reference_output), 1e-12)
    gradient_error = jnp.linalg.norm(candidate_grad - reference_grad)
    gradient_error /= jnp.maximum(jnp.linalg.norm(reference_grad), 1e-12)
    gradient_cosine = jnp.vdot(candidate_grad.ravel(), reference_grad.ravel())
    gradient_cosine /= jnp.linalg.norm(candidate_grad) * jnp.linalg.norm(reference_grad)

    assert float(forward_error) < forward_tolerance
    assert float(gradient_error) < gradient_tolerance
    assert float(gradient_cosine) > min_cosine


@pytest.mark.parametrize(
    ("low_dtype", "forward_tolerance", "gradient_tolerance"),
    [
        (jnp.float16, 0.002, 0.003),
        (jnp.bfloat16, 0.01, 0.02),
    ],
)
def test_mixed_mode_supports_float16_and_bfloat16_with_fp32_parity(
    low_dtype, forward_tolerance, gradient_tolerance
):
    width = 32
    x_low = jax.random.normal(jax.random.key(1), (3, width), dtype=low_dtype)
    reference = YatNMN(
        width,
        24,
        dtype=jnp.float32,
        param_dtype=low_dtype,
        compute_mode="fp32",
        epsilon=1e-3,
        rngs=nnx.Rngs(42),
    )
    candidate = YatNMN(
        width,
        24,
        dtype=low_dtype,
        param_dtype=low_dtype,
        compute_mode="mixed",
        epsilon=1e-3,
        rngs=nnx.Rngs(42),
    )

    x_fp32 = x_low.astype(jnp.float32)
    reference_output = reference(x_fp32)
    candidate_output = candidate(x_low).astype(jnp.float32)
    reference_grad = jax.grad(lambda x: _loss(reference, x))(x_fp32)
    candidate_grad = jax.grad(lambda x: _loss(candidate, x))(x_low).astype(jnp.float32)

    forward_error = jnp.linalg.norm(candidate_output - reference_output)
    forward_error /= jnp.maximum(jnp.linalg.norm(reference_output), 1e-12)
    gradient_error = jnp.linalg.norm(candidate_grad - reference_grad)
    gradient_error /= jnp.maximum(jnp.linalg.norm(reference_grad), 1e-12)

    assert jnp.isfinite(candidate_output).all()
    assert jnp.isfinite(candidate_grad).all()
    assert float(forward_error) < forward_tolerance
    assert float(gradient_error) < gradient_tolerance


def test_mixed_mode_uses_bf16_dot_operands_with_fp32_output():
    calls = []

    def recording_dot_general(lhs, rhs, dimension_numbers, **kwargs):
        calls.append((lhs.dtype, rhs.dtype, kwargs.get("preferred_element_type")))
        return lax.dot_general(lhs, rhs, dimension_numbers, **kwargs)

    layer = _layer(16, "mixed", dot_general=recording_dot_general)
    x = jnp.ones((2, 16), dtype=jnp.bfloat16)
    layer(x)

    assert calls == [(jnp.bfloat16, jnp.bfloat16, jnp.float32)]


def test_mixed_lowering_keeps_bf16_dot_operands():
    layer = _layer(16, "mixed")
    x = jnp.ones((2, 16), dtype=jnp.bfloat16)
    lowered = jax.jit(lambda inputs: layer(inputs)).lower(x).as_text()
    dot_lines = [line for line in lowered.splitlines() if "dot_general" in line]

    assert any(
        "(tensor<2x16xbf16>, tensor<16x24xbf16>) -> tensor<2x24xf32>" in line
        for line in dot_lines
    ), "mixed mode did not lower to BF16 dot operands with an FP32 result"


def test_bf16_mode_uses_strict_bf16_dot_operands():
    calls = []

    def recording_dot_general(lhs, rhs, dimension_numbers, **kwargs):
        calls.append((lhs.dtype, rhs.dtype, kwargs.get("preferred_element_type")))
        return lax.dot_general(lhs, rhs, dimension_numbers, **kwargs)

    layer = _layer(16, "bf16", dot_general=recording_dot_general)
    x = jnp.ones((2, 16), dtype=jnp.bfloat16)
    layer(x)

    assert calls == [(jnp.bfloat16, jnp.bfloat16, None)]


@pytest.mark.parametrize("fused", [False, True])
def test_bf16_distance_floor_keeps_near_collision_finite(fused):
    width = 64
    layer = YatNMN(
        width,
        1,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        compute_mode="bf16",
        fused=fused,
        epsilon=1e-5,
        distance_floor=1e-2,
        learnable_epsilon=True,
        rngs=nnx.Rngs(0),
    )
    x = jax.random.normal(jax.random.key(7), (1, width), dtype=jnp.bfloat16)
    layer.kernel[...] = x.T

    output = layer(x)
    _, grads = jax.value_and_grad(_loss)(layer, x)

    assert jnp.isfinite(output).all()
    for leaf in jax.tree.leaves(grads):
        if hasattr(leaf, "dtype"):
            assert jnp.isfinite(leaf).all()


def test_fp32_explicit_mode_preserves_default_behavior():
    default = YatNMN(16, 8, rngs=nnx.Rngs(5))
    explicit = YatNMN(16, 8, compute_mode="fp32", rngs=nnx.Rngs(5))
    x = jax.random.normal(jax.random.key(6), (2, 16))

    np.testing.assert_array_equal(default(x), explicit(x))


@pytest.mark.parametrize("mode", ["float16", "auto", ""])
def test_invalid_compute_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="compute_mode"):
        YatNMN(4, 2, compute_mode=mode, rngs=nnx.Rngs(0))


@pytest.mark.parametrize("distance_floor", [-1e-3, np.inf, np.nan])
def test_invalid_distance_floor_is_rejected(distance_floor):
    with pytest.raises(ValueError, match="distance_floor"):
        YatNMN(4, 2, distance_floor=distance_floor, rngs=nnx.Rngs(0))
