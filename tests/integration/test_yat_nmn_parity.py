"""Cross-framework numerical parity tests for YatNMN.

For each combination of optional features (spherical, weight_normalized,
learnable_epsilon, constant_bias), this test instantiates `YatNMN` in each
available framework with **identical kernel weights, bias values, and inputs**,
then asserts that all framework outputs are numerically close.

This catches the class of bug where one framework silently disagrees with
the others — the kind of regression that hit Keras `YatNMN.spherical` and
`YatNMN.weight_normalized` (both used the wrong axis), since those were
mathematically broken in only one framework.

Each framework's `YatNMN` lays out neurons differently in its kernel:
- NNX, Keras: kernel shape (in_features, out_features) — neurons are columns
- PyTorch, TF, Linen: kernel shape (out_features, in_features) — neurons are rows

We use a single canonical "logical" weight matrix W of shape
(in_features, out_features) and transpose it for the row-based frameworks.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ── Logical inputs / parameters used by every framework ───────────────────

IN_FEATURES = 5
OUT_FEATURES = 4
BATCH = 3
RNG = np.random.default_rng(2026)

# Use a non-trivial input so all features (spherical, weight-norm) actually
# differ from the trivial case.
X = RNG.normal(size=(BATCH, IN_FEATURES)).astype(np.float32)
W_LOGICAL = RNG.normal(size=(IN_FEATURES, OUT_FEATURES)).astype(np.float32)
B = RNG.normal(size=(OUT_FEATURES,)).astype(np.float32)
EPSILON = 1e-5


def reference_yat(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray | None,
    *,
    spherical: bool,
    weight_normalized: bool,
    epsilon: float,
) -> np.ndarray:
    """Pure-numpy reference YatNMN implementation.

    All frameworks should agree with this and with each other.
    """
    if spherical:
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        # Each column of w is one neuron — normalize columns to unit norm.
        w = w / (np.linalg.norm(w, axis=0, keepdims=True) + 1e-8)
    if weight_normalized:
        w = w / (np.linalg.norm(w, axis=0, keepdims=True) + 1e-8)

    dot = x @ w  # (batch, out_features)

    if spherical:
        # Both x rows and w columns are unit vectors → ||x - W||² = 2 - 2(x·W)
        dist = np.maximum(2.0 - 2.0 * dot, 0.0)
    else:
        x_sq = np.sum(x ** 2, axis=-1, keepdims=True)
        if weight_normalized:
            w_sq = np.ones((1, w.shape[-1]), dtype=w.dtype)
        else:
            w_sq = np.sum(w ** 2, axis=0, keepdims=True)
        dist = np.maximum(x_sq + w_sq - 2.0 * dot, 0.0)

    if b is not None:
        dot = dot + b

    return (dot ** 2) / (dist + epsilon)


# Parameter matrix: every combination of the 4 booleans + None/value for bias.
PARAMS = []
for spherical in [False, True]:
    for weight_normalized in [False, True]:
        for learnable_epsilon in [False, True]:
            for bias_mode in ["learnable", "constant", "none"]:
                PARAMS.append((spherical, weight_normalized, learnable_epsilon, bias_mode))


def _bias_for(mode: str):
    if mode == "learnable":
        return B, None
    if mode == "constant":
        # Constant bias must be a scalar; broadcast to all neurons.
        return None, 0.42
    return None, None


# ── Per-framework helpers ────────────────────────────────────────────────


def _run_nnx(spherical, weight_normalized, learnable_epsilon, bias_mode):
    pytest.importorskip("flax")
    from flax import nnx
    import jax
    import jax.numpy as jnp
    from nmn.nnx.layers import YatNMN as NnxYatNMN

    bias_arr, const_bias = _bias_for(bias_mode)
    use_bias = bias_arr is not None or const_bias is not None

    layer = NnxYatNMN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        use_bias=use_bias,
        constant_bias=const_bias,
        use_alpha=False,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
        learnable_epsilon=learnable_epsilon,
        rngs=nnx.Rngs(0),
    )
    # Inject canonical weights — NNX kernel is (in_features, out_features).
    layer.kernel[...] = jnp.asarray(W_LOGICAL)
    if bias_arr is not None:
        layer.bias[...] = jnp.asarray(bias_arr)

    return np.asarray(layer(jnp.asarray(X)))


def _run_linen(spherical, weight_normalized, learnable_epsilon, bias_mode):
    pytest.importorskip("flax")
    import jax
    import jax.numpy as jnp
    from nmn.linen.nmn import YatNMN as LinenYatNMN

    bias_arr, const_bias = _bias_for(bias_mode)
    use_bias = bias_arr is not None or const_bias is not None

    m = LinenYatNMN(
        features=OUT_FEATURES,
        use_bias=use_bias,
        constant_bias=const_bias,
        use_alpha=False,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
        learnable_epsilon=learnable_epsilon,
    )
    params = m.init(jax.random.key(0), jnp.asarray(X))
    # Linen kernel is (features, in_features) — transpose the canonical W.
    new_params = {"params": dict(params["params"])}
    new_params["params"]["kernel"] = jnp.asarray(W_LOGICAL.T)
    if bias_arr is not None:
        new_params["params"]["bias"] = jnp.asarray(bias_arr)

    return np.asarray(m.apply(new_params, jnp.asarray(X)))


def _run_torch(spherical, weight_normalized, learnable_epsilon, bias_mode):
    torch = pytest.importorskip("torch")
    from nmn.torch.nmn import YatNMN as TorchYatNMN

    bias_arr, const_bias = _bias_for(bias_mode)
    use_bias = bias_arr is not None or const_bias is not None

    layer = TorchYatNMN(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=use_bias,
        constant_bias=const_bias,
        alpha=False,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
        learnable_epsilon=learnable_epsilon,
    )
    # PyTorch kernel is (out_features, in_features) — transpose canonical W.
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(W_LOGICAL.T))
        if bias_arr is not None:
            layer.bias.copy_(torch.from_numpy(bias_arr))

    layer.eval()
    out = layer(torch.from_numpy(X))
    return out.detach().numpy()


def _run_keras(spherical, weight_normalized, learnable_epsilon, bias_mode):
    pytest.importorskip("tensorflow")
    from nmn.keras.nmn import YatNMN as KerasYatNMN

    bias_arr, const_bias = _bias_for(bias_mode)
    use_bias = bias_arr is not None or const_bias is not None

    layer = KerasYatNMN(
        units=OUT_FEATURES,
        use_bias=use_bias,
        constant_bias=const_bias,
        use_alpha=False,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
        learnable_epsilon=learnable_epsilon,
    )
    _ = layer(X)  # build
    # Keras kernel is (in_features, units) — same shape as canonical W.
    layer.kernel.assign(W_LOGICAL)
    if bias_arr is not None:
        layer.bias.assign(bias_arr)

    return np.asarray(layer(X))


def _run_tf(spherical, weight_normalized, learnable_epsilon, bias_mode):
    tf = pytest.importorskip("tensorflow")
    from nmn.tf.nmn import YatNMN as TfYatNMN

    bias_arr, const_bias = _bias_for(bias_mode)
    use_bias = bias_arr is not None or const_bias is not None

    layer = TfYatNMN(
        features=OUT_FEATURES,
        use_bias=use_bias,
        constant_bias=const_bias,
        use_alpha=False,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
        learnable_epsilon=learnable_epsilon,
    )
    layer.build((BATCH, IN_FEATURES))
    # TF kernel is (features, in_features) — transpose canonical W.
    layer.kernel.assign(tf.constant(W_LOGICAL.T))
    if bias_arr is not None:
        layer.bias.assign(tf.constant(bias_arr))

    return layer(tf.constant(X)).numpy()


def _run_mlx(spherical, weight_normalized, learnable_epsilon, bias_mode):
    mx = pytest.importorskip("mlx.core")
    # Pin to CPU: MLX's Metal matmul accumulates at lower precision than
    # numpy fp32, so the < 1e-4 rtol used here is only meaningful on CPU.
    prev_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from nmn.mlx.nmn import YatNMN as MlxYatNMN

        bias_arr, const_bias = _bias_for(bias_mode)
        use_bias = bias_arr is not None or const_bias is not None

        layer = MlxYatNMN(
            features=OUT_FEATURES,
            use_bias=use_bias,
            constant_bias=const_bias,
            use_alpha=False,
            spherical=spherical,
            weight_normalized=weight_normalized,
            epsilon=EPSILON,
            learnable_epsilon=learnable_epsilon,
        )
        layer.build(IN_FEATURES)
        # MLX kernel is (features, in_features) — transpose canonical W.
        layer.kernel = mx.array(W_LOGICAL.T)
        if bias_arr is not None:
            layer.bias = mx.array(bias_arr)
        out = layer(mx.array(X))
        return np.asarray(out)
    finally:
        mx.set_default_device(prev_device)


FRAMEWORKS = {
    "nnx": _run_nnx,
    "linen": _run_linen,
    "torch": _run_torch,
    "keras": _run_keras,
    "tf": _run_tf,
    "mlx": _run_mlx,
}


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spherical,weight_normalized,learnable_epsilon,bias_mode",
    PARAMS,
)
def test_yat_nmn_matches_reference(
    spherical, weight_normalized, learnable_epsilon, bias_mode
):
    """Every framework must agree with the pure-numpy reference."""
    bias_arr, const_bias = _bias_for(bias_mode)
    if bias_arr is not None:
        b_for_ref = bias_arr
    elif const_bias is not None:
        b_for_ref = np.full((OUT_FEATURES,), const_bias, dtype=np.float32)
    else:
        b_for_ref = None

    expected = reference_yat(
        X, W_LOGICAL, b_for_ref,
        spherical=spherical,
        weight_normalized=weight_normalized,
        epsilon=EPSILON,
    )

    any_ran = False
    skip_exc = pytest.skip.Exception if hasattr(pytest.skip, "Exception") else type("X", (), {})
    for name, runner in FRAMEWORKS.items():
        try:
            actual = runner(spherical, weight_normalized, learnable_epsilon, bias_mode)
        except (ImportError, ModuleNotFoundError, skip_exc) as e:
            continue
        any_ran = True
        np.testing.assert_allclose(
            actual, expected,
            rtol=2e-4, atol=2e-4,
            err_msg=(
                f"{name} YatNMN disagrees with reference for "
                f"spherical={spherical}, weight_normalized={weight_normalized}, "
                f"learnable_epsilon={learnable_epsilon}, bias_mode={bias_mode}"
            ),
        )

    if not any_ran:
        pytest.skip("No framework available to test")
