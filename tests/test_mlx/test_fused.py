"""Tests for the fused metal-kernel YAT score path."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")
mlx_optim = pytest.importorskip("mlx.optimizers")

from nmn.mlx import YatNMN, fused_yat_score, is_gpu_available  # noqa: E402


# ---------------------------------------------------------------------------
# Functional fused_yat_score
# ---------------------------------------------------------------------------


def _ref_yat(x_np, w_np, b_np, alpha, eps=1e-5):
    dot = x_np @ w_np.T
    x_sq = (x_np ** 2).sum(axis=-1, keepdims=True)
    w_sq = (w_np ** 2).sum(axis=-1)[None, :]
    dist = x_sq + w_sq - 2 * dot
    num = dot + b_np
    return alpha * (num ** 2) / (dist + eps)


def test_fused_yat_score_executes_metal_kernel_on_gpu():
    """The public Apple Silicon CI runner must execute the fused Metal path."""
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        assert is_gpu_available()
        x = mx.array([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]])
        w = mx.array([[0.3, -0.2, 0.6], [-0.5, 0.4, 0.2]])
        bias = mx.array([0.1, -0.2])
        alpha = mx.array([1.25])
        actual = fused_yat_score(x, w, bias=bias, alpha=alpha, epsilon=0.07)
        mx.eval(actual)

        expected = _ref_yat(
            np.array(x), np.array(w), np.array(bias), 1.25, eps=0.07
        )
        assert np.allclose(np.array(actual), expected, rtol=2e-5, atol=2e-6)
    finally:
        mx.set_default_device(previous_device)


def test_fused_yat_score_2d_matches_numpy_on_cpu():
    """On the CPU fallback the fused score must match a numpy reference
    to fp32 ULP."""
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        x = mx.random.normal(shape=(4, 8))
        w = mx.random.normal(shape=(6, 8))
        b = mx.random.normal(shape=(6,))
        alpha = mx.array([1.5])
        y = np.array(fused_yat_score(x, w, bias=b, alpha=alpha, epsilon=1e-5))
        ref = _ref_yat(np.array(x), np.array(w), np.array(b), 1.5)
        assert np.max(np.abs(y - ref)) < 1e-5
    finally:
        mx.set_default_device(mx.cpu)


def test_fused_yat_score_higher_rank_input():
    """``(...., in_features)`` shapes flatten and restore correctly."""
    mx.set_default_device(mx.cpu)
    try:
        x = mx.random.normal(shape=(2, 3, 4, 5))
        w = mx.random.normal(shape=(7, 5))
        y = fused_yat_score(x, w)
        assert y.shape == (2, 3, 4, 7)
    finally:
        mx.set_default_device(mx.cpu)


def test_fused_yat_score_rejects_dim_mismatch():
    mx.set_default_device(mx.cpu)
    try:
        x = mx.zeros((1, 4))
        w = mx.zeros((6, 5))
        with pytest.raises(ValueError):
            fused_yat_score(x, w)
    finally:
        mx.set_default_device(mx.cpu)


# ---------------------------------------------------------------------------
# Autograd via custom_function.vjp
# ---------------------------------------------------------------------------


def test_fused_yat_gradient_matches_finite_difference():
    """Compare the analytic gradient to a centred finite-difference at
    every primal element (small dim so it's fast)."""
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        x = mx.random.normal(shape=(2, 3))
        w = mx.random.normal(shape=(2, 3))
        b = mx.random.normal(shape=(2,))
        alpha = mx.array([1.2])

        def loss_fn(x, w, b, alpha):
            return mx.sum(fused_yat_score(x, w, bias=b, alpha=alpha))

        _, (gx, gw, gb, ga) = mx.value_and_grad(loss_fn, argnums=(0, 1, 2, 3))(
            x, w, b, alpha
        )

        def loss_np(x_np, w_np, b_np, a_np):
            return float(_ref_yat(x_np, w_np, b_np, float(a_np[0])).sum())

        h = 1e-2
        xn, wn, bn, an = (np.array(t).copy() for t in (x, w, b, alpha))

        def fd(arr, idx):
            arr_p = arr.copy(); arr_m = arr.copy()
            arr_p.flat[idx] += h; arr_m.flat[idx] -= h
            return arr_p, arr_m

        gx_np = np.array(gx)
        gw_np = np.array(gw)
        gb_np = np.array(gb)
        ga_np = np.array(ga)

        # Check a handful of elements per gradient — full grid would be slow
        # in pure-Python finite-difference, the spot-checks catch sign / scale
        # errors equally well.
        for idx in [0, gx_np.size - 1, gx_np.size // 2]:
            xp, xm = fd(xn, idx)
            ref = (loss_np(xp, wn, bn, an) - loss_np(xm, wn, bn, an)) / (2 * h)
            assert abs(ref - gx_np.flat[idx]) < 0.1, f"gx[{idx}] mismatch"

        for idx in [0, gw_np.size - 1]:
            wp, wm = fd(wn, idx)
            ref = (loss_np(xn, wp, bn, an) - loss_np(xn, wm, bn, an)) / (2 * h)
            assert abs(ref - gw_np.flat[idx]) < 0.2, f"gw[{idx}] mismatch"

        for idx in range(gb_np.size):
            bp, bm = fd(bn, idx)
            ref = (loss_np(xn, wn, bp, an) - loss_np(xn, wn, bm, an)) / (2 * h)
            assert abs(ref - gb_np.flat[idx]) < 0.1, f"gb[{idx}] mismatch"

        ap, am = fd(an, 0)
        ref = (loss_np(xn, wn, bn, ap) - loss_np(xn, wn, bn, am)) / (2 * h)
        assert abs(ref - ga_np.flat[0]) < 0.1
    finally:
        mx.set_default_device(mx.cpu)


def test_fused_yat_array_epsilon_all_gradients_match_eager_and_compile():
    """An array epsilon remains a differentiable argument, including when
    the value-and-gradient function is compiled."""
    mx.set_default_device(mx.cpu)
    try:
        x = mx.array([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]])
        w = mx.array([[0.3, -0.2, 0.6], [-0.5, 0.4, 0.2]])
        b = mx.array([0.1, -0.2])
        alpha = mx.array([1.25])
        eps = mx.array([0.07])

        def eager_loss(x, w, b, alpha, eps):
            dot = x @ w.T
            dist = mx.maximum(
                mx.sum(x * x, axis=-1, keepdims=True)
                + mx.sum(w * w, axis=-1)[None, :]
                - 2.0 * dot,
                0.0,
            )
            return mx.sum(alpha * (dot + b) ** 2 / (dist + eps))

        def fused_loss(x, w, b, alpha, eps):
            return mx.sum(
                fused_yat_score(x, w, bias=b, alpha=alpha, epsilon=eps)
            )

        argnums = (0, 1, 2, 3, 4)
        eager_value, eager_grads = mx.value_and_grad(
            eager_loss, argnums=argnums
        )(x, w, b, alpha, eps)
        fused_value, fused_grads = mx.value_and_grad(
            fused_loss, argnums=argnums
        )(x, w, b, alpha, eps)
        compiled_value, compiled_grads = mx.compile(
            mx.value_and_grad(fused_loss, argnums=argnums)
        )(x, w, b, alpha, eps)

        assert np.allclose(np.array(fused_value), np.array(eager_value), atol=1e-6)
        assert np.allclose(np.array(compiled_value), np.array(eager_value), atol=1e-6)
        for eager_grad, fused_grad, compiled_grad in zip(
            eager_grads, fused_grads, compiled_grads
        ):
            assert np.allclose(
                np.array(fused_grad), np.array(eager_grad), rtol=2e-5, atol=2e-6
            )
            assert np.allclose(
                np.array(compiled_grad), np.array(eager_grad), rtol=2e-5, atol=2e-6
            )
        assert abs(float(np.array(fused_grads[-1])[0])) > 0.0
    finally:
        mx.set_default_device(mx.cpu)


def test_fused_yat_scalar_array_epsilon_vjp_matches_eager():
    """A zero-dimensional MLX epsilon is a direct differentiable primal."""
    mx.set_default_device(mx.cpu)
    try:
        x = mx.array([[0.2, -0.4], [0.5, 0.1]])
        w = mx.array([[0.3, -0.2], [-0.5, 0.4]])
        b = mx.array([0.1, -0.2])
        alpha = mx.array([1.25])
        eps = mx.array(0.07)

        def eager_loss(value):
            dot = x @ w.T
            dist = mx.maximum(
                mx.sum(x * x, axis=-1, keepdims=True)
                + mx.sum(w * w, axis=-1)[None, :]
                - 2.0 * dot,
                0.0,
            )
            return mx.sum(alpha * (dot + b) ** 2 / (dist + value))

        def fused_loss(value):
            return mx.sum(
                fused_yat_score(x, w, bias=b, alpha=alpha, epsilon=value)
            )

        expected = mx.grad(eager_loss)(eps)
        actual = mx.grad(fused_loss)(eps)
        assert actual.shape == ()
        assert np.allclose(np.array(actual), np.array(expected), rtol=2e-5, atol=2e-6)
        assert np.isfinite(float(actual))
        assert float(actual) != 0.0
    finally:
        mx.set_default_device(mx.cpu)


# ---------------------------------------------------------------------------
# YatNMN(fused=True) integration
# ---------------------------------------------------------------------------


def test_yat_nmn_fused_matches_plain_on_cpu():
    """fused=True must produce numerically identical output to fused=False
    when only the basic features are in use."""
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        plain = YatNMN(features=8)
        plain.build(16)
        fused = YatNMN(features=8, fused=True)
        fused.build(16)
        fused.kernel = plain.kernel
        fused.bias = plain.bias
        fused.alpha = plain.alpha

        x = mx.random.normal(shape=(4, 16))
        y_plain = np.array(plain(x))
        y_fused = np.array(fused(x))
        assert np.max(np.abs(y_plain - y_fused)) < 1e-5
    finally:
        mx.set_default_device(mx.cpu)


def test_yat_nmn_fused_supports_constant_bias():
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(0)
        plain = YatNMN(features=4, constant_bias=0.25)
        plain.build(6)
        fused = YatNMN(features=4, constant_bias=0.25, fused=True)
        fused.build(6)
        fused.kernel = plain.kernel
        fused.alpha = plain.alpha

        x = mx.random.normal(shape=(3, 6))
        assert np.max(np.abs(np.array(plain(x)) - np.array(fused(x)))) < 1e-5
    finally:
        mx.set_default_device(mx.cpu)


def test_yat_nmn_fused_supports_constant_alpha():
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(1)
        plain = YatNMN(features=4, constant_alpha=True)
        plain.build(6)
        fused = YatNMN(features=4, constant_alpha=True, fused=True)
        fused.build(6)
        fused.kernel = plain.kernel
        fused.bias = plain.bias

        x = mx.random.normal(shape=(3, 6))
        assert np.max(np.abs(np.array(plain(x)) - np.array(fused(x)))) < 1e-5
    finally:
        mx.set_default_device(mx.cpu)


def test_yat_nmn_fused_falls_back_when_spherical():
    """spherical / weight_normalized / return_weights aren't supported in
    the fused path — the layer should silently fall back to the eager
    forward (no kernel launch) and still produce the right output."""
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(2)
        plain = YatNMN(features=4, spherical=True)
        plain.build(6)
        fused = YatNMN(features=4, spherical=True, fused=True)
        fused.build(6)
        fused.kernel = plain.kernel
        fused.bias = plain.bias
        fused.alpha = plain.alpha

        x = mx.random.normal(shape=(3, 6))
        assert np.max(np.abs(np.array(plain(x)) - np.array(fused(x)))) < 1e-5
    finally:
        mx.set_default_device(mx.cpu)


def test_yat_nmn_fused_gradient_flow():
    """One AdamW step through the fused path reduces a squared-error loss."""

    def loss_fn(model, x, y):
        return mx.mean((model(x) - y) ** 2)

    mx.set_default_device(mx.cpu)
    try:
        layer = YatNMN(features=4, fused=True)
        x = mx.random.normal(shape=(8, 6))
        y = mx.random.normal(shape=(8, 4))
        _ = layer(x)  # build

        grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
        loss, grads = grad_fn(layer, x, y)
        assert set(grads.keys()) >= {"kernel", "bias", "alpha"}

        opt = mlx_optim.AdamW(learning_rate=1e-2)
        opt.update(layer, grads)
        mx.eval(layer.parameters())
        assert float(loss_fn(layer, x, y)) < float(loss)
    finally:
        mx.set_default_device(mx.cpu)


@pytest.mark.parametrize("lazy", [False, True])
def test_yat_nmn_fused_learnable_epsilon_gradient_parity(lazy):
    """Fused module gradients include the softplus epsilon parameter;
    lazy mode freezes only the kernel."""

    def loss_fn(model, x):
        return mx.sum(model(x))

    mx.set_default_device(mx.cpu)
    try:
        plain = YatNMN(
            features=2, learnable_epsilon=True, epsilon=0.07, lazy=lazy
        )
        fused = YatNMN(
            features=2, learnable_epsilon=True, epsilon=0.07,
            fused=True, lazy=lazy,
        )
        plain.build(3)
        fused.build(3)
        fused.kernel = plain.kernel
        fused.bias = plain.bias
        fused.alpha = plain.alpha
        fused.epsilon_param = plain.epsilon_param
        x = mx.array([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]])

        eager_out = plain(x)
        fused_out = fused(x)
        _, eager_grads = mlx_nn.value_and_grad(plain, loss_fn)(plain, x)
        _, fused_grads = mlx_nn.value_and_grad(fused, loss_fn)(fused, x)

        expected_keys = {"bias", "alpha", "epsilon_param"}
        if not lazy:
            expected_keys.add("kernel")
        assert set(eager_grads) == expected_keys
        assert set(fused_grads) == expected_keys
        assert np.allclose(np.array(fused_out), np.array(eager_out), atol=1e-6)
        for name in expected_keys:
            assert np.allclose(
                np.array(fused_grads[name]),
                np.array(eager_grads[name]),
                rtol=2e-5,
                atol=2e-6,
            )
        eps_grad = np.array(fused_grads["epsilon_param"])
        assert np.all(np.isfinite(eps_grad))
        assert np.any(eps_grad != 0.0)
    finally:
        mx.set_default_device(mx.cpu)


def test_compiled_yat_nmn_preserves_softplus_epsilon_parameter_chain():
    """Compilation keeps the module's epsilon_param -> softplus -> fused VJP
    chain intact, rather than merely differentiating a precomputed epsilon."""

    def loss_fn(model, x):
        return mx.sum(model(x))

    mx.set_default_device(mx.cpu)
    try:
        layer = YatNMN(
            features=2,
            fused=True,
            learnable_epsilon=True,
            epsilon=0.07,
        )
        layer.build(3)
        layer.kernel = mx.array([[0.3, -0.2, 0.6], [-0.5, 0.4, 0.2]])
        layer.bias = mx.array([0.1, -0.2])
        layer.alpha = mx.array([1.25])
        inputs = mx.array([[0.2, -0.4, 0.7], [0.5, 0.1, -0.3]])

        grad_fn = mlx_nn.value_and_grad(layer, loss_fn)
        _, eager_grads = grad_fn(layer, inputs)
        compiled_grad_fn = mx.compile(
            lambda value: grad_fn(layer, value),
            inputs=layer.trainable_parameters(),
        )
        _, compiled_grads = compiled_grad_fn(inputs)

        constrained_epsilon = mlx_nn.softplus(layer.epsilon_param)
        epsilon_grad = mx.grad(
            lambda eps: mx.sum(
                fused_yat_score(
                    inputs,
                    layer.kernel,
                    bias=layer.bias,
                    alpha=layer.alpha,
                    epsilon=eps,
                )
            )
        )(constrained_epsilon)
        expected_param_grad = epsilon_grad * mx.sigmoid(layer.epsilon_param)

        assert np.allclose(
            np.array(compiled_grads["epsilon_param"]),
            np.array(eager_grads["epsilon_param"]),
            rtol=2e-5,
            atol=2e-6,
        )
        assert np.allclose(
            np.array(compiled_grads["epsilon_param"]),
            np.array(expected_param_grad),
            rtol=2e-5,
            atol=2e-6,
        )
        assert np.any(np.array(compiled_grads["epsilon_param"]) != 0.0)
    finally:
        mx.set_default_device(mx.cpu)
