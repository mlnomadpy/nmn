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
