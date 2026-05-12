"""Regression tests for issue #3 (PyTorch backend): ``constant_alpha=False``
and ``constant_bias=False`` must not be silently coerced to ``0.0``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nmn.torch import YatNMN


def _new_layer(**kwargs):
    base = dict(in_features=8, out_features=16)
    base.update(kwargs)
    return YatNMN(**base)


class TestConstantAlphaFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(alpha=True, constant_alpha=False)
        assert getattr(layer, "_constant_alpha_value", None) is None

    def test_false_keeps_learnable_alpha(self) -> None:
        layer = _new_layer(alpha=True, constant_alpha=False)
        alpha = getattr(layer, "alpha", None)
        assert isinstance(alpha, torch.nn.Parameter)

    def test_false_does_not_zero_output(self) -> None:
        torch.manual_seed(0)
        layer = _new_layer(alpha=True, constant_alpha=False)
        x = torch.randn(4, 8)
        y = layer(x)
        assert torch.isfinite(y).all()
        assert y.norm().item() > 0.0, (
            "output collapsed to zero — constant_alpha=False silently coerced "
            "to 0.0 (issue #3 regression)"
        )

    def test_zero_float_is_still_respected(self) -> None:
        layer = _new_layer(alpha=True, constant_alpha=0.0)
        assert layer._constant_alpha_value == 0.0


class TestConstantBiasFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(bias=True, constant_bias=False)
        assert getattr(layer, "_constant_bias_value", None) is None
        bias = getattr(layer, "bias", None)
        assert bias is None or isinstance(bias, torch.nn.Parameter)
