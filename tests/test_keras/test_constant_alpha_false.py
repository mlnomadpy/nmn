"""Regression tests for issue #3 (Keras backend): ``constant_alpha=False``
and ``constant_bias=False`` must not be silently coerced to ``0.0``.
"""

from __future__ import annotations

import pytest

keras = pytest.importorskip("keras")
import numpy as np

from nmn.keras import YatNMN


def _new_layer(**kwargs):
    base = dict(units=16)
    base.update(kwargs)
    return YatNMN(**base)


class TestConstantAlphaFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(use_alpha=True, constant_alpha=False)
        layer.build((None, 8))
        assert getattr(layer, "_constant_alpha_value", None) is None

    def test_false_does_not_zero_output(self) -> None:
        rng = np.random.default_rng(0)
        layer = _new_layer(use_alpha=True, constant_alpha=False)
        x = keras.ops.convert_to_tensor(rng.standard_normal((4, 8)).astype("float32"))
        y = layer(x)
        y_np = keras.ops.convert_to_numpy(y)
        assert np.isfinite(y_np).all()
        assert float(np.linalg.norm(y_np)) > 0.0, (
            "output collapsed to zero — constant_alpha=False silently coerced "
            "to 0.0 (issue #3 regression)"
        )


class TestConstantBiasFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(use_bias=True, constant_bias=False)
        layer.build((None, 8))
        assert getattr(layer, "_constant_bias_value", None) is None
