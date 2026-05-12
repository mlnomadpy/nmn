"""Regression tests for issue #3 (TensorFlow backend): ``constant_alpha=False``
and ``constant_bias=False`` must not be silently coerced to ``0.0``.
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")
import numpy as np

from nmn.tf import YatNMN


def _new_layer(**kwargs):
    base = dict(features=16)
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
        x = tf.convert_to_tensor(rng.standard_normal((4, 8)).astype("float32"))
        y = layer(x).numpy()
        assert np.isfinite(y).all()
        assert float(np.linalg.norm(y)) > 0.0, (
            "output collapsed to zero — constant_alpha=False silently coerced "
            "to 0.0 (issue #3 regression)"
        )


class TestConstantBiasFalseTrap:
    def test_false_is_treated_as_none(self) -> None:
        layer = _new_layer(use_bias=True, constant_bias=False)
        layer.build((None, 8))
        assert getattr(layer, "_constant_bias_value", None) is None
