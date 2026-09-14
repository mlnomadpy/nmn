"""Distinguish finite observations from unsupported function-space claims."""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch import YatExpansion, YatNMN  # noqa: E402
from nmn.torch.diagnostics import diagnose_layer, sensor_diagnostics  # noqa: E402


def test_supported_and_incompatible_layer_semantics():
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    layer = YatExpansion(1, 1, dtype=torch.float64)
    with torch.no_grad():
        layer.centers.fill_(1)
    report = diagnose_layer(layer, x)
    assert report["scope"] == "fixed-unbiased-section-snapshot"
    assert report["expansion"]["rkhs_inner_products"].item() == 1
    assert report["actual_zero_output"].item() == 0
    incompatible = YatNMN(1, 1, constant_bias=1.0, lazy=True, param_dtype=torch.float64)
    report = diagnose_layer(incompatible, x)
    assert report["scope"] == "unsupported-function-space"
    assert "sampled_gram" not in report
    assert report["actual_zero_output"].item() > 0
    assert not report["flags"]["trainable_parameters"]["weight"]
    assert report["flags"]["trainable_parameters"]["alpha"]


def test_sensor_null_directions_and_origin_degeneracy():
    centers = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    report = sensor_diagnostics(centers, points, noise_radius=0.01)
    assert report["input_space_singular_values"].shape == (3, 2)
    assert report["local_minimum_input_gain"].abs().max() == 0
    assert report["empirical_minimum_separation_ratio"] == 0
    assert report["jacobian"][0].abs().max() == 0
    assert not report["sampled_noise_ball_separation"][0, 2]
    with pytest.raises(ValueError, match="nonnegative"):
        sensor_diagnostics(centers, points, noise_radius=-1)
