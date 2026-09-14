"""Scoped numerical kernel and finite sensor-bank diagnostics."""

import math

import torch

from .interpretable import YatExpansion
from .nmn import YatNMN
from .research import expansion_geometry, input_jacobian, yat_gram


def sensor_diagnostics(centers, points, *, epsilon=1.0, noise_radius=0.0):
    """Measure finite-bank observation geometry on a declared point population.

    Noise radius is a supplied Euclidean bound in observation space. A pair of
    disjoint radius-r observation balls requires distance > 2r. The returned
    predicate describes only sampled pairs and floating-point calculations; it
    does not prove a uniform inverse bound or realizable adversarial noise.
    All-pairs arrays cost O(N²); Jacobians cost O(N * centers * input_dimension).
    """
    if not math.isfinite(noise_radius) or noise_radius < 0:
        raise ValueError("noise_radius must be finite and nonnegative")
    if centers.ndim != 2 or points.ndim != 2 or not len(centers) or not len(points):
        raise ValueError("centers and points must be nonempty matrices")
    if not centers.is_floating_point() or not points.is_floating_point():
        raise ValueError("centers and points must be floating point")
    centers = centers.detach().to(dtype=torch.float64)
    points = points.detach().to(device=centers.device, dtype=torch.float64)
    if not bool(torch.isfinite(centers).all() & torch.isfinite(points).all()):
        raise ValueError("centers and points must be finite")

    def observe(point):
        return yat_gram(point.unsqueeze(0), centers, epsilon=epsilon).squeeze(0)

    values = yat_gram(points, centers, epsilon=epsilon)
    jacobian = input_jacobian(observe, points)
    singular_values = torch.linalg.svdvals(jacobian)
    # A wide Jacobian has input null directions even when its returned singular
    # values are all positive: pad those missing input-space singular values.
    if centers.shape[0] < points.shape[1]:
        singular_values = torch.cat(
            (
                singular_values,
                points.new_zeros(len(points), points.shape[1] - centers.shape[0]),
            ),
            dim=-1,
        )
    source_distance = (points[:, None] - points[None, :]).square().sum(-1).sqrt()
    observed_distance = (values[:, None] - values[None, :]).square().sum(-1).sqrt()
    distinct = source_distance > 0
    pairs = torch.triu(distinct, diagonal=1)
    ratios = observed_distance[pairs] / source_distance[pairs]
    centered = values - values.mean(0)
    return {
        "points": points,
        "centers": centers,
        "epsilon": epsilon,
        "observations": values,
        "zero_observation": observe(torch.zeros_like(points[0])),
        "jacobian": jacobian,
        "input_space_singular_values": singular_values,
        "local_minimum_input_gain": singular_values[:, -1],
        "source_pair_distances": source_distance,
        "observation_pair_distances": observed_distance,
        "distinct_pair_mask": pairs,
        "empirical_minimum_separation_ratio": ratios.min() if ratios.numel() else None,
        "noise_radius": noise_radius,
        "sampled_noise_ball_separation": (observed_distance > 2 * noise_radius) & pairs,
        "observation_mean": values.mean(0),
        "observation_covariance": (
            centered.T @ centered / (len(points) - 1) if len(points) > 1 else None
        ),
        "assurance": "finite floating-point diagnostics",
        "limitations": [
            "A positive sampled separation ratio is not a uniform inverse bound.",
            "Positive local Jacobian gain does not establish global injectivity.",
            "Covariance depends on this population, not an inferred data distribution.",
        ],
    }


def diagnose_layer(layer, inputs):
    """Inspect actual NMN semantics before assigning function-space quantities.

    Supported norm calculation: YatExpansion with an unmodified unbiased,
    non-spherical, untied, fixed-epsilon kernel and no hidden alpha scaling.
    Other YatNMN settings still return actual outputs/flags but carry explicit
    unsupported reasons rather than inheriting an unrelated Mercer/RKHS claim.
    """
    if not isinstance(layer, (YatExpansion, YatNMN)):
        raise TypeError("diagnostics support YatExpansion and YatNMN")
    kernel = layer.kernel if isinstance(layer, YatExpansion) else layer
    if inputs.ndim != 2 or not len(inputs) or inputs.shape[1] != kernel.in_features:
        raise ValueError("inputs must be a nonempty matrix matching layer input width")
    if not inputs.is_floating_point() or not bool(torch.isfinite(inputs).all()):
        raise ValueError("inputs must be finite floating-point values")
    reasons = []
    if kernel.bias is not None or kernel._constant_bias_value not in (None, 0):
        reasons.append(
            "numerator bias: unsupported by the unbiased fixed-kernel diagnostic"
        )
    if kernel.spherical or kernel.weight_normalized:
        reasons.append(
            "normalized geometry requires a separate function-space interpretation"
        )
    if kernel.learnable_epsilon:
        reasons.append(
            "learnable epsilon: this diagnostic does not freeze its effective kernel"
        )
    if kernel.tie_kernel_bank:
        reasons.append("tied bank/sliced geometry is not supported by this diagnostic")
    if kernel.alpha is not None or kernel._constant_alpha_value is not None:
        reasons.append(
            "hidden alpha scaling is not interpreted as an expansion coefficient here"
        )
    flags = {
        "spherical": kernel.spherical,
        "weight_normalized": kernel.weight_normalized,
        "distance_mode": kernel.distance_mode,
        "epsilon": kernel.epsilon,
        "learnable_epsilon": kernel.learnable_epsilon,
        "lazy": kernel.lazy,
        "tied_kernel_bank": kernel.tie_kernel_bank,
        "constant_bias": kernel._constant_bias_value,
        "constant_alpha": kernel._constant_alpha_value,
        "trainable_parameters": {
            name: parameter.requires_grad
            for name, parameter in layer.named_parameters()
        },
    }
    with torch.no_grad():
        result = {
            "flags": flags,
            "inputs": inputs.detach().clone(),
            "precision": {"observed": str(inputs.dtype), "geometry": "torch.float64"},
            "parameter_values": {
                name: parameter.detach().clone()
                for name, parameter in layer.named_parameters()
            },
            "actual_outputs": layer(inputs),
            "actual_zero_output": layer(torch.zeros_like(inputs[:1])),
            "scope": (
                "unsupported-function-space"
                if reasons
                else "fixed-unbiased-section-snapshot"
            ),
            "unsupported_reasons": reasons,
        }
        if not reasons:
            gram = yat_gram(inputs.double(), epsilon=kernel.epsilon)
            eigenvalues = torch.linalg.eigvalsh(gram)
            tolerance = (
                torch.finfo(gram.dtype).eps * len(inputs) * eigenvalues.abs().max()
            )
            result["sampled_gram"] = {
                "matrix": gram,
                "symmetry_residual": (gram - gram.T).abs().max(),
                "eigenvalues": eigenvalues,
                "roundoff_scale": tolerance,
                "negative_beyond_roundoff_scale": (eigenvalues < -tolerance).any(),
                "numerical_rank": (eigenvalues > tolerance).sum(),
                "condition_number": (
                    eigenvalues[-1] / eigenvalues[0]
                    if bool(eigenvalues[0] > tolerance)
                    else eigenvalues.new_tensor(float("inf"))
                ),
            }
            if isinstance(layer, YatExpansion):
                result["expansion"] = expansion_geometry(layer, inputs)
            else:
                result["rkhs_norm_status"] = (
                    "not computed: use YatExpansion for explicit coefficients"
                )
        result["limitations"] = [
            "Finite Gram observations do not certify global PSD, universality or semantics.",
            "Roundoff scale is a diagnostic heuristic, not a certified eigenvalue error bound.",
            "Fixed-snapshot norms do not assert a common RKHS throughout parameter training.",
            "Lazy mode freezes kernel directions only; inspect every parameter flag.",
        ]
    return result
