"""Explicit finite IMQ and conventional tanh blocks for matched routing studies."""

import math

import torch
from torch import nn


class IMQExpansion(nn.Module):
    """Finite inverse-multiquadric expansion using 1/(distance² + epsilon).

    Centers and signed coefficients are trainable; epsilon is fixed and shared.
    This uses generalized IMQ exponent 1, matching the rational denominator
    comparison; it is distinct from the exponent-1/2 convention.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_centers=1,
        *,
        epsilon=1.0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (in_features, out_features, num_centers)
        ):
            raise ValueError("dimensions must be positive integers")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.in_features, self.out_features, self.num_centers = (
            in_features,
            out_features,
            num_centers,
        )
        self.epsilon = epsilon
        self.centers = nn.Parameter(
            torch.empty(num_centers, in_features, device=device, dtype=dtype)
        )
        nn.init.xavier_normal_(self.centers)
        self.coefficients = nn.Parameter(
            torch.full(
                (out_features, num_centers), 1 / num_centers, device=device, dtype=dtype
            )
        )

    def features(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError(
                "input must be floating point with the declared feature width"
            )
        distance = (x.unsqueeze(-2) - self.centers.to(x.dtype)).square().sum(-1)
        return torch.reciprocal(distance + self.epsilon)

    def contributions(self, x):
        return self.features(x).unsqueeze(-2) * self.coefficients.to(x.dtype)

    def forward(self, x):
        return self.contributions(x).sum(-1)


class LinearExpansion(nn.Module):
    """Trainable linear-kernel expansion with inspectable signed contributions.

    Feature i is x dot center[i]; output j sums coefficient[j,i] times that
    feature. num_centers bounds factorization rank, not nonlinearity. No biases
    are hidden. epsilon is accepted only for graph-spec compatibility and unused.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_centers=1,
        *,
        epsilon=1.0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (in_features, out_features, num_centers)
        ):
            raise ValueError("dimensions must be positive integers")
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.centers = nn.Parameter(
            torch.empty(num_centers, in_features, device=device, dtype=dtype)
        )
        nn.init.xavier_normal_(self.centers)
        self.coefficients = nn.Parameter(
            torch.full(
                (out_features, num_centers), 1 / num_centers, device=device, dtype=dtype
            )
        )
        del epsilon

    @property
    def effective_weight(self):
        """Represented linear map, shape (outputs, inputs), with autograd."""
        return self.coefficients @ self.centers

    def features(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError(
                "input must be floating point with the declared feature width"
            )
        return x @ self.centers.to(x.dtype).T

    def contributions(self, x):
        return self.features(x).unsqueeze(-2) * self.coefficients.to(x.dtype)

    def forward(self, x):
        return self.contributions(x).sum(-1)


class TanhMLPBlock(nn.Module):
    """Linear+bias → tanh → bias-free linear, exposing hidden-unit contributions.

    Hidden biases add trainable parameters compared with the finite kernel banks.
    Equal hidden width is not asserted to mean equal capacity or compute.
    """

    def __init__(
        self,
        in_features,
        out_features,
        num_centers=1,
        *,
        epsilon=1.0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (in_features, out_features, num_centers)
        ):
            raise ValueError("dimensions must be positive integers")
        self.in_features, self.out_features, self.num_centers = (
            in_features,
            out_features,
            num_centers,
        )
        self.hidden = nn.Linear(
            in_features, num_centers, bias=True, device=device, dtype=dtype
        )
        self.readout = nn.Linear(
            num_centers, out_features, bias=False, device=device, dtype=dtype
        )
        # epsilon is a graph-spec field used by kernel families, not tanh.
        del epsilon

    def features(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError(
                "input must be floating point with the declared feature width"
            )
        return torch.tanh(
            torch.nn.functional.linear(
                x, self.hidden.weight.to(x.dtype), self.hidden.bias.to(x.dtype)
            )
        )

    def contributions(self, x):
        return self.features(x).unsqueeze(-2) * self.readout.weight.to(x.dtype)

    def forward(self, x):
        return self.contributions(x).sum(-1)


def baseline_geometry(block, inputs):
    """Distinguish kernel-module geometry from conventional MLP feature observations."""
    if isinstance(block, IMQExpansion):
        centers = block.centers.double()
        gram = torch.reciprocal(
            (centers[:, None] - centers[None, :]).square().sum(-1) + block.epsilon
        )
        coefficients = block.coefficients.double()
        return {
            "family": "imq",
            "formula": "(squared_distance + epsilon)^(-1)",
            "epsilon": block.epsilon,
            "centers": centers,
            "coefficients": coefficients,
            "kernel_values": block.features(inputs),
            "center_gram": gram,
            "rkhs_inner_products": coefficients @ gram @ coefficients.T,
        }
    if isinstance(block, LinearExpansion):
        centers = block.centers.double()
        coefficients = block.coefficients.double()
        gram = centers @ centers.T
        return {
            "family": "linear",
            "formula": "dot(input, center)",
            "centers": centers,
            "coefficients": coefficients,
            "kernel_values": block.features(inputs),
            "center_gram": gram,
            "effective_weight": coefficients @ centers,
            "rkhs_inner_products": coefficients @ gram @ coefficients.T,
            "scope": "local linear-kernel function norms; not a norm for the full nonlinear graph",
        }
    if isinstance(block, TanhMLPBlock):
        return {
            "family": "tanh",
            "hidden_features": block.features(inputs),
            "rkhs_status": "not assigned: conventional neural feature block",
            "hidden_bias": True,
            "output_bias": False,
        }
    raise TypeError("unsupported baseline block")
