"""Trainable ⵟ expansions with explicit, intervenable computation paths."""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Union, cast

import torch
from torch import nn

from .nmn import YatNMN

Control = Union[float, torch.Tensor]


@dataclass(frozen=True)
class Intervention:
    """Scale a module output, or replace it after scaling.

    Controls are scalars or tensors broadcastable to the module output without
    expanding its shape. For per-example controls use shape ``(batch, 1)``.
    Replacement cuts gradients through the replaced computation; gradients into
    a tensor replacement are preserved. A replacement takes precedence over gate.
    """

    gate: Control = 1.0
    replacement: Optional[Control] = None


def _control(value: Control, output: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=output.dtype, device=output.device)
    try:
        return torch.broadcast_to(value, output.shape)
    except RuntimeError as exc:
        raise ValueError("control must broadcast to the module output shape") from exc


class YatExpansion(nn.Module):
    """A trainable finite kernel expansion with exact additive contributions.

    ``output[j] = sum_i coefficient[j, i] * k(center[i], input)``.
    Centers and coefficients are trainable. Epsilon is fixed and positive;
    numerator bias and hidden alpha scaling are disabled. Contributions have
    shape ``(..., out_features, num_centers)`` and sum to the output.
    This decomposition describes computation, not learned semantic meaning.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 1,
        *,
        epsilon: float = 1.0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("num_centers", num_centers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.in_features = in_features
        self.out_features = out_features
        self.num_centers = num_centers
        self.kernel = YatNMN(
            in_features,
            num_centers,
            bias=False,
            alpha=False,
            epsilon=epsilon,
            device=device,
            param_dtype=dtype,
        )
        self.coefficients = nn.Parameter(
            torch.full(
                (out_features, num_centers),
                1.0 / num_centers,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def centers(self) -> nn.Parameter:
        """Prototype coordinates, shape ``(num_centers, in_features)``."""
        return self.kernel.weight

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected input with last dimension {self.in_features}")
        if not x.is_floating_point():
            raise ValueError("input must be a floating-point tensor")
        return cast(torch.Tensor, self.kernel(x))

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Return signed center contributions without detaching autograd."""
        features = self._features(x)
        return features.unsqueeze(-2) * self.coefficients.to(features.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._features(x)
        return torch.nn.functional.linear(
            features, self.coefficients.to(features.dtype)
        )


class ThreeNeuronYat(nn.Module):
    """An explicit trainable network with one protected computation path.

    Input ``[..., 2]`` contains ``(u, v)``. Output ``[..., 2]`` contains
    ``(target, protected)``. The routing is fixed:
    ``h = H(u)``, ``p = P(v)``, ``y = Y(concat(h, v))``.

    Interventions can address ``h``, ``p``, or ``y``. Changing ``h`` recomputes
    ``y`` and cannot affect ``p`` because that path has no dependency on ``h``.
    This architectural property holds for arbitrary parameter values. It is
    not a robustness guarantee against parameter edits or an intervention on p.

    With one center per module there are seven trainable scalar parameters
    (four center coordinates and three coefficients). Larger center banks keep
    the same inspectable routing. Use ``reference()`` for the fixed all-ones
    initialization corresponding to the rational three-neuron example.
    """

    state_names = ("h", "p", "y")
    output_names = ("target", "protected")

    def __init__(
        self,
        num_centers: int = 1,
        *,
        epsilon: float = 1.0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        options = dict(
            num_centers=num_centers, epsilon=epsilon, device=device, dtype=dtype
        )
        self.h = YatExpansion(1, 1, **options)
        self.p = YatExpansion(1, 1, **options)
        self.y = YatExpansion(2, 1, **options)

    @classmethod
    def reference(cls, *, device=None, dtype=torch.float32):
        """Construct the epsilon=1, unit-center, unit-coefficient reference."""
        model = cls(device=device, dtype=dtype)
        with torch.no_grad():
            for module in (model.h, model.p, model.y):
                module.centers.fill_(1.0)
                module.coefficients.fill_(1.0)
        return model

    def forward_with_trace(
        self,
        x: torch.Tensor,
        interventions: Optional[Mapping[str, Intervention]] = None,
    ):
        """Return ``(outputs, trace)`` with live pre/post intervention tensors.

        Trace contains module inputs, center contributions, raw outputs and
        effective states. Raw contributions reconstruct pre-intervention outputs.
        No cached state is reused between calls, and no parameters are modified.
        """
        if x.ndim < 1 or x.shape[-1] != 2:
            raise ValueError("expected input with last dimension 2: (u, v)")
        interventions = {} if interventions is None else dict(interventions)
        unknown = set(interventions) - set(self.state_names)
        if unknown:
            raise ValueError(f"unknown intervention states: {sorted(unknown)}")
        if any(not isinstance(item, Intervention) for item in interventions.values()):
            raise TypeError(
                "interventions must map state names to Intervention objects"
            )
        trace: Dict[str, torch.Tensor] = {}

        def evaluate(name, module, inputs):
            contributions = module.contributions(inputs)
            raw = contributions.sum(dim=-1)
            control = interventions.get(name, Intervention())
            effective = (
                _control(control.replacement, raw)
                if control.replacement is not None
                else raw * _control(control.gate, raw)
            )
            trace[f"{name}.input"] = inputs
            trace[f"{name}.contributions"] = contributions
            trace[f"{name}.raw"] = raw
            trace[name] = effective
            return effective

        h = evaluate("h", self.h, x[..., :1])
        p = evaluate("p", self.p, x[..., 1:])
        y = evaluate("y", self.y, torch.cat((h, x[..., 1:]), dim=-1))
        return torch.cat((y, p), dim=-1), trace

    def forward(
        self,
        x: torch.Tensor,
        interventions: Optional[Mapping[str, Intervention]] = None,
    ) -> torch.Tensor:
        """Compute target and protected outputs with optional interventions."""
        output, _ = self.forward_with_trace(x, interventions)
        return cast(torch.Tensor, output)
