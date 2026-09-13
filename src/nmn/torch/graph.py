"""Explicit-state residual ⵟ graphs with simultaneous reads within each layer."""

from dataclasses import asdict, dataclass
from typing import List, Mapping, Optional, Sequence, cast

import torch
from torch import nn

from .interpretable import Intervention, YatExpansion, _control


@dataclass(frozen=True)
class YatModuleSpec:
    """Fixed coordinate routing and trainable expansion size for one module."""

    name: str
    reads: tuple
    writes: tuple
    num_centers: int = 1
    epsilon: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        if not self.name or not self.name.isidentifier():
            raise ValueError("module name must be a nonempty Python identifier")
        if not self.reads or not self.writes:
            raise ValueError("modules require at least one read and write slot")
        if len(set(self.reads)) != len(self.reads) or len(set(self.writes)) != len(
            self.writes
        ):
            raise ValueError("read and write slot lists must each be unique")


class YatGraph(nn.Module):
    """Trainable residual graph with fixed coordinate encoder and readout.

    Every layer reads the SAME incoming state and adds its gated module writes:
    ``z_next = z + sum_m scatter(gate_m * expansion_m(z[reads_m])))``.
    Overlapping writes add. Input slots are filled from x; other slots start at
    zero. Readout selects named state coordinates without a hidden learned head.
    Replacement overrides a module's write vector, not the entire state slot.

    Trace and intervention keys are globally unique module names. State snapshots
    are under ``state.0``, ``state.1``, etc. Routing is checkpointed and mismatched
    routing is rejected when loading state_dict, even if tensor shapes agree.
    """

    def __init__(
        self,
        slots: Sequence[str],
        input_names: Sequence[str],
        output_names: Sequence[str],
        layers: Sequence[Sequence[YatModuleSpec]],
        *,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.slots = tuple(slots)
        self.input_names = tuple(input_names)
        self.output_names = tuple(output_names)
        self.layer_specs = tuple(tuple(layer) for layer in layers)
        for label, names in (
            ("slots", self.slots),
            ("input_names", self.input_names),
            ("output_names", self.output_names),
        ):
            if not names or any(not isinstance(s, str) or not s for s in names):
                raise ValueError(f"{label} must contain nonempty strings")
            if len(set(names)) != len(names):
                raise ValueError(f"{label} must be unique")
        if not set(self.input_names + self.output_names) <= set(self.slots):
            raise ValueError("input/output names must belong to slots")
        if not self.layer_specs or any(not layer for layer in self.layer_specs):
            raise ValueError("provide nonempty layers")
        specs = [spec for layer in self.layer_specs for spec in layer]
        if any(not isinstance(spec, YatModuleSpec) for spec in specs):
            raise TypeError("layers must contain YatModuleSpec objects")
        self.state_names = tuple(spec.name for spec in specs)
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("module names must be globally unique")
        self.blocks = nn.ModuleDict()
        for spec in specs:
            if not set(spec.reads + spec.writes) <= set(self.slots):
                raise ValueError(f"unknown slot in module {spec.name}")
            self.blocks[spec.name] = YatExpansion(
                len(spec.reads),
                len(spec.writes),
                spec.num_centers,
                epsilon=spec.epsilon,
                device=device,
                dtype=dtype,
            )
        self._slot_indices = {name: i for i, name in enumerate(self.slots)}

    def configuration(self):
        """JSON-compatible full construction contract (excluding dtype/device)."""
        return {
            "class": "nmn.torch.YatGraph",
            "slots": list(self.slots),
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
            "layers": [
                [
                    {
                        **asdict(spec),
                        "reads": list(spec.reads),
                        "writes": list(spec.writes),
                    }
                    for spec in layer
                ]
                for layer in self.layer_specs
            ],
            "update": "simultaneous-residual-add",
            "distance_mode": "direct",
        }

    @classmethod
    def from_configuration(cls, config, *, device=None, dtype=torch.float32):
        """Reconstruct topology before loading tensor weights."""
        if config.get("class") != "nmn.torch.YatGraph" or (
            config.get("update") != "simultaneous-residual-add"
            or config.get("distance_mode") != "direct"
        ):
            raise ValueError("unsupported graph configuration")
        layers = [
            [YatModuleSpec(**spec) for spec in layer] for layer in config["layers"]
        ]
        return cls(
            config["slots"],
            config["input_names"],
            config["output_names"],
            layers,
            device=device,
            dtype=dtype,
        )

    def get_extra_state(self):
        return self.configuration()

    def set_extra_state(self, state):
        if state != self.configuration():
            raise ValueError(
                "checkpoint routing/configuration does not match this graph"
            )

    def dependencies(self):
        """Conservative input/module ancestors of each output; no numerical pruning."""
        ancestors = {
            slot: ({f"input:{slot}"} if slot in self.input_names else set())
            for slot in self.slots
        }
        for layer in self.layer_specs:
            updates = []
            for spec in layer:
                inherited = set().union(*(ancestors[s] for s in spec.reads))
                updates.append((spec, inherited | {f"module:{spec.name}"}))
            for spec, inherited in updates:
                for slot in spec.writes:
                    ancestors[slot] = ancestors[slot] | inherited
        return {name: sorted(ancestors[name]) for name in self.output_names}

    def forward_with_trace(
        self,
        x: torch.Tensor,
        interventions: Optional[Mapping[str, Intervention]] = None,
    ):
        if (
            x.ndim < 1
            or x.shape[-1] != len(self.input_names)
            or not x.is_floating_point()
        ):
            raise ValueError(
                f"expected floating input with last dimension {len(self.input_names)}"
            )
        controls = {} if interventions is None else dict(interventions)
        unknown = set(controls) - set(self.state_names)
        if unknown:
            raise ValueError(f"unknown intervention modules: {sorted(unknown)}")
        if any(not isinstance(c, Intervention) for c in controls.values()):
            raise TypeError("controls must be Intervention objects")
        inputs = {name: x[..., i] for i, name in enumerate(self.input_names)}
        zero = torch.zeros_like(x[..., 0])
        state = torch.stack([inputs.get(name, zero) for name in self.slots], dim=-1)
        trace = {"state.0": state}
        for layer_index, layer in enumerate(self.layer_specs):
            writes: List[List[torch.Tensor]] = [[] for _ in self.slots]
            for spec in layer:
                read = state[..., [self._slot_indices[name] for name in spec.reads]]
                contributions = cast(
                    YatExpansion, self.blocks[spec.name]
                ).contributions(read)
                raw = contributions.sum(-1)
                control = controls.get(spec.name, Intervention())
                effective = (
                    _control(control.replacement, raw)
                    if control.replacement is not None
                    else raw * _control(control.gate, raw)
                )
                trace.update(
                    {
                        f"{spec.name}.input": read,
                        f"{spec.name}.contributions": contributions,
                        f"{spec.name}.raw": raw,
                        spec.name: effective,
                    }
                )
                for i, name in enumerate(spec.writes):
                    writes[self._slot_indices[name]].append(effective[..., i])
            state = state + torch.stack(
                [sum(values, torch.zeros_like(zero)) for values in writes], dim=-1
            )
            trace[f"state.{layer_index + 1}"] = state
        output = state[..., [self._slot_indices[name] for name in self.output_names]]
        return output, trace

    def forward(self, x, interventions=None):
        return self.forward_with_trace(x, interventions)[0]
