"""Exact rational enclosures of supported native real-valued graph functions."""

import hashlib
from pathlib import Path
from typing import Union, cast

from ..research import intervals
from ..research.intervals import RationalInterval as I
from ..research.intervals import rational
from .baselines import IMQExpansion
from .graph import YatGraph
from .interpretable import ThreeNeuronYat, YatExpansion
from .research import model_from_snapshot


def enclose_native(snapshot, box, *, controls=None):
    """Enclose every output for all real inputs in one closed rational box.

    Supports fixed Yat and reciprocal-distance IMQ expansions and residual sums.
    Stored floating parameters are interpreted as their exact binary rational
    values. This does not enclose floating runtime roundoff. No branch search or
    contract verdict is performed; unsupported operations raise an error.
    """
    model = model_from_snapshot(snapshot)
    controls = {} if controls is None else dict(controls)
    if set(controls) - set(model.state_names):
        raise ValueError("unknown enclosure control module")
    names = model.input_names if isinstance(model, YatGraph) else ("u", "v")
    if set(box) != set(names):
        raise ValueError("box must contain exactly the model input names")
    inputs = {name: I(*box[name]) for name in names}
    blocks = (
        dict(model.blocks.items())
        if isinstance(model, YatGraph)
        else {name: getattr(model, name) for name in model.state_names}
    )
    if any(
        not isinstance(block, (YatExpansion, IMQExpansion)) for block in blocks.values()
    ):
        raise ValueError("enclosure supports only fixed Yat and IMQ expansions")
    trace = {}
    denominators = {}

    def execute(name, values):
        block = cast(Union[YatExpansion, IMQExpansion], blocks[name])
        centers = block.centers.detach().cpu().tolist()
        coefficients = block.coefficients.detach().cpu().tolist()
        epsilon = (
            block.kernel.epsilon if isinstance(block, YatExpansion) else block.epsilon
        )
        if I.point(epsilon).lower <= 0:
            raise ValueError("epsilon must be strictly positive")
        features, ds = [], []
        for center in centers:
            denominator = sum(
                ((value - weight).square() for value, weight in zip(values, center)),
                I.point(epsilon),
            )
            numerator = (
                sum(
                    (value * weight for value, weight in zip(values, center)),
                    I.point(0),
                ).square()
                if isinstance(block, YatExpansion)
                else I.point(1)
            )
            features.append(numerator * denominator.positive_reciprocal())
            ds.append(denominator)
        raw = [
            sum(
                (feature * weight for feature, weight in zip(features, row)), I.point(0)
            )
            for row in coefficients
        ]
        control = controls.get(name, {})
        if set(control) - {"gate", "replacement"}:
            raise ValueError("only gate and replacement controls are supported")
        replacement = control.get("replacement")
        value = replacement if replacement is not None else control.get("gate", 1)
        values_control = value if isinstance(value, list) else [value] * len(raw)
        if len(values_control) != len(raw):
            raise ValueError("control must be scalar or one value per module write")
        effective = [
            I.point(v) if replacement is not None else r * v
            for r, v in zip(raw, values_control)
        ]
        trace.update({name + ".input": values, name + ".raw": raw, name: effective})
        denominators[name] = ds
        return effective

    if isinstance(model, ThreeNeuronYat):
        h = execute("h", [inputs["u"]])
        p = execute("p", [inputs["v"]])
        y = execute("y", [h[0], inputs["v"]])
        outputs = [y[0], p[0]]
    else:
        state = {slot: inputs.get(slot, I.point(0)) for slot in model.slots}
        trace["state.0"] = [state[slot] for slot in model.slots]
        for index, layer in enumerate(model.layer_specs):
            updates = [
                (spec, execute(spec.name, [state[slot] for slot in spec.reads]))
                for spec in layer
            ]
            for spec, values in updates:
                for slot, value in zip(spec.writes, values):
                    state[slot] = state[slot] + value
            trace[f"state.{index+1}"] = [state[slot] for slot in model.slots]
        outputs = [state[name] for name in model.output_names]
    return dict(
        schema="nmn.rational-enclosure.v1",
        model_snapshot=snapshot,
        input_box={name: value.to_list() for name, value in inputs.items()},
        controls={
            name: {
                key: (
                    None
                    if value is None
                    else (
                        [str(rational(v)) for v in value]
                        if isinstance(value, list)
                        else str(rational(value))
                    )
                )
                for key, value in control.items()
            }
            for name, control in controls.items()
        },
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(intervals.__file__))
        },
        output_bounds={
            name: value.to_list() for name, value in zip(model.output_names, outputs)
        },
        trace={
            name: [value.to_list() for value in values]
            for name, values in trace.items()
        },
        denominator_bounds={
            name: [value.to_list() for value in values]
            for name, values in denominators.items()
        },
        assurance="exact rational enclosure of the real-valued function with stored binary-rational parameters",
        limitations=[
            "No floating-point runtime roundoff bound is included.",
            "One box only; no branch-and-bound or contract verification verdict.",
            "Tanh, read patches, parameter edits and input-dependent controls are unsupported.",
            "Dependency overestimation may make intervals loose; an oversized enclosure is not a counterexample.",
        ],
    )
