"""Exact rational enclosures of supported native real-valued graph functions."""

import hashlib
from pathlib import Path
from typing import Union, cast

from ..research import intervals
from ..research.intervals import RationalInterval as Interval
from ..research.intervals import rational
from .baselines import IMQExpansion, LinearExpansion
from .graph import YatGraph
from .interpretable import ThreeNeuronYat, YatExpansion
from .research import model_from_snapshot


def enclose_native(snapshot, box, *, controls=None):
    """Enclose every output for all real inputs in one closed rational box.

    Supports fixed Yat, reciprocal-distance IMQ, linear expansions and residual sums.
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
    inputs = {name: Interval(*box[name]) for name in names}
    blocks = (
        dict(model.blocks.items())
        if isinstance(model, YatGraph)
        else {name: getattr(model, name) for name in model.state_names}
    )
    if any(
        not isinstance(block, (YatExpansion, IMQExpansion, LinearExpansion))
        for block in blocks.values()
    ):
        raise ValueError(
            "enclosure supports only fixed Yat and IMQ expansions or linear modules"
        )
    trace = {}
    denominators = {}

    def execute(name, values):
        block = cast(Union[YatExpansion, IMQExpansion, LinearExpansion], blocks[name])
        centers = block.centers.detach().cpu().tolist()
        coefficients = block.coefficients.detach().cpu().tolist()
        if isinstance(block, LinearExpansion):
            # Contract trainable factors as rationals, not rounded tensor products.
            # A fixed linear form attains box extrema at coordinate endpoints.
            raw = [
                sum(
                    (
                        value
                        * sum(
                            (
                                rational(coefficient) * rational(center[k])
                                for coefficient, center in zip(row, centers)
                            ),
                            rational(0),
                        )
                        for k, value in enumerate(values)
                    ),
                    Interval.point(0),
                )
                for row in coefficients
            ]
            ds: list[Interval] = []  # Linear modules contain no denominator.
        else:
            epsilon = (
                block.kernel.epsilon
                if isinstance(block, YatExpansion)
                else block.epsilon
            )
            if Interval.point(epsilon).lower <= 0:
                raise ValueError("epsilon must be strictly positive")
            features, ds = [], []
            for center in centers:
                denominator = sum(
                    (
                        (value - weight).square()
                        for value, weight in zip(values, center)
                    ),
                    Interval.point(epsilon),
                )
                numerator = (
                    sum(
                        (value * weight for value, weight in zip(values, center)),
                        Interval.point(0),
                    ).square()
                    if isinstance(block, YatExpansion)
                    else Interval.point(1)
                )
                features.append(numerator * denominator.positive_reciprocal())
                ds.append(denominator)
            raw = [
                sum(
                    (feature * weight for feature, weight in zip(features, row)),
                    Interval.point(0),
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
            Interval.point(v) if replacement is not None else r * v
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
        state = {slot: inputs.get(slot, Interval.point(0)) for slot in model.slots}
        trace["state.0"] = [state[slot] for slot in model.slots]
        for index, layer in enumerate(model.layer_specs):
            updates = [
                (spec, execute(spec.name, [state[slot] for slot in spec.reads]))
                for spec in layer
            ]
            for spec, values in updates:
                for slot, value in zip(spec.writes, values):
                    state[slot] = state[slot] + value
            trace[f"state.{index + 1}"] = [state[slot] for slot in model.slots]
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


def enclose_native_difference(snapshot, box, *, controls=None, reference_controls=None):
    """Enclose edited minus reference outputs on the same real input box.

    Both executions use the same fixed parameters. Conservative structural
    dependencies establish exact zero for outputs unaffected by every changed
    module control. Other outputs use subtraction of the two valid enclosures.
    No numerical equality or pointwise sampling is used to infer independence.
    """
    reference = enclose_native(snapshot, box, controls=reference_controls)
    edited = enclose_native(snapshot, box, controls=controls)
    model = model_from_snapshot(snapshot)
    if isinstance(model, YatGraph):
        dependencies = model.dependencies()
        widths = {
            name: cast(
                Union[YatExpansion, IMQExpansion, LinearExpansion], block
            ).out_features
            for name, block in model.blocks.items()
        }
    else:
        dependencies = {"target": ["module:h", "module:y"], "protected": ["module:p"]}
        widths = {name: 1 for name in model.state_names}

    def action(mapping, name):
        control = mapping.get(name, {})
        replacement = control.get("replacement")
        value = replacement if replacement is not None else control.get("gate", "1")
        values = value if isinstance(value, list) else [value] * widths[name]
        return ("replacement" if replacement is not None else "gate", tuple(values))

    changed = [
        name
        for name in model.state_names
        if action(reference["controls"], name) != action(edited["controls"], name)
    ]
    zero_outputs, bounds = [], {}
    for name in model.output_names:
        if not {f"module:{module}" for module in changed} & set(dependencies[name]):
            bounds[name] = ["0", "0"]
            zero_outputs.append(name)
        else:
            bounds[name] = (
                Interval(*edited["output_bounds"][name])
                - Interval(*reference["output_bounds"][name])
            ).to_list()
    return dict(
        schema="nmn.rational-difference.v1",
        model_snapshot=snapshot,
        input_box=edited["input_box"],
        controls=edited["controls"],
        reference_controls=reference["controls"],
        output_bounds=bounds,
        changed_modules=changed,
        structural_zero_outputs=zero_outputs,
        dependencies=dependencies,
        reference_enclosure=reference,
        edited_enclosure=edited,
        assurance="exact rational bounds on edited minus reference real-valued outputs for shared inputs and parameters",
        limitations=[
            "No floating-point runtime roundoff coverage.",
            "Affected-output subtraction may overestimate because input correlation is discarded.",
            "Structural zeros use fixed graph routing, not inferred semantic or empirical independence.",
        ],
    )
