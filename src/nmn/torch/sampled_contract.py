"""Execute explicitly bound sampled target/protection contracts on native models."""

import hashlib
import json
import math
from pathlib import Path

import torch

from .interpretable import Intervention
from .research import collect_research_data


def evaluate_sampled_contract(model, dataset, contract):
    """Check shared native writes on named samples, never issue a certificate.

    Targets compare edited predictions with supplied semantic references.
    Protection compares edited and baseline coordinates. Optional baseline_targets
    separately require ordinary prediction accuracy. All measurements remain raw.
    """
    contract = json.loads(json.dumps(contract, allow_nan=False))
    required = {
        "schema",
        "scope",
        "arithmetic",
        "model_sha256",
        "dataset_sha256",
        "sample_ids",
        "action_domain",
        "controls",
        "targets",
        "protected",
        "provenance",
    }
    if (
        not isinstance(contract, dict)
        or not required <= set(contract)
        or set(contract) - required - {"baseline_targets"}
    ):
        raise ValueError("contract requires exactly the documented fields")
    if (
        contract["schema"] != "nmn.sampled-contract.v1"
        or contract["scope"] != "sampled"
        or contract["arithmetic"] != "floating-point"
    ):
        raise ValueError("only sampled floating-point contracts are supported here")
    if (
        not isinstance(contract["provenance"], str)
        or not contract["provenance"].strip()
    ):
        raise ValueError("contract provenance is required")
    if contract["dataset_sha256"] != dataset.sha256:
        raise ValueError("contract dataset identity mismatch")
    ids = contract["sample_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(s, str) for s in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("sample_ids must be nonempty unique IDs in declared order")
    samples = [dataset.sample(s) for s in ids]
    domain = contract["action_domain"]
    if (
        not isinstance(domain, dict)
        or set(domain) != {"kind", "gate_bounds", "allow_replacements"}
        or domain["kind"] != "shared-module-writes"
        or type(domain["allow_replacements"]) is not bool
    ):
        raise ValueError(
            "action_domain must declare shared-module-writes, gate_bounds and allow_replacements"
        )

    def finite(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )

    bounds = domain["gate_bounds"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(finite(v) for v in bounds)
        or bounds[0] > bounds[1]
    ):
        raise ValueError("gate_bounds must be an ordered pair of finite numbers")
    controls = contract["controls"]
    if not isinstance(controls, dict) or not set(controls) <= set(model.state_names):
        raise ValueError("controls must name native modules")
    native = {}
    for name in model.state_names:
        control = controls.get(name, {})
        if not isinstance(control, dict) or set(control) - {"gate", "replacement"}:
            raise ValueError(
                f"controls.{name}: only gate and replacement are supported"
            )
        gate = control.get("gate", 1.0)
        if not finite(gate) or not bounds[0] <= gate <= bounds[1]:
            raise ValueError(f"controls.{name}.gate: outside declared gate bounds")
        replacement = control.get("replacement")
        if replacement is not None:
            if not domain["allow_replacements"]:
                raise ValueError(f"controls.{name}: replacements are forbidden")
            block = (
                model.blocks[name] if hasattr(model, "blocks") else getattr(model, name)
            )
            if not (
                finite(replacement)
                or (
                    isinstance(replacement, list)
                    and len(replacement) == block.out_features
                    and all(finite(v) for v in replacement)
                )
            ):
                raise ValueError(
                    f"controls.{name}.replacement: expected shared scalar or output-width vector"
                )
        if name in controls:
            native[name] = Intervention(gate=gate, replacement=replacement)
    output_index = {name: i for i, name in enumerate(model.output_names)}

    def validate_measurement(section, *, reference):
        value = contract[section]
        fields = {"output_names", "absolute_tolerance"} | (
            {"expected"} if reference else set()
        )
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"{section}: requires {sorted(fields)}")
        names = value["output_names"]
        tolerances = value["absolute_tolerance"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)
            or not set(names) <= set(output_index)
            or not isinstance(tolerances, list)
            or len(tolerances) != len(names)
            or any(not finite(v) or v < 0 for v in tolerances)
        ):
            raise ValueError(
                f"{section}: invalid output names or nonnegative coordinate tolerances"
            )
        if reference:
            expected = value["expected"]
            if (
                not isinstance(expected, dict)
                or set(expected) != set(ids)
                or any(
                    not isinstance(v, list)
                    or len(v) != len(names)
                    or not all(finite(x) for x in v)
                    for v in expected.values()
                )
            ):
                raise ValueError(
                    f"{section}.expected: must cover exactly the declared samples and outputs"
                )

    validate_measurement("targets", reference=True)
    validate_measurement("protected", reference=False)
    if "baseline_targets" in contract:
        validate_measurement("baseline_targets", reference=True)
    parameter = next(model.parameters())
    if parameter.dtype not in (torch.float32, torch.float64):
        raise ValueError("sampled contracts require float32 or float64")
    inputs = torch.tensor(
        [s.inputs for s in samples], dtype=parameter.dtype, device=parameter.device
    )
    # This snapshot verifies the model binding before executing any edited run.
    initial = collect_research_data(model, inputs, sample_ids=ids, derivatives=False)
    if initial["model_sha256"] != contract["model_sha256"]:
        raise ValueError("contract model identity mismatch")
    execution = collect_research_data(
        model, inputs, sample_ids=ids, edits={"contract": native}, derivatives=False
    )
    baseline = execution["observations"]["baseline"]
    edited = execution["observations"]["edits"]["contract"]["outputs"]
    rows = []
    for i, sid in enumerate(ids):
        measurements = {}
        for section in ["targets", "protected"] + (
            ["baseline_targets"] if "baseline_targets" in contract else []
        ):
            spec = contract[section]
            observed = baseline[i] if section == "baseline_targets" else edited[i]
            values = [observed[output_index[n]] for n in spec["output_names"]]
            expected = (
                [baseline[i][output_index[n]] for n in spec["output_names"]]
                if section == "protected"
                else spec["expected"][sid]
            )
            errors = [abs(a - b) for a, b in zip(values, expected)]
            measurements[section] = dict(
                observed=values,
                reference=expected,
                absolute_error=errors,
                within_tolerance=[
                    finite(e) and e <= t
                    for e, t in zip(errors, spec["absolute_tolerance"])
                ],
            )
        rows.append(
            dict(sample_id=sid, split=samples[i].split, measurements=measurements)
        )
    violations = [
        dict(
            sample_id=r["sample_id"],
            measurement=section,
            output=contract[section]["output_names"][j],
        )
        for r in rows
        for section, m in r["measurements"].items()
        for j, ok in enumerate(m["within_tolerance"])
        if not ok
    ]
    return dict(
        schema="nmn.sampled-contract-evidence.v1",
        status="observed",
        assessment=(
            "observed-violations" if violations else "observed-within-tolerances"
        ),
        contract=contract,
        contract_sha256=hashlib.sha256(
            json.dumps(contract, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset=dataset.to_dict(),
        dataset_sha256=dataset.sha256,
        model_snapshot=execution,
        sample_ids=ids,
        rows=rows,
        violations=violations,
        coverage=dict(samples_checked=len(ids), domain="listed sample IDs only"),
        limitations=[
            "Sampled compliance is not exhaustive finite or continuous-domain certification.",
            "Supplied semantic references and split independence are not validated as scientific claims.",
            "Only input-independent shared module writes are supported; no read/edge or parameter edits.",
        ],
    )
