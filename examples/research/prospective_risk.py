"""Freeze a native edit family before fresh fixed-size validation sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from nmn.research.dashboard import build_dashboard
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.io import write_json_exclusive
from nmn.research.native_export import export_native_record
from nmn.research.risk import validate_risk
from nmn.torch.replay import replay_native_record
from nmn.torch.research import model_from_snapshot
from nmn.torch.sampled_contract import evaluate_sampled_contract


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def run(model_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(Path(model_path).read_text())
    model = model_from_snapshot(snapshot)
    if (
        tuple(model.input_names) != ("u", "v")
        or tuple(model.output_names) != ("target", "protected")
        or "h" not in model.state_names
    ):
        raise ValueError(
            "example requires u,v inputs, target/protected outputs and native h write"
        )
    n, seed = 256, 20260918
    common = dict(
        model_sha256=snapshot["model_sha256"],
        action_domain=dict(
            kind="shared-module-writes",
            gate_bounds=[0.0, 1.0],
            allow_replacements=False,
        ),
        reference_id="polynomial-removal-v1",
        reference_description="ordinary target=u^2+0.5v, protected=v; removal target=0.5v",
        criteria=dict(
            targets=dict(output_names=["target"], absolute_tolerance=[0.1]),
            protected=dict(output_names=["protected"], absolute_tolerance=[1e-12]),
            baseline_targets=dict(
                output_names=["target", "protected"], absolute_tolerance=[0.1, 0.1]
            ),
        ),
    )
    definitions = {
        name: dict(common, controls={"h": {"gate": gate}})
        for name, gate in [("remove-h", 0.0), ("retain-h", 1.0)]
    }
    plan = dict(
        schema="nmn.risk-plan.v1",
        delta="1/20",
        candidates={name: digest(value) for name, value in definitions.items()},
        clauses={
            name: dict(alpha="1/20", sample_count=n) for name in common["criteria"]
        },
        provenance="Fixed native edit family and criteria saved before sampling; no fitting or retuning",
        sampling_law="Declared IID uniform[-1,1]^2 sampling model, implemented with reproducible Torch pseudorandom draws",
        freeze_provenance="definitions.json and plan.json written before generating validation inputs",
        assumptions=dict(
            iid_within_clause=True,
            family_independent_of_validation=True,
            fixed_sample_counts=True,
        ),
    )
    protocol = dict(
        data_seed=seed,
        samples=n,
        plan_sha256=digest(plan),
        definitions_sha256=digest(definitions),
        sampling_scope="Probability statement is conditional on the declared IID sampling model; PRNG independence is not formally certified",
        stopping="Exactly 256 samples; no early stopping, candidate expansion or threshold retuning",
    )
    # All choices above are persisted before any new input or loss is generated.
    for name, value in [
        ("model", snapshot),
        ("definitions", definitions),
        ("plan", plan),
        ("protocol", protocol),
    ]:
        write_json_exclusive(value, destination / (name + ".json"), sort_keys=True)
    generator = torch.Generator().manual_seed(seed)
    values = (
        2 * torch.rand(n, 2, generator=generator, dtype=torch.float64) - 1
    ).tolist()
    dataset = ResearchDataset(
        [
            ResearchSample(f"validation-{i}", x, "validation", f"unit-{i}")
            for i, x in enumerate(values)
        ],
        name="Prospective fixed-family synthetic risk validation",
        provenance=f"Fresh seed {seed}; plan {digest(plan)} saved before sampling",
    )
    write_json_exclusive(
        dataset.to_dict(), destination / "dataset.json", sort_keys=True
    )
    ids = dataset.sample_ids()
    observations = dict(schema="nmn.risk-observations.v1", candidates={})
    sources = []
    bindings = {}
    for name, definition in definitions.items():
        contract = dict(
            schema="nmn.sampled-contract.v1",
            scope="sampled",
            arithmetic="floating-point",
            model_sha256=snapshot["model_sha256"],
            dataset_sha256=dataset.sha256,
            sample_ids=ids,
            action_domain=definition["action_domain"],
            controls=definition["controls"],
            provenance=f'Frozen candidate {plan["candidates"][name]}; plan {digest(plan)}',
            **{key: dict(value) for key, value in definition["criteria"].items()},
        )
        contract["targets"]["expected"] = {
            s: [0.5 * dataset.sample(s).inputs[1]] for s in ids
        }
        contract["baseline_targets"]["expected"] = {
            s: [
                dataset.sample(s).inputs[0] ** 2 + 0.5 * dataset.sample(s).inputs[1],
                dataset.sample(s).inputs[1],
            ]
            for s in ids
        }
        evidence = evaluate_sampled_contract(model, dataset, contract)
        replay = replay_native_record(evidence)
        for suffix, record in [("contract", evidence), ("replay", replay)]:
            path = destination / (name + "-" + suffix + ".json")
            write_json_exclusive(record, path, sort_keys=True)
            sources.append(path)
        if replay["status"] != "matched":
            raise RuntimeError("numerical replay mismatch; evidence retained")
        observations["candidates"][name] = dict(
            identity=plan["candidates"][name],
            clauses={
                clause: [
                    dict(
                        sample_id=row["sample_id"],
                        group_id=dataset.sample(row["sample_id"]).group_id,
                        failed=not all(row["measurements"][clause]["within_tolerance"]),
                    )
                    for row in evidence["rows"]
                ]
                for clause in plan["clauses"]
            },
        )
        bindings[name] = dict(
            frozen_definition_sha256=plan["candidates"][name],
            contract_sha256=evidence["contract_sha256"],
            evidence_sha256=digest(evidence),
        )
    write_json_exclusive(
        observations, destination / "risk-observations.json", sort_keys=True
    )
    write_json_exclusive(bindings, destination / "bindings.json", sort_keys=True)
    report = validate_risk(plan, observations)
    write_json_exclusive(report, destination / "risk.json", sort_keys=True)
    sources.append(destination / "risk.json")
    for path in sources:
        export_native_record(path, destination / "notes" / path.stem)
    build_dashboard(sources, destination / "dashboard")
    print(
        json.dumps(
            dict(
                status=report["status"],
                accepted=report["accepted_under_declared_assumptions"],
                failures={
                    name: {c: r["failures"] for c, r in row["clauses"].items()}
                    for name, row in report["results"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.model, args.output)
