"""Bounded synthetic task fitting with an untouched final evaluation population.

Run: python examples/research/matched_training.py OUTPUT
Nine CPU fits, each limited to 300 updates and 20 optimization seconds.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from nmn.research.dashboard import build_dashboard
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.benchmark import benchmark_models
from nmn.torch.replay import replay_native_record
from nmn.torch.research import (
    collect_research_data,
    model_from_snapshot,
    save_research_data,
)
from nmn.torch.training import TrainingConfig, train_native


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(20260913)
    samples, labels = [], {}
    for split, count in [("train", 128), ("validation", 32), ("evaluation", 128)]:
        for i, (u, v) in enumerate(
            (
                2 * torch.rand(count, 2, generator=generator, dtype=torch.float64) - 1
            ).tolist()
        ):
            sid = f"{split}-{i}"
            samples.append(ResearchSample(sid, [u, v], split, sid))
            labels[sid] = [u * u + 0.5 * v, v]
    dataset = ResearchDataset(
        samples,
        name="Matched polynomial task training",
        provenance="Deterministic synthetic uniform points; y=u^2+0.5v, protected=v",
    )
    config = TrainingConfig(
        max_steps=300,
        batch_size=32,
        learning_rate=0.01,
        evaluate_every=20,
        max_seconds=20.0,
    )
    contract = {
        "architecture_id": "shared-routing-four-units-per-module",
        "semantic_specification": "inputs u,v in [-1,1]; outputs target=u^2+0.5v and protected=v",
        "intervention_specification": "delete h module output; intended counterfactual u=0 is evaluated separately",
        "guarantee_scope": "empirical fitting and finite native effects only; no statistical or semantic certification",
        "worked_example": "u=1,v=1 gives target=1.5, protected=1; intended u=0 target=0.5",
    }
    protocol = {
        "dataset_seed": 20260913,
        "initialization_seed": 17,
        "minibatch_seeds": [0, 1, 2],
        "config": asdict(config),
        "contract": contract,
        "families": ["yat", "imq", "tanh"],
        "initialization": "same center/hidden-weight arrays and 1/4 readout coefficients; tanh biases zero",
        "caveats": [
            "Equal width/routing is not equal capacity: tanh has extra biases.",
            "Seeds vary minibatches, not initialization or data.",
            "Native h deletion need not realize the intended semantic u=0 counterfactual.",
            "Protected path independence is imposed for every family.",
        ],
    }
    save_research_data(protocol, destination / "protocol.json")
    save_research_data(dataset.to_dict(), destination / "dataset.json")
    save_research_data(labels, destination / "labels.json")
    fitting_targets = {
        s.sample_id: labels[s.sample_id] for s in samples if s.split != "evaluation"
    }
    models, rows, sources = {}, [], []
    for family in protocol["families"]:
        print("Fitting " + family, flush=True)
        specs = [
            [
                YatModuleSpec("h", ["u"], ["h"], num_centers=4, family=family),
                YatModuleSpec("p", ["v"], ["protected"], num_centers=4, family=family),
            ],
            [YatModuleSpec("y", ["h", "v"], ["target"], num_centers=4, family=family)],
        ]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = YatGraph(
                ["u", "v", "h", "target", "protected"],
                ["u", "v"],
                ["target", "protected"],
                specs,
                dtype=torch.float64,
            )
        init_generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for block in model.blocks.values():
                weights = (
                    torch.randn(
                        4,
                        block.in_features,
                        generator=init_generator,
                        dtype=torch.float64,
                    )
                    * 0.5
                )
                if family == "tanh":
                    block.hidden.weight.copy_(weights)
                    block.hidden.bias.zero_()
                    block.readout.weight.fill_(0.25)
                else:
                    block.centers.copy_(weights)
                    block.coefficients.fill_(0.25)
        initial = collect_research_data(
            model,
            torch.zeros(1, 2, dtype=torch.float64),
            sample_ids=["definition"],
            derivatives=False,
        )
        training = train_native(
            initial,
            dataset,
            fitting_targets,
            config=config,
            architecture_contract=contract,
            target_provenance="Fixed analytic polynomial task; no teacher model",
            seeds=(0, 1, 2),
        )
        path = destination / (family + "-training.json")
        save_research_data(training, path)
        sources.append(path)
        for result in training["runs"]:
            name = f'{family}-seed-{result["seed"]}'
            rows.append(
                {
                    "method": name,
                    "training_status": result["status"],
                    "steps_completed": result["steps_completed"],
                    "selected_step": result["best_step"],
                }
            )
            if result["selected_checkpoint"] is not None:
                models[name] = model_from_snapshot(result["selected_checkpoint"])
    evaluation_ids = dataset.sample_ids(split="evaluation")
    expected = [labels[sid] for sid in evaluation_ids]
    benchmark = benchmark_models(
        models,
        dataset,
        edits={"delete-h": {"h": Intervention(gate=0)}},
        expected_outputs=expected,
        protected_outputs=["protected"],
        split="evaluation",
        repeats=3,
        warmup=1,
    )
    for row in rows:
        measured = benchmark["methods"].get(row["method"], {})
        row["evaluation_status"] = measured.get("status", "unavailable")
        if measured.get("status") == "measured":
            outputs = torch.tensor(measured["baseline_outputs"], dtype=torch.float64)
            edited = torch.tensor(
                measured["edits"]["delete-h"]["outputs"], dtype=torch.float64
            )
            target = torch.tensor(expected, dtype=torch.float64)
            counterfactual = torch.tensor(
                [
                    [0.5 * dataset.sample(sid).inputs[1], dataset.sample(sid).inputs[1]]
                    for sid in evaluation_ids
                ],
                dtype=torch.float64,
            )
            row.update(
                parameter_count=measured["parameter_count"],
                per_output_mse=((outputs - target) ** 2).mean(0).tolist(),
                deletion_counterfactual_mse=((edited - counterfactual) ** 2)
                .mean(0)
                .tolist(),
                max_protected_change=float((edited[:, 1] - outputs[:, 1]).abs().max()),
            )
    save_research_data(
        {"protocol": protocol, "rows": rows}, destination / "summary.json"
    )
    for name, record in [
        ("evaluation", benchmark),
        ("evaluation-replay", replay_native_record(benchmark)),
    ]:
        path = destination / (name + ".json")
        save_research_data(record, path)
        sources.append(path)
        if name == "evaluation-replay" and record["status"] != "matched":
            raise RuntimeError("evaluation replay mismatch; evidence retained")
    for path in sources:
        export_native_record(path, destination / "notes" / path.stem)
    build_dashboard(sources, destination / "dashboard")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.output)
