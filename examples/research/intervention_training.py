"""Compare task-only and detached-donor supervision on a fresh hybrid task."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from nmn.research.dashboard import build_dashboard
from nmn.research.datasets import DonorPair, ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.replay import replay_native_record
from nmn.torch.research import (
    collect_research_data,
    model_from_snapshot,
    save_research_data,
)
from nmn.torch.studies import donor_study
from nmn.torch.training import TrainingConfig, train_native


def run(destination, *, gradient_comparison=False, fixed_edit_comparison=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data_seed = (
        20260919
        if fixed_edit_comparison
        else (20260916 if gradient_comparison else 20260915)
    )
    generator = torch.Generator().manual_seed(data_seed)
    samples, targets = [], {}
    for split, count in [("train", 128), ("validation", 32), ("evaluation", 128)]:
        for i, (u, v) in enumerate(
            (
                2 * torch.rand(count, 2, generator=generator, dtype=torch.float64) - 1
            ).tolist()
        ):
            sid = f"{split}-{i}"
            samples.append(ResearchSample(sid, [u, v], split, sid))
            targets[sid] = [u * u + 0.5 * v, v]
    dataset = ResearchDataset(
        samples,
        name=(
            "Fixed edit supervision comparison"
            if fixed_edit_comparison
            else (
                "Donor gradient comparison"
                if gradient_comparison
                else "Detached donor supervision ablation"
            )
        ),
        provenance=f"Fresh analytic polynomial task, deterministic seed {data_seed}",
    )

    def pairs(split):
        ids = dataset.sample_ids(split=split)
        return [
            DonorPair(
                f"{split}-pair-{i}",
                base,
                ids[(i + 17) % len(ids)],
                ("h",),
                {
                    "target": dataset.sample(ids[(i + 17) % len(ids)]).inputs[0] ** 2
                    + 0.5 * dataset.sample(base).inputs[1]
                },
            )
            for i, base in enumerate(ids)
        ]

    training_pairs, evaluation_pairs = pairs("train"), pairs("evaluation")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = YatGraph(
            ["u", "v", "h", "target", "protected"],
            ["u", "v"],
            ["target", "protected"],
            [
                [
                    YatModuleSpec("h", ["u"], ["h"], num_centers=4),
                    YatModuleSpec(
                        "p", ["v"], ["protected"], num_centers=4, family="linear"
                    ),
                ],
                [YatModuleSpec("y", ["h", "v"], ["target"], num_centers=4)],
            ],
            dtype=torch.float64,
        )
    init_rng = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.copy_(
                0.5
                * torch.randn(
                    4, block.in_features, generator=init_rng, dtype=torch.float64
                )
            )
            block.coefficients.fill_(0.25)
    initial = collect_research_data(
        model,
        torch.zeros(1, 2, dtype=torch.float64),
        sample_ids=["definition"],
        derivatives=False,
    )
    contract = {
        "architecture_id": "hybrid-h-y-yat-p-linear-four-factors",
        "semantic_specification": "target=u^2+0.5v; protected=v",
        "intervention_specification": "replace base h with donor h; target=donor_u^2+0.5base_v",
        "guarantee_scope": "empirical task and supplied donor correspondence; no causal or population certificate",
        "worked_example": "base=(1,1), donor=(0,-1): expected transferred target=0.5",
    }
    conditions = (
        [("detached-donor", 1.0, True), ("joint-donor", 1.0, False)]
        if gradient_comparison
        else [("task-only", 0.0, True), ("task-plus-donor", 1.0, True)]
    )
    configs = {
        name: TrainingConfig(
            max_steps=300,
            batch_size=16,
            learning_rate=0.01,
            evaluate_every=20,
            max_seconds=60.0,
            intervention_weight=weight,
            separate_pair_rng=True,
            detach_donor=detach,
        )
        for name, weight, detach in conditions
    }
    fixed_objective = None
    if fixed_edit_comparison:
        fixed_objective = {
            "schema": "nmn.fixed-edit-objective.v1",
            "controls": {"h": {"gate": 0.0}},
            "output_names": ["target"],
            "protected_outputs": ["protected"],
            "targets": {
                s.sample_id: [0.5 * s.inputs[1]]
                for s in samples
                if s.split != "evaluation"
            },
            "provenance": "Analytic h deletion target=0.5v; evaluation labels excluded",
        }
        contract["intervention_specification"] = (
            "Set h gate to zero; edited target=0.5v; retain protected output"
        )
        configs = {
            name: TrainingConfig(
                max_steps=300,
                batch_size=16,
                learning_rate=0.01,
                evaluate_every=20,
                max_seconds=60.0,
                separate_pair_rng=True,
                fixed_edit_weight=weight,
                protection_weight=weight,
                checkpoint_objective="task-plus-fixed-edit" if weight else "task",
            )
            for name, weight in [("task-only", 0.0), ("task-plus-fixed-edit", 1.0)]
        }
    save_research_data(
        {
            "configs": {k: asdict(v) for k, v in configs.items()},
            "contract": contract,
            "data_seed": data_seed,
            "comparison": (
                "fixed-edit-supervision"
                if fixed_edit_comparison
                else "donor-gradient" if gradient_comparison else "donor-supervision"
            ),
            "initialization_seed": 17,
            "seeds": [0, 1, 2],
            "selection": "Per-config checkpoint objective on validation only; evaluation excluded. Fixed-edit comparison changes both training loss and selection objective.",
            "fixed_objective": fixed_objective,
        },
        destination / "protocol.json",
    )
    save_research_data(dataset.to_dict(), destination / "dataset.json")
    rows, sources = [], []
    for condition, config in configs.items():
        print("Training " + condition, flush=True)
        training = train_native(
            initial,
            dataset,
            {
                s.sample_id: targets[s.sample_id]
                for s in samples
                if s.split != "evaluation"
            },
            config=config,
            architecture_contract=contract,
            target_provenance="Analytic task and declared donor correspondence",
            seeds=(0, 1, 2),
            pairs=training_pairs if config.intervention_weight else (),
            fixed_edit=fixed_objective if config.fixed_edit_weight else None,
        )
        path = destination / (condition + "-training.json")
        save_research_data(training, path)
        sources.append(path)
        for result in training["runs"]:
            label = f'{condition}-seed-{result["seed"]}'
            fitted = model_from_snapshot(result["selected_checkpoint"])
            ids = dataset.sample_ids(split="evaluation")
            x = torch.tensor(
                [dataset.sample(s).inputs for s in ids], dtype=torch.float64
            )
            observations = collect_research_data(
                fitted,
                x,
                sample_ids=ids,
                derivatives=False,
                edits={"remove-h": {"h": Intervention(gate=0.0)}},
            )
            output = torch.tensor(
                observations["observations"]["baseline"], dtype=torch.float64
            )
            expected = torch.tensor([targets[s] for s in ids], dtype=torch.float64)
            study = donor_study(
                fitted, dataset, evaluation_pairs, protected_outputs=["protected"]
            )
            transferred = torch.tensor(
                [r["edited_outputs"][0] for r in study["rows"]], dtype=torch.float64
            )
            reference = torch.tensor(
                [p.expected["target"] for p in evaluation_pairs], dtype=torch.float64
            )
            with torch.no_grad():
                removed = fitted(x, {"h": Intervention(gate=0.0)})
            rows.append(
                {
                    "condition": condition,
                    "seed": result["seed"],
                    "training_status": result["status"],
                    "steps": result["steps_completed"],
                    "selected_step": result["best_step"],
                    "task_mse": ((output - expected) ** 2).mean(0).tolist(),
                    "removed_h_target_mse": float(
                        ((removed[:, 0] - 0.5 * x[:, 1]) ** 2).mean()
                    ),
                    "removed_h_protected_change_mse": float(
                        ((removed[:, 1] - output[:, 1]) ** 2).mean()
                    ),
                    "donor_target_mse": float(((transferred - reference) ** 2).mean()),
                }
            )
            for suffix, record in [("observations", observations), ("donors", study)]:
                path = destination / (label + "-" + suffix + ".json")
                save_research_data(record, path)
                sources.append(path)
            replay = replay_native_record(study)
            path = destination / (label + "-donor-replay.json")
            save_research_data(replay, path)
            sources.append(path)
            if replay["status"] != "matched":
                raise RuntimeError("donor replay mismatch; records retained")
    save_research_data({"rows": rows}, destination / "summary.json")
    for path in sources:
        export_native_record(path, destination / "notes" / path.stem)
    build_dashboard(sources, destination / "dashboard")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--gradient-comparison", action="store_true")
    modes.add_argument("--fixed-edit-comparison", action="store_true")
    args = parser.parse_args()
    run(
        args.output,
        gradient_comparison=args.gradient_comparison,
        fixed_edit_comparison=args.fixed_edit_comparison,
    )
