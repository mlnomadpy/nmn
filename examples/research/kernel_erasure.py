"""Fit finite-bank erasure targets and measure a bounded native realization."""

import argparse
import json
from pathlib import Path

import torch

from nmn.research.dashboard import build_dashboard
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record
from nmn.torch.erasure import erasure_study
from nmn.torch.preimage import execute_preimage_study, preimage_study
from nmn.torch.replay import replay_native_record
from nmn.torch.research import model_from_snapshot, save_research_data


def run(checkpoint, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    model = model_from_snapshot(json.loads(Path(checkpoint).read_text()))
    protocol = dict(
        data_seed=20260920,
        fit_count=64,
        evaluation_count=64,
        label="u squared",
        module="y",
        rtol=1e-10,
        search_steps=100,
        search_seconds=30.0,
        learning_rate=0.05,
        bounds="baseline receiver h plus/minus 1; receiver v fixed",
        scope="fit-only finite-bank projection, followed by per-example evaluation preimage optimization; no learned native mapper",
    )
    save_research_data(protocol, destination / "protocol.json")
    rng = torch.Generator().manual_seed(protocol["data_seed"])
    samples, labels = [], {}
    for split in ["fit", "evaluation"]:
        for i, uv in enumerate(
            (2 * torch.rand(64, 2, generator=rng, dtype=torch.float64) - 1).tolist()
        ):
            sid = f"{split}-{i}"
            samples.append(ResearchSample(sid, uv, split, sid))
            labels[sid] = [uv[0] ** 2]
    dataset = ResearchDataset(
        samples,
        name="Finite-bank erasure study",
        provenance="Fresh analytic inputs; seed20260920",
    )
    save_research_data(dataset.to_dict(), destination / "dataset.json")
    save_research_data(labels, destination / "labels.json")
    projection = erasure_study(
        model,
        dataset,
        module_name="y",
        labels=labels,
        provenance="Analytic u squared label",
        rtol=protocol["rtol"],
        fit_split="fit",
        evaluation_split="evaluation",
    )
    # Save the frozen projection before optimizing any evaluation native inputs.
    save_research_data(projection, destination / "projection.json")
    points = torch.tensor(
        projection["evaluation_snapshot"]["observations"]["baseline_trace"]["y.input"],
        dtype=torch.float64,
    )
    lower, upper = points.clone(), points.clone()
    lower[:, 0] -= 1
    upper[:, 0] += 1
    proposal = preimage_study(
        model,
        dataset,
        module_name="y",
        targets=projection["targets"],
        lower=lower,
        upper=upper,
        provenance="Frozen fit-only covariance projection targets",
        max_steps=protocol["search_steps"],
        max_seconds=protocol["search_seconds"],
        learning_rate=protocol["learning_rate"],
        split="evaluation",
    )
    execution = execute_preimage_study(proposal)
    replay = replay_native_record(execution)
    records = {
        "projection": projection,
        "proposal": proposal,
        "execution": execution,
        "replay": replay,
    }
    for name, record in records.items():
        path = destination / (name + ".json")
        if name != "projection":
            save_research_data(record, path)
        export_native_record(path, destination / "notes" / name)
    build_dashboard(
        [destination / (name + ".json") for name in records], destination / "dashboard"
    )
    actual = torch.tensor(execution["executed_features"], dtype=torch.float64)
    y = torch.tensor(projection["evaluation"]["labels"], dtype=torch.float64)
    covariance = (actual - actual.mean(0)).T @ (y - y.mean(0)) / len(y)
    outputs = torch.tensor(execution["outputs"], dtype=torch.float64)
    delta = torch.tensor(execution["output_delta"], dtype=torch.float64)
    ids = dataset.sample_ids(split="evaluation")
    ideal = torch.tensor(
        [0.5 * dataset.sample(s).inputs[1] for s in ids], dtype=torch.float64
    )
    summary = dict(
        fit_covariance_after=projection["fit"]["covariance_after_norm"],
        evaluation_covariance_before=projection["evaluation"]["covariance_before_norm"],
        projected_covariance_after=projection["evaluation"]["covariance_after_norm"],
        executed_covariance_after=torch.linalg.vector_norm(covariance).item(),
        feature_realization_mse=torch.tensor(
            execution["feature_residuals"], dtype=torch.float64
        )
        .square()
        .mean()
        .item(),
        target_removal_mse=(outputs[:, 0] - ideal).square().mean().item(),
        protected_change_mse=delta[:, 1].square().mean().item(),
        replay_status=replay["status"],
    )
    save_research_data(summary, destination / "summary.json")
    print(json.dumps(summary, indent=2))
    if replay["status"] != "matched":
        raise RuntimeError("Execution replay mismatch; all records retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
