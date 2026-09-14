"""Fit a native module correspondence on fresh data for a saved hybrid model."""

import argparse
import json
from pathlib import Path

import torch

from nmn.research.dashboard import build_dashboard
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record
from nmn.research.semantics import TabulatedReference
from nmn.torch.alignment import alignment_study
from nmn.torch.replay import replay_native_record
from nmn.torch.research import model_from_snapshot, save_research_data


def run(model_path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = json.loads(Path(model_path).read_text())
    model = model_from_snapshot(snapshot)
    generator = torch.Generator().manual_seed(20260917)
    samples, baseline, cases = [], {}, []
    for split, count in [("tuning", 64), ("validation", 128)]:
        values = (
            2 * torch.rand(count, 2, generator=generator, dtype=torch.float64) - 1
        ).tolist()
        for i, (u, v) in enumerate(values):
            sid = f"{split}-{i}"
            samples.append(ResearchSample(sid, [u, v], split, sid))
            baseline[sid] = dict(target=u * u + 0.5 * v, protected=v)
            cases.append(
                dict(
                    case_id=f"{split}-pair-{i}",
                    base_id=sid,
                    donor_id=f"{split}-{(i+17)%count}",
                    variables=["square"],
                    outputs=dict(
                        target=values[(i + 17) % count][0] ** 2 + 0.5 * v, protected=v
                    ),
                )
            )
    dataset = ResearchDataset(
        samples,
        name="Fresh supervised module alignment",
        provenance="Analytic polynomial task, generated seed 20260917",
    )
    reference = TabulatedReference(
        dict(
            schema="nmn.semantic-reference.v1",
            reference_id="polynomial-square-transfer-v1",
            provenance="Supplied u squared semantics and untouched base v",
            variables=["square"],
            baseline=baseline,
            cases=cases,
        )
    )
    protocol = dict(
        data_seed=20260917,
        module_pool=["p", "y", "h"],
        max_candidates=3,
        provenance="Supervised whole-module assignment on a frozen hybrid checkpoint",
    )
    for name, value in [
        ("model", snapshot),
        ("dataset", dataset.to_dict()),
        ("reference", reference.to_dict()),
        ("protocol", protocol),
    ]:
        save_research_data(value, destination / (name + ".json"))
    study = alignment_study(
        model,
        dataset,
        reference=reference,
        **{k: v for k, v in protocol.items() if k != "data_seed"},
    )
    save_research_data(study, destination / "alignment.json")
    replay = replay_native_record(study)
    save_research_data(replay, destination / "replay.json")
    for name in ("alignment", "replay"):
        export_native_record(
            destination / (name + ".json"), destination / "notes" / name
        )
    build_dashboard(
        [destination / "alignment.json", destination / "replay.json"],
        destination / "dashboard",
    )
    print(
        json.dumps(
            {
                k: study[k]
                for k in ("status", "coverage", "candidates", "selected", "exact_ties")
            },
            indent=2,
        )
    )
    print("Held-out MSE:", study["evaluation"]["mse"], "Replay:", replay["status"])
    if replay["status"] != "matched":
        raise RuntimeError("alignment replay mismatch; records retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.model, args.output)
