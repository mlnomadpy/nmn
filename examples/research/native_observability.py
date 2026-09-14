"""Run a self-contained summary/probe workflow on a fixed arithmetic graph.

Usage: python examples/research/native_observability.py --output /tmp/nmn-observability
No downloads, optimizer training, or existing model files are needed.
"""

import argparse
import json
from pathlib import Path

import torch

from nmn.research.components import extract_native_component
from nmn.research.dashboard import build_dashboard, load_dashboard_sources
from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.native_export import export_native_record
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.probes import probe_study
from nmn.torch.reduction import fit_reduction_study
from nmn.torch.replay import replay_native_record
from nmn.torch.research import collect_research_data, save_research_data


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Shared writable residual slots, with nonlinear propagation through depth.
    model = YatGraph(
        ["x", "h", "y"],
        ["x"],
        ["y"],
        [[YatModuleSpec("a", ["x"], ["h"])], [YatModuleSpec("b", ["h"], ["y"])]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    values = [("tuning", [0.0, 0.25, 0.75, 1.0]), ("validation", [0.1, 0.3, 0.7, 0.9])]
    samples = [
        ResearchSample(f"{split}-{i}", [x], split, f"{split}-{i}", {})
        for split, points in values
        for i, x in enumerate(points)
    ]
    dataset = ResearchDataset(
        samples,
        name="Native observability arithmetic example",
        provenance="Supplied scalar inputs and threshold labels; no research benchmark claim",
    )
    labels = {
        sample.sample_id: "low" if sample.inputs[0] < 0.5 else "high"
        for sample in samples
    }
    inputs = torch.tensor([s.inputs for s in samples], dtype=torch.float64)
    snapshot = collect_research_data(
        model, inputs, sample_ids=[s.sample_id for s in samples], derivatives=False
    )
    save_research_data(snapshot, destination / "model.json")
    save_research_data(dataset.to_dict(), destination / "dataset.json")
    summary = fit_reduction_study(model, dataset, start_layer=1, rank=1, ridge=0.01)
    probe = probe_study(
        model,
        dataset,
        feature="a",
        labels=labels,
        classes=["low", "high"],
        provenance="Supplied x < 0.5 labels; feature and ridge fixed in this example",
        ridge=0.01,
        refit_edits=True,
        edits={
            "negate": {"a": Intervention(gate=-1)},
            "remove": {"a": Intervention(gate=0)},
        },
    )
    records = {
        "summary": summary,
        "summary-evaluation": summary["evaluation"],
        "probe": probe,
    }
    for name in ("summary-evaluation", "probe"):
        replay = replay_native_record(records[name])
        records[name + "-replay"] = replay
        if replay["status"] != "matched":
            save_research_data(replay, destination / (name + "-replay.json"))
            raise RuntimeError("numerical replay mismatch; partial evidence retained")
    for name, record in records.items():
        path = destination / (name + ".json")
        save_research_data(record, path)
        export_native_record(path, destination / "notes" / name)
    for component in ("model", "dataset", "maps", "evaluation"):
        extract_native_component(
            destination / "summary.json",
            component,
            destination / "components" / component,
        )
    recipe = destination / "sources.json"
    recipe.write_text(
        json.dumps(
            {
                "schema": "nmn.evidence-sources.v1",
                "sources": [name + ".json" for name in records],
            },
            indent=2,
        )
        + "\n"
    )
    report = build_dashboard(load_dashboard_sources(recipe), destination / "dashboard")
    if report["unavailable"]:
        raise RuntimeError(
            "dashboard has unavailable records; inspect retained evidence"
        )
    return {
        "output": str(destination),
        "dashboard": str(destination / "dashboard/index.html"),
        "records": report["records"],
        "replay": "matched",
        "interpretation": "Arithmetic workflow demonstration; no semantic or erasure guarantee",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
