"""Finite arithmetic fixture: correct predictions do not validate slot labels."""

import argparse
from pathlib import Path

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.research.semantics import TabulatedReference
from nmn.torch import YatGraph, YatModuleSpec
from nmn.torch.research import save_research_data
from nmn.torch.semantics import semantic_study


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = YatGraph(
        ["u", "v", "target", "protected"],
        ["u", "v"],
        ["target", "protected"],
        [
            [
                YatModuleSpec("a", ["u"], ["target"]),
                YatModuleSpec("b", ["v"], ["protected"]),
            ]
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in model.blocks.values():
            block.centers.fill_(1)
            block.coefficients.fill_(1)
    dataset = ResearchDataset(
        [
            ResearchSample("base", (1.0, 0.0), "evaluation", "a"),
            ResearchSample("donor", (0.0, 1.0), "evaluation", "b"),
        ],
        name="Two-bit copy correspondence fixture",
        provenance="Supplied Boolean-valued arithmetic examples, no trained semantics",
    )
    reference = TabulatedReference(
        {
            "schema": "nmn.semantic-reference.v1",
            "reference_id": "two-bit-copy-table-v1",
            "provenance": "Copy u to target and v to protected; transfer only u from donor",
            "variables": ["u", "v"],
            "baseline": {
                "base": {"target": 1.0, "protected": 0.0},
                "donor": {"target": 0.0, "protected": 1.0},
            },
            "cases": [
                {
                    "case_id": "transfer-u",
                    "base_id": "base",
                    "donor_id": "donor",
                    "variables": ["u"],
                    "outputs": {"target": 0.0, "protected": 0.0},
                }
            ],
        }
    )
    correspondence = dict(
        origin="supplied",
        provenance="Manually assigned arithmetic routing",
        anchors=[],
        ambiguities=["Finite contexts cannot identify a unique mechanism"],
        mapping={"u": ["a"], "v": ["b"]},
    )
    for name, mapping in [
        ("correct", {"u": ["a"], "v": ["b"]}),
        ("swapped", {"u": ["b"], "v": ["a"]}),
    ]:
        record = semantic_study(
            model,
            dataset,
            reference=reference,
            correspondence={**correspondence, "mapping": mapping},
        )
        save_research_data(record, output / (name + ".json"))
        print(
            name,
            record["status"],
            "baseline agrees:",
            all(row["agrees"] for row in record["baseline"]),
        )
    save_research_data(record["model_snapshot"], output / "model.json")
    save_research_data(dataset.to_dict(), output / "dataset.json")
    save_research_data(reference.to_dict(), output / "reference.json")
    save_research_data(correspondence, output / "correspondence.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
