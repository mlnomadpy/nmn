"""Write a tiny receiving-slot donor study; no fitting or downloaded data."""

import argparse

import torch

from nmn.research.datasets import DonorPair, ResearchDataset, ResearchSample
from nmn.torch import YatGraph, YatModuleSpec
from nmn.torch.research import save_research_data
from nmn.torch.studies import donor_study

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
model = YatGraph(
    ["x", "h", "y", "p"],
    ["x"],
    ["y", "p"],
    [
        [YatModuleSpec("a", ["x"], ["h"])],
        [YatModuleSpec("b", ["h"], ["y"]), YatModuleSpec("c", ["h"], ["p"])],
    ],
    dtype=torch.float64,
)
with torch.no_grad():
    for block in model.blocks.values():
        block.centers.fill_(1)
        block.coefficients.fill_(1)
dataset = ResearchDataset(
    [
        ResearchSample("base", (1.0,), "evaluation", "base"),
        ResearchSample("donor", (0.0,), "evaluation", "donor"),
    ],
    name="Receiving-slot donor arithmetic fixture",
    provenance="Supplied arithmetic inputs and expected outputs; not discovered semantics",
)
record = donor_study(
    model,
    dataset,
    [
        DonorPair("transfer", "base", "donor", ("b",), {"y": 0.0}),
        DonorPair("self", "base", "base", ("b",), {"y": 1.0}),
    ],
    read_slots={"b": ["h"]},
    protected_outputs=["p"],
)
save_research_data(record, args.output)
print(
    {
        "output": args.output,
        "edited_outputs": [r["edited_outputs"] for r in record["rows"]],
    }
)
