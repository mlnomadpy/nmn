"""Compare untrained ⵟ, exponent-1 IMQ, and tanh blocks under identical routing."""

import argparse

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.benchmark import benchmark_models
from nmn.torch.research import save_research_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
models = {}
for family in ("yat", "imq", "tanh"):
    torch.manual_seed(0)
    models[family] = YatGraph(
        ["u", "v", "h", "p", "y"],
        ["u", "v"],
        ["y", "p"],
        [
            [
                YatModuleSpec("hidden", ["u"], ["h"], family=family),
                YatModuleSpec("protected", ["v"], ["p"], family=family),
            ],
            [YatModuleSpec("target", ["h", "v"], ["y"], family=family)],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        for block in models[family].blocks.values():
            if family == "tanh":
                block.hidden.weight.fill_(1)
                block.hidden.bias.zero_()
                block.readout.weight.fill_(1)
            else:
                block.centers.fill_(1)
                block.coefficients.fill_(1)

data = ResearchDataset(
    [
        ResearchSample("a", (1.0, 1.0), "diagnostic", "a"),
        ResearchSample("b", (0.5, 1.0), "diagnostic", "b"),
    ],
    name="untrained matched-routing comparison",
    provenance="designed inputs; parameters supplied, no learning",
)
result = benchmark_models(
    models,
    data,
    edits={"remove_hidden": {"hidden": Intervention(gate=0)}},
    protected_outputs=["p"],
    repeats=3,
    warmup=1,
)
save_research_data(result, args.output)
for name, record in result["methods"].items():
    print(
        name,
        record["status"],
        "parameters:",
        record.get("parameter_count"),
        "outputs:",
        record.get("baseline_outputs"),
    )
print("Untrained diagnostic only; no kernel advantage is established.")
