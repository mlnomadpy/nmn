"""Execute donor transfer against an independently supplied arithmetic reference."""

import argparse

import torch

from nmn.research.datasets import DonorPair, ResearchDataset, ResearchSample
from nmn.torch import ThreeNeuronYat
from nmn.torch.research import save_research_data
from nmn.torch.studies import donor_study


def reference(base, donor, modules):
    # A specified arithmetic operation; no model outputs are read here.
    assert modules == ("h",)
    u = donor.semantics["u"]
    v = base.semantics["v"]
    h = u * u / ((u - 1) ** 2 + 1)
    return {"target": (h + v) ** 2 / ((h - 1) ** 2 + (v - 1) ** 2 + 1)}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
dataset = ResearchDataset(
    [
        ResearchSample(
            "base", (1.0, 1.0), "evaluation", "context-a", {"u": 1.0, "v": 1.0}
        ),
        ResearchSample(
            "donor", (0.5, 1.0), "evaluation", "context-b", {"u": 0.5, "v": 1.0}
        ),
    ],
    name="designed scalar transfer",
    provenance="explicit arithmetic labels; no learned semantics",
)
pairs = [
    DonorPair("transfer", "base", "donor", ("h",)),
    DonorPair("identity-control", "base", "base", ("h",)),
]
result = donor_study(
    ThreeNeuronYat.reference(dtype=torch.float64),
    dataset,
    pairs,
    protected_outputs=["protected"],
    reference=reference,
    reference_id="scalar-h-transfer-v1",
)
save_research_data(result, args.output)
for row in result["rows"]:
    print(
        row["pair"]["pair_id"],
        "reference error:",
        row["absolute_reference_error"],
        "protected change:",
        row["protected_delta"],
    )
print("Saved", args.output)
