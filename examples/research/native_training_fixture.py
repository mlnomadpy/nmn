"""Explicitly run a tiny CPU optimizer fixture, not a semantic research study."""

import argparse

import torch

from nmn.research.datasets import ResearchDataset, ResearchSample
from nmn.torch import ThreeNeuronYat
from nmn.torch.research import collect_research_data, save_research_data
from nmn.torch.training import TrainingConfig, train_native

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
model = ThreeNeuronYat.reference(dtype=torch.float64)
model.p.requires_grad_(False)
snapshot = collect_research_data(
    model,
    torch.ones(1, 2, dtype=torch.float64),
    sample_ids=["initial"],
    derivatives=False,
)
dataset = ResearchDataset(
    [
        ResearchSample("train-a", (1.0, 1.0), "train", "a"),
        ResearchSample("train-b", (0.5, 1.0), "train", "b"),
        ResearchSample("selection", (0.75, 1.0), "validation", "v"),
    ],
    name="bounded CPU optimizer fixture",
    provenance="synthetic implementation fixture",
)
contract = {
    "architecture_id": "ThreeNeuronYat; protected p parameters frozen",
    "semantic_specification": "supplied scalar coordinates; no semantic discovery",
    "intervention_specification": "native h gates/replacements are specified; this run is task-only",
    "guarantee_scope": "fixed routing isolates p from h; no learned recovery or risk guarantee",
    "worked_example": "all-ones reference: gate h=0 changes (4,1) to (.5,1)",
}
result = train_native(
    snapshot,
    dataset,
    {sid: [0.0, 1.0] for sid in dataset.sample_ids()},
    config=TrainingConfig(max_steps=2, evaluate_every=1, learning_rate=0.01),
    architecture_contract=contract,
    target_provenance="synthetic zero-target fixture",
    seeds=[0, 1],
)
save_research_data(result, args.output)
for run in result["runs"]:
    print(run["seed"], run["status"], "selected step:", run["best_step"])
