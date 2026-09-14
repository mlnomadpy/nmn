"""Collect native model observations for an Obsidian research note.

Usage: python examples/research/collect_native_data.py --output measurements.json
"""

import argparse

import torch

from nmn.torch import Intervention, ThreeNeuronYat
from nmn.torch.research import collect_research_data, save_research_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
model = ThreeNeuronYat.reference(dtype=torch.float64)
grid = torch.linspace(0, 1, 5, dtype=torch.float64)
inputs = torch.cartesian_prod(grid, grid)
data = collect_research_data(
    model,
    inputs,
    sample_ids=[f"grid-{i:02}" for i in range(len(inputs))],
    edits={
        "remove_h": {"h": Intervention(gate=0)},
        "half_h": {"h": Intervention(gate=0.5)},
        "joint_h_y": {"h": Intervention(gate=0), "y": Intervention(gate=0)},
    },
    metadata={
        "task": "designed three-neuron reference",
        "population": "25-point Cartesian grid [0,.25,.5,.75,1]^2",
        "split": "diagnostic-only; no learning or candidate selection",
        "semantic_status": "roles supplied by architecture; no learned semantics",
        "vault_notes": [
            "Intervention research - Modular architectures",
            "Kernel theory - explainability gap map",
        ],
    },
)
save_research_data(data, args.output)
print(f"Saved native observations for {len(inputs)} inputs to {args.output}")
print("Model identity:", data["model_sha256"])
print("Contains: parameters, traces, geometry, RKHS norms, input Jacobians,")
print("gate Jacobians/Hessians, per-example intervention effects and provenance.")
