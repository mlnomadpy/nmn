"""Compare native joint gate effects with derivative/path predictions."""

import argparse

import torch

from nmn.torch import ThreeNeuronYat
from nmn.torch.paths import gate_path
from nmn.torch.research import _json_value, collect_research_data, save_research_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
model = ThreeNeuronYat.reference(dtype=torch.float64)
x = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
path = gate_path(model, x, [1, 1, 1], [0, 1, 1], steps=64)
result = {
    "schema": "nmn.gate-path-study.v1",
    "model_snapshot": collect_research_data(
        model, x, sample_ids=["one"], derivatives=False
    ),
    "path": _json_value(path),
}
save_research_data(result, args.output)
print("Actual effect:", path["actual_delta"].tolist())
for name, prediction in path["predictions"].items():
    print(name, prediction.tolist(), "residual:", path["residuals"][name].tolist())
print("Saved", args.output)
