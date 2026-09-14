"""Build an explicit-state graph, intervene, and collect native research data."""

import argparse

import torch

from nmn.torch import Intervention, YatGraph, YatModuleSpec
from nmn.torch.research import collect_research_data, save_research_data

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", help="optional new JSON path")
args = parser.parse_args()
model = YatGraph(
    slots=["u", "v", "h", "p", "y"],
    input_names=["u", "v"],
    output_names=["y", "p"],
    layers=[
        [
            YatModuleSpec("hidden", ["u"], ["h"], num_centers=2),
            YatModuleSpec("protected", ["v"], ["p"]),
        ],
        [YatModuleSpec("target", ["h", "v"], ["y"])],
    ],
    dtype=torch.float64,
)
with torch.no_grad():
    for block in model.blocks.values():
        block.centers.fill_(1)
x = torch.tensor([[1.0, 1.0], [0.5, 1.0]], dtype=torch.float64)
y, trace = model.forward_with_trace(x)
edits = {"remove_hidden": {"hidden": Intervention(gate=0)}}
print("Outputs:", y.detach().tolist())
print("Remove hidden:", model(x, edits["remove_hidden"]).detach().tolist())
print("Dependencies:", model.dependencies())
print(
    "Layer states:",
    {k: v.detach().tolist() for k, v in trace.items() if k.startswith("state.")},
)
if args.output:
    data = collect_research_data(
        model,
        x,
        sample_ids=["a", "b"],
        edits=edits,
        metadata={"split": "diagnostic", "trained": False},
    )
    save_research_data(data, args.output)
    print("Saved", args.output)
