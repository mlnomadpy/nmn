"""Run the actual NMN network: python examples/research/native_three_neuron.py.

Requires nmn[torch]. This demonstrates forward passes and backpropagation;
no training run or dataset download is performed.
"""

import torch

from nmn.torch import Intervention, ThreeNeuronYat

model = ThreeNeuronYat.reference(dtype=torch.float64)
x = torch.tensor([[1.0, 1.0], [0.5, 1.0]], dtype=torch.float64)
normal, trace = model.forward_with_trace(x)
gated = model(x, {"h": Intervention(gate=0.0)})
restored = model(x, {"h": Intervention(replacement=trace["h"])})

print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
print("Outputs (target, protected):", normal.detach().tolist())
print("Gate h=0:", gated.detach().tolist())
print("Restore h:", restored.detach().tolist())
print("Target center contributions:", trace["y.contributions"].detach().tolist())

# A tensor control remains differentiable, including downstream recomputation.
gate = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
controlled = model(x, {"h": Intervention(gate=gate)})
controlled[:, 0].sum().backward()
print("Target sensitivity to gate:", gate.grad.item())
print(
    "Parameter gradients:",
    {
        name: None if p.grad is None else p.grad.detach().tolist()
        for name, p in model.named_parameters()
    },
)
