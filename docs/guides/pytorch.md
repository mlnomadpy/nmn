# PyTorch Guide

End-to-end walkthrough for using NMN with **PyTorch**.

- **Module path**: `nmn.torch`
- **Minimum versions**: Python 3.10+, PyTorch ≥ 1.11

---

## 1. Install

```bash
pip install "nmn[torch]"
```

Verify:

```python
import torch
import nmn
from nmn.torch import YatNMN

print(nmn.__version__, torch.__version__)
layer = YatNMN(in_features=128, out_features=64)
print(layer(torch.randn(32, 128)).shape)  # torch.Size([32, 64])
```

If you need CUDA, install a CUDA build of PyTorch *first* from [pytorch.org](https://pytorch.org/get-started/locally/), then `pip install "nmn[torch]"`.

---

## 2. Hello world — replace one `Linear+ReLU`

```python
import torch
import torch.nn as nn
from nmn.torch import YatNMN

# Before:
# fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

# After:
fc = YatNMN(in_features=128, out_features=64)

x = torch.randn(32, 128)
y = fc(x)
print(y.shape, y.min().item(), y.max().item())
# torch.Size([32, 64]) — outputs are non-negative (Yat is a squared ratio)
```

The non-linearity is baked into the layer. No `ReLU` needed.

---

## 3. MNIST end-to-end

A minimal trainable model. Drop into a Python file and run.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nmn.torch import YatNMN

device = "cuda" if torch.cuda.is_available() else "cpu"

class YatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            YatNMN(in_features=28 * 28, out_features=256),
            YatNMN(in_features=256,     out_features=128),
            nn.Linear(128, 10),  # keep final logits linear
        )

    def forward(self, x):
        return self.net(x)

train_ds = datasets.MNIST(".data", train=True, download=True,
                          transform=transforms.ToTensor())
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

model = YatMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(3):
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch}: loss={loss.item():.4f}")
```

Expect ~97–98% test accuracy in 3 epochs.

---

## 4. CNN on CIFAR — replacing `Conv2d+ReLU`

```python
from nmn.torch import YatConv2D, YatNMN
import torch.nn as nn

class YatCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            YatConv2D(in_channels=3,  out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            YatConv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            YatConv2D(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            YatNMN(in_features=128, out_features=64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
```

For a fully-worked ResNet-style training script see
[`src/nmn/torch/examples/vision/resnet_training.py`](../../src/nmn/torch/examples/vision/resnet_training.py).

---

## 5. Attention block

```python
from nmn.torch import MultiHeadYatAttention

attn = MultiHeadYatAttention(embed_dim=512, num_heads=8).to(device)

x = torch.randn(4, 16, 512, device=device)  # (batch, seq, dim)
y = attn(x, x, x)
print(y.shape)  # torch.Size([4, 16, 512])
```

Use this as a drop-in for `torch.nn.MultiheadAttention` in a transformer block. Pair with a `YatNMN` MLP for a fully Yat block.

---

## 6. Saving & loading

Same as any `nn.Module`:

```python
torch.save(model.state_dict(), "yat_mlp.pt")

# later …
m = YatMLP()
m.load_state_dict(torch.load("yat_mlp.pt", map_location="cpu"))
m.eval()
```

### TorchScript

```python
scripted = torch.jit.script(model)
scripted.save("yat_mlp.script.pt")
```

### ONNX export (experimental)

ONNX export *should* work for `YatNMN` and `YatConv*` since they only use standard tensor ops, but it is **not yet covered by tests**. Track progress in [`TODO.md`](../../TODO.md).

```python
torch.onnx.export(model, torch.randn(1, 1, 28, 28), "yat_mlp.onnx",
                  input_names=["x"], output_names=["logits"], opset_version=17)
```

If you successfully export a non-trivial Yat model to ONNX, please open a PR adding a test to `tests/test_torch/`.

---

## 7. Distributed training

`YatNMN` and `YatConv*` are plain `nn.Module`s, so they work unchanged with:

- `torch.nn.DataParallel`
- `torch.nn.parallel.DistributedDataParallel`
- [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) — compile passes cleanly; expect speedups similar to a stock model.
- Mixed precision via `torch.cuda.amp.autocast`. **Bump `epsilon` to `1e-3`** for fp16 (see Troubleshooting).

---

## 8. Troubleshooting

| Symptom                                       | Likely cause                                             | Fix                                                                          |
| --------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `NaN` loss within first ~100 steps            | `epsilon` too small for input/weight scale               | Use `YatNMN(..., epsilon=1e-3)` or `1e-2` for fp16/bf16                      |
| Loss plateaus very early                      | α uninitialized / fixed at 1.0                           | `YatNMN(..., use_alpha=True)`, or `constant_alpha=True` (fixes α = √2)       |
| Output range very small                       | Weights too small at init                                | Default init usually works; if not, increase init scale or use `use_alpha`    |
| Gradient explodes in late training            | Unnormalized inputs reaching a Yat layer                 | Add `LayerNorm` *before* the first Yat layer (not between Yat layers)        |
| `torch.compile` graph break on Yat layer      | Old PyTorch version                                       | Upgrade to PyTorch ≥ 2.1                                                     |

---

## 9. Next steps

- [Architecture & theory](../architecture.md)
- [Migration cheat sheet](../migration.md)
- [Full PyTorch examples in repo](../../src/nmn/torch/examples/)
- [EXAMPLES.md](../../EXAMPLES.md) — more snippets
