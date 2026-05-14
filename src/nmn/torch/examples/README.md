# PyTorch YAT Examples

Runnable training scripts using NMN's PyTorch layers.

## Files

| Path | What it does |
| ---- | ------------ |
| `quick_example.py` | Smoke test for `YatNMN` — weight normalization, α tuning, basic forward/backward. |
| `vision/mnist.py` | 2-layer Yat MLP on MNIST. 3 epochs, ~95 % test acc on CPU in ~5 s. |
| `vision/resnet_training.py` | Yat vs. standard ResNet comparison on CIFAR / TinyImageNet / HF datasets. |

## Quick start — MNIST

```bash
PYTHONPATH=src python -m nmn.torch.examples.vision.mnist \
    --epochs 3 --batch-size 128 --lr 3e-4 \
    --report .context/torch_mnist.json
```

Output (Apple Silicon CPU, fp32, seed=0):

```
device=cpu params=235,148
epoch 0: train_loss=0.9561 test_loss=0.2656 test_acc=0.9207 time=1.4s
epoch 1: train_loss=0.2391 test_loss=0.1978 test_acc=0.9388 time=1.4s
epoch 2: train_loss=0.1818 test_loss=0.1584 test_acc=0.9497 time=1.4s
```

The `--report` JSON contains the full per-epoch history and is the source of
truth for the numbers cited in [`docs/guides/pytorch.md`](../../../../docs/guides/pytorch.md)
and the [cross-framework benchmark blog post](../../../../website/blog-pages/20-cross-framework-mnist.html).

## Tips for using YAT layers

- **Start with the defaults**: `use_alpha=True`, `epsilon=1e-5` covers most cases.
- **fp16 / bf16**: bump `epsilon` to `1e-3` to avoid NaNs in early steps.
- **Normalization**: add `LayerNorm` *before* the first Yat layer if inputs are unbounded.
- **DropConnect**: start at `drop_rate=0.05–0.1`.
- **No activation function**: the non-linearity is intrinsic — no ReLU/GELU between Yat layers.

See [`docs/guides/pytorch.md`](../../../../docs/guides/pytorch.md) for a full
walkthrough including attention, save/load, and troubleshooting.
