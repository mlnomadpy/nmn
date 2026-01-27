---
sidebar_position: 2
---

# CIFAR-10 Classification

Image classification with YatConv on CIFAR-10.

## Model

```python
from nmn.nnx.conv import YatConv
from nmn.nnx.nmn import YatNMN

class CIFAR10Net(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = YatConv(3, 64, (3, 3), constant_alpha=True, rngs=rngs)
        self.conv2 = YatConv(64, 128, (3, 3), strides=(2, 2), constant_alpha=True, rngs=rngs)
        self.conv3 = YatConv(128, 256, (3, 3), strides=(2, 2), constant_alpha=True, rngs=rngs)
        self.conv4 = YatConv(256, 512, (3, 3), strides=(2, 2), constant_alpha=True, rngs=rngs)
        self.fc = YatNMN(512 * 4 * 4, 10, constant_alpha=True, rngs=rngs)
    
    def __call__(self, x):
        x = self.conv1(x)  # (B, 32, 32, 64)
        x = self.conv2(x)  # (B, 16, 16, 128)
        x = self.conv3(x)  # (B, 8, 8, 256)
        x = self.conv4(x)  # (B, 4, 4, 512)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
```

## Training Tips

- Use `constant_alpha=True` for stability
- Learning rate: `1e-3` to `3e-4` work well
- No need for batch normalization
