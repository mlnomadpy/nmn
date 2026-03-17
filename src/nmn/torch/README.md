# PyTorch NMN (Neural Matter Networks) - YAT Layers

Comprehensive PyTorch implementation of YAT (Yet Another Transformation) neural layers for building efficient and expressive neural networks.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
  - [YatConv (1D, 2D, 3D)](#yatconv-1d-2d-3d)
  - [YatNMN (Linear YAT Layer)](#yatnmn-linear-yat-layer)
  - [Weight Normalization](#weight-normalization)
  - [Weight Tying & Kernel Banks](#weight-tying--kernel-banks)
  - [Constant Alpha Scaling](#constant-alpha-scaling)
  - [Spherical Mode](#spherical-mode)
  - [DropConnect Regularization](#dropconnect-regularization)
- [Advanced Usage](#advanced-usage)
- [Training Example](#training-example)
- [Performance Tips](#performance-tips)
- [API Reference](#api-reference)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd nmn

# Install in development mode
pip install -e .

# Or install with torch support specifically
pip install -e ".[torch]"
```

## Quick Start

### Basic YatConv2D Usage

```python
import torch
import torch.nn as nn
from nmn.torch.layers import YatConv2D

# Simple YAT convolutional layer
layer = YatConv2D(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    padding=1,
    use_alpha=True,  # Learnable alpha scaling
    epsilon=1e-5     # Stability constant
)

# Forward pass
x = torch.randn(4, 3, 32, 32)  # batch, channels, height, width
y = layer(x)  # Output: (4, 64, 32, 32)
```

### Basic YatNMN Usage

```python
from nmn.torch.nmn import YatNMN

# YAT linear transformation layer
layer = YatNMN(
    in_features=256,
    out_features=128,
    use_alpha=True,
    epsilon=1e-5
)

x = torch.randn(4, 256)
y = layer(x)  # Output: (4, 128)
```

## Core Features

### YatConv (1D, 2D, 3D)

YAT convolution applies the YAT transformation to convolution operations:

$$y = \frac{(x \ast W + b)^2}{||x - W||^2 + \epsilon}$$

where $\ast$ denotes convolution and distances are computed patch-wise.

**Available variants:**
- `YatConv1D` - For sequence/temporal data
- `YatConv2D` - For images
- `YatConv3D` - For volumetric data

```python
from nmn.torch.layers import YatConv2D

layer = YatConv2D(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
    use_bias=True,
    use_alpha=True,
    epsilon=1e-5,
    drop_rate=0.0
)

output = layer(input_tensor)
```

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels/filters
- `kernel_size`: Size of convolutional kernel
- `stride`: Stride of convolution
- `padding`: Padding mode ('zeros', 'reflect', etc.)
- `use_bias`: Whether to include bias term
- `use_alpha`: Whether to apply alpha scaling
- `epsilon`: Small constant for numerical stability

### YatNMN (Linear YAT Layer)

YAT applied to linear transformations:

$$y = \frac{(x \cdot W + b)^2}{||x - W||^2 + \epsilon}$$

```python
from nmn.torch.nmn import YatNMN

layer = YatNMN(
    in_features=512,
    out_features=256,
    bias=True,
    use_alpha=True,
    epsilon=1e-5,
    spherical=False
)

output = layer(input_tensor)
```

**Parameters:**
- `in_features`: Input dimension
- `out_features`: Output dimension
- `bias`: Whether to include bias
- `use_alpha`: Whether to apply alpha scaling
- `constant_alpha`: Fixed alpha value (see below)
- `epsilon`: Numerical stability constant
- `spherical`: Enable spherical normalization (see below)

### Weight Normalization

Normalize each neuron/filter to have unit L2-norm. This provides:

1. **Computational efficiency**: Avoids recomputing kernel norms in YAT distance formula (they are guaranteed to be 1.0)
2. **Numerical stability**: Prevents weight explosion
3. **Regularization effect**: Implicitly regularizes weight magnitudes

```python
from nmn.torch.layers import YatConv2D
from nmn.torch.nmn import YatNMN

# Enable weight normalization
conv = YatConv2D(
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    weight_normalized=True  # Enable weight normalization
)

linear = YatNMN(
    in_features=256,
    out_features=128,
    weight_normalized=True  # Enable weight normalization
)
```

**Benefits:**
- Faster inference (skips norm calculation each forward pass)
- More stable training
- Implicit regularization

**How it works:**
- Each kernel filter or neuron is normalized: $W' = \frac{W}{||W||_2}$
- In YAT distance calculation: $||W'||^2 = 1.0$ (always)
- Instead of computing: `kernel_norm = sqrt(sum(w^2))`, use constant `1.0`

### Weight Tying & Kernel Banks

Share kernel parameters across multiple layers to **reduce model size** and **encourage feature learning**.

**Key concepts:**
- **Kernel Bank**: A shared parameter store for multiple layers
- **Auto-expansion**: Automatically grows when a larger layer needs more features
- **First-k slicing**: Layers extract the first k filters they need

```python
from nmn.torch.layers import YatConv2D

# Create multiple layers sharing same kernel bank
layer1 = YatConv2D(
    in_channels=3,
    out_channels=32,  # Use first 32 filters from bank
    kernel_size=3,
    tie_kernel_bank=True,           # Enable weight tying
    kernel_bank_id='resnet-conv'    # Bank namespace
)

layer2 = YatConv2D(
    in_channels=32,
    out_channels=64,  # Automatically expands bank to 64
    kernel_size=3,
    tie_kernel_bank=True,
    kernel_bank_id='resnet-conv'
)

layer3 = YatConv2D(
    in_channels=64,
    out_channels=128,  # Automatically expands bank to 128
    kernel_size=3,
    tie_kernel_bank=True,
    kernel_bank_id='resnet-conv'
)

# All three layers share kernels, bank auto-expanded: 32 -> 64 -> 128
```

**Output:**
```
Auto-expanding kernel bank 'resnet-conv': 32 -> 64 filters
Auto-expanding kernel bank 'resnet-conv': 64 -> 128 filters
```

**For YatNMN:**

```python
from nmn.torch.nmn import YatNMN

head1 = YatNMN(
    in_features=512,
    out_features=256,
    tie_kernel_bank=True,
    kernel_bank_id='classifier'
)

head2 = YatNMN(
    in_features=256,
    out_features=128,
    tie_kernel_bank=True,
    kernel_bank_id='classifier'
)
```

**Multiple independent banks:**

```python
# Create separate banks for conv and classification layers
conv_layer = YatConv2D(
    64, 128, 3,
    tie_kernel_bank=True,
    kernel_bank_id='conv-layers'
)

linear_layer = YatNMN(
    256, 128,
    tie_kernel_bank=True,
    kernel_bank_id='linear-layers'
)

# Two completely separate banks
```

**Parameter reduction example:**

```
Without tying:
- layer1: 32 filters (seed bank size)
- layer2: 64 filters
- layer3: 128 filters
Total: 224 filters

With tying:
- Single shared bank: 128 filters (final size)
Reduction: 224 -> 128 (42.9% fewer parameters)
```

### Constant Alpha Scaling

Control the alpha scaling factor in the YAT transformation.

**Three modes:**

1. **Learnable alpha** (default):
```python
layer = YatConv2D(
    64, 128, 3,
    use_alpha=True           # Alpha is a learnable parameter
)
```

2. **Constant alpha (default value = √2):**
```python
layer = YatConv2D(
    64, 128, 3,
    constant_alpha=True      # Use sqrt(2) ≈ 1.414 as fixed multiplier
)
```

3. **Custom constant alpha:**
```python
layer = YatConv2D(
    64, 128, 3,
    constant_alpha=2.0       # Use custom value: 2.0
)
```

4. **No alpha scaling:**
```python
layer = YatConv2D(
    64, 128, 3,
    use_alpha=False          # No alpha scaling
)
```

**Why use constant alpha?**
- **Faster**: No parameter to optimize
- **Stable gradients**: Fixed scaling factor
- **Empirically effective**: √2 often works well in practice

### Spherical Mode

Normalize both inputs and kernels to lie on a unit sphere. Useful for distance-based computations.

```python
layer = YatNMN(
    in_features=256,
    out_features=128,
    spherical=True  # Enable spherical normalization
)

# Inputs: x = x / ||x||
# Kernels: W = W / ||W||
# Distance: ||x - W||² = 2 - 2(x·W)  [simplified formula]
```

In **spherical mode** with **weight_normalized**:
```python
layer = YatNMN(
    in_features=256,
    out_features=128,
    spherical=True,           # Normalize at each forward pass
    weight_normalized=True    # Also normalize at init
)
# Both inputs and kernels are unit norm
# Maximum efficiency and numerical stability
```

### DropConnect Regularization

Randomly drop connections during training for regularization.

```python
layer = YatConv2D(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    use_dropconnect=True,  # Enable DropConnect
    drop_rate=0.1          # Drop 10% of connections during training
)

# During training: random connections dropped
# During inference (deterministic=True): no dropout
output = layer(x, deterministic=False)  # Training
output = layer(x, deterministic=True)   # Inference
```

## Advanced Usage

### Combining Features

Build efficient models by combining multiple features:

```python
from nmn.torch.layers import YatConv2D
from nmn.torch.nmn import YatNMN
import torch.nn as nn

class EfficientYATNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Use weight tying + normalization + constant alpha for efficiency
        self.conv1 = YatConv2D(
            3, 64, 3, padding=1,
            weight_normalized=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv-bank',
            constant_alpha=True
        )
        
        self.conv2 = YatConv2D(
            64, 128, 3, stride=2, padding=1,
            weight_normalized=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv-bank',
            constant_alpha=True
        )
        
        self.conv3 = YatConv2D(
            128, 256, 3, stride=2, padding=1,
            weight_normalized=True,
            tie_kernel_bank=True,
            kernel_bank_id='conv-bank',
            constant_alpha=True
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with weight tying
        self.proj = nn.Linear(256, 128)
        self.head = YatNMN(
            128, num_classes,
            weight_normalized=True,
            tie_kernel_bank=True,
            kernel_bank_id='head-bank',
            constant_alpha=True
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).view(x.size(0), -1)
        x = self.proj(x)
        x = self.head(x)
        return x
```

### Custom Initialization

Control weight initialization:

```python
from nmn.torch.layers import YatConv2D
import torch.nn as nn

layer = YatConv2D(64, 128, 3)

# Apply custom initialization before weight tying/normalization
nn.init.orthogonal_(layer.weight)

# Then normalize if enabled
if hasattr(layer, 'weight_normalized') and layer.weight_normalized:
    reduce_dims = tuple(range(1, layer.weight.dim()))
    norm = torch.sqrt(torch.sum(layer.weight**2, dim=reduce_dims, keepdim=True))
    layer.weight.data = layer.weight.data / (norm + 1e-8)
```

## Training Example

Complete training script with YAT layers featuring weight tying, normalization, and constant alpha:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nmn.torch.layers import YatConv2D
from nmn.torch.nmn import YatNMN

# Model with all optimizations
class OptimizedYATNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            YatConv2D(3, 64, 3, padding=1,
                     weight_normalized=True,
                     constant_alpha=True,
                     tie_kernel_bank=True,
                     kernel_bank_id='features'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            YatConv2D(64, 128, 3, stride=2, padding=1,
                     weight_normalized=True,
                     constant_alpha=True,
                     tie_kernel_bank=True,
                     kernel_bank_id='features'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            YatConv2D(128, 256, 3, stride=2, padding=1,
                     weight_normalized=True,
                     constant_alpha=True,
                     tie_kernel_bank=True,
                     kernel_bank_id='features'),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            YatNMN(128, num_classes,
                   weight_normalized=True,
                   constant_alpha=True,
                   tie_kernel_bank=True,
                   kernel_bank_id='classifier')
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OptimizedYATNet(num_classes=10).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

optimizer = optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Data loading
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10('.', train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

# Training loop
model.train()
for epoch in range(10):
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}")
```

## Performance Tips

### 1. **Use Weight Normalization for Speed**
```python
# Weight normalization reduces norm computation (~10% forward pass speedup)
layer = YatConv2D(64, 128, 3, weight_normalized=True)
```

### 2. **Use Constant Alpha to Save Parameters**
```python
# Constant alpha saves 1 parameter per layer
layer = YatConv2D(64, 128, 3, constant_alpha=True)  # Good default
```

### 3. **Combine Weight Tying with Large Networks**
```python
# For ResNet-50 style networks, 30-50% parameter reduction possible
for i in range(num_stages):
    layers.append(
        YatConv2D(
            in_ch, out_ch, 3,
            tie_kernel_bank=True,
            kernel_bank_id='shared-conv'
        )
    )
```

### 4. **Use DropConnect During Training**
```python
# Add regularization without slowing down inference
layer = YatConv2D(64, 128, 3, use_dropconnect=True, drop_rate=0.1)
output = layer(x, deterministic=not self.training)
```

### 5. **Spherical Mode for Distance-Critical Tasks**
```python
# Use when distance metrics matter (e.g., metric learning)
layer = YatNMN(256, 128, spherical=True, weight_normalized=True)
```

## API Reference

### YatConv2D

```python
YatConv2D(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, str, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    use_alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    use_dropconnect: bool = False,
    mask: Optional[Tensor] = None,
    epsilon: float = 1e-5,
    drop_rate: float = 0.0,
    weight_normalized: bool = False,
    tie_kernel_bank: bool = False,
    kernel_bank_size: Optional[int] = None,
    kernel_bank_id: str = 'default',
    device=None,
    dtype=None,
)
```

### YatNMN

```python
YatNMN(
    in_features: int,
    out_features: int,
    bias: bool = True,
    alpha: bool = True,
    constant_alpha: Optional[Union[bool, float]] = None,
    dtype: Optional[torch.dtype] = None,
    param_dtype: torch.dtype = torch.float32,
    epsilon: float = 1e-5,
    spherical: bool = False,
    positive_init: bool = False,
    weight_normalized: bool = False,
    tie_kernel_bank: bool = False,
    kernel_bank_size: Optional[int] = None,
    kernel_bank_id: str = 'default',
)
```

## References

- YAT (Yet Another Transformation): Distance-based neural activations
- Weight Normalization: Accelerates deep network convergence
- Kernel Banking: Parameter sharing for efficient learning
- NMN (Neural Matter Networks): Universal approximation framework
