# YAT Convolution Examples

This directory contains example scripts demonstrating the usage of YAT (Yet Another Transformer) convolution layers implemented in PyTorch.

## Overview

YAT convolutions use a distance-based attention mechanism that computes:
```
output = (dot_product)² / (||patch||² + ||kernel||² - 2*dot_product + epsilon)
```

This creates an attention-like mechanism where the response is stronger when the input patch is similar to the convolution kernel.

## Files

### `yat_examples.py`
A comprehensive demonstration script showing:
- Basic YAT convolution usage
- Comparison with standard convolutions
- YAT-specific features (alpha scaling, DropConnect, masking)
- Grouped convolutions
- Simple network examples

**Usage:**
```bash
python yat_examples.py
```

### `yat_cifar10.py`
A complete training script for CIFAR-10 classification that demonstrates:
- YAT convolution in a real deep learning scenario
- Comparison between YAT and standard convolutions
- Proper training loop with validation
- Model saving and loading
- Performance evaluation

**Usage:**
```bash
# Train with YAT convolutions
python yat_cifar10.py --model yat --epochs 20 --batch-size 128 --use-alpha --use-dropconnect

# Train with standard convolutions for comparison
python yat_cifar10.py --model standard --epochs 20 --batch-size 128

# Full options
python yat_cifar10.py --help
```

**Training Options:**
- `--model {yat,standard}`: Choose between YAT or standard convolutions
- `--batch-size INT`: Training batch size (default: 128)
- `--epochs INT`: Number of training epochs (default: 20)
- `--lr FLOAT`: Learning rate (default: 0.001)
- `--use-alpha`: Enable alpha scaling in YAT layers
- `--use-dropconnect`: Enable DropConnect regularization
- `--drop-rate FLOAT`: DropConnect probability (default: 0.1)

## YAT Convolution Features

### 1. Alpha Scaling
Applies a learnable scaling factor based on the number of output channels:
```python
YatConv2d(in_channels=3, out_channels=16, kernel_size=3, use_alpha=True)
```

### 2. DropConnect Regularization
Randomly drops connections during training for better generalization:
```python
YatConv2d(in_channels=3, out_channels=16, kernel_size=3, 
          use_dropconnect=True, drop_rate=0.1)
```

### 3. Weight Masking
Allows masking specific weights in the convolution kernels:
```python
mask = torch.ones(16, 3, 3, 3)  # Shape: (out_channels, in_channels, height, width)
mask[:, :, 0, 0] = 0  # Zero out top-left corner
YatConv2d(in_channels=3, out_channels=16, kernel_size=3, mask=mask)
```

### 4. Grouped Convolutions
Supports grouped convolutions for parameter efficiency:
```python
YatConv2d(in_channels=8, out_channels=16, kernel_size=3, groups=2)
```

### 5. Deterministic Mode
Controls whether DropConnect is applied during forward pass:
```python
# Training mode (applies DropConnect)
output = yat_conv(input, deterministic=False)

# Inference mode (no DropConnect)
output = yat_conv(input, deterministic=True)
```

## Expected Results

### Performance Characteristics
- **YAT convolutions** tend to be slower than standard convolutions due to the distance computation
- **Memory usage** is higher due to intermediate computations for distance calculations
- **Training stability** may be improved due to the attention-like mechanism
- **Generalization** may be better with DropConnect enabled

### CIFAR-10 Training Results
Expected test accuracies after 20 epochs:
- **Standard ConvNet**: ~75-80% accuracy
- **YAT ConvNet**: ~70-85% accuracy (varies based on hyperparameters)

Results may vary based on:
- Alpha scaling settings
- DropConnect rate
- Learning rate and optimization settings
- Random initialization

## Requirements

- PyTorch >= 1.8.0
- torchvision >= 0.9.0
- numpy
- Python >= 3.7

For CIFAR-10 training:
```bash
pip install torch torchvision
```

## Tips for Using YAT Convolutions

1. **Start with basic settings**: Use `use_alpha=True` and `use_dropconnect=False` initially
2. **Adjust epsilon**: If you see numerical instabilities, increase the epsilon parameter
3. **Learning rate**: YAT convolutions may benefit from slightly lower learning rates
4. **DropConnect**: Start with low drop rates (0.05-0.1) and increase if needed
5. **Batch normalization**: YAT works well with batch normalization layers
6. **Mixed precision**: Consider using mixed precision training to speed up computation

## Debugging

If you encounter issues:

1. **Import errors**: Make sure the `nmn.torch.conv` module is in your Python path
2. **CUDA errors**: YAT convolutions should work with both CPU and GPU
3. **Numerical issues**: Try increasing the epsilon parameter (default: 1e-5)
4. **Performance issues**: YAT convolutions are computationally more expensive than standard convolutions

## References

The YAT convolution implementation is based on the concept of distance-based attention mechanisms in neural networks, providing an alternative to standard convolution operations with learnable attention-like properties.
