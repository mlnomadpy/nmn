#!/usr/bin/env python3
"""
Simple YAT Convolution Usage Examples

This script demonstrates basic usage patterns for YAT (Yet Another Transformer) 
convolution layers, including:

1. Basic YAT convolution forward pass
2. Comparison with standard convolution
3. Using different YAT features (alpha scaling, DropConnect, masking)
4. Grouped convolutions
5. Performance comparison

Usage:
    python yat_examples.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys

# Add the parent directory to path to import YAT convolutions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from nmn.torch.conv import YatConv2d


def example_basic_usage():
    """Demonstrate basic YAT convolution usage."""
    print("=" * 60)
    print("Example 1: Basic YAT Convolution Usage")
    print("=" * 60)
    
    # Create a YAT convolution layer
    yat_conv = YatConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1,
        use_alpha=True,
        use_dropconnect=False
    )
    
    # Create input tensor (batch_size=4, channels=3, height=32, width=32)
    x = torch.randn(4, 3, 32, 32)
    
    print(f"Input shape: {x.shape}")
    print(f"YAT Conv parameters: {sum(p.numel() for p in yat_conv.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        output = yat_conv(x, deterministic=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"Output mean: {output.mean().item():.4f}")
    print(f"Output std: {output.std().item():.4f}")
    print()


def example_vs_standard_conv():
    """Compare YAT convolution with standard convolution."""
    print("=" * 60)
    print("Example 2: YAT vs Standard Convolution Comparison")
    print("=" * 60)
    
    # Create layers with same configuration
    yat_conv = YatConv2d(3, 16, 3, padding=1, use_alpha=True, use_dropconnect=False)
    std_conv = nn.Conv2d(3, 16, 3, padding=1)
    
    # Use same input
    x = torch.randn(2, 3, 16, 16)
    
    print(f"Input shape: {x.shape}")
    
    # Time both forward passes
    with torch.no_grad():
        # YAT convolution
        start_time = time.time()
        yat_output = yat_conv(x, deterministic=True)
        yat_time = time.time() - start_time
        
        # Standard convolution
        start_time = time.time()
        std_output = std_conv(x)
        std_time = time.time() - start_time
    
    print(f"YAT Conv output shape: {yat_output.shape}")
    print(f"Std Conv output shape: {std_output.shape}")
    print(f"YAT Conv time: {yat_time * 1000:.2f}ms")
    print(f"Std Conv time: {std_time * 1000:.2f}ms")
    print(f"Time ratio (YAT/Std): {yat_time / std_time:.2f}x")
    
    # Compare output statistics
    print(f"\nOutput Statistics:")
    print(f"YAT Conv - Mean: {yat_output.mean().item():.4f}, Std: {yat_output.std().item():.4f}")
    print(f"Std Conv - Mean: {std_output.mean().item():.4f}, Std: {std_output.std().item():.4f}")
    print()


def example_yat_features():
    """Demonstrate different YAT features."""
    print("=" * 60)
    print("Example 3: YAT Feature Demonstrations")
    print("=" * 60)
    
    x = torch.randn(2, 8, 16, 16)
    
    # 1. Alpha scaling
    print("3a. Alpha Scaling Effect:")
    yat_with_alpha = YatConv2d(8, 16, 3, padding=1, use_alpha=True)
    yat_without_alpha = YatConv2d(8, 16, 3, padding=1, use_alpha=False)
    
    with torch.no_grad():
        out_with_alpha = yat_with_alpha(x, deterministic=True)
        out_without_alpha = yat_without_alpha(x, deterministic=True)
    
    print(f"With alpha - Mean: {out_with_alpha.mean().item():.4f}, Std: {out_with_alpha.std().item():.4f}")
    print(f"Without alpha - Mean: {out_without_alpha.mean().item():.4f}, Std: {out_without_alpha.std().item():.4f}")
    
    # 2. DropConnect effect
    print("\n3b. DropConnect Effect (training mode):")
    yat_dropconnect = YatConv2d(8, 16, 3, padding=1, use_dropconnect=True, drop_rate=0.2)
    yat_dropconnect.train()  # Set to training mode
    
    with torch.no_grad():
        # Multiple forward passes to see variation
        outputs = []
        for i in range(3):
            out = yat_dropconnect(x, deterministic=False)  # Non-deterministic for DropConnect
            outputs.append(out)
            print(f"DropConnect run {i+1} - Mean: {out.mean().item():.4f}, Std: {out.std().item():.4f}")
    
    # 3. Weight masking
    print("\n3c. Weight Masking:")
    yat_conv = YatConv2d(8, 16, 3, padding=1)
    
    # Create a mask that zeros out some weights
    mask = torch.ones_like(yat_conv.weight)
    mask[:, :, 0, 0] = 0  # Zero out top-left corner of all kernels
    
    yat_masked = YatConv2d(8, 16, 3, padding=1, mask=mask)
    
    with torch.no_grad():
        out_normal = yat_conv(x, deterministic=True)
        out_masked = yat_masked(x, deterministic=True)
    
    print(f"Normal conv - Mean: {out_normal.mean().item():.4f}, Std: {out_normal.std().item():.4f}")
    print(f"Masked conv - Mean: {out_masked.mean().item():.4f}, Std: {out_masked.std().item():.4f}")
    print()


def example_grouped_convolution():
    """Demonstrate grouped YAT convolution."""
    print("=" * 60)
    print("Example 4: Grouped YAT Convolution")
    print("=" * 60)
    
    # Create grouped convolution (groups=2 means input/output channels are split)
    yat_grouped = YatConv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        padding=1,
        groups=2,  # Split 8 input channels into 2 groups of 4, and 16 output channels into 2 groups of 8
        use_alpha=True
    )
    
    # Regular convolution for comparison
    yat_regular = YatConv2d(8, 16, 3, padding=1, groups=1, use_alpha=True)
    
    x = torch.randn(2, 8, 16, 16)
    
    with torch.no_grad():
        out_grouped = yat_grouped(x, deterministic=True)
        out_regular = yat_regular(x, deterministic=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Grouped conv output: {out_grouped.shape}")
    print(f"Regular conv output: {out_regular.shape}")
    
    # Compare parameter counts
    grouped_params = sum(p.numel() for p in yat_grouped.parameters())
    regular_params = sum(p.numel() for p in yat_regular.parameters())
    
    print(f"Grouped conv parameters: {grouped_params:,}")
    print(f"Regular conv parameters: {regular_params:,}")
    print(f"Parameter reduction: {regular_params / grouped_params:.2f}x")
    print()


def example_simple_network():
    """Demonstrate YAT convolution in a simple network."""
    print("=" * 60)
    print("Example 5: Simple YAT Network")
    print("=" * 60)
    
    class SimpleYATNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = YatConv2d(3, 32, 3, padding=1, use_alpha=True, use_dropconnect=True, drop_rate=0.1)
            self.conv2 = YatConv2d(32, 64, 3, padding=1, use_alpha=True, use_dropconnect=True, drop_rate=0.1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 8 * 8, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x, deterministic=not self.training))
            x = self.pool(x)
            x = F.relu(self.conv2(x, deterministic=not self.training))
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc(x)
            return x
    
    # Create network and test
    net = SimpleYATNet()
    x = torch.randn(4, 3, 32, 32)  # CIFAR-10 like input
    
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test in both training and evaluation modes
    net.train()
    with torch.no_grad():
        train_output = net(x)
    
    net.eval()
    with torch.no_grad():
        eval_output = net(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {train_output.shape}")
    print(f"Training mode output mean: {train_output.mean().item():.4f}")
    print(f"Evaluation mode output mean: {eval_output.mean().item():.4f}")
    print(f"Output difference (train vs eval): {(train_output - eval_output).abs().mean().item():.6f}")
    print()


def main():
    """Run all examples."""
    print("YAT Convolution Examples")
    print("========================")
    print("This script demonstrates various features of YAT (Yet Another Transformer) convolutions.")
    print("YAT uses a distance-based attention mechanism: (dot_product)^2 / (distance_squared + epsilon)")
    print()
    
    # Set random seed for reproducible results
    torch.manual_seed(42)
    
    # Run examples
    example_basic_usage()
    example_vs_standard_conv()
    example_yat_features()
    example_grouped_convolution()
    example_simple_network()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
