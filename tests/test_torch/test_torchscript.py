"""TorchScript compatibility tests for NMN PyTorch layers.

These tests verify that NMN layers can be exported via torch.jit.script
or torch.jit.trace, enabling deployment in C++ / TorchServe environments.
"""

import pytest


def test_torchscript_trace_yat_nmn():
    """YatNMN can be traced with torch.jit.trace."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatNMN(in_features=16, out_features=8)
    layer.eval()

    example = torch.randn(4, 16)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5), (
        "Traced output differs from eager output"
    )


def test_torchscript_trace_yat_nmn_no_bias():
    """YatNMN without bias can be traced."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatNMN(in_features=16, out_features=8, bias=False, alpha=False)
    layer.eval()

    example = torch.randn(4, 16)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_trace_yat_nmn_batched():
    """Traced YatNMN handles different batch sizes at runtime."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatNMN(in_features=16, out_features=8)
    layer.eval()

    example = torch.randn(4, 16)
    traced = torch.jit.trace(layer, example)

    # Different batch size — trace should handle this
    x_other = torch.randn(8, 16)
    out_eager = layer(x_other)
    out_traced = traced(x_other)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_trace_yat_conv2d():
    """YatConv2D can be traced with torch.jit.trace."""
    try:
        import torch
        from nmn.torch.layers import YatConv2D
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatConv2D(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    layer.eval()

    example = torch.randn(2, 3, 16, 16)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_trace_yat_conv1d():
    """YatConv1D can be traced with torch.jit.trace."""
    try:
        import torch
        from nmn.torch.layers import YatConv1D
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatConv1D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
    layer.eval()

    example = torch.randn(2, 4, 32)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_trace_yat_conv3d():
    """YatConv3D can be traced with torch.jit.trace."""
    try:
        import torch
        from nmn.torch.layers import YatConv3D
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatConv3D(in_channels=2, out_channels=4, kernel_size=3, padding=1)
    layer.eval()

    example = torch.randn(1, 2, 8, 8, 8)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_trace_yat_conv_transpose2d():
    """YatConvTranspose2D can be traced with torch.jit.trace."""
    try:
        import torch
        from nmn.torch.layers import YatConvTranspose2D
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatConvTranspose2D(
        in_channels=8, out_channels=4, kernel_size=2, stride=2
    )
    layer.eval()

    example = torch.randn(2, 8, 8, 8)
    traced = torch.jit.trace(layer, example)

    out_eager = layer(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)


def test_torchscript_save_load(tmp_path):
    """Traced YatNMN model can be saved and loaded."""
    try:
        import torch
        from nmn.torch.nmn import YatNMN
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    layer = YatNMN(in_features=16, out_features=8)
    layer.eval()

    example = torch.randn(4, 16)
    traced = torch.jit.trace(layer, example)

    model_path = tmp_path / "yat_nmn_traced.pt"
    torch.jit.save(traced, str(model_path))

    loaded = torch.jit.load(str(model_path))
    out_original = traced(example)
    out_loaded = loaded(example)
    assert torch.allclose(out_original, out_loaded, atol=1e-6), (
        "Loaded model output differs from saved model output"
    )


def test_torchscript_multilayer_model():
    """A multi-layer NMN model can be traced end-to-end."""
    try:
        import torch
        import torch.nn as nn
        from nmn.torch.nmn import YatNMN
    except ImportError:
        pytest.skip("PyTorch dependencies not available")

    class TinyNMN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = YatNMN(32, 64)
            self.l2 = YatNMN(64, 16)
            self.l3 = YatNMN(16, 4)

        def forward(self, x):
            return self.l3(self.l2(self.l1(x)))

    model = TinyNMN()
    model.eval()

    example = torch.randn(8, 32)
    traced = torch.jit.trace(model, example)

    out_eager = model(example)
    out_traced = traced(example)
    assert torch.allclose(out_eager, out_traced, atol=1e-5)
    assert out_traced.shape == (8, 4)
