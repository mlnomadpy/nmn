"""Tests for YatNMN lazy mode (issue #37, PyTorch backend).

lazy=True freezes ONLY the kernel (excluded from gradients/optimizer); bias,
alpha, and the learnable epsilon stay trainable. lazy=False is unchanged.
"""

import pytest

torch = pytest.importorskip("torch")

from nmn.torch.nmn import YatNMN


def _trainable_names(layer):
    return {n for n, p in layer.named_parameters() if p.requires_grad}


def test_lazy_freezes_only_kernel():
    layer = YatNMN(
        in_features=8, out_features=4,
        bias=True, alpha=True, learnable_epsilon=True, lazy=True,
    )
    assert layer.weight.requires_grad is False
    assert layer.bias.requires_grad is True
    assert layer.alpha.requires_grad is True
    assert layer.epsilon_param.requires_grad is True


def test_lazy_kernel_excluded_from_trainable_params():
    layer = YatNMN(
        in_features=8, out_features=4,
        bias=True, alpha=True, learnable_epsilon=True, lazy=True,
    )
    names = _trainable_names(layer)
    assert "weight" not in names
    assert "bias" in names
    assert "alpha" in names
    assert "epsilon_param" in names


def test_freeze_kernel_alias():
    layer = YatNMN(in_features=8, out_features=4, freeze_kernel=True)
    assert layer.lazy is True
    assert layer.weight.requires_grad is False
    assert layer.bias.requires_grad is True


def test_lazy_default_false_backward_compat():
    layer = YatNMN(in_features=8, out_features=4)
    assert layer.lazy is False
    assert layer.weight.requires_grad is True
    assert "weight" in _trainable_names(layer)


def test_lazy_training_step_kernel_unchanged_others_change():
    torch.manual_seed(0)
    # Use a larger epsilon so softplus(raw) has a non-vanishing slope and the
    # smoke test can observe epsilon moving within a couple of SGD steps.
    layer = YatNMN(
        in_features=8, out_features=4,
        bias=True, alpha=True, learnable_epsilon=True, epsilon=0.5, lazy=True,
    )
    kernel_before = layer.weight.detach().clone()
    bias_before = layer.bias.detach().clone()
    alpha_before = layer.alpha.detach().clone()
    eps_before = layer.epsilon_param.detach().clone()

    opt = torch.optim.SGD(
        [p for p in layer.parameters() if p.requires_grad], lr=0.1
    )
    x = torch.randn(16, 8)
    target = torch.randn(16, 4)

    eps_got_grad = False
    for _ in range(3):
        opt.zero_grad()
        out = layer(x)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        # Kernel must get no gradient.
        assert layer.weight.grad is None
        # epsilon IS in the autograd graph (it is trainable in lazy mode).
        if layer.epsilon_param.grad is not None and layer.epsilon_param.grad.abs().sum() > 0:
            eps_got_grad = True
        opt.step()

    assert torch.allclose(layer.weight, kernel_before), "kernel must stay frozen"
    assert not torch.allclose(layer.bias, bias_before), "bias should train"
    assert not torch.allclose(layer.alpha, alpha_before), "alpha should train"
    assert eps_got_grad, "epsilon should receive gradient (trainable in lazy mode)"
    assert not torch.allclose(layer.epsilon_param, eps_before), "epsilon should train"


def test_non_lazy_kernel_does_change():
    torch.manual_seed(0)
    layer = YatNMN(in_features=8, out_features=4, lazy=False)
    kernel_before = layer.weight.detach().clone()
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    x = torch.randn(16, 8)
    target = torch.randn(16, 4)
    for _ in range(3):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(layer.weight, kernel_before)


def test_lazy_forward_matches_nonlazy_same_init():
    # With identical params, the forward output is identical regardless of lazy.
    torch.manual_seed(123)
    eager = YatNMN(in_features=8, out_features=4, lazy=False)
    torch.manual_seed(123)
    frozen = YatNMN(in_features=8, out_features=4, lazy=True)
    x = torch.randn(5, 8)
    with torch.no_grad():
        assert torch.allclose(eager(x), frozen(x), atol=1e-6)


def test_lazy_composes_with_no_bias():
    layer = YatNMN(in_features=8, out_features=4, bias=False, lazy=True)
    assert layer.weight.requires_grad is False
    assert layer.bias is None
