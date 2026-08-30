"""Regression tests for PyTorch dtype and shared-bank issue tranches."""

import copy
import threading

import pytest
import torch

from nmn.torch import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    YatNMN,
)
from nmn.torch._precision import saturating_upcast


def test_tied_dense_construction_preserves_live_peer():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4, 2, tie_kernel_bank=True, kernel_bank_size=3,
        kernel_bank_id="issue-71",
    )
    x = torch.randn(3, 4)
    weight_before = first.weight.detach().clone()
    output_before = first(x).detach().clone()

    second = YatNMN(4, 3, tie_kernel_bank=True, kernel_bank_id="issue-71")

    torch.testing.assert_close(first.weight, weight_before)
    torch.testing.assert_close(first(x), output_before)
    assert second.weight is first.weight
    assert first.weight.requires_grad


def test_tied_dense_auto_expands_before_first_use_and_preserves_existing_slice():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4, 2, tie_kernel_bank=True, alpha=False, bias=False,
        kernel_bank_id="issue-71-pre-use-expand",
    )
    existing_slice = first.weight.detach().clone()

    second = YatNMN(
        4, 3, tie_kernel_bank=True, alpha=False, bias=False,
        kernel_bank_id="issue-71-pre-use-expand",
    )

    assert first.weight is second.weight
    assert first.weight.shape == (3, 4)
    torch.testing.assert_close(first.weight[:2], existing_slice)


def test_tied_dense_rejects_expansion_without_mutating_stale_gradient():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4, 2, tie_kernel_bank=True, alpha=False, bias=False,
        kernel_bank_id="issue-71-stale-grad",
    )
    first(torch.randn(3, 4)).sum().backward()
    parameter = first.weight
    value_before = parameter.detach().clone()
    gradient_before = parameter.grad.detach().clone()

    with pytest.raises(ValueError, match="capacity is frozen"):
        YatNMN(
            4, 3, tie_kernel_bank=True, alpha=False, bias=False,
            kernel_bank_id="issue-71-stale-grad",
        )

    assert first.weight is parameter
    torch.testing.assert_close(first.weight, value_before)
    torch.testing.assert_close(first.weight.grad, gradient_before)


def test_tied_dense_rejects_expansion_without_mutating_adam_state():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4, 2, tie_kernel_bank=True, alpha=False, bias=False,
        kernel_bank_id="issue-71-adam",
    )
    optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    first(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    parameter = first.weight
    value_before = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    with pytest.raises(ValueError, match="capacity is frozen"):
        YatNMN(
            4, 3, tie_kernel_bank=True, alpha=False, bias=False,
            kernel_bank_id="issue-71-adam",
        )

    assert first.weight is parameter
    torch.testing.assert_close(first.weight, value_before)
    for key, expected in state_before.items():
        actual = optimizer.state[parameter][key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected


def test_tied_dense_rejects_incompatible_lazy_consumer():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(4, 2, tie_kernel_bank=True, kernel_bank_id="issue-71-lazy")
    with pytest.raises(ValueError, match="same lazy"):
        YatNMN(4, 2, tie_kernel_bank=True, lazy=True, kernel_bank_id="issue-71-lazy")
    assert first.weight.requires_grad


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConv1D, (2, 2, 7)),
        (YatConv2D, (2, 2, 5, 5)),
        (YatConv3D, (2, 2, 4, 4, 4)),
    ],
)
def test_tied_conv_bank_accumulates_gradients_and_uses_actual_bias_width(
    conv_cls, input_shape
):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-{conv_cls.__name__}"
    narrow = conv_cls(
        2, 2, 1, tie_kernel_bank=True, kernel_bank_size=4,
        kernel_bank_id=bank_id,
    )
    wide = conv_cls(
        2, 4, 1, tie_kernel_bank=True, kernel_bank_size=4,
        kernel_bank_id=bank_id,
    )
    assert narrow.weight is wide.weight
    assert narrow.out_channels == 2
    assert wide.out_channels == 4
    assert narrow.bias.shape == (2,)
    assert wide.bias.shape == (4,)

    reference = conv_cls(2, 2, 1)
    with torch.no_grad():
        reference.weight.copy_(narrow.weight[:2])
        reference.bias.copy_(narrow.bias)
        reference.alpha.copy_(narrow.alpha)
    x_tied = torch.randn(*input_shape, requires_grad=True)
    x_reference = x_tied.detach().clone().requires_grad_()
    tied_output = narrow(x_tied)
    reference_output = reference(x_reference)
    torch.testing.assert_close(tied_output, reference_output)
    tied_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(x_tied.grad, x_reference.grad)
    torch.testing.assert_close(narrow.weight.grad[:2], reference.weight.grad)

    narrow.zero_grad(set_to_none=True)
    wide.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(narrow.parameters(), lr=1e-3)
    before = narrow.weight.detach().clone()
    x = torch.randn(*input_shape)
    (narrow(x).sum() + wide(x).sum()).backward()
    assert narrow.weight.grad is not None
    assert torch.isfinite(narrow.weight.grad).all()
    assert torch.count_nonzero(narrow.weight.grad[2:]) > 0
    optimizer.step()
    assert not torch.equal(before, narrow.weight)


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv2D, YatConv3D])
def test_tied_conv_auto_expands_before_first_use_and_preserves_slice(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-pre-use-expand-{conv_cls.__name__}"
    first = conv_cls(
        2, 2, 1, tie_kernel_bank=True, bias=False, use_alpha=False,
        kernel_bank_id=bank_id,
    )
    existing_slice = first.weight.detach().clone()

    second = conv_cls(
        2, 4, 1, tie_kernel_bank=True, bias=False, use_alpha=False,
        kernel_bank_id=bank_id,
    )

    assert first.weight is second.weight
    assert first.weight.shape[0] == 4
    torch.testing.assert_close(first.weight[:2], existing_slice)
    assert first.out_channels == 2
    assert second.out_channels == 4


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConv1D, (1, 2, 3)),
        (YatConv2D, (1, 2, 2, 2)),
        (YatConv3D, (1, 2, 2, 2, 2)),
    ],
)
def test_tied_conv_rejects_expansion_without_mutating_adam_state(
    conv_cls, input_shape
):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-immutable-{conv_cls.__name__}"
    first = conv_cls(
        2, 2, 1, tie_kernel_bank=True, bias=False, use_alpha=False,
        kernel_bank_id=bank_id,
    )
    optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    first(torch.randn(*input_shape)).sum().backward()
    optimizer.step()
    parameter = first.weight
    value_before = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    with pytest.raises(ValueError, match="capacity is frozen"):
        conv_cls(
            2, 4, 1, tie_kernel_bank=True, bias=False, use_alpha=False,
            kernel_bank_id=bank_id,
        )

    assert first.weight is parameter
    assert first.out_channels == 2
    torch.testing.assert_close(first.weight, value_before)
    for key, expected in state_before.items():
        torch.testing.assert_close(optimizer.state[parameter][key], expected)


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv2D, YatConv3D])
def test_tied_conv_bank_is_device_scoped_and_preserves_public_width(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-device-{conv_cls.__name__}"
    cpu_layer = conv_cls(
        2, 2, 1, tie_kernel_bank=True, kernel_bank_size=4,
        kernel_bank_id=bank_id, device="cpu",
    )
    meta_layer = conv_cls(
        2, 2, 1, tie_kernel_bank=True, kernel_bank_size=4,
        kernel_bank_id=bank_id, device="meta",
    )

    assert cpu_layer.weight.device.type == "cpu"
    assert meta_layer.weight.device.type == "meta"
    assert cpu_layer.weight is not meta_layer.weight
    assert cpu_layer.out_channels == meta_layer.out_channels == 2


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv3D])
def test_tied_conv_concurrent_construction_is_atomic(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-construction-race-{conv_cls.__name__}"
    barrier = threading.Barrier(3)
    layers = []
    errors = []

    def construct(width):
        barrier.wait()
        try:
            layers.append(conv_cls(
                2, width, 1, tie_kernel_bank=True, bias=False,
                use_alpha=False, kernel_bank_id=bank_id,
            ))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [
        threading.Thread(target=construct, args=(2,)),
        threading.Thread(target=construct, args=(4,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(layers) == 2
    assert layers[0].weight is layers[1].weight
    assert layers[0].weight.shape[0] == 4


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [(YatConv1D, (1, 2, 3)), (YatConv3D, (1, 2, 2, 2, 2))],
)
def test_tied_conv_first_use_and_expansion_race_is_serialized(
    conv_cls, input_shape
):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-use-race-{conv_cls.__name__}"
    first = conv_cls(
        2, 2, 1, tie_kernel_bank=True, bias=False, use_alpha=False,
        kernel_bank_id=bank_id,
    )
    parameter = first.weight
    barrier = threading.Barrier(3)
    outputs = []
    expanded = []
    errors = []

    def execute():
        barrier.wait()
        outputs.append(first(torch.randn(*input_shape)))

    def expand():
        barrier.wait()
        try:
            expanded.append(conv_cls(
                2, 4, 1, tie_kernel_bank=True, bias=False,
                use_alpha=False, kernel_bank_id=bank_id,
            ))
        except ValueError as error:
            errors.append(error)

    threads = [threading.Thread(target=execute), threading.Thread(target=expand)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outputs) == 1 and torch.isfinite(outputs[0]).all()
    assert first.weight is parameter
    assert len(expanded) + len(errors) == 1
    if expanded:
        assert expanded[0].weight is parameter
        assert parameter.shape[0] == 4
    else:
        assert "capacity is frozen" in str(errors[0])
        assert parameter.shape[0] == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda tied: YatNMN(4, 2, tie_kernel_bank=tied),
        lambda tied: YatConv1D(2, 2, 1, tie_kernel_bank=tied),
        lambda tied: YatConv2D(2, 2, 1, tie_kernel_bank=tied),
        lambda tied: YatConv3D(2, 2, 1, tie_kernel_bank=tied),
    ],
)
def test_tied_bank_rejects_apply_migration_but_untied_module_migrates(factory):
    tied = factory(True)
    parameter = tied.weight
    with pytest.raises(RuntimeError, match="migration is unsupported"):
        tied.double()
    with pytest.raises(RuntimeError, match="migration is unsupported"):
        tied.to("meta")
    assert tied.weight is parameter
    assert tied.weight.device.type == "cpu"
    assert tied.weight.dtype == torch.float32

    untied = factory(False).double()
    assert untied.weight.dtype == torch.float64


def test_tied_conv_attachment_rejects_stale_registry_dtype():
    YatConv1D._KERNEL_BANKS.clear()
    first = YatConv1D(
        2, 2, 1, tie_kernel_bank=True,
        kernel_bank_id="issue-72-stale-registry",
    )
    first.weight.data = first.weight.data.double()

    with pytest.raises(RuntimeError, match="registry is stale"):
        YatConv1D(
            2, 2, 1, tie_kernel_bank=True,
            kernel_bank_id="issue-72-stale-registry",
        )


def test_yat_nmn_device_constructor_covers_all_owned_state_and_default():
    default_layer = YatNMN(4, 2, learnable_epsilon=True)
    assert default_layer.weight.device.type == "cpu"
    assert all(value.device.type == "cpu" for value in default_layer.state_dict().values())

    meta_layer = YatNMN(
        4, 2, learnable_epsilon=True, device="meta", param_dtype=torch.float64
    )
    state = meta_layer.state_dict()
    assert set(state) == {"weight", "alpha", "bias", "epsilon_param"}
    assert all(value.device.type == "meta" for value in state.values())
    assert all(value.dtype == torch.float64 for value in state.values())


def test_tied_yat_nmn_banks_are_device_separated_and_constructor_route_runs():
    YatNMN._KERNEL_BANKS.clear()
    bank_id = "issue-71-device-constructor"
    cpu_layer = YatNMN(
        4, 2, tie_kernel_bank=True, learnable_epsilon=True,
        kernel_bank_id=bank_id, device="cpu", dtype=torch.float32,
        param_dtype=torch.float64,
    )
    meta_layer = YatNMN(
        4, 2, tie_kernel_bank=True, learnable_epsilon=True,
        kernel_bank_id=bank_id, device="meta", dtype=torch.float32,
        param_dtype=torch.float64,
    )

    assert cpu_layer.weight is not meta_layer.weight
    assert all(value.device.type == "cpu" for value in cpu_layer.state_dict().values())
    assert all(value.device.type == "meta" for value in meta_layer.state_dict().values())
    output = cpu_layer(torch.randn(3, 4, dtype=torch.float32))
    assert output.device.type == "cpu"
    assert output.dtype == torch.float32


def test_untied_yat_nmn_supports_constructor_device_and_later_migration():
    layer = YatNMN(4, 2, learnable_epsilon=True, device="cpu")
    migrated = layer.to(dtype=torch.float64)
    assert migrated is layer
    state = layer.state_dict()
    assert state["epsilon_param"].dtype == torch.float32
    assert all(
        value.dtype == torch.float64
        for name, value in state.items()
        if name != "epsilon_param"
    )


def test_attention_compute_and_parameter_dtypes_round_trip():
    torch.manual_seed(0)
    layer = MultiHeadYatAttention(
        4, 2, dtype=torch.float32, param_dtype=torch.float64, dropout=0.0
    )
    x = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    expected = layer(x, deterministic=True)
    expected.square().sum().backward()

    assert expected.dtype == torch.float32
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.dtype == torch.float64
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    restored = MultiHeadYatAttention(
        4, 2, dtype=torch.float32, param_dtype=torch.float64, dropout=0.0
    )
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    torch.testing.assert_close(restored(x.detach(), deterministic=True), expected.detach())

    reference = MultiHeadYatAttention(4, 2, dtype=torch.float32, param_dtype=torch.float32)
    reference.load_state_dict({key: value.float() for key, value in layer.state_dict().items()})
    x_split = x.detach().clone().requires_grad_()
    x_reference = x.detach().clone().requires_grad_()
    split_output = layer(x_split, deterministic=True)
    reference_output = reference(x_reference, deterministic=True)
    torch.testing.assert_close(split_output, reference_output, rtol=1e-5, atol=1e-6)
    split_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(x_split.grad, x_reference.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("spherical", [False, True])
def test_low_precision_conv_and_embedding_exact_match_is_finite(dtype, spherical):
    conv = YatConv1D(
        1,
        1,
        3,
        bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        dtype=dtype,
    )
    with torch.no_grad():
        # The finite score is representable in fp16, but its derivative is
        # about 75,000 and exercises the saturating fp32->fp16 grad boundary.
        conv.weight.fill_(0.5)
    # Two identical patches also force the learnable-epsilon cotangent to be
    # reduced above fp16's finite range before it reaches the storage boundary.
    conv_input = torch.full((1, 1, 4), 0.5, dtype=dtype, requires_grad=True)
    conv_output = conv(conv_input)
    conv_output.sum().backward()
    assert torch.isfinite(conv_output).all() and (conv_output >= 0).all()
    assert torch.isfinite(conv_input.grad).all()
    assert torch.isfinite(conv.weight.grad).all()
    assert torch.isfinite(conv.epsilon_param.grad).all()

    embed = YatEmbed(
        1, 3, use_alpha=False, epsilon=1e-5, dtype=dtype, spherical=spherical
    )
    with torch.no_grad():
        embed.embedding.fill_(0.5)
    query = embed.embedding.detach().clone().requires_grad_()
    embed_output = embed.attend(query)
    embed_output.sum().backward()
    assert torch.isfinite(embed_output).all() and (embed_output >= 0).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(embed.embedding.grad).all()


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64, None])
def test_split_precision_exact_match_saturates_storage_gradients(compute_dtype):
    conv = YatConv1D(
        1,
        1,
        3,
        bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        dtype=compute_dtype,
        param_dtype=torch.float16,
    )
    with torch.no_grad():
        conv.weight.fill_(0.5)
    input_dtype = torch.float64 if compute_dtype is None else compute_dtype
    x = torch.full((1, 1, 3), 0.5, dtype=input_dtype, requires_grad=True)
    output = conv(x)
    output.sum().backward()

    assert output.dtype == input_dtype and torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert conv.weight.grad.dtype == torch.float16
    assert torch.isfinite(conv.weight.grad).all()
    assert torch.isfinite(conv.epsilon_param.grad).all()


def test_saturating_upcast_preserves_nan_gradient_diagnostics():
    x = torch.ones(1, dtype=torch.float16, requires_grad=True)
    (saturating_upcast(x) * torch.tensor(float("nan"))).sum().backward()
    assert torch.isnan(x.grad).all()


@pytest.mark.parametrize("kind", ["conv", "embed"])
@pytest.mark.skipif(not hasattr(torch, "func"), reason="torch.func requires PyTorch 2")
def test_low_precision_saturating_upcast_supports_func_transforms(kind):
    if kind == "conv":
        layer = YatConv1D(1, 1, 1, bias=False, use_alpha=False, dtype=torch.float16)
        x = torch.full((2, 1, 1), 0.25, dtype=torch.float16)
        call = layer
    else:
        layer = YatEmbed(2, 1, use_alpha=False, dtype=torch.float16)
        x = torch.full((2, 1), 0.25, dtype=torch.float16)
        call = layer.attend

    primal, tangent = torch.func.jvp(call, (x,), (torch.ones_like(x),))
    batched = torch.vmap(call)(x)
    assert torch.isfinite(primal).all() and torch.isfinite(tangent).all()
    assert torch.isfinite(batched).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConvTranspose1D, (1, 1, 1)),
        (YatConvTranspose2D, (1, 1, 1, 1)),
        (YatConvTranspose3D, (1, 1, 1, 1, 1)),
    ],
)
def test_low_precision_transpose_conv_exact_match_is_finite(
    dtype, conv_cls, input_shape
):
    layer = conv_cls(
        1, 1, 1, bias=False, use_alpha=False, epsilon=1e-5, dtype=dtype
    )
    with torch.no_grad():
        layer.weight.fill_(0.7)
    x = torch.full(input_shape, 0.7, dtype=dtype, requires_grad=True)
    output = layer(x)
    output.sum().backward()
    assert torch.isfinite(output).all() and (output >= 0).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_conv_and_embedding_match_fp32_away_from_collision(dtype):
    conv32 = YatConv1D(1, 2, 3, bias=False, use_alpha=False, epsilon=1e-2)
    conv_low = YatConv1D(
        1, 2, 3, bias=False, use_alpha=False, epsilon=1e-2, dtype=dtype
    )
    conv_low.load_state_dict({k: v.to(dtype) for k, v in conv32.state_dict().items()})
    x32 = torch.tensor([[[0.2, -0.4, 0.7, 0.1, -0.3]]])
    torch.testing.assert_close(
        conv_low(x32.to(dtype)).float(), conv32(x32), rtol=6e-2, atol=2e-2
    )

    embed32 = YatEmbed(3, 4, use_alpha=False, epsilon=1e-2)
    embed_low = YatEmbed(3, 4, use_alpha=False, epsilon=1e-2, dtype=dtype)
    embed_low.load_state_dict({k: v.to(dtype) for k, v in embed32.state_dict().items()})
    query32 = torch.tensor([[0.3, -0.2, 0.6, 0.9]])
    torch.testing.assert_close(
        embed_low.attend(query32.to(dtype)).float(),
        embed32.attend(query32),
        rtol=6e-2,
        atol=2e-2,
    )
