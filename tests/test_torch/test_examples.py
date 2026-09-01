"""Smoke tests for documented PyTorch examples."""


def test_quick_example_basic_layers_run(capsys):
    from nmn.torch.examples.quick_example import example_1_basic_yat_layers

    example_1_basic_yat_layers()

    output = capsys.readouterr().out
    assert "Conv forward pass" in output
    assert "Linear forward pass" in output
