"""Tests for the import-light ``nmn`` CLI (``nmn.cli``)."""

from __future__ import annotations

import importlib
import sys

import pytest

from nmn import cli


# ---------------------------------------------------------------------------
# Import-lightness: importing nmn.cli must not pull in any heavy framework.
# ---------------------------------------------------------------------------

def test_cli_import_is_light():
    # Force a clean (re)import of nmn.cli in a subprocess-free way: drop it and
    # any heavy modules that may already be loaded by other tests, then import.
    heavy = ["torch", "tensorflow", "jax", "flax", "keras", "mlx"]
    saved = {name: sys.modules.get(name) for name in heavy + ["nmn.cli"]}
    for name in heavy + ["nmn.cli"]:
        sys.modules.pop(name, None)
    try:
        importlib.import_module("nmn.cli")
        # The spec requires torch/tensorflow specifically to stay unimported.
        assert "torch" not in sys.modules
        assert "tensorflow" not in sys.modules
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["info"],
        ["version"],
        ["frameworks"],
        ["features"],
        ["doctor"],
        ["examples"],
        ["guide", "torch"],
    ],
)
def test_commands_exit_zero(argv):
    assert cli.main(argv) == 0


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code == 2


def test_unknown_guide_framework_exits_two():
    assert cli.main(["guide", "cobol"]) == 2


# ---------------------------------------------------------------------------
# Content: key substrings
# ---------------------------------------------------------------------------

def test_info_default_banner(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "nmn" in out
    # All six framework extras advertised.
    for extra in ("nmn[torch]", "nmn[nnx]", "nmn[keras]", "nmn[tf]", "nmn[linen]", "nmn[mlx]"):
        assert extra in out
    assert "nmn guide" in out


def test_version_prints_only_version(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    from nmn import __version__

    assert out == __version__


def test_frameworks_shows_import_and_ctor(capsys):
    assert cli.main(["frameworks"]) == 0
    out = capsys.readouterr().out
    assert "from nmn.torch import YatNMN" in out
    assert "in_features=128, out_features=256" in out  # torch / nnx ctor
    assert "rngs=nnx.Rngs(0)" in out  # nnx-specific kwarg
    assert "from nmn.linen import YatNMN" in out
    assert "units=256" in out  # keras ctor
    assert "from nmn.mlx import YatNMN" in out


@pytest.mark.parametrize(
    "alias,needle",
    [
        ("torch", "from nmn.torch import YatNMN"),
        ("pytorch", "from nmn.torch import YatNMN"),
        ("nnx", "rngs"),
        ("flax-nnx", "rngs"),
        ("linen", "from nmn.linen import YatNMN"),
        ("flax-linen", "from nmn.linen import YatNMN"),
        ("keras", "units="),
        ("tf", "from nmn.tf import YatNMN"),
        ("tensorflow", "from nmn.tf import YatNMN"),
        ("mlx", "from nmn.mlx import YatNMN"),
    ],
)
def test_guide_aliases(alias, needle, capsys):
    assert cli.main(["guide", alias]) == 0
    out = capsys.readouterr().out
    assert needle in out
    assert "Full guide" in out


def test_guide_attention_kwargs_match_signatures(capsys):
    """Every embedded attention quickstart must use the real ctor kwargs.

    keras/tf/mlx -> MultiHeadYatAttention(embed_dim=..., num_heads=...) (NO key_dim)
    linen        -> MultiHeadAttention(num_heads=..., qkv_features=..., out_features=...)
    nnx          -> MultiHeadAttention(num_heads=..., in_features=..., rngs=...)
    torch        -> MultiHeadYatAttention(embed_dim=..., num_heads=...)
    """
    guides = {}
    for fw in ("torch", "nnx", "linen", "keras", "tf", "mlx"):
        assert cli.main(["guide", fw]) == 0
        guides[fw] = capsys.readouterr().out

    # key_dim was the bug: it must not appear in ANY guide.
    for fw, text in guides.items():
        assert "key_dim" not in text, f"{fw} guide still mentions key_dim"

    # keras/tf/mlx and torch -> embed_dim, no in_features in their attention.
    for fw in ("torch", "keras", "tf", "mlx"):
        assert "embed_dim=128" in guides[fw], f"{fw} guide missing embed_dim"

    # linen attention must use qkv_features/out_features, never in_features.
    assert "qkv_features=128" in guides["linen"]
    assert "out_features=128" in guides["linen"]
    assert "in_features=128" not in guides["linen"]
    # The dead `class MLP(YatNMN): pass` snippet must be gone.
    assert "class MLP" not in guides["linen"]

    # nnx attention does take in_features (that is its real signature).
    assert "in_features=128" in guides["nnx"]
    assert "rngs=" in guides["nnx"]


def test_guide_yatnmn_ctor_kwargs_match_signatures(capsys):
    """The embedded YatNMN constructor line must use the per-framework kwarg."""
    expected = {
        "torch": "in_features=",
        "nnx": "in_features=",
        "linen": "features=128",
        "keras": "units=",
        "tf": "features=128",
        "mlx": "features=128",
    }
    for fw, needle in expected.items():
        assert cli.main(["guide", fw]) == 0
        out = capsys.readouterr().out
        assert needle in out, f"{fw} guide missing YatNMN kwarg {needle!r}"
        # keras must NOT use features= (it is `units=`).
        if fw == "keras":
            assert "YatNMN(features=" not in out


def _extract_lines(text, prefix_token):
    """Return source lines (dedented) from a guide containing ``prefix_token``."""
    return [
        line.strip()
        for line in text.splitlines()
        if prefix_token in line
    ]


def test_torch_guide_snippets_construct(capsys):
    """For the torch backend (importable locally), the emitted YatNMN ctor and
    attention lines must actually construct using the exact guide kwargs."""
    try:
        importlib.import_module("torch")
    except Exception:
        pytest.skip("torch backend not importable locally")

    assert cli.main(["guide", "torch"]) == 0
    text = capsys.readouterr().out

    from nmn.torch import YatNMN, MultiHeadYatAttention

    # Exec the YatNMN ctor lines verbatim from the guide.
    for line in _extract_lines(text, "YatNMN(in_features="):
        eval(line.rstrip(","), {"YatNMN": YatNMN})
    line = _extract_lines(text, "MultiHeadYatAttention(embed_dim=")[0]
    eval(
        line.split("=", 1)[1].strip(),
        {"MultiHeadYatAttention": MultiHeadYatAttention},
    )


@pytest.mark.parametrize("fw", ["nnx", "linen", "mlx"])
def test_guide_attention_ctor_constructs_for_importable(fw, capsys):
    """Construct the attention object from each importable backend's guide
    using the exact kwargs the guide emits."""
    backends = {
        "nnx": ["jax", "flax"],
        "linen": ["jax", "flax"],
        "mlx": ["mlx.core"],
    }
    try:
        for m in backends[fw]:
            importlib.import_module(m)
    except Exception:
        pytest.skip(f"{fw} backend not importable locally")

    assert cli.main(["guide", fw]) == 0
    text = capsys.readouterr().out

    if fw == "nnx":
        from flax import nnx
        from nmn.nnx import MultiHeadAttention
        rngs = nnx.Rngs(0)
        line = _extract_lines(text, "MultiHeadAttention(num_heads=")[0]
        expr = line.split("=", 1)[1].strip()
        eval(expr, {"MultiHeadAttention": MultiHeadAttention, "rngs": rngs, "nnx": nnx})
    elif fw == "linen":
        from nmn.linen import MultiHeadAttention
        line = _extract_lines(text, "MultiHeadAttention(num_heads=")[0]
        expr = line.split("=", 1)[1].strip()
        eval(expr, {"MultiHeadAttention": MultiHeadAttention})
    elif fw == "mlx":
        from nmn.mlx import MultiHeadYatAttention
        line = _extract_lines(text, "MultiHeadYatAttention(embed_dim=")[0]
        expr = line.split("=", 1)[1].strip()
        eval(expr, {"MultiHeadYatAttention": MultiHeadYatAttention})


def test_features_mentions_may_ray_and_lazy(capsys):
    assert cli.main(["features"]) == 0
    out = capsys.readouterr().out
    assert "create_maclaurin_projection" in out
    assert "radial_yat_attention" in out
    assert "performer_kind" in out
    assert "lazy=True" in out
    # canonical kwargs
    assert "bias=" in out
    assert "epsilon=" in out


def test_doctor_lists_all_backends(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    for key in ("torch", "nnx", "linen", "keras", "tf", "mlx"):
        assert key in out
    assert "Python" in out


def test_examples_points_to_examples_md(capsys):
    assert cli.main(["examples"]) == 0
    out = capsys.readouterr().out
    assert "EXAMPLES.md" in out
    assert "nmn guide" in out


# ---------------------------------------------------------------------------
# Programmatic API in nmn/__init__.py
# ---------------------------------------------------------------------------

def test_nmn_help(capsys):
    import nmn

    nmn.help()
    out = capsys.readouterr().out
    assert "nmn[torch]" in out


def test_nmn_doctor_returns_dict(capsys):
    import nmn

    report = nmn.doctor()
    capsys.readouterr()  # drain
    assert isinstance(report, dict)
    assert set(report) == {"torch", "nnx", "linen", "keras", "tf", "mlx"}
    # values are either a version string or None; never raises on missing.
    for value in report.values():
        assert value is None or isinstance(value, str)
