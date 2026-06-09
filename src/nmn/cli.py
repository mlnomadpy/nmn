"""Command-line interface for Neural-Matter Network (NMN).

This module is deliberately *import-light*: importing ``nmn.cli`` must NOT
import any heavy framework (torch / jax / flax / tensorflow / keras / mlx).
Only the :func:`doctor` command attempts framework imports, each guarded in a
``try``/``except`` so a missing backend never raises.

Entry points::

    nmn               # console script (see [project.scripts])
    python -m nmn     # via src/nmn/__main__.py

All output is plain text (no color dependencies) so it stays readable in a
terminal and greppable by coding agents.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Static metadata (kept here so the CLI needs no heavy imports)
# ---------------------------------------------------------------------------

DESCRIPTION = (
    "Neural-Matter Network (NMN) - YAT neural network layers and attention "
    "for six frameworks."
)

ONLINE_GUIDES = "https://github.com/azettaai/nmn/tree/master/docs/guides"
ONLINE_EXAMPLES = "https://github.com/azettaai/nmn/blob/master/EXAMPLES.md"

# Canonical framework order. Each entry: pip extra, import line, the YatNMN
# constructor signature for that framework, and the doctor probe module(s).
FRAMEWORKS = {
    "torch": {
        "extra": "nmn[torch]",
        "label": "PyTorch",
        "import": "from nmn.torch import YatNMN",
        "ctor": "YatNMN(in_features=128, out_features=256)",
        "probes": ["torch"],
    },
    "nnx": {
        "extra": "nmn[nnx]",
        "label": "Flax NNX",
        "import": "from nmn.nnx import YatNMN",
        "ctor": "YatNMN(in_features=128, out_features=256, rngs=nnx.Rngs(0))",
        "probes": ["jax", "flax"],
    },
    "linen": {
        "extra": "nmn[linen]",
        "label": "Flax Linen",
        "import": "from nmn.linen import YatNMN",
        "ctor": "YatNMN(features=256)",
        "probes": ["jax", "flax"],
    },
    "keras": {
        "extra": "nmn[keras]",
        "label": "Keras",
        "import": "from nmn.keras import YatNMN",
        "ctor": "YatNMN(units=256)",
        "probes": ["keras"],
    },
    "tf": {
        "extra": "nmn[tf]",
        "label": "TensorFlow",
        "import": "from nmn.tf import YatNMN",
        "ctor": "YatNMN(features=256)",
        "probes": ["tensorflow"],
    },
    "mlx": {
        "extra": "nmn[mlx]",
        "label": "MLX (Apple Silicon)",
        "import": "from nmn.mlx import YatNMN",
        "ctor": "YatNMN(features=256)",
        # mlx exposes __version__ on mlx.core, not the top-level package.
        "probes": ["mlx.core"],
    },
}

# Aliases accepted by `nmn guide <framework>`.
_ALIASES = {
    "torch": "torch",
    "pytorch": "torch",
    "nnx": "nnx",
    "flax-nnx": "nnx",
    "flax_nnx": "nnx",
    "linen": "linen",
    "flax-linen": "linen",
    "flax_linen": "linen",
    "keras": "keras",
    "tf": "tf",
    "tensorflow": "tf",
    "mlx": "mlx",
}


def _version() -> str:
    """Return the installed nmn version string (import-light)."""
    try:
        from nmn import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "0.0.0"


# ---------------------------------------------------------------------------
# Embedded per-framework quickstarts (must NOT read docs/ — not shipped in wheel)
# ---------------------------------------------------------------------------

_GUIDES = {
    "torch": """\
NMN quickstart — PyTorch  (pip install nmn[torch])

    import torch
    from nmn.torch import YatNMN

    # A small YAT MLP: each YatNMN layer computes
    #   y = (x . W)^2 / (||x - W||^2 + epsilon) * alpha
    mlp = torch.nn.Sequential(
        YatNMN(in_features=64, out_features=128),
        YatNMN(in_features=128, out_features=10),
    )
    y = mlp(torch.randn(8, 64))            # -> (8, 10)

    # Attention (one-liner):
    from nmn.torch import MultiHeadYatAttention
    attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)

Full guide: {guides}/pytorch.md
""",
    "nnx": """\
NMN quickstart — Flax NNX  (pip install nmn[nnx])

    import jax.numpy as jnp
    from flax import nnx
    from nmn.nnx import YatNMN, MultiHeadAttention

    rngs = nnx.Rngs(0)

    # A small YAT MLP (nnx needs in_features/out_features and rngs):
    layer1 = YatNMN(in_features=64, out_features=128, rngs=rngs)
    layer2 = YatNMN(in_features=128, out_features=10, rngs=rngs)
    x = jnp.ones((8, 64))
    y = layer2(layer1(x))                  # -> (8, 10)

    # Attention (one-liner):
    attn = MultiHeadAttention(num_heads=8, in_features=128, rngs=rngs)

Full guide: {guides}/flax-nnx.md
""",
    "linen": """\
NMN quickstart — Flax Linen  (pip install nmn[linen])

    import jax, jax.numpy as jnp
    from nmn.linen import YatNMN, MultiHeadAttention

    # Linen uses `features=` and lazy params (no rngs in the constructor):
    model = YatNMN(features=128)
    x = jnp.ones((8, 64))
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)             # -> (8, 128)

    # Attention (one-liner):
    attn = MultiHeadAttention(num_heads=8, qkv_features=128, out_features=128)

Full guide: {guides}/flax-linen.md
""",
    "keras": """\
NMN quickstart — Keras  (pip install nmn[keras])

    import keras
    from nmn.keras import YatNMN, MultiHeadYatAttention

    # Keras uses `units=` like keras.layers.Dense:
    model = keras.Sequential([
        YatNMN(units=128),
        YatNMN(units=10),
    ])
    # model.build(input_shape=(None, 64))

    # Attention (one-liner):
    attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)

Full guide: {guides}/keras.md
""",
    "tf": """\
NMN quickstart — TensorFlow  (pip install nmn[tf])

    import tensorflow as tf
    from nmn.tf import YatNMN, MultiHeadYatAttention

    # The tf backend uses `features=`:
    layer1 = YatNMN(features=128)
    layer2 = YatNMN(features=10)
    x = tf.ones((8, 64))
    y = layer2(layer1(x))                  # -> (8, 10)

    # Attention (one-liner):
    attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)

Full guide: {guides}/tensorflow.md
""",
    "mlx": """\
NMN quickstart — MLX (Apple Silicon)  (pip install nmn[mlx])

    import mlx.core as mx
    from nmn.mlx import YatNMN, MultiHeadYatAttention

    # The mlx backend uses `features=`:
    layer1 = YatNMN(features=128)
    layer2 = YatNMN(features=10)
    x = mx.ones((8, 64))
    y = layer2(layer1(x))                  # -> (8, 10)

    # Attention (one-liner):
    attn = MultiHeadYatAttention(embed_dim=128, num_heads=8)

Full guide: {guides}/mlx.md
""",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_info() -> str:
    lines: List[str] = []
    lines.append(f"nmn {_version()}")
    lines.append(DESCRIPTION)
    lines.append("")
    lines.append("Frameworks (install only the one you need):")
    for key, meta in FRAMEWORKS.items():
        lines.append(
            f"  {key:<6} {meta['label']:<20} pip install {meta['extra']}"
        )
    lines.append("")
    lines.append("Get a self-contained quickstart for a framework with:")
    lines.append("  nmn guide <framework>      e.g. nmn guide torch")
    lines.append("")
    lines.append("Other commands: version, frameworks, features, doctor, examples")
    return "\n".join(lines)


def _render_frameworks() -> str:
    lines: List[str] = []
    lines.append("Framework import lines and YatNMN constructor signatures:")
    lines.append("")
    for key, meta in FRAMEWORKS.items():
        lines.append(f"{key} ({meta['label']}) — pip install {meta['extra']}")
        lines.append(f"  import: {meta['import']}")
        lines.append(f"  ctor:   {meta['ctor']}")
        lines.append("")
    lines.append(
        "Note: each backend is an independent optional install; importing one "
        "never pulls in another."
    )
    return "\n".join(lines)


def _render_guide(name: str) -> Optional[str]:
    canonical = _ALIASES.get(name.lower())
    if canonical is None:
        return None
    return _GUIDES[canonical].format(guides=ONLINE_GUIDES)


def _render_features() -> str:
    return """\
NMN performer feature maps (MAY / RAY / SLAY) and lazy YatNMN

Linear-attention feature maps approximate the spherical YAT kernel
    kappa(s) = (s + b)^2 / ((2 + epsilon) - 2s)
so attention runs in O(n) instead of O(n^2). Three variants:
  - SLAY   : bias-free anchor (b = 0)
  - MAY    : Random-Maclaurin, bias-aware (faithful for b > 0)
  - RAY    : radial / quadrature, bias-aware

Framework-agnostic API (present in nnx / linen / keras / tf / torch / mlx):

    from nmn.nnx import (              # same names in other backends
        create_maclaurin_projection, maclaurin_features, maclaurin_yat_attention,
        create_radial_projection,    radial_features,    radial_yat_attention,
    )

    import jax
    params = create_maclaurin_projection(
        jax.random.PRNGKey(0), head_dim=64, num_features=256,
        bias=1.0, epsilon=1e-5,      # canonical kwargs: bias, epsilon
    )
    out = maclaurin_yat_attention(q, k, v, params)   # q,k,v: [..., L, H, D]

nnx attention selector — pick the feature map with `performer_kind`:

    from nmn.nnx import RotaryYatAttention
    attn = RotaryYatAttention(
        embed_dim=128, num_heads=8,
        use_performer=True,
        performer_kind="maclaurin",  # one of: slay, maclaurin, radial
        rngs=nnx.Rngs(0),
    )

Lazy YatNMN (freeze ONLY the kernel; bias / alpha / epsilon stay trainable):

    YatNMN(..., lazy=True)           # alias: freeze_kernel=True
"""


def _doctor_report() -> Dict[str, Optional[str]]:
    """Probe each backend; return {framework: version_or_None}. Never raises."""
    result: Dict[str, Optional[str]] = {}
    for key, meta in FRAMEWORKS.items():
        version: Optional[str] = None
        ok = True
        for mod_name in meta["probes"]:
            try:
                mod = importlib.import_module(mod_name)
                version = getattr(mod, "__version__", "unknown")
            except Exception:
                ok = False
                version = None
                break
        result[key] = version if ok else None
    return result


def _render_doctor() -> str:
    report = _doctor_report()
    lines: List[str] = []
    lines.append(f"nmn {_version()}")
    lines.append(f"Python {platform.python_version()} ({sys.executable})")
    lines.append("")
    lines.append("Backend import check:")
    for key, meta in FRAMEWORKS.items():
        version = report[key]
        if version is None:
            status = f"MISSING  (pip install {meta['extra']})"
        else:
            status = f"OK       version {version}"
        lines.append(f"  {key:<6} {meta['label']:<20} {status}")
    return "\n".join(lines)


def _render_examples() -> str:
    return f"""\
NMN examples

Runnable examples live alongside each backend in the source tree:
  src/nmn/<framework>/examples/   (torch, nnx, linen, keras, tf, mlx)
A curated walkthrough is in EXAMPLES.md:
  {ONLINE_EXAMPLES}

Quickstart for the most popular backend (Flax NNX):

    import jax.numpy as jnp
    from flax import nnx
    from nmn.nnx import YatNMN

    rngs = nnx.Rngs(0)
    layer = YatNMN(in_features=64, out_features=128, rngs=rngs)
    y = layer(jnp.ones((8, 64)))          # -> (8, 128)

For a per-framework quickstart run:  nmn guide <framework>
"""


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nmn",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="banner: version, frameworks, pointers")
    sub.add_parser("version", help="print the version string only")
    sub.add_parser("frameworks", help="import lines + YatNMN constructor signatures")

    guide = sub.add_parser("guide", help="self-contained quickstart for a framework")
    guide.add_argument(
        "framework",
        help="one of: torch/pytorch, nnx/flax-nnx, linen/flax-linen, keras, "
        "tf/tensorflow, mlx",
    )

    sub.add_parser("features", help="MAY/RAY performer maps + lazy YatNMN usage")
    sub.add_parser("doctor", help="report which backends import + their versions")
    sub.add_parser("examples", help="where to find runnable examples")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    command = args.command or "info"

    if command in ("info",):
        print(_render_info())
        return 0
    if command == "version":
        print(_version())
        return 0
    if command == "frameworks":
        print(_render_frameworks())
        return 0
    if command == "guide":
        text = _render_guide(args.framework)
        if text is None:
            valid = ", ".join(sorted(set(_ALIASES)))
            print(
                f"Unknown framework {args.framework!r}. Valid choices: {valid}",
                file=sys.stderr,
            )
            return 2
        print(text)
        return 0
    if command == "features":
        print(_render_features())
        return 0
    if command == "doctor":
        print(_render_doctor())
        return 0
    if command == "examples":
        print(_render_examples())
        return 0

    # Should be unreachable (argparse rejects unknown subcommands with exit 2).
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
