"""Neural-Matter Network (NMN) - beyond blinded neurons."""

try:
    from nmn._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"


def help() -> None:
    """Print the ``nmn info`` banner (version, frameworks, pointers).

    Import-light: ``nmn.cli`` is imported lazily so importing ``nmn`` does not
    pull in any framework.
    """
    from nmn import cli

    print(cli._render_info())


def doctor():
    """Print the ``nmn doctor`` report and return ``{framework: version_or_None}``.

    Import-light: ``nmn.cli`` is imported lazily.
    """
    from nmn import cli

    print(cli._render_doctor())
    return cli._doctor_report()


__all__ = [
    "__version__",
    # Discovery / diagnostics helpers (mirror the `nmn` CLI):
    "help",
    "doctor",
    # Framework subpackages — import the one you need:
    #   from nmn.torch import YatNMN, YatConv2D, ...
    #   from nmn.nnx import YatNMN, YatConv, ...
    #   from nmn.keras import YatNMN, YatConv2D, ...
    #   from nmn.tf import YatNMN, YatConv2D, ...
    #   from nmn.linen import YatNMN, YatConv2D, ...
    #   from nmn.mlx import YatNMN, YatConv2D, ...
    "torch",
    "nnx",
    "keras",
    "tf",
    "linen",
    "mlx",
]
