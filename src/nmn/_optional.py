"""Helpers for optional backend dependency boundaries.

This module intentionally uses only the Python standard library so importing
the root :mod:`nmn` package remains dependency-free.
"""

from __future__ import annotations

from importlib.util import find_spec


def require_optional_dependency(module: str, *, backend: str, extra: str) -> None:
    """Raise an actionable error when an optional backend dependency is absent.

    Only absence discovered before loading NMN's backend implementation is
    translated.  Import errors raised while importing an installed framework
    or NMN implementation are deliberately left untouched, so broken installs
    and programming defects are never misreported as a missing extra.
    """
    if find_spec(module) is None:
        message = (
            f"The {backend} backend requires the optional dependency "
            f'{module!r}. Install it with `pip install "nmn[{extra}]"`.'
        )
        raise ModuleNotFoundError(message, name=module)


__all__ = ["require_optional_dependency"]
