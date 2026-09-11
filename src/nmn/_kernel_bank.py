"""Shared helpers for lifecycle-safe kernel-bank signatures."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def _freeze(value: Any) -> object:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return ("list", tuple(_freeze(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(key), _freeze(item)) for key, item in value.items())),
        )
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if isinstance(value, functools.partial):
        return (
            "partial",
            initializer_signature(value.func),
            _freeze(value.args),
            _freeze(value.keywords),
        )
    if callable(value):
        return initializer_signature(value)
    attributes = getattr(value, "__dict__", None)
    if attributes:
        return (
            "object",
            type(value).__module__,
            type(value).__qualname__,
            _freeze(attributes),
        )
    return ("value", type(value).__module__, type(value).__qualname__, str(value))


def initializer_signature(initializer: Callable[..., object]) -> tuple[object, ...]:
    """Return a pickle-stable signature for a parameter initializer."""
    module = getattr(initializer, "__module__", type(initializer).__module__)
    qualname = getattr(initializer, "__qualname__", type(initializer).__qualname__)
    defaults = _freeze(getattr(initializer, "__defaults__", None))
    keyword_defaults = _freeze(getattr(initializer, "__kwdefaults__", None))
    closure = getattr(initializer, "__closure__", None)
    closure_values = (
        tuple(_freeze(cell.cell_contents) for cell in closure) if closure else ()
    )
    code = getattr(initializer, "__code__", None)
    instance_state = (
        _freeze(vars(initializer))
        if code is None and hasattr(initializer, "__dict__")
        else None
    )
    local_code = (
        (code.co_code, _freeze(code.co_consts))
        if code is not None and ("<locals>" in qualname or qualname == "<lambda>")
        else None
    )
    return (
        "initializer",
        module,
        qualname,
        defaults,
        keyword_defaults,
        closure_values,
        local_code,
        instance_state,
    )
