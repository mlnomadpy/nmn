"""Lazy, actionable imports for optional runnable-example dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class LazyExampleDependency:
    """Resolve an example-only module on first use, never on model import."""

    def __init__(self, module: str, install: str, purpose: str) -> None:
        self._module_name = module
        self._install = install
        self._purpose = purpose
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            try:
                self._module = importlib.import_module(self._module_name)
            except ModuleNotFoundError as error:
                raise ModuleNotFoundError(
                    f"{self._purpose} requires the optional dependency "
                    f"{self._module_name!r}. Install it with "
                    f'`pip install "{self._install}"`.'
                ) from error
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)


def lazy_example_dependency(
    module: str, *, install: str, purpose: str
) -> LazyExampleDependency:
    """Create a module proxy with an exact installation remedy."""
    return LazyExampleDependency(module, install, purpose)
