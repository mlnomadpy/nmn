"""Lifecycle-safe ownership for tied PyTorch YAT kernel banks."""

from __future__ import annotations

import threading
from typing import Any, cast

from torch.nn import Parameter

__all__ = ["KernelBank"]


class KernelBank:
    """Own parameters shared by explicitly associated YAT layers.

    Pass the same instance as ``kernel_bank=`` to layers that should share a
    tied kernel. Separate instances isolate models even when their
    ``kernel_bank_id`` values are equal.
    """

    def __init__(self) -> None:
        self._parameters: dict[tuple[Any, ...], Parameter] = {}
        self._used: dict[int, bool] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Return the number of live banks owned by this context."""
        return len(self._parameters)

    def __getstate__(self) -> dict[str, object]:
        """Serialize owned parameters without the process-local lock."""
        return {
            "_parameters": self._parameters,
            "_used_keys": {
                key: self._used.get(id(parameter), False)
                for key, parameter in self._parameters.items()
            },
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore owned parameters, usage state, and a process-local lock."""
        self._parameters = cast(dict[tuple[Any, ...], Parameter], state["_parameters"])
        used_keys = cast(dict[tuple[Any, ...], bool], state.get("_used_keys", {}))
        self._used = {
            id(parameter): used_keys.get(key, False)
            for key, parameter in self._parameters.items()
        }
        self._lock = threading.RLock()
