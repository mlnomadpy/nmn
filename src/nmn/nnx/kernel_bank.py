"""Lifecycle-safe ownership for tied Flax NNX YAT kernel banks."""

from __future__ import annotations

import threading

from flax import nnx

__all__ = ["KernelBank"]

_KERNEL_BANK_LOCK = threading.RLock()


class _BankParam(nnx.Param):
    """A normal NNX parameter that can be held by a weak legacy registry."""

    __slots__ = ("__weakref__",)


class _BankEntry(nnx.Module):
    def __init__(self, key: tuple[object, ...], parameter: nnx.Param) -> None:
        self.key = key
        self.parameter = parameter


class KernelBank(nnx.Module):
    """Own parameters shared by explicitly associated NNX YAT layers.

    Pass the same instance as ``kernel_bank=`` to layers that should share a
    tied kernel. Separate instances isolate models even when their
    ``kernel_bank_id`` values are equal.
    """

    def __init__(self) -> None:
        self._entries = nnx.List()

    @property
    def _lock(self):
        return _KERNEL_BANK_LOCK

    def __len__(self) -> int:
        """Return the number of live banks owned by this context."""
        return len(self._entries)

    def _get(self, key: tuple[object, ...]) -> nnx.Param | None:
        for entry in self._entries:
            if entry.key == key:
                return entry.parameter
        return None

    def _set(self, key: tuple[object, ...], parameter: nnx.Param) -> None:
        self._entries.append(_BankEntry(key, parameter))
