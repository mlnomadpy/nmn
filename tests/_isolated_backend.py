"""Safe availability probes for optional native test backends.

This module intentionally contains no imports from a machine-learning runtime.
Some optional backends can terminate the interpreter from native code during
initialization, so availability must be decided by a bounded child process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence


_PROBE_MARKER = "__NMN_TEST_BACKEND_READY__"
_PROBE_TIMEOUT_SECONDS = 15.0
_PROBE_SCRIPT = f"""\
import importlib
import json
import sys

request = json.loads(sys.argv[1])
modules = {{name: importlib.import_module(name) for name in request["modules"]}}
if request.get("readiness") == "mlx":
    # Importing mlx.core alone is insufficient: on an installed but unusable
    # runtime, the first device query may abort in native Metal code.
    modules["mlx.core"].default_device()
print({_PROBE_MARKER!r})
"""


def isolated_import_succeeds(
    modules: Sequence[str], *, readiness: str | None = None
) -> bool:
    """Return whether optional modules initialize safely in a child process."""

    request = json.dumps({"modules": list(modules), "readiness": readiness})
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SCRIPT, request],
            capture_output=True,
            check=False,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return completed.returncode == 0 and _PROBE_MARKER in completed.stdout.splitlines()


def mlx_is_usable() -> bool:
    """Return whether MLX imports and can initialize its default device."""

    return isolated_import_succeeds(["mlx.core"], readiness="mlx")
