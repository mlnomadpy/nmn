"""Safe availability probes for optional native test backends.

This module intentionally contains no imports from a machine-learning runtime.
Some optional backends can terminate the interpreter from native code during
initialization, so availability must be decided by a bounded child process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


_PROBE_MARKER = "__NMN_TEST_BACKEND_READY__"
_PROBE_MARKER_BYTES = _PROBE_MARKER.encode("ascii")
_PROBE_TIMEOUT_SECONDS = 15.0
_PROBE_REAP_SECONDS = 1.0
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


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate the isolated probe and every descendant, with bounded reap."""

    if os.name == "posix":
        group_signalled = False
        try:
            os.killpg(process.pid, signal.SIGTERM)
            group_signalled = True
        except (ProcessLookupError, PermissionError):
            pass
        if group_signalled:
            deadline = time.monotonic() + _PROBE_REAP_SECONDS
            while time.monotonic() < deadline:
                if not _process_group_exists(process.pid):
                    break
                time.sleep(0.01)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
    else:  # pragma: no cover - exercised on Windows CI
        # CREATE_NEW_PROCESS_GROUP isolates the probe.  taskkill /T is the
        # standard Windows mechanism that also follows and kills descendants.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=_PROBE_REAP_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass

    try:
        process.communicate(timeout=_PROBE_REAP_SECONDS)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_PROBE_REAP_SECONDS)
        except (OSError, subprocess.SubprocessError):
            pass


def _popen_options() -> dict:
    options = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "posix":
        options["start_new_session"] = True
    else:  # pragma: no cover - exercised on Windows CI
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return options


def isolated_import_succeeds(
    modules: Sequence[str], *, readiness: str | None = None
) -> bool:
    """Return whether optional modules initialize safely in a child process."""

    request = json.dumps({"modules": list(modules), "readiness": readiness})
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE_SCRIPT, request],
            **_popen_options(),
        )
    except OSError:
        return False

    try:
        stdout, _stderr = process.communicate(timeout=_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        _stop_process_tree(process)
        return False

    if stdout is None:
        stdout = b""
    elif not isinstance(stdout, (bytes, bytearray)):
        _stop_process_tree(process)
        return False
    success = (
        process.returncode == 0
        and _PROBE_MARKER_BYTES in stdout.splitlines()
    )
    if not success:
        # The direct child may have exited while a grandchild remains alive.
        _stop_process_tree(process)
    return success


def mlx_is_usable() -> bool:
    """Return whether MLX imports and can initialize its default device."""

    return isolated_import_succeeds(["mlx.core"], readiness="mlx")
