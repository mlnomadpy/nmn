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
import tempfile
import time
from collections.abc import Sequence

_PROBE_MARKER = "__NMN_TEST_BACKEND_READY__"
_PROBE_MARKER_BYTES = _PROBE_MARKER.encode("ascii")
_PROBE_TIMEOUT_SECONDS = 15.0
_PROBE_REAP_SECONDS = 1.0
_PROBE_SCRIPT = f"""\
import importlib
import json
import os
import sys
import time

request = json.loads(sys.argv[1])
gate = request.get("gate")
while gate and not os.path.exists(gate):
    time.sleep(0.005)
modules = {{name: importlib.import_module(name) for name in request["modules"]}}
if request.get("readiness") == "mlx":
    # Importing mlx.core alone is insufficient: on an installed but unusable
    # runtime, the first device query may abort in native Metal code.
    modules["mlx.core"].default_device()
print({_PROBE_MARKER!r})
"""


def _assign_windows_kill_job(process: subprocess.Popen) -> None:
    """Own the probe tree with a Windows kill-on-close Job Object."""

    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(information), ctypes.sizeof(information)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(handle, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    process._nmn_job_handle = handle
    process._nmn_close_job_handle = kernel32.CloseHandle


def _close_windows_kill_job(process: subprocess.Popen) -> bool:
    handle = getattr(process, "_nmn_job_handle", None)
    if handle is None:
        return False
    process._nmn_job_handle = None
    process._nmn_close_job_handle(handle)
    return True


def _remove_gate(gate: str | None) -> None:
    if gate is None:
        return
    try:
        os.unlink(gate)
    except OSError:
        pass


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
        # Job membership survives root-process exit, unlike taskkill's
        # parent/child discovery. Closing the last handle atomically terminates
        # every process still owned by the job.
        if not _close_windows_kill_job(process):
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

    gate = None
    request_data = {"modules": list(modules), "readiness": readiness}
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        try:
            descriptor, gate = tempfile.mkstemp(prefix="nmn-backend-gate-")
            os.close(descriptor)
            os.unlink(gate)
        except OSError:
            _remove_gate(gate)
            return False
        request_data["gate"] = gate
    request = json.dumps(request_data)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", _PROBE_SCRIPT, request],
            **_popen_options(),
        )
    except OSError:
        _remove_gate(gate)
        return False

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        try:
            _assign_windows_kill_job(process)
            with open(gate, "wb"):
                pass
        except (OSError, AttributeError):
            _stop_process_tree(process)
            _remove_gate(gate)
            return False

    try:
        stdout, _stderr = process.communicate(timeout=_PROBE_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        _stop_process_tree(process)
        _remove_gate(gate)
        return False
    finally:
        _remove_gate(gate)

    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        # Release durable ownership after the root finishes. This also removes
        # any background descendants left behind by an otherwise successful
        # import.
        _close_windows_kill_job(process)

    if stdout is None:
        stdout = b""
    elif not isinstance(stdout, (bytes, bytearray)):
        _stop_process_tree(process)
        return False
    success = process.returncode == 0 and _PROBE_MARKER_BYTES in stdout.splitlines()
    if not success:
        # The direct child may have exited while a grandchild remains alive.
        _stop_process_tree(process)
    return success


def mlx_is_usable() -> bool:
    """Return whether MLX imports and can initialize its default device."""

    return isolated_import_succeeds(["mlx.core"], readiness="mlx")
