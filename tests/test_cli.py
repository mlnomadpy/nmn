"""Tests for the import-light ``nmn`` CLI (``nmn.cli``)."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from nmn import cli
from tests import _isolated_backend

# ---------------------------------------------------------------------------
# Import-lightness: importing nmn.cli must not pull in any heavy framework.
# ---------------------------------------------------------------------------


def test_cli_import_is_light():
    # Force a clean (re)import of nmn.cli in a subprocess-free way: drop it and
    # any heavy modules that may already be loaded by other tests, then import.
    heavy = ["torch", "tensorflow", "jax", "flax", "keras", "mlx"]
    saved = {name: sys.modules.get(name) for name in heavy + ["nmn.cli"]}
    for name in heavy + ["nmn.cli"]:
        sys.modules.pop(name, None)
    try:
        importlib.import_module("nmn.cli")
        # The spec requires torch/tensorflow specifically to stay unimported.
        assert "torch" not in sys.modules
        assert "tensorflow" not in sys.modules
    finally:
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["info"],
        ["version"],
        ["frameworks"],
        ["features"],
        ["doctor"],
        ["examples"],
        ["guide", "torch"],
    ],
)
def test_commands_exit_zero(argv):
    assert cli.main(argv) == 0


def test_unknown_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code == 2


def test_unknown_guide_framework_exits_two():
    assert cli.main(["guide", "cobol"]) == 2


# ---------------------------------------------------------------------------
# Content: key substrings
# ---------------------------------------------------------------------------


def test_info_default_banner(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "nmn" in out
    # All six framework extras advertised.
    for extra in (
        "nmn[torch]",
        "nmn[nnx]",
        "nmn[keras]",
        "nmn[tf]",
        "nmn[linen]",
        "nmn[mlx]",
    ):
        assert extra in out
    assert "nmn guide" in out


def test_version_prints_only_version(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out.strip()
    from nmn import __version__

    assert out == __version__


def test_frameworks_shows_import_and_ctor(capsys):
    assert cli.main(["frameworks"]) == 0
    out = capsys.readouterr().out
    assert "from nmn.torch import YatNMN" in out
    assert "in_features=128, out_features=256" in out  # torch / nnx ctor
    assert "rngs=nnx.Rngs(0)" in out  # nnx-specific kwarg
    assert "from nmn.linen import YatNMN" in out
    assert "units=256" in out  # keras ctor
    assert "from nmn.mlx import YatNMN" in out


@pytest.mark.parametrize(
    "alias,needle",
    [
        ("torch", "from nmn.torch import YatNMN"),
        ("pytorch", "from nmn.torch import YatNMN"),
        ("nnx", "rngs"),
        ("flax-nnx", "rngs"),
        ("linen", "from nmn.linen import YatNMN"),
        ("flax-linen", "from nmn.linen import YatNMN"),
        ("keras", "units="),
        ("tf", "from nmn.tf import YatNMN"),
        ("tensorflow", "from nmn.tf import YatNMN"),
        ("mlx", "from nmn.mlx import YatNMN"),
    ],
)
def test_guide_aliases(alias, needle, capsys):
    assert cli.main(["guide", alias]) == 0
    out = capsys.readouterr().out
    assert needle in out
    assert "Full guide" in out


def test_guide_attention_kwargs_match_signatures(capsys):
    """Every embedded attention quickstart must use the real ctor kwargs.

    keras/tf/mlx -> MultiHeadYatAttention(embed_dim=..., num_heads=...) (NO key_dim)
    linen        -> MultiHeadAttention(num_heads=..., qkv_features=..., out_features=...)
    nnx          -> MultiHeadAttention(num_heads=..., in_features=..., rngs=...)
    torch        -> MultiHeadYatAttention(embed_dim=..., num_heads=...)
    """
    guides = {}
    for fw in ("torch", "nnx", "linen", "keras", "tf", "mlx"):
        assert cli.main(["guide", fw]) == 0
        guides[fw] = capsys.readouterr().out

    # key_dim was the bug: it must not appear in ANY guide.
    for fw, text in guides.items():
        assert "key_dim" not in text, f"{fw} guide still mentions key_dim"

    # keras/tf/mlx and torch -> embed_dim, no in_features in their attention.
    for fw in ("torch", "keras", "tf", "mlx"):
        assert "embed_dim=128" in guides[fw], f"{fw} guide missing embed_dim"

    # linen attention must use qkv_features/out_features, never in_features.
    assert "qkv_features=128" in guides["linen"]
    assert "out_features=128" in guides["linen"]
    assert "in_features=128" not in guides["linen"]
    # The dead `class MLP(YatNMN): pass` snippet must be gone.
    assert "class MLP" not in guides["linen"]

    # nnx attention does take in_features (that is its real signature).
    assert "in_features=128" in guides["nnx"]
    assert "rngs=" in guides["nnx"]


def test_guide_yatnmn_ctor_kwargs_match_signatures(capsys):
    """The embedded YatNMN constructor line must use the per-framework kwarg."""
    expected = {
        "torch": "in_features=",
        "nnx": "in_features=",
        "linen": "features=128",
        "keras": "units=",
        "tf": "features=128",
        "mlx": "features=128",
    }
    for fw, needle in expected.items():
        assert cli.main(["guide", fw]) == 0
        out = capsys.readouterr().out
        assert needle in out, f"{fw} guide missing YatNMN kwarg {needle!r}"
        # keras must NOT use features= (it is `units=`).
        if fw == "keras":
            assert "YatNMN(features=" not in out


def _extract_lines(text, prefix_token):
    """Return source lines (dedented) from a guide containing ``prefix_token``."""
    return [line.strip() for line in text.splitlines() if prefix_token in line]


def test_torch_guide_snippets_construct(capsys):
    """For the torch backend (importable locally), the emitted YatNMN ctor and
    attention lines must actually construct using the exact guide kwargs."""
    try:
        importlib.import_module("torch")
    except Exception:
        pytest.skip("torch backend not importable locally")

    assert cli.main(["guide", "torch"]) == 0
    text = capsys.readouterr().out

    from nmn.torch import MultiHeadYatAttention, YatNMN

    # Exec the YatNMN ctor lines verbatim from the guide.
    for line in _extract_lines(text, "YatNMN(in_features="):
        eval(line.rstrip(","), {"YatNMN": YatNMN})
    line = _extract_lines(text, "MultiHeadYatAttention(embed_dim=")[0]
    eval(
        line.split("=", 1)[1].strip(),
        {"MultiHeadYatAttention": MultiHeadYatAttention},
    )


@pytest.mark.parametrize("fw", ["nnx", "linen"])
def test_guide_attention_ctor_constructs_for_importable(fw, capsys):
    """Construct the attention object from each importable backend's guide
    using the exact kwargs the guide emits."""
    backends = {
        "nnx": ["jax", "flax"],
        "linen": ["jax", "flax"],
    }
    try:
        for m in backends[fw]:
            importlib.import_module(m)
    except Exception:
        pytest.skip(f"{fw} backend not importable locally")

    assert cli.main(["guide", fw]) == 0
    text = capsys.readouterr().out

    if fw == "nnx":
        from flax import nnx

        from nmn.nnx import MultiHeadAttention

        rngs = nnx.Rngs(0)
        line = _extract_lines(text, "MultiHeadAttention(num_heads=")[0]
        expr = line.split("=", 1)[1].strip()
        eval(expr, {"MultiHeadAttention": MultiHeadAttention, "rngs": rngs, "nnx": nnx})
    elif fw == "linen":
        from nmn.linen import MultiHeadAttention

        line = _extract_lines(text, "MultiHeadAttention(num_heads=")[0]
        expr = line.split("=", 1)[1].strip()
        eval(expr, {"MultiHeadAttention": MultiHeadAttention})


def test_mlx_guide_attention_ctor_constructs_in_isolated_process(capsys):
    """Never initialize the optional native MLX runtime in the pytest process."""
    if not _isolated_backend.mlx_is_usable():
        pytest.skip("MLX runtime is not usable in an isolated probe")

    assert cli.main(["guide", "mlx"]) == 0
    text = capsys.readouterr().out
    line = _extract_lines(text, "MultiHeadYatAttention(embed_dim=")[0]
    expr = line.split("=", 1)[1].strip()
    script = "from nmn.mlx import MultiHeadYatAttention\n" f"{expr}\n"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_features_mentions_may_ray_and_lazy(capsys):
    assert cli.main(["features"]) == 0
    out = capsys.readouterr().out
    assert "create_maclaurin_projection" in out
    assert "radial_yat_attention" in out
    assert "performer_kind" in out
    assert "lazy=True" in out
    # canonical kwargs
    assert "bias=" in out
    assert "epsilon=" in out


def test_doctor_lists_all_backends(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    for key in ("torch", "nnx", "linen", "keras", "tf", "mlx"):
        assert key in out
    assert "Python" in out


# ---------------------------------------------------------------------------
# Doctor isolation: optional backends cannot crash or hang the caller.
# ---------------------------------------------------------------------------


def _probe_result(*, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=[sys.executable],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class _FakeProbeProcess:
    def __init__(self, *, returncode=0, stdout=b"", stderr=b""):
        self.pid = 999_999_999
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self, timeout=None):
        del timeout
        return self._stdout, self._stderr


def _patch_fake_windows_job(monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(
            _isolated_backend, "_assign_windows_kill_job", lambda process: None
        )
        monkeypatch.setattr(
            _isolated_backend, "_close_windows_kill_job", lambda process: True
        )


def test_optional_backend_probe_success(monkeypatch):
    _patch_fake_windows_job(monkeypatch)
    process = _FakeProbeProcess(stdout=_isolated_backend._PROBE_MARKER_BYTES + b"\n")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(_isolated_backend.subprocess, "Popen", fake_popen)

    assert _isolated_backend.isolated_import_succeeds(["mlx.core"], readiness="mlx")
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-c", _isolated_backend._PROBE_SCRIPT]
    request = json.loads(command[3])
    assert request["modules"] == ["mlx.core"]
    assert request["readiness"] == "mlx"
    assert ("gate" in request) == (os.name == "nt")
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.PIPE
    if os.name == "posix":
        assert kwargs["start_new_session"] is True
    else:
        assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP


def test_optional_backend_probe_handles_none_and_malformed_output(monkeypatch):
    _patch_fake_windows_job(monkeypatch)
    processes = iter(
        [
            _FakeProbeProcess(stdout=None, stderr=None),
            _FakeProbeProcess(stdout=_isolated_backend._PROBE_MARKER),
            _FakeProbeProcess(
                stdout=b"\xff\xfe\n" + _isolated_backend._PROBE_MARKER_BYTES + b"\n",
                stderr=b"\x80",
            ),
        ]
    )
    monkeypatch.setattr(
        _isolated_backend.subprocess, "Popen", lambda *args, **kwargs: next(processes)
    )
    monkeypatch.setattr(_isolated_backend, "_stop_process_tree", lambda process: None)

    assert not _isolated_backend.isolated_import_succeeds(["unused"])
    assert not _isolated_backend.isolated_import_succeeds(["unused"])
    assert _isolated_backend.isolated_import_succeeds(["unused"])


def test_optional_backend_probe_python_failure_is_unavailable(monkeypatch):
    _patch_fake_windows_job(monkeypatch)
    monkeypatch.setattr(
        _isolated_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProbeProcess(
            returncode=1, stderr=b"RuntimeError: backend initialization failed"
        ),
    )
    monkeypatch.setattr(_isolated_backend, "_stop_process_tree", lambda process: None)
    assert not _isolated_backend.mlx_is_usable()


@pytest.mark.parametrize("returncode", [-signal.SIGABRT, 128 + signal.SIGABRT])
def test_optional_backend_probe_native_failure_is_unavailable(monkeypatch, returncode):
    _patch_fake_windows_job(monkeypatch)
    monkeypatch.setattr(
        _isolated_backend.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProbeProcess(returncode=returncode),
    )
    monkeypatch.setattr(_isolated_backend, "_stop_process_tree", lambda process: None)
    assert not _isolated_backend.mlx_is_usable()


def _prepend_pythonpath(monkeypatch, directory):
    existing = os.environ.get("PYTHONPATH")
    value = str(directory)
    if existing:
        value += os.pathsep + existing
    monkeypatch.setenv("PYTHONPATH", value)


def test_optional_backend_probe_really_imports_in_child(tmp_path, monkeypatch):
    (tmp_path / "healthy_backend.py").write_text("VALUE = 1\n", encoding="utf-8")
    _prepend_pythonpath(monkeypatch, tmp_path)

    assert _isolated_backend.isolated_import_succeeds(["healthy_backend"])


def test_optional_backend_probe_really_contains_python_failure(tmp_path, monkeypatch):
    (tmp_path / "python_failure_backend.py").write_text(
        "raise RuntimeError('initialization failed')\n", encoding="utf-8"
    )
    _prepend_pythonpath(monkeypatch, tmp_path)

    assert not _isolated_backend.isolated_import_succeeds(["python_failure_backend"])


def test_optional_backend_probe_really_contains_native_abort(tmp_path, monkeypatch):
    (tmp_path / "native_abort_backend.py").write_text(
        "import os\nos.abort()\n", encoding="utf-8"
    )
    _prepend_pythonpath(monkeypatch, tmp_path)

    assert not _isolated_backend.isolated_import_succeeds(["native_abort_backend"])


def test_optional_backend_probe_really_handles_invalid_bytes(tmp_path, monkeypatch):
    (tmp_path / "invalid_bytes_backend.py").write_text(
        "import os\nos.write(1, b'\\xff\\xfe\\n')\n" "os.write(2, b'\\x80\\n')\n",
        encoding="utf-8",
    )
    _prepend_pythonpath(monkeypatch, tmp_path)

    assert _isolated_backend.isolated_import_succeeds(["invalid_bytes_backend"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group assertion")
def test_optional_backend_probe_timeout_kills_descendant_group(tmp_path, monkeypatch):
    pid_file = tmp_path / "descendant.pid"
    (tmp_path / "descendant_backend.py").write_text(
        "import pathlib, signal, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "def stop(signum, frame):\n"
        "    child.wait(timeout=2)\n"
        "    raise SystemExit(128 + signum)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    _prepend_pythonpath(monkeypatch, tmp_path)
    monkeypatch.setattr(_isolated_backend, "_PROBE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(_isolated_backend, "_PROBE_REAP_SECONDS", 0.5)

    assert not _isolated_backend.isolated_import_succeeds(["descendant_backend"])
    descendant_pid = int(pid_file.read_text(encoding="utf-8"))

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"probe descendant {descendant_pid} remained alive")


def _windows_pid_is_running(pid):
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == 259
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object assertion")
@pytest.mark.parametrize("termination", ["timeout", "nonzero", "abort"])
def test_optional_backend_probe_windows_job_kills_descendants(
    tmp_path, monkeypatch, termination
):
    pid_file = tmp_path / f"windows-{termination}.pid"
    ending = {
        "timeout": "time.sleep(60)",
        "nonzero": "raise SystemExit(23)",
        "abort": "os.abort()",
    }[termination]
    (tmp_path / "windows_descendant_backend.py").write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        f"{ending}\n",
        encoding="utf-8",
    )
    _prepend_pythonpath(monkeypatch, tmp_path)
    monkeypatch.setattr(_isolated_backend, "_PROBE_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(_isolated_backend, "_PROBE_REAP_SECONDS", 0.5)

    assert not _isolated_backend.isolated_import_succeeds(
        ["windows_descendant_backend"]
    )
    descendant_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _windows_pid_is_running(descendant_pid):
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"job descendant {descendant_pid} remained alive")


def test_doctor_probe_success_ignores_backend_output(monkeypatch):
    marker = cli._DOCTOR_PROBE_MARKER
    completed = _probe_result(
        stdout=f"backend initialization message\n{marker}{json.dumps('2.4.1')}\n"
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return completed

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._probe_backend(["first", "second.core"]) == "2.4.1"
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-c", cli._DOCTOR_PROBE_SCRIPT]
    assert json.loads(command[3]) == ["first", "second.core"]
    assert kwargs["timeout"] == cli._DOCTOR_PROBE_TIMEOUT_SECONDS
    assert kwargs["capture_output"] is True
    assert kwargs["check"] is False


def test_doctor_probe_missing_import_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(
            returncode=1,
            stderr="ModuleNotFoundError: No module named 'optional_backend'",
        ),
    )
    assert cli._probe_backend(["optional_backend"]) is None


def test_doctor_probe_python_exception_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(
            returncode=1, stderr="RuntimeError: backend initialization failed"
        ),
    )
    assert cli._probe_backend(["broken_backend"]) is None


def test_doctor_probe_timeout_is_unavailable(monkeypatch):
    def time_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(cli.subprocess, "run", time_out)
    assert cli._probe_backend(["hanging_backend"]) is None


def test_doctor_probe_nonzero_exit_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(returncode=23),
    )
    assert cli._probe_backend(["exiting_backend"]) is None


def test_doctor_probe_native_abort_is_unavailable(monkeypatch):
    # On POSIX, a negative return code identifies the terminating signal.  A
    # native extension abort therefore remains data in the parent pytest
    # process rather than aborting pytest itself.
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(returncode=-signal.SIGABRT),
    )
    assert cli._probe_backend(["aborting_backend"]) is None


def test_doctor_probe_really_isolates_signal_termination(tmp_path, monkeypatch):
    probe = tmp_path / "signal_backend.py"
    probe.write_text(
        "import os\nimport signal\nos.kill(os.getpid(), signal.SIGTERM)\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH")
    pythonpath = str(tmp_path)
    if existing:
        pythonpath += os.pathsep + existing
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    assert cli._probe_backend(["signal_backend"]) is None


def test_doctor_report_preserves_public_shape(monkeypatch):
    versions = iter(["1", "2", "4", None, "6"])
    monkeypatch.setattr(cli, "_probe_backend", lambda probes: next(versions))

    report = cli._doctor_report()

    assert list(report) == ["torch", "nnx", "linen", "keras", "tf", "mlx"]
    assert report == {
        "torch": "1",
        "nnx": "2",
        "linen": "2",
        "keras": "4",
        "tf": None,
        "mlx": "6",
    }


def test_examples_points_to_examples_md(capsys):
    assert cli.main(["examples"]) == 0
    out = capsys.readouterr().out
    assert "EXAMPLES.md" in out
    assert "nmn guide" in out


# ---------------------------------------------------------------------------
# Programmatic API in nmn/__init__.py
# ---------------------------------------------------------------------------


def test_nmn_help(capsys):
    import nmn

    nmn.help()
    out = capsys.readouterr().out
    assert "nmn[torch]" in out


def test_nmn_doctor_returns_dict(capsys):
    import nmn

    report = nmn.doctor()
    capsys.readouterr()  # drain
    assert isinstance(report, dict)
    assert set(report) == {"torch", "nnx", "linen", "keras", "tf", "mlx"}
    # values are either a version string or None; never raises on missing.
    for value in report.values():
        assert value is None or isinstance(value, str)


def test_nmn_doctor_probes_once_and_renders_same_report(monkeypatch, capsys):
    import nmn

    # ``test_cli_import_is_light`` deliberately reloads this module, so patch
    # the currently registered instance used by ``nmn.doctor``.
    active_cli = nmn.cli

    expected = {
        "torch": "2.0",
        "nnx": None,
        "linen": None,
        "keras": None,
        "tf": None,
        "mlx": None,
    }
    calls = 0

    def fake_report():
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(active_cli, "_doctor_report", fake_report)

    assert nmn.doctor() is expected
    output = capsys.readouterr().out
    assert calls == 1
    assert "torch  PyTorch" in output
    assert "OK       version 2.0" in output
    assert "nnx    Flax NNX" in output
    assert "MISSING" in output
