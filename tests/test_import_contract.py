"""Tests for the dependency-light package and optional-backend boundaries."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nmn import _optional

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src"


@pytest.fixture(scope="module")
def base_only_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a Python executable in an environment with no installed extras."""
    environment = tmp_path_factory.mktemp("base-only") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
        capture_output=True,
        text=True,
    )
    executable = environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    assert executable.is_file()
    return executable


def _run_base_only(python: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE)
    return subprocess.run(
        [str(python), *arguments],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_root_imports_and_cli_work_without_frameworks(base_only_python: Path) -> None:
    commands = [
        (
            "-c",
            "import nmn, sys; "
            "assert nmn.__all__ == ['__version__', 'help', 'doctor']; "
            "assert not {'torch', 'jax', 'flax', 'keras', 'tensorflow', 'mlx'} "
            "& sys.modules.keys()",
        ),
        (
            "-c",
            "from nmn import *; "
            "assert callable(help) and callable(doctor) and isinstance(__version__, str)",
        ),
        ("-m", "nmn", "info"),
        ("-m", "nmn", "doctor"),
    ]

    for command in commands:
        completed = _run_base_only(base_only_python, *command)
        assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("backend", "dependency", "extra"),
    [
        ("torch", "torch", "torch"),
        ("nnx", "jax", "nnx"),
        ("linen", "jax", "linen"),
        ("keras", "keras", "keras"),
        ("tf", "tensorflow", "tf"),
        ("mlx", "mlx", "mlx"),
    ],
)
def test_missing_backend_dependency_has_install_guidance(
    base_only_python: Path, backend: str, dependency: str, extra: str
) -> None:
    completed = _run_base_only(base_only_python, "-c", f"import nmn.{backend}")

    assert completed.returncode != 0
    assert f"optional dependency '{dependency}'" in completed.stderr
    assert f'pip install "nmn[{extra}]"' in completed.stderr


def test_dependency_guard_only_translates_a_confirmed_absence(monkeypatch) -> None:
    monkeypatch.setattr(_optional, "find_spec", lambda name: object())

    assert (
        _optional.require_optional_dependency(
            "installed_framework", backend="Example", extra="example"
        )
        is None
    )


@pytest.mark.parametrize("backend", ["keras", "tf", "linen", "mlx"])
def test_backend_initializer_does_not_catch_implementation_import_errors(
    backend: str,
) -> None:
    """An installed backend's internal import defects must reach the caller."""
    initializer = SOURCE / "nmn" / backend / "__init__.py"
    tree = ast.parse(initializer.read_text(encoding="utf-8"))

    caught_names = {
        handler.type.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if isinstance(handler.type, ast.Name)
    }
    assert "ImportError" not in caught_names
    assert "ModuleNotFoundError" not in caught_names


def test_missing_dependency_error_preserves_module_name(monkeypatch) -> None:
    monkeypatch.setattr(_optional, "find_spec", lambda name: None)

    with pytest.raises(ModuleNotFoundError) as caught:
        _optional.require_optional_dependency(
            "example_framework", backend="Example", extra="example"
        )

    assert caught.value.name == "example_framework"
    assert 'pip install "nmn[example]"' in str(caught.value)
