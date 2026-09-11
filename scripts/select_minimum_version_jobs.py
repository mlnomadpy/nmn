"""Select minimum-version CI jobs from pull-request file changes.

The selector is deliberately standard-library-only so the workflow can run it
before installing NMN or any optional machine-learning framework.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

JOBS = ("torch", "tensorflow", "keras", "mlx")

_BACKEND_PREFIXES = {
    "torch": ("src/nmn/torch/", "tests/test_torch/", "tests/test_torch_"),
    "tensorflow": ("src/nmn/tf/", "tests/test_tf/", "tests/test_tf_"),
    "keras": ("src/nmn/keras/", "tests/test_keras/", "tests/test_keras_"),
    "mlx": ("src/nmn/mlx/", "tests/test_mlx/", "tests/test_mlx_"),
}

# Changes to dependency declarations, this policy, shared package modules, or
# cross-framework test support can invalidate every backend's lower bound.
_ALL_JOB_FILES = {
    ".github/workflows/minimum-versions.yml",
    "pyproject.toml",
    "scripts/select_minimum_version_jobs.py",
    "tests/__init__.py",
    "tests/_isolated_backend.py",
    "tests/conftest.py",
}
_ALL_JOB_PREFIXES = ("src/nmn/_", "tests/integration/")


def select_jobs(paths: Iterable[str]) -> set[str]:
    """Return the minimum-version jobs affected by repository-relative *paths*."""

    selected: set[str] = set()
    for raw_path in paths:
        path = raw_path.removeprefix("./")
        if path in _ALL_JOB_FILES or path.startswith(_ALL_JOB_PREFIXES):
            return set(JOBS)
        for job, prefixes in _BACKEND_PREFIXES.items():
            if path.startswith(prefixes):
                selected.add(job)
    return selected


def _changed_paths_from_stdin() -> list[str]:
    data = sys.stdin.buffer.read()
    if not data:
        return []
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in data.rstrip(b"\0").split(b"\0")
    ]


def main(argv: list[str] | None = None) -> int:
    """Write GitHub Actions boolean outputs for every minimum-version job."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="select every job (used by schedules and manual dispatches)",
    )
    args = parser.parse_args(argv)

    selected = set(JOBS) if args.all else select_jobs(_changed_paths_from_stdin())
    for job in JOBS:
        print(f"{job}={'true' if job in selected else 'false'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the workflow
    raise SystemExit(main())
