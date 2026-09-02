"""Offline policy checks for user- and agent-facing installation docs."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
OPTIONAL_DEPENDENCIES = PYPROJECT["project"]["optional-dependencies"]
NNX_README = (ROOT / "src/nmn/nnx/README.md").read_text(encoding="utf-8")
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _minimum_version(extra: str, package: str) -> str:
    requirement = next(
        item
        for item in OPTIONAL_DEPENDENCIES[extra]
        if re.match(rf"{re.escape(package)}(?:\[.*\])?>=", item, re.IGNORECASE)
    )
    return requirement.split(">=", maxsplit=1)[1]


def test_nnx_readme_only_advertises_defined_project_extras():
    documented = re.findall(
        r"pip install(?: --upgrade)?(?: -e)? [\"']?\.\[([^\]]+)\]", NNX_README
    )
    assert documented, "expected at least one local editable-install example"

    undefined = {
        extra.strip()
        for group in documented
        for extra in group.split(",")
        if extra.strip() not in OPTIONAL_DEPENDENCIES
    }
    assert not undefined, f"undefined project extras in NNX README: {sorted(undefined)}"


def test_nnx_accelerator_commands_use_official_jax_extras():
    jax_minimum = _minimum_version("nnx", "jax")
    assert f'"jax[tpu]>={jax_minimum}"' in NNX_README
    assert f'"jax[cuda13]>={jax_minimum}"' in NNX_README
    assert '".[tpu]"' not in NNX_README
    assert '".[gpu]"' not in NNX_README


def test_nnx_dependency_minimums_follow_project_metadata():
    assert f"JAX >= {_minimum_version('nnx', 'jax')}" in NNX_README
    assert f"Flax >= {_minimum_version('nnx', 'flax')}" in NNX_README


def test_optional_grain_example_documents_its_install():
    assert "pip install grain" in NNX_README


def test_agent_docs_link_to_release_sources_instead_of_hard_coding_version():
    for relative_path in ("AGENTS.md", "llms.txt"):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "Current release:" not in content
        assert "https://pypi.org/project/nmn/" in content
        assert "https://github.com/azettaai/nmn/releases" in content


def test_changelog_has_unreleased_and_descending_dated_releases():
    headers = re.findall(
        r"^## \[([^\]]+)\](?: — (\d{4}-\d{2}-\d{2}))?$",
        CHANGELOG,
        re.MULTILINE,
    )
    assert headers and headers[0] == ("Unreleased", "")

    releases = headers[1:]
    versions = [version for version, _date in releases]
    assert len(versions) == len(set(versions))
    assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in versions)
    assert all(date for _version, date in releases)

    version_tuples = [tuple(map(int, version.split("."))) for version in versions]
    assert version_tuples == sorted(version_tuples, reverse=True)
