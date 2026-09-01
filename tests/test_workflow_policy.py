"""Regression tests for CI/CD workflow policy."""

import json
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
DOCUSAURUS = WORKFLOWS.parents[1] / "website" / "docusaurus"
ROOT = WORKFLOWS.parents[1]
CHECKOUT_REF = re.compile(r"actions/checkout@([^\s'\"#]+)")


def test_publish_uploads_only_version_tags():
    workflow = (WORKFLOWS / "publish.yml").read_text()
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "branches:" not in trigger_block
    assert "- 'v*.*.*'" in trigger_block
    assert "if: startsWith(github.ref, 'refs/tags/v')" in workflow


def test_ci_actions_use_node24_compatible_releases():
    workflow_paths = [
        *WORKFLOWS.glob("*.yml"),
        *WORKFLOWS.glob("*.yaml"),
    ]
    workflows = "\n".join(path.read_text() for path in workflow_paths)

    checkout_versions = CHECKOUT_REF.findall(workflows)
    assert checkout_versions
    assert set(checkout_versions) == {"v7"}
    assert CHECKOUT_REF.findall("uses: actions/checkout@main") == ["main"]
    assert "actions/setup-python@v5" not in workflows
    assert "codecov/codecov-action@v4" not in workflows
    assert "codecov/codecov-action@v5" not in workflows
    assert "actions/upload-artifact@v4" not in workflows
    assert "actions/download-artifact@v4" not in workflows


def test_codecov_uses_current_files_input():
    workflow = (WORKFLOWS / "test.yml").read_text()

    assert "        file: ./coverage.xml" not in workflow
    assert workflow.count("        files: ./coverage.xml") == 3


def test_jax_ci_covers_minimum_and_latest_dependency_sets():
    workflow = (WORKFLOWS / "test.yml").read_text()
    jax_job = workflow.split("  test-jax:", 1)[1].split("  test-torch:", 1)[0]

    assert jax_job.count("dependencies:") == 2
    assert "dependencies: minimum" in jax_job
    assert "dependencies: latest" in jax_job
    assert '"jax==0.9.1"' in jax_job
    assert 'pip install -e ".[dev,nnx,linen]" optax' in jax_job
    assert "tests/scripts" not in jax_job
    assert "Run minimum-version JAX tests with coverage" in jax_job
    assert "Run latest-version JAX backend tests" in jax_job
    assert jax_job.count("--cov=nmn") == 1


def test_clean_checkout_jobs_do_not_delete_python_caches():
    workflow = (WORKFLOWS / "test.yml").read_text()

    assert "Clear pycache" not in workflow
    assert 'find . -name "*.pyc"' not in workflow


def test_lint_toolchain_is_reproducible_and_skips_generated_version_file():
    workflow = (WORKFLOWS / "test.yml").read_text()
    project = (WORKFLOWS.parents[1] / "pyproject.toml").read_text()

    assert '"flake8==7.3.0"' in workflow
    assert '"black==26.5.1"' in workflow
    assert '"isort==9.0.1"' in workflow
    assert "extend-exclude = 'src/nmn/_version\\.py'" in project


def test_mypy_checks_the_package_from_one_drift_resistant_config():
    workflow = (WORKFLOWS / "test.yml").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    config = project["tool"]["mypy"]
    package_files = sorted((ROOT / "src" / "nmn").rglob("*.py"))

    assert config["files"] == ["src/nmn"]
    assert "follow_imports" not in config
    assert "exclude" not in config
    assert "mypy==2.3.1" in project["project"]["optional-dependencies"]["dev"]
    assert len(package_files) >= 99
    assert workflow.count("mypy --no-error-summary") == 1
    mypy_job = workflow.split("  mypy:", 1)[1]
    assert 'pip install -e ".[dev]" "numpy<2.3"' in mypy_job
    assert "src/nmn/torch/" not in mypy_job
    assert "src/nmn/nnx/" not in mypy_job


def test_mirror_fails_and_verifies_when_sync_is_unavailable():
    workflow = (WORKFLOWS / "mirror.yml").read_text()

    assert 'if [ -z "${MIRROR_PAT}" ]; then' in workflow
    assert "exit 0" not in workflow
    assert "exit 1" in workflow
    assert "git ls-remote" in workflow
    assert "mirrored_head" in workflow


def test_website_is_built_on_pull_requests_with_node24():
    workflow = (WORKFLOWS / "website.yml").read_text()

    assert "pull_request:" in workflow
    assert workflow.count("website/**") == 2
    assert "node-version: '24'" in workflow
    assert "npm ci" in workflow
    assert "bash website/prepare-docusaurus-static.sh" in workflow
    assert "npm run build" in workflow

    config = (DOCUSAURUS / "docusaurus.config.js").read_text()
    assert "onBrokenLinks: 'throw'" in config
    assert "onBrokenMarkdownLinks: 'throw'" in config

    deploy_workflow = (WORKFLOWS / "deploy.yml").read_text()
    assert "bash website/prepare-docusaurus-static.sh" in deploy_workflow


def test_website_manifest_and_lockfile_use_coherent_versions():
    manifest = json.loads((DOCUSAURUS / "package.json").read_text())
    lockfile = json.loads((DOCUSAURUS / "package-lock.json").read_text())
    expected = {
        "@docusaurus/core": "3.10.2",
        "@docusaurus/preset-classic": "3.10.2",
        "@docusaurus/module-type-aliases": "3.10.2",
        "@docusaurus/types": "3.10.2",
        "react": "19.2.8",
        "react-dom": "19.2.8",
    }
    manifest_packages = manifest["dependencies"] | manifest["devDependencies"]
    locked_packages = lockfile["packages"]
    locked_root = locked_packages[""]

    for package, version in expected.items():
        assert manifest_packages[package] == version
        locked_requirement = locked_root["dependencies"].get(package)
        locked_requirement = locked_requirement or locked_root["devDependencies"].get(
            package
        )
        assert locked_requirement == version
        assert locked_packages[f"node_modules/{package}"]["version"] == version
