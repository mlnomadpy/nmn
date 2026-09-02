"""Regression tests for CI/CD workflow policy."""

import json
import re
from pathlib import Path

import nmn

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
DOCUSAURUS = WORKFLOWS.parents[1] / "website" / "docusaurus"
ROOT = WORKFLOWS.parents[1]
CHECKOUT_REF = re.compile(r"actions/checkout@([^\s'\"#]+)")
ACTION_REF = re.compile(r"uses:\s+([^\s@]+)@([^\s#]+)")


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


def test_third_party_actions_are_immutable_and_dependabot_updates_them():
    workflow_paths = [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]
    action_refs = [
        (owner, ref)
        for path in workflow_paths
        for owner, ref in ACTION_REF.findall(path.read_text())
        if not owner.startswith("actions/")
    ]

    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", ref) for _, ref in action_refs)
    assert {owner for owner, _ in action_refs} == {
        "codecov/codecov-action",
        "pypa/gh-action-pypi-publish",
    }
    dependabot = (ROOT / ".github" / "dependabot.yml").read_text()
    assert 'package-ecosystem: "github-actions"' in dependabot


def test_codecov_uses_current_files_input():
    workflow = (WORKFLOWS / "test.yml").read_text()

    assert "        file: ./coverage.xml" not in workflow
    assert workflow.count("        files: ./coverage.xml") == 3


def test_policy_ci_runs_for_every_pull_request_and_push():
    workflow = (WORKFLOWS / "test.yml").read_text()
    trigger = workflow.split("concurrency:", 1)[0]

    assert "pull_request:" in trigger
    assert "push:" in trigger
    assert "paths:" not in trigger


def test_release_and_deployment_permissions_are_job_scoped_and_bounded():
    publish = (WORKFLOWS / "publish.yml").read_text()
    deploy = (WORKFLOWS / "deploy.yml").read_text()
    mirror = (WORKFLOWS / "mirror.yml").read_text()

    publish_global = publish.split("jobs:", 1)[0]
    deploy_global = deploy.split("jobs:", 1)[0]
    publish_build = publish.split("  build:", 1)[1].split("  publish-to-testpypi:", 1)[
        0
    ]
    deploy_build = deploy.split("  build:", 1)[1].split("  deploy:", 1)[0]

    assert "id-token: write" not in publish_global
    assert "id-token: write" not in publish_build
    assert publish.count("id-token: write") == 2
    assert "id-token: write" not in deploy_global
    assert "id-token: write" not in deploy_build
    assert deploy.count("id-token: write") == 1
    assert publish.count("timeout-minutes:") == 3
    assert deploy.count("timeout-minutes:") == 2
    assert mirror.count("timeout-minutes:") == 1


def test_minimum_backend_policy_is_scheduled_and_matches_metadata():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    workflow = (WORKFLOWS / "minimum-versions.yml").read_text()
    docs = (ROOT / "tests" / "README.md").read_text()
    extras = project["project"]["optional-dependencies"]

    assert "schedule:" in workflow and "workflow_dispatch:" in workflow
    assert "torch==1.11.0+cpu" in workflow and "torch>=1.11.0" in extras["torch"]
    assert "tensorflow==2.10.0" in workflow and "tensorflow>=2.10.0" in extras["tf"]
    assert "keras==3.0.0" in workflow and "keras>=3.0.0" in extras["keras"]
    assert "mlx==0.18.1" in workflow and "mlx>=0.18.1" in extras["mlx"]
    assert "native TPU Mosaic and CUDA" in docs
    assert "real Apple Silicon GPU" in docs


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


def test_local_checkout_precedes_any_installed_nmn_package():
    package_path = Path(nmn.__file__).resolve()

    assert package_path.is_relative_to(ROOT / "src")


def test_developer_commands_and_tool_versions_match_ci():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    makefile = (ROOT / "Makefile").read_text()
    precommit = (ROOT / ".pre-commit-config.yaml").read_text()
    dev = set(project["project"]["optional-dependencies"]["dev"])

    assert "build>=1.2.2" in dev
    assert {"black==26.5.1", "isort==9.0.1", "flake8==7.3.0"} <= dev
    assert "$(PYTHON) -m mypy --no-error-summary" in makefile
    assert "rev: 26.5.1" in precommit
    assert "rev: 9.0.1" in precommit
    assert "rev: 7.3.0" in precommit


def test_only_documented_pytest_markers_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert project["tool"]["pytest"]["ini_options"]["markers"] == [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    ]


def test_contribution_templates_cover_every_backend():
    expected = {"nmn.torch", "nmn.nnx", "nmn.linen", "nmn.keras", "nmn.tf", "nmn.mlx"}
    templates = [
        ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml",
        ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.yml",
    ]

    for template in templates:
        contents = template.read_text()
        assert all(backend in contents for backend in expected), template


def test_mypy_checks_the_package_from_one_drift_resistant_config():
    workflow = (WORKFLOWS / "test.yml").read_text()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    config = project["tool"]["mypy"]
    package_files = sorted(
        path
        for path in (ROOT / "src" / "nmn").rglob("*.py")
        if path.name != "_version.py"
    )

    assert config["files"] == ["src/nmn"]
    assert config["follow_imports"] == "skip"
    assert "exclude" not in config
    assert "mypy==2.3.1" in project["project"]["optional-dependencies"]["dev"]
    # hatch-vcs materializes the ignored ``_version.py`` during builds. Count
    # only committed package sources so this invariant is identical in a clean
    # checkout and an already-built developer tree.
    assert len(package_files) >= 98
    assert workflow.count("mypy --no-error-summary") == 1
    mypy_job = workflow.split("  mypy:", 1)[1]
    assert 'pip install -e ".[dev]" "numpy<2.3"' in mypy_job
    assert "src/nmn/torch/" not in mypy_job
    assert "src/nmn/nnx/" not in mypy_job


def test_sdist_excludes_the_local_dacli_evidence_ledger():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    sdist = project["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert "/.dacli" in sdist["exclude"]


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
