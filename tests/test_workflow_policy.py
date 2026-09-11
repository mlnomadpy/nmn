"""Regression tests for CI/CD workflow policy."""

import importlib.util
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


def _load_minimum_version_selector():
    path = ROOT / "scripts" / "select_minimum_version_jobs.py"
    spec = importlib.util.spec_from_file_location("minimum_version_selector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_uploads_only_version_tags():
    workflow = (WORKFLOWS / "publish.yml").read_text()
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "branches:" not in trigger_block
    assert "- 'v*.*.*'" in trigger_block
    assert "if: startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "git fetch --no-tags origin master" in workflow
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/master' in workflow


def test_release_integrity_controls_are_documented():
    security = (ROOT / "SECURITY.md").read_text()
    security = " ".join(security.split())

    for control in (
        "`master` branch ruleset",
        "version tags matching `v*.*.*`",
        "secret-scanning push protection",
        "Dependabot security updates",
        "deleted automatically",
        "contained in `origin/master`",
        "protected `pypi` environment",
    ):
        assert control in security


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


def test_codecov_uploads_are_oidc_authenticated_and_failure_blocking():
    workflow = (WORKFLOWS / "test.yml").read_text()
    readme = (ROOT / "README.md").read_text()
    policy = (ROOT / "codecov.yml").read_text()

    assert workflow.count("        use_oidc: true") == 3
    assert workflow.count("        fail_ci_if_error: true") == 3
    assert workflow.count("        disable_search: true") == 3
    assert workflow.count("      id-token: write") == 3
    assert "CODECOV_TOKEN" not in workflow
    for upload in workflow.split("- name: Upload coverage")[1:]:
        assert "continue-on-error" not in upload.split("\n\n", 1)[0]
    assert "branch=master" in readme
    assert "target: auto" in policy
    assert "target: 80%" in policy
    assert "carryforward: true" in policy


def test_local_coverage_policy_is_fail_closed_and_combines_backend_data():
    workflow = (WORKFLOWS / "test.yml").read_text()
    project = (ROOT / "pyproject.toml").read_text()

    assert workflow.count("uses: actions/upload-artifact@v7") == 6
    assert workflow.count("uses: actions/download-artifact@v8") == 6
    assert workflow.count("include-hidden-files: true") == 5
    assert workflow.count("if-no-files-found: error") == 6
    assert "cp .coverage .coverage.jax" in workflow
    assert "cp .coverage .coverage.torch" in workflow
    assert "cp .coverage .coverage.keras" in workflow
    assert "cp .coverage .coverage.mlx" in workflow
    assert "cp .coverage .coverage.conformance" in workflow
    assert (
        "needs: [test-jax, test-torch, test-keras, conformance-linux, test-mlx]"
        in workflow
    )
    assert "name: coverage-conformance" in workflow
    assert "relative_files = true" in project
    assert "NMN_CONFORMANCE_PLATFORM: linux" in workflow
    assert "NMN_CONFORMANCE_PLATFORM: macos" in workflow
    assert "NMN_CONFORMANCE_FIXTURE: conformance-fixtures/dense-v1.npz" in workflow
    assert "name: conformance-fixtures" in workflow
    assert "coverage combine coverage-data" in workflow
    assert "coverage report --fail-under=70" in workflow
    assert "--compare-branch=origin/${{ github.base_ref }}" in workflow
    assert "--fail-under=80" in workflow
    assert '"*/_version.py"' in project
    coverage_job = workflow.split("  coverage-policy:", 1)[1].split(
        "\n  test-keras-multibackend:", 1
    )[0]
    assert "continue-on-error" not in coverage_job


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
    assert deploy.count("timeout-minutes:") == 1
    assert (WORKFLOWS / "website.yml").read_text().count("timeout-minutes:") == 1
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
    assert workflow.count("python -m venv .venv-minimum") == 4
    assert workflow.count(".venv-minimum/bin/python -m pip check") == 4
    assert workflow.count(".venv-minimum/bin/python -m pytest") == 4
    assert "native TPU Mosaic and CUDA" in docs
    assert "real Apple Silicon GPU" in docs


def test_minimum_backend_workflow_triggers_on_affected_sources_and_tests():
    workflow = (WORKFLOWS / "minimum-versions.yml").read_text()
    trigger = workflow.split("permissions:", 1)[0]

    expected_paths = {
        "src/nmn/torch/**",
        "src/nmn/tf/**",
        "src/nmn/keras/**",
        "src/nmn/mlx/**",
        "tests/test_torch/**",
        "tests/test_torch_*.py",
        "tests/test_tf/**",
        "tests/test_tf_*.py",
        "tests/test_keras/**",
        "tests/test_keras_*.py",
        "tests/test_mlx/**",
        "tests/test_mlx_*.py",
    }
    assert all(f"- '{path}'" in trigger for path in expected_paths)
    assert "docs/**" not in trigger
    assert "README.md" not in trigger
    assert "needs: changes" in workflow
    assert workflow.count("if: needs.changes.outputs.") == 4
    assert "python scripts/select_minimum_version_jobs.py" in workflow


def test_minimum_backend_selector_is_precise_and_fail_closed_for_shared_code():
    selector = _load_minimum_version_selector()

    for job, source, test in (
        ("torch", "src/nmn/torch/layers/linear.py", "tests/test_torch/test_basic.py"),
        ("tensorflow", "src/nmn/tf/conv.py", "tests/test_tf/test_basic.py"),
        ("keras", "src/nmn/keras/nmn.py", "tests/test_keras/test_basic.py"),
        ("mlx", "src/nmn/mlx/attention.py", "tests/test_mlx/test_basic.py"),
    ):
        assert selector.select_jobs([source]) == {job}
        assert selector.select_jobs([test]) == {job}

    assert selector.select_jobs(["tests/test_mlx_goat_source.py"]) == {"mlx"}

    assert selector.select_jobs(["docs/guides/pytorch.md"]) == set()
    assert selector.select_jobs(["README.md"]) == set()
    assert selector.select_jobs(["src/nmn/_epsilon.py"]) == set(selector.JOBS)
    assert selector.select_jobs(["tests/integration/test_cross_framework.py"]) == set(
        selector.JOBS
    )
    assert selector.select_jobs(["pyproject.toml"]) == set(selector.JOBS)


def test_jax_ci_covers_minimum_and_latest_dependency_sets():
    workflow = (WORKFLOWS / "test.yml").read_text()
    jax_job = workflow.split("  test-jax:", 1)[1].split("  test-torch:", 1)[0]

    assert jax_job.count("dependencies:") == 2
    assert "dependencies: minimum" in jax_job
    assert "dependencies: latest" in jax_job
    assert '"jax==0.9.1"' in jax_job
    assert 'pip install -e ".[dev,nnx,linen]" optax' in jax_job
    assert "tests/scripts" not in jax_job
    assert "tests/benchmarks" not in jax_job
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
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "benchmark: explicit performance measurements excluded from correctness CI",
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


def test_mirror_uses_repository_scoped_self_sync_and_verifies_every_ref():
    workflow = (WORKFLOWS / "mirror.yml").read_text()
    sync_script = (ROOT / "scripts" / "sync-public-mirror.sh").read_text()

    assert "github.repository == 'mlnomadpy/nmn'" in workflow
    assert "actions/create-github-app-token@v3" in workflow
    assert "client-id: ${{ vars.MIRROR_APP_CLIENT_ID }}" in workflow
    assert "private-key: ${{ secrets.MIRROR_APP_PRIVATE_KEY }}" in workflow
    assert "permission-contents: write" in workflow
    assert "permission-workflows: write" in workflow
    token_step = workflow.split("id: mirror-token", 1)[1].split("\n\n", 1)[0]
    assert "owner:" not in token_step
    assert "repositories:" not in token_step
    assert "persist-credentials: false" in workflow
    assert "MIRROR_PUSH_REMOTE: https://x-access-token:" in workflow
    assert "${{ github.repository }}.git" in workflow
    assert workflow.count("${{ steps.mirror-token.outputs.token }}") == 1
    assert 'mirror_push_remote="${MIRROR_PUSH_REMOTE:-${mirror_remote}}"' in sync_script
    assert workflow.count("contents: read") == 1
    assert "\n      contents: write\n" not in workflow
    assert "MIRROR_PAT" not in workflow
    assert "DEPLOY_KEY" not in workflow
    assert "https://github.com/azettaai/nmn.git" in workflow
    assert "bash scripts/sync-public-mirror.sh" in workflow
    assert "git merge-base --is-ancestor" in sync_script
    assert 'git push --atomic "${mirror_push_remote}"' in sync_script
    assert '"${canonical_ref}:refs/heads/${branch}" --tags' in sync_script
    assert "git ls-remote" in sync_script
    assert "mirrored_head" in sync_script
    assert "mirrored_tag" in sync_script
    assert "continue-on-error" not in workflow


def test_website_is_built_on_pull_requests_with_node24():
    workflow = (WORKFLOWS / "website.yml").read_text()

    assert "pull_request:" in workflow
    assert workflow.count("website/**") == 2
    assert "node-version: '24'" in workflow
    assert "npm ci" in workflow
    assert "bash website/prepare-docusaurus-static.sh" in workflow
    assert "npm run build" in workflow
    assert "workflow_call:" in workflow

    config = (DOCUSAURUS / "docusaurus.config.js").read_text()
    assert "onBrokenLinks: 'throw'" in config
    assert "onBrokenMarkdownLinks: 'throw'" in config

    deploy_workflow = (WORKFLOWS / "deploy.yml").read_text()
    assert "uses: ./.github/workflows/website.yml" in deploy_workflow


def test_deployment_consumes_the_fail_closed_audited_website_artifact():
    website = (WORKFLOWS / "website.yml").read_text()
    deploy = (WORKFLOWS / "deploy.yml").read_text()
    build_steps = website.split("    steps:", 1)[1]

    audit = build_steps.index("run: npm run audit:ci")
    prepare = build_steps.index("run: bash website/prepare-docusaurus-static.sh")
    build = build_steps.index("run: npm run build")
    upload = build_steps.index("uses: actions/upload-pages-artifact@v5")

    assert audit < prepare < build < upload
    assert "cache-dependency-path: website/docusaurus/package-lock.json" in website
    assert "continue-on-error" not in build_steps
    assert "if: always()" not in build_steps

    deploy_build = deploy.split("  build:", 1)[1].split("\n  deploy:", 1)[0]
    deploy_job = deploy.split("\n  deploy:", 1)[1]
    assert "uses: ./.github/workflows/website.yml" in deploy_build
    assert "needs: build" in deploy_job
    assert "npm ci" not in deploy
    assert "npm run build" not in deploy
    assert "actions/upload-pages-artifact" not in deploy
    assert "'.github/workflows/website.yml'" in deploy


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


def test_website_dependency_audit_has_exact_reviewed_allowlist():
    manifest = json.loads((DOCUSAURUS / "package.json").read_text())
    lockfile = json.loads((DOCUSAURUS / "package-lock.json").read_text())
    workflow = (WORKFLOWS / "website.yml").read_text()
    script = (ROOT / "website" / "audit-dependencies.mjs").read_text()
    rationale = (ROOT / "website" / "DEPENDENCY_SECURITY.md").read_text()

    assert manifest["scripts"]["audit:ci"] == "node ../audit-dependencies.mjs"
    assert manifest["overrides"] == {
        "qs": "6.16.0",
        "serialize-javascript": "7.1.1",
        "uuid": "11.1.1",
    }
    assert (
        lockfile["packages"]["node_modules/serialize-javascript"]["version"] == "7.1.1"
    )
    assert lockfile["packages"]["node_modules/qs"]["version"] == "6.16.0"
    assert lockfile["packages"]["node_modules/uuid"]["version"] == "11.1.1"
    assert "npm run audit:ci" in workflow
    assert set(re.findall(r"'(GHSA-[\w-]+)'", script)) == {
        "GHSA-w3rx-r6r6-pgpr",
        "GHSA-5p2g-fcmc-qvqq",
    }
    assert "critical > 0" in script
    assert "Unreviewed high advisories" in script
    assert "report.error" in script
    assert "did not return a vulnerability report" in script
    assert "reviewed repository images" in rationale
