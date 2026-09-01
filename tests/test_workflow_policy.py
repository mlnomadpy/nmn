"""Regression tests for CI/CD workflow policy."""

import json
import re
from pathlib import Path

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
DOCUSAURUS = WORKFLOWS.parents[1] / "website" / "docusaurus"


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

    checkout_versions = re.findall(r"actions/checkout@(v\d+)", workflows)
    assert checkout_versions
    assert set(checkout_versions) == {"v7"}
    assert "actions/setup-python@v5" not in workflows
    assert "codecov/codecov-action@v4" not in workflows
    assert "codecov/codecov-action@v5" not in workflows
    assert "actions/upload-artifact@v4" not in workflows
    assert "actions/download-artifact@v4" not in workflows


def test_codecov_uses_current_files_input():
    workflow = (WORKFLOWS / "test.yml").read_text()

    assert "        file: ./coverage.xml" not in workflow
    assert workflow.count("        files: ./coverage.xml") == 3


def test_website_is_built_on_pull_requests_with_node24():
    workflow = (WORKFLOWS / "website.yml").read_text()

    assert "pull_request:" in workflow
    assert "website/docusaurus/**" in workflow
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
