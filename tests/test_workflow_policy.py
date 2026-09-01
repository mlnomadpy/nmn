"""Regression tests for CI/CD workflow policy."""

from pathlib import Path

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


def test_publish_uploads_only_version_tags():
    workflow = (WORKFLOWS / "publish.yml").read_text()
    trigger_block = workflow.split("permissions:", 1)[0]

    assert "branches:" not in trigger_block
    assert "- 'v*.*.*'" in trigger_block
    assert "if: startsWith(github.ref, 'refs/tags/v')" in workflow


def test_ci_actions_use_node24_compatible_releases():
    workflows = "\n".join(path.read_text() for path in WORKFLOWS.glob("*.yml"))

    assert "actions/setup-python@v5" not in workflows
    assert "codecov/codecov-action@v4" not in workflows
    assert "codecov/codecov-action@v5" not in workflows
    assert "actions/upload-artifact@v4" not in workflows
    assert "actions/download-artifact@v4" not in workflows


def test_codecov_uses_current_files_input():
    workflow = (WORKFLOWS / "test.yml").read_text()

    assert "        file: ./coverage.xml" not in workflow
    assert workflow.count("        files: ./coverage.xml") == 3
