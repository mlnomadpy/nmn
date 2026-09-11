"""Local integration regressions for the public-mirror sync transaction."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SYNC_SCRIPT = ROOT / "scripts" / "sync-public-mirror.sh"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repositories(tmp_path):
    canonical = tmp_path / "canonical.git"
    mirror = tmp_path / "mirror.git"
    author = tmp_path / "author"
    runner = tmp_path / "runner"
    _git(tmp_path, "init", "--bare", str(canonical))
    _git(tmp_path, "init", "--bare", str(mirror))
    _git(canonical, "symbolic-ref", "HEAD", "refs/heads/master")
    _git(mirror, "symbolic-ref", "HEAD", "refs/heads/master")
    _git(tmp_path, "init", "-b", "master", str(author))
    _git(author, "config", "user.name", "Mirror Test")
    _git(author, "config", "user.email", "mirror@example.invalid")
    (author / "README.md").write_text("base\n")
    _git(author, "add", "README.md")
    _git(author, "commit", "-m", "base")
    _git(author, "tag", "-a", "v0.3.5", "-m", "v0.3.5")
    _git(author, "remote", "add", "canonical", str(canonical))
    _git(author, "remote", "add", "mirror", str(mirror))
    _git(author, "push", "canonical", "master", "--tags")
    _git(author, "push", "mirror", "master", "--tags")
    _git(tmp_path, "clone", str(mirror), str(runner))
    _git(runner, "remote", "add", "canonical", str(canonical))
    return canonical, mirror, author, runner


def _sync(runner: Path, *, check: bool = True):
    environment = os.environ.copy()
    environment.update(
        CANONICAL_REMOTE="canonical", MIRROR_REMOTE="origin", MIRROR_BRANCH="master"
    )
    return subprocess.run(
        ["bash", str(SYNC_SCRIPT)],
        cwd=runner,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def _remote_ref(repository: Path, ref: str) -> str:
    return _git(repository, "rev-parse", ref).stdout.strip()


def test_sync_fast_forwards_across_workflow_file_change(repositories):
    canonical, mirror, author, runner = repositories
    workflow = author / ".github" / "workflows" / "changed.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: changed\non: workflow_dispatch\n")
    _git(author, "add", str(workflow.relative_to(author)))
    _git(author, "commit", "-m", "change workflow")
    _git(author, "push", "canonical", "master")

    _sync(runner)

    assert _remote_ref(mirror, "refs/heads/master") == _remote_ref(
        canonical, "refs/heads/master"
    )
    assert _remote_ref(mirror, "refs/tags/v0.3.5") == _remote_ref(
        canonical, "refs/tags/v0.3.5"
    )


def test_sync_refuses_divergent_master(repositories):
    _canonical, mirror, author, runner = repositories
    (author / "canonical.txt").write_text("canonical\n")
    _git(author, "add", "canonical.txt")
    _git(author, "commit", "-m", "canonical change")
    _git(author, "push", "canonical", "master")

    mirror_writer = runner.parent / "mirror-writer"
    _git(runner.parent, "clone", str(mirror), str(mirror_writer))
    _git(mirror_writer, "config", "user.name", "Mirror Test")
    _git(mirror_writer, "config", "user.email", "mirror@example.invalid")
    (mirror_writer / "mirror.txt").write_text("mirror\n")
    _git(mirror_writer, "add", "mirror.txt")
    _git(mirror_writer, "commit", "-m", "divergent mirror change")
    _git(mirror_writer, "push", "origin", "master")
    _git(runner, "fetch", "origin", "master")
    before = _remote_ref(mirror, "refs/heads/master")

    result = _sync(runner, check=False)

    assert result.returncode != 0
    assert "has diverged" in result.stderr
    assert _remote_ref(mirror, "refs/heads/master") == before


def test_sync_refuses_tag_rewrite_before_branch_push(repositories):
    _canonical, mirror, author, runner = repositories
    (author / "canonical.txt").write_text("canonical\n")
    _git(author, "add", "canonical.txt")
    _git(author, "commit", "-m", "canonical change")
    _git(author, "tag", "-f", "-a", "v0.3.5", "-m", "rewritten v0.3.5")
    _git(author, "push", "canonical", "master")
    _git(author, "push", "--force", "canonical", "refs/tags/v0.3.5")
    branch_before = _remote_ref(mirror, "refs/heads/master")
    tag_before = _remote_ref(mirror, "refs/tags/v0.3.5")

    result = _sync(runner, check=False)

    assert result.returncode != 0
    assert _remote_ref(mirror, "refs/heads/master") == branch_before
    assert _remote_ref(mirror, "refs/tags/v0.3.5") == tag_before


def test_sync_atomically_refuses_branch_when_server_rejects_tag(repositories):
    _canonical, mirror, author, runner = repositories
    (author / "canonical.txt").write_text("canonical\n")
    _git(author, "add", "canonical.txt")
    _git(author, "commit", "-m", "canonical change")
    _git(author, "tag", "-a", "v0.3.6", "-m", "v0.3.6")
    _git(author, "push", "canonical", "master", "--tags")
    branch_before = _remote_ref(mirror, "refs/heads/master")

    hook = mirror / "hooks" / "pre-receive"
    hook.write_text(
        "#!/usr/bin/env bash\n"
        "while read -r old new ref; do\n"
        '  if [ "${ref}" = refs/tags/v0.3.6 ]; then exit 1; fi\n'
        "done\n"
    )
    hook.chmod(0o755)

    result = _sync(runner, check=False)

    assert result.returncode != 0
    assert _remote_ref(mirror, "refs/heads/master") == branch_before
    assert (
        _git(mirror, "show-ref", "--verify", "refs/tags/v0.3.6", check=False).returncode
        != 0
    )
