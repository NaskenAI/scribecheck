"""The Gate 2 runbook's guard (docs/runbooks/gate2_guard.sh), in a throwaway repository.

Only the sourced guard functions run here, never the runbook itself, and nothing reads
benchmark data. Skipped where bash or git is missing (Windows runners).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).parent.parent / "docs" / "runbooks" / "gate2_guard.sh"
PREREG = "docs/evidence/gate2_preregistration.md"
TAG = "m4-candidate-v4"
DIGEST = "a" * 64

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("git") is None,
    reason="needs bash and git",
)


def _env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE" and not k.startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.org",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.org",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    return env


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=repo,
        env=_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def write(repo: Path, path: str, text: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def commit(repo: Path, *paths: str, message: str = "c") -> str:
    git(repo, "add", *paths)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def guard(repo: Path, script: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{GUARD}"; {script}'],
        cwd=repo,
        env={**_env(), **env},
        capture_output=True,
        text=True,
    )


CHECK = f'GATE2_PREREG_FILE={PREREG}; gate2_check_git {PREREG} {TAG}; echo "prereg=$GATE2_PREREG"'


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Tagged code, a later script change, then the pre-registration commit on its own."""
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/pkg/a.py", "x = 1\n")
    write(tmp_path, "docs/runbooks/run.sh", "echo run\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    write(tmp_path, "scripts/score.py", "y = 2\n")
    commit(tmp_path, ".")
    write(tmp_path, PREREG, f"# Pre-registration\n\nheld-out-archive-sha256: {DIGEST}\n")
    commit(tmp_path, PREREG, message="prereg")
    return tmp_path


def test_head_at_the_preregistration_commit_passes(repo: Path) -> None:
    done = guard(repo, CHECK)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"prereg={git(repo, 'rev-parse', 'HEAD')}"


def test_a_docs_only_descendant_passes(repo: Path) -> None:
    prereg = git(repo, "rev-parse", "HEAD")
    write(repo, "docs/evidence/notes.md", "later\n")
    write(repo, PREREG, f"# Pre-registration\n\nheld-out-archive-sha256: {DIGEST}\n\nDeviation.\n")
    commit(repo, ".")
    done = guard(repo, CHECK)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"prereg={prereg}"


@pytest.mark.parametrize(
    "path", ["scripts/score.py", "src/pkg/a.py", "docs/runbooks/run.sh", "README.md"]
)
def test_a_change_outside_docs_after_the_preregistration_refuses(repo: Path, path: str) -> None:
    write(repo, path, "changed\n")
    commit(repo, ".")
    done = guard(repo, CHECK)
    assert done.returncode != 0
    assert f"{path} changed after the pre-registration commit" in done.stderr


def test_uncommitted_changes_refuse(repo: Path) -> None:
    write(repo, "src/pkg/a.py", "x = 3\n")
    done = guard(repo, CHECK)
    assert done.returncode != 0 and "uncommitted" in done.stderr
    git(repo, "add", ".")
    done = guard(repo, CHECK)
    assert done.returncode != 0 and "uncommitted" in done.stderr


def test_untracked_files_do_not_refuse(repo: Path) -> None:
    write(repo, "docs/evidence/gate2_claude.txt", "counts\n")  # the run's own output
    assert guard(repo, CHECK).returncode == 0


def test_no_preregistration_refuses(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "x\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    done = guard(tmp_path, CHECK)
    assert done.returncode != 0 and "commit the pre-registration first" in done.stderr


def test_a_preregistration_commit_with_other_files_refuses(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "x\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    write(tmp_path, PREREG, "p\n")
    write(tmp_path, "scripts/b.py", "y\n")
    commit(tmp_path, ".")
    done = guard(tmp_path, CHECK)
    assert done.returncode != 0 and "changes more than" in done.stderr


def test_a_preregistration_on_another_branch_refuses(repo: Path) -> None:
    git(repo, "checkout", "-q", "-b", "side", "HEAD~1")  # before the pre-registration
    write(repo, "docs/x.md", "x\n")
    commit(repo, "docs/x.md")
    done = guard(repo, CHECK)
    assert done.returncode != 0 and "commit the pre-registration first" in done.stderr


def test_src_differing_from_the_tag_refuses(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "x\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    write(tmp_path, "src/a.py", "changed before the pre-registration\n")
    commit(tmp_path, ".")
    write(tmp_path, PREREG, "p\n")
    commit(tmp_path, PREREG)
    done = guard(tmp_path, CHECK)
    assert done.returncode != 0 and f"src/ differs from {TAG}" in done.stderr


def test_a_missing_tag_refuses(repo: Path) -> None:
    git(repo, "tag", "-d", TAG)
    done = guard(repo, CHECK)
    assert done.returncode != 0 and f"no tag {TAG}" in done.stderr


def test_frozen_values_come_from_the_preregistration_commit(repo: Path) -> None:
    write(repo, PREREG, f"held-out-archive-sha256: {'b' * 64}\n")
    commit(repo, PREREG)  # a later docs-only edit cannot change a frozen value
    done = guard(repo, f"{CHECK} > /dev/null; gate2_frozen held-out-archive-sha256")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == DIGEST
    done = guard(repo, f"{CHECK} > /dev/null; gate2_frozen raw-archive-sha256")
    assert done.returncode != 0 and "no single 'raw-archive-sha256: <hex>' line" in done.stderr


def test_check_digest(tmp_path: Path) -> None:
    import hashlib

    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "x\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    good = hashlib.sha256(b"ciphertext").hexdigest()
    write(tmp_path, PREREG, f"held-out-archive-sha256: {good}\nraw-archive-sha256: {DIGEST}\n")
    commit(tmp_path, PREREG)
    (tmp_path / "held.gpg").write_bytes(b"ciphertext")
    (tmp_path / "raw.gpg").write_bytes(b"other")
    ok = guard(
        tmp_path, f"{CHECK} > /dev/null; gate2_check_digest held.gpg held-out-archive-sha256"
    )
    assert ok.returncode == 0, ok.stderr
    bad = guard(tmp_path, f"{CHECK} > /dev/null; gate2_check_digest raw.gpg raw-archive-sha256")
    assert bad.returncode != 0 and "not the pre-registered" in bad.stderr
    assert "raw.gpg" not in bad.stderr  # names the value, not the file


def test_sha256(repo: Path) -> None:
    import hashlib

    (repo / "f.bin").write_bytes(b"abc")
    done = guard(repo, "gate2_sha256 f.bin")
    assert done.stdout.strip() == hashlib.sha256(b"abc").hexdigest()


def test_refuses_inside_claude_code(repo: Path) -> None:
    done = guard(repo, "gate2_refuse_agent", CLAUDECODE="1")
    assert done.returncode != 0 and "never by Claude Code" in done.stderr
    assert guard(repo, "gate2_refuse_agent").returncode == 0


@pytest.mark.parametrize("path", ["scripts/json.py", "src/pkg/b.py", "docs/runbooks/extra.toml"])
def test_an_untracked_file_among_the_code_refuses(repo: Path, path: str) -> None:
    write(repo, path, "shadow\n")
    done = guard(repo, CHECK)
    assert done.returncode != 0 and f"{path} is not committed" in done.stderr


def test_an_ignored_module_among_the_code_refuses_but_caches_do_not(repo: Path) -> None:
    write(repo, ".git/info/exclude", "scripts/local.py\n")
    write(repo, "scripts/__pycache__/score.cpython-311.pyc", "cache\n")
    assert guard(repo, CHECK).returncode == 0
    write(repo, "scripts/local.py", "shadow\n")
    done = guard(repo, CHECK)
    assert done.returncode != 0 and "scripts/local.py is not committed" in done.stderr


def _pinned(repo: Path) -> None:
    tagged = git(repo, "rev-parse", f"{TAG}^{{commit}}")
    write(repo, PREREG, f"{TAG}-commit: {tagged}\r\nheld-out-archive-sha256: {DIGEST}\r\n")


def test_the_tag_must_point_at_the_preregistered_commit(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "src/a.py", "x\n")
    commit(tmp_path, ".")
    git(tmp_path, "tag", TAG)
    _pinned(tmp_path)  # with CRLF line ends, which the guard tolerates
    commit(tmp_path, PREREG)
    check_tag = f"{CHECK} > /dev/null; gate2_check_tag {TAG}"
    done = guard(tmp_path, check_tag)
    assert done.returncode == 0, done.stderr
    # Re-pointing the tag at another commit with the same src/ is refused.
    write(tmp_path, "docs/x.md", "x\n")
    commit(tmp_path, "docs/x.md")
    git(tmp_path, "tag", "-f", TAG, "HEAD")
    done = guard(tmp_path, check_tag)
    assert done.returncode != 0 and f"tag {TAG} points at" in done.stderr
