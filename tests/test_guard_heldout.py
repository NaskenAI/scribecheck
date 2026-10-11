"""The PreToolUse guard that keeps agents away from held-out benchmark data.

Folder names come from the hook's own constants, so this file never names a protected
folder (the guard would block writing it). Every run points the home folder and both
environment variables into a temporary directory, so the real folders are never involved.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_heldout.py"
VARIABLES = ("SCRIBECHECK_HELDOUT_DIR", "SCRIBECHECK_RAW_DIR")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("guard_heldout", HOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = _load()
PROTECTED: tuple[tuple[str, tuple[str, ...]], ...] = GUARD.PROTECTED


def _env(home: Path, **overrides: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    env.update(HOME=str(home), USERPROFILE=str(home), **overrides)
    return env


def _run(home: Path, payload: object, **overrides: str) -> subprocess.CompletedProcess[str]:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=data,
        capture_output=True,
        text=True,
        env=_env(home, **overrides),
        check=False,
    )


def _blocked(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode == 2:
        assert "Blocked" in result.stderr and "held-out" in result.stderr
        return True
    assert result.returncode == 0, result.stderr
    return False


def test_both_folders_are_protected() -> None:
    assert [variable for variable, _ in PROTECTED] == list(VARIABLES)
    assert GUARD.RAW_DEFAULT_PARTS[0] == "data"
    assert GUARD.heldout_markers is GUARD.protected_markers


@pytest.mark.parametrize(("variable", "default"), PROTECTED, ids=VARIABLES)
def test_default_path_resolves_under_home(
    variable: str, default: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert os.path.normpath(os.path.join(tmp_path, *default)) in GUARD.protected_folders()


@pytest.mark.parametrize(("variable", "default"), PROTECTED, ids=VARIABLES)
def test_default_folder_is_blocked(variable: str, default: tuple[str, ...], tmp_path: Path) -> None:
    folder = os.path.join(tmp_path, *default)
    read = {"tool_name": "Read", "tool_input": {"file_path": os.path.join(folder, "a.json")}}
    assert _blocked(_run(tmp_path, read))
    bash = {"tool_name": "Bash", "tool_input": {"command": f"ls ../{default[-1]}/"}}
    assert _blocked(_run(tmp_path, bash))
    grep = {"tool_name": "Grep", "tool_input": {"pattern": "x", "path": folder}}
    assert _blocked(_run(tmp_path, grep))


@pytest.mark.parametrize("variable", VARIABLES)
def test_environment_override_is_blocked(variable: str, tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere" / "custom-protected"
    payload = {"tool_name": "Read", "tool_input": {"file_path": str(custom / "notes.txt")}}
    assert _blocked(_run(tmp_path, payload, **{variable: str(custom)}))


@pytest.mark.parametrize("variable", VARIABLES)
def test_windows_style_path_is_blocked_in_escaped_json(variable: str, tmp_path: Path) -> None:
    custom = r"C:\Users\someone\custom-protected"
    payload = {"tool_name": "Read", "tool_input": {"file_path": custom + r"\notes.txt"}}
    assert _blocked(_run(tmp_path, payload, **{variable: custom}))


@pytest.mark.parametrize(("variable", "default"), PROTECTED, ids=VARIABLES)
def test_empty_override_falls_back_to_the_default(
    variable: str, default: tuple[str, ...], tmp_path: Path
) -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": f"cat {default[-1]}/x"}}
    assert _blocked(_run(tmp_path, payload, **{variable: ""}))


@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Bash", "tool_input": {"command": "python -m pytest -q"}},
        {"tool_name": "Bash", "tool_input": {"command": "echo $SCRIBECHECK_RAW_DIR"}},
        {"tool_name": "Bash", "tool_input": {"command": "echo $SCRIBECHECK_HELDOUT_DIR"}},
        {"tool_name": "Read", "tool_input": {"file_path": "src/scribecheck/rules/lexicon.py"}},
        {"tool_name": "Read", "tool_input": {"file_path": "data/omissionbench/dev/case-001.json"}},
        "not json",
    ],
)
def test_other_calls_are_allowed(payload: object, tmp_path: Path) -> None:
    assert not _blocked(_run(tmp_path, payload))
