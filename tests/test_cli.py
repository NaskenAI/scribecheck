from __future__ import annotations

import subprocess
import sys

import pytest

from scribecheck import __version__
from scribecheck.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: scribecheck" in capsys.readouterr().out


def test_unknown_command_is_a_tool_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["no-such-command"])
    assert exc.value.code == 2


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scribecheck", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
