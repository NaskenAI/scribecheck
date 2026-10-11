"""Stop hook: Claude can't end a turn while lint or tests fail.

Runs ruff and pytest with the project's virtual environment when one exists.
On failure, exit code 2 blocks the stop and shows the output to Claude. If Claude
is already continuing because of this hook, the stop is allowed, so a session
can't loop forever; the failure stays visible in the transcript.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def project_python(project: Path) -> str:
    for candidate in (
        project / ".venv" / "bin" / "python",
        project / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if payload.get("stop_hook_active"):
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
    python = project_python(project)
    checks = [
        ("ruff", [python, "-m", "ruff", "check", "."]),
        ("pytest", [python, "-m", "pytest", "-q", "-x", "--no-header"]),
    ]
    for name, command in checks:
        result = subprocess.run(command, cwd=project, capture_output=True, text=True, check=False)
        output = (result.stdout + result.stderr).strip()
        if "No module named" in output and name in output:
            print(
                f'{name} is not installed; run: python -m pip install -e ".[dev]"', file=sys.stderr
            )
            return 0
        # pytest exit code 5 means no tests were collected, which is not a failure here.
        if result.returncode not in (0, 5):
            print(f"{name} failed. Fix this before finishing:\n{output[-4000:]}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
