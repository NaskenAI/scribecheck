"""PreToolUse hook: block any tool call that touches held-out benchmark data.

Two folders hold it, and both are protected the same way:

- the OmissionBench held-out subset: default ~/scribecheck-heldout, override with the
  SCRIBECHECK_HELDOUT_DIR environment variable;
- the raw OmissionBench download, which contains held-out material: default
  ~/data/omissionbench-raw, override with SCRIBECHECK_RAW_DIR.

The held-out subset decides the launch claim at Gate 2. If an agent reads either
folder while tuning, the result is contaminated. A call is blocked when its input
mentions a folder's full path, its JSON-escaped Windows form, or its name. Exit code 2
blocks the call and shows the message to Claude. Code may mention the variable names;
it may not touch the folders.
"""

from __future__ import annotations

import json
import os
import sys

HELDOUT_DEFAULT_NAME = "scribecheck-heldout"
RAW_DEFAULT_PARTS = ("data", "omissionbench-raw")

# (environment variable, default path under the home folder) for each protected folder.
PROTECTED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SCRIBECHECK_HELDOUT_DIR", (HELDOUT_DEFAULT_NAME,)),
    ("SCRIBECHECK_RAW_DIR", RAW_DEFAULT_PARTS),
)

MESSAGE = (
    "Blocked: this touches held-out benchmark data (the held-out subset or the raw "
    "OmissionBench download). Only the frozen Gate 2 run, started by a person, may use it. "
    "Tune on development data instead."
)


def protected_folders() -> list[str]:
    """Each protected folder: its environment variable if set and non-empty, else its default."""
    home = os.path.expanduser("~")
    return [
        os.path.normpath(os.environ.get(variable) or os.path.join(home, *default))
        for variable, default in PROTECTED
    ]


def protected_markers() -> set[str]:
    markers: set[str] = set()
    for folder in protected_folders():
        markers |= {folder, os.path.basename(folder)}
        # JSON escapes Windows backslashes, so also match the escaped form.
        markers.add(folder.replace("\\", "\\\\"))
    return {m for m in markers if m and m not in {".", os.sep}}


heldout_markers = protected_markers
"""The earlier name, kept for anything that still calls it."""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = json.dumps(payload.get("tool_input", {}))
    if any(marker in tool_input for marker in protected_markers()):
        print(MESSAGE, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
