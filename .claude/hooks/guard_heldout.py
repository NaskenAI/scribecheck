"""PreToolUse hook: block any tool call that touches the held-out benchmark data.

The held-out subset decides the launch claim at Gate 2. If an agent reads it while
tuning, the result is contaminated. Exit code 2 blocks the call and shows the
message to Claude.

The held-out folder defaults to ~/scribecheck-heldout; set SCRIBECHECK_HELDOUT_DIR
to change it. Code may mention the variable name; it may not touch the folder.
"""

from __future__ import annotations

import json
import os
import sys


def heldout_markers() -> set[str]:
    default = os.path.join(os.path.expanduser("~"), "scribecheck-heldout")
    folder = os.path.normpath(os.environ.get("SCRIBECHECK_HELDOUT_DIR", default))
    markers = {folder, os.path.basename(folder)}
    # JSON escapes Windows backslashes, so also match the escaped form.
    markers.add(folder.replace("\\", "\\\\"))
    return {m for m in markers if m and m not in {".", os.sep}}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = json.dumps(payload.get("tool_input", {}))
    if any(marker in tool_input for marker in heldout_markers()):
        print(
            "Blocked: this touches the held-out benchmark data. Only the frozen Gate 2 run, "
            "started by a person, may use it. Tune on development data instead.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
