"""The decision rule: pass, warn or fail, from per-flag severity and confidence.

Following docs/spec.md ("Severity, decision rule and calibration"):

- In rules mode every finding is a warning, and nothing fails.
- In full mode failing is opt-in: a note fails only when a flag has a severity in
  `fail_on` and a confidence in `fail_confidence`.
- Otherwise critical and supporting flags, and partial verdicts, warn; peripheral flags
  (kept only by the strict standard) are reported without warning.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from scribecheck.config import Mode, Standard
from scribecheck.flags import Confidence, Flag, Severity

Decision = Literal["pass", "warn", "fail"]

_WARNING_SEVERITIES = frozenset({"critical", "supporting"})


def apply_standard(flags: Iterable[Flag], standard: Standard) -> tuple[Flag, ...]:
    """The flags the standard checks: `clinical` drops peripheral omissions; `strict` keeps all."""
    if standard == "strict":
        return tuple(flags)
    return tuple(f for f in flags if not (f.type == "omission" and f.severity == "peripheral"))


def decide(
    flags: Sequence[Flag],
    *,
    mode: Mode,
    fail_on: Iterable[Severity] = (),
    fail_confidence: Iterable[Confidence] = ("certain", "confirmed"),
) -> Decision:
    """Pass, warn or fail for one note's flags (already filtered by the standard)."""
    failing, levels = frozenset(fail_on), frozenset(fail_confidence)
    if mode == "full" and any(f.severity in failing and f.confidence in levels for f in flags):
        return "fail"
    if any(f.severity in _WARNING_SEVERITIES or f.verdict == "partial" for f in flags):
        return "warn"
    return "pass"
