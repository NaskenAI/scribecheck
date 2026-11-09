"""The blindness rail: a transcript-only stage must never see a note under judgement.

Ported from the OmissionBench harness's `assert_blind` (judges/w2_pipeline.py; Composo,
MIT, see prompts/route_one/LICENSE): normalised slices of each note (start, middle, end)
are searched for in the built prompt before the call is made, and a hit raises.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


class BlindnessError(AssertionError):
    """A transcript-only prompt contains text from a note it will be used to judge."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def assert_blind(prompt: str, notes: Iterable[str], stage: str) -> None:
    """Raise `BlindnessError` if any note's text appears in `prompt`.

    As in the harness: notes shorter than 80 normalised characters are skipped, and longer
    ones are probed with three slices (start, middle, end) of up to 120 characters, each at
    least 80 long.
    """
    haystack = _norm(prompt)
    for note in notes:
        text = _norm(note)
        if not text:
            continue
        if len(text) < 80:
            continue
        starts = (0, max(0, len(text) // 2 - 60), max(0, len(text) - 120))
        for probe in (text[s : s + 120] for s in starts):
            if len(probe) >= 80 and probe in haystack:
                raise BlindnessError(
                    f"blindness violation in {stage}: a note appears in the prompt"
                )
