"""Pass B on the 65 development clean notes: every certain flag there is false.

A clean twin was verified against its own transcript, so a certain changed-fact flag on
one is a false flag by definition. This test runs Pass B's rules over every development
clean note and its transcript and requires none.

It runs only where the OmissionBench development copy exists (`data/omissionbench`,
written by scripts/split_omissionbench.py and gitignored), so CI skips it. It reads that
copy at run time, embeds no benchmark text, and reports only consultation keys, rule ids
and counts. It reads development data only, never the held-out subset.

Run with `-s` to see the unchecked counts by reason.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scribecheck.note import parse_note
from scribecheck.rules.changed import CERTAIN_DISABLED, check_changed_facts
from scribecheck.transcript import parse_transcript_file

DEV = Path(__file__).resolve().parents[1] / "data" / "omissionbench"
PAIRS = DEV / "pairs" / "dataset_v2.json"
DEVELOPMENT_CONSULTATIONS = 65

pytestmark = pytest.mark.skipif(
    not PAIRS.is_file(), reason="the OmissionBench development copy is not present"
)


def _clean_twins() -> dict[str, tuple[str, str, str]]:
    """key -> (stratum, consultation id, clean note), one per development consultation."""
    twins: dict[str, tuple[str, str, str]] = {}
    for pair in json.loads(PAIRS.read_text(encoding="utf-8"))["pairs"]:
        if pair.get("eval_set") is True:
            twins.setdefault(pair["key"], (pair["stratum"], pair["id"], pair["clean"]))
    return twins


def test_no_certain_flags_on_development_clean_notes() -> None:
    twins = _clean_twins()
    assert len(twins) == DEVELOPMENT_CONSULTATIONS
    flagged: list[tuple[str, str | None]] = []
    candidates: list[tuple[str, str]] = []
    reasons: Counter[str] = Counter()
    for key, (stratum, consultation, clean) in sorted(twins.items()):
        transcript = parse_transcript_file(DEV / "transcripts" / stratum / f"{consultation}.txt")
        result = check_changed_facts(transcript, parse_note(clean))
        flagged += [(key, flag.rule_id) for flag in result.flags]
        # While certain flags are disabled, a would-be flag is an unchecked item whose only
        # reason is the disabled marker; on a clean note it would be just as false.
        candidates += [
            (key, item.category) for item in result.unchecked if item.reasons == (CERTAIN_DISABLED,)
        ]
        reasons.update(item.reason for item in result.unchecked)
    summary = ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items()))
    print(
        f"\n{len(twins)} clean notes; certain flags {len(flagged)}; would-be flags "
        f"{len(candidates)}; unchecked by primary reason: {summary}"
    )
    assert flagged == [], f"false certain flags (consultation, rule): {flagged}"
    assert candidates == [], f"false would-be flags (consultation, category): {candidates}"
