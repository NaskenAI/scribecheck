"""Shared assertion for finder tests: every match must slice back to its own text."""

from __future__ import annotations

from collections.abc import Sequence

from scribecheck.rules._match import Match


def assert_round_trip(text: str, matches: Sequence[Match]) -> None:
    previous_end = 0
    for match in matches:
        assert text[match.start : match.end] == match.text
        assert previous_end <= match.start < match.end
        previous_end = match.end
