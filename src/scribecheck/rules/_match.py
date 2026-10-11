"""Shared base for finder results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict

Comparison = Literal["equal", "different", "not_comparable"]


class Match(BaseModel):
    """A value found in text. `text` is always `source[start:end]`."""

    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    text: str


M = TypeVar("M", bound=Match)


def non_overlapping(candidates: Iterable[M]) -> tuple[M, ...]:
    """Keep the leftmost, then longest, candidate wherever candidates overlap."""
    chosen: list[M] = []
    for candidate in sorted(candidates, key=lambda m: (m.start, -(m.end - m.start))):
        if not chosen or candidate.start >= chosen[-1].end:
            chosen.append(candidate)
    return tuple(chosen)
