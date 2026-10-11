"""Transcript parsing: plain text with optional speaker prefixes, or JSON.

Plain-text speaker prefixes are "Doctor:" or the bracketed ACI style "[doctor]" (an
optional colon may follow the bracket). Only labels in the speaker map count, so a
bracketed annotation such as "[inaudible]" stays part of the turn text.

Turns are numbered from 1 and keep their exact text, so an evidence quote can be
located by turn number plus character offsets within `Turn.text`. Plain-text turns
also record where they sit in `Transcript.text`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

from scribecheck._text import normalise_text, read_text_file, strip_span
from scribecheck.errors import InputError

Modality = Literal["in-person", "video", "telephone"]
TranscriptFormat = Literal["auto", "json", "text"]

MODALITIES: tuple[str, ...] = get_args(Modality)
UNKNOWN_SPEAKER = "unknown"

DEFAULT_SPEAKER_MAP: Mapping[str, str] = {
    "doctor": "clinician",
    "dr": "clinician",
    "clinician": "clinician",
    "patient": "patient",
    # Other people in the room keep their own role, so their answers never count as the
    # patient's (dialogue negation binds only clinician and patient turns).
    "relative": "relative",
    "carer": "carer",
    "parent": "parent",
    "nurse": "nurse",
}


class Turn(BaseModel):
    """One speaker turn."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    speaker: str
    """Canonical role from the speaker map, the lowercased label if unmapped, or "unknown"."""
    label: str | None = None
    """The speaker label as written, without brackets or colon; None when there was none."""
    text: str
    """The turn's exact text; evidence offsets index into this."""
    start: int | None = None
    """Offset of `text` in `Transcript.text`; None for JSON input."""
    end: int | None = None


class Transcript(BaseModel):
    """A parsed consultation transcript."""

    model_config = ConfigDict(frozen=True)

    consultation_id: str | None = None
    modality: Modality | None = None
    turns: tuple[Turn, ...]
    text: str | None = None
    """The normalised plain-text source; None for JSON input."""

    @property
    def has_speaker_labels(self) -> bool:
        return any(turn.label is not None for turn in self.turns)

    def quote(self, turn: int, start: int, end: int) -> str:
        """Return the text at `start:end` within turn number `turn` (from 1)."""
        if not 1 <= turn <= len(self.turns):
            raise IndexError(f"turn {turn} out of range 1..{len(self.turns)}")
        return self.turns[turn - 1].text[start:end]


def parse_transcript(
    text: str,
    *,
    format: TranscriptFormat = "auto",
    speaker_map: Mapping[str, str] | None = None,
    consultation_id: str | None = None,
) -> Transcript:
    """Parse a transcript from text.

    `format="auto"` treats input whose first non-blank character is "{" as JSON, and
    rejects input that is a JSON array; other input starting with "[" stays plain text.
    `speaker_map` maps speaker labels (case-insensitive) to roles and replaces
    `DEFAULT_SPEAKER_MAP`. `consultation_id` applies to plain text; JSON carries its own.
    Raises `InputError` on malformed input.
    """
    if format not in get_args(TranscriptFormat):
        raise ValueError(f"unknown transcript format {format!r}")
    text = normalise_text(text)
    roles = _normalise_speaker_map(DEFAULT_SPEAKER_MAP if speaker_map is None else speaker_map)
    if format == "json" or (format == "auto" and _looks_like_json(text)):
        return _parse_json(text, roles)
    return _parse_plain(text, roles, consultation_id)


def _looks_like_json(text: str) -> bool:
    start = text.lstrip()[:1]
    if start == "{":
        return True
    if start == "[":
        # "[00:01] ..." timestamps are plain text; a valid JSON array is a wrong shape.
        try:
            json.loads(text)
        except (json.JSONDecodeError, RecursionError):
            return False
        return True
    return False


def parse_transcript_file(
    path: str | Path,
    *,
    speaker_map: Mapping[str, str] | None = None,
) -> Transcript:
    """Parse a transcript file: `.json` as JSON, anything else as plain text.

    A plain-text transcript's consultation_id is the file name without its extension.
    """
    path = Path(path)
    text = read_text_file(path)
    is_json = path.suffix.lower() == ".json"
    try:
        return parse_transcript(
            text,
            format="json" if is_json else "text",
            speaker_map=speaker_map,
            consultation_id=None if is_json else path.stem,
        )
    except InputError as exc:
        raise InputError(f"{path}: {exc}") from exc


def _normalise_label(label: str) -> str:
    return " ".join(label.split()).rstrip(".").casefold()


def _normalise_speaker_map(speaker_map: Mapping[str, str]) -> dict[str, str]:
    roles = {_normalise_label(label): role for label, role in speaker_map.items()}
    roles.pop("", None)
    return roles


def _speaker_for(label: str | None, roles: Mapping[str, str]) -> str:
    if label is None:
        return UNKNOWN_SPEAKER
    key = _normalise_label(label)
    return roles.get(key, key)


# Plain text


def _prefix_pattern(roles: Mapping[str, str]) -> re.Pattern[str] | None:
    if not roles:
        return None
    labels = sorted(roles, key=len, reverse=True)
    alternatives = "|".join(r"\s+".join(map(re.escape, label.split())) for label in labels)
    return re.compile(
        rf"^[ \t]*(?:\[[ \t]*(?P<bracketed>{alternatives})[ \t]*\][ \t]*:?"
        rf"|(?P<label>(?:{alternatives})\.?)[ \t]*:)",
        re.IGNORECASE | re.MULTILINE,
    )


def _parse_plain(text: str, roles: Mapping[str, str], consultation_id: str | None) -> Transcript:
    if not text.strip():
        raise InputError("transcript is empty")
    pattern = _prefix_pattern(roles)
    prefixes = list(pattern.finditer(text)) if pattern else []

    # (label, start, end) for each candidate turn, before trimming.
    spans: list[tuple[str | None, int, int]] = []
    if not prefixes:
        position = 0
        for line in text.split("\n"):
            spans.append((None, position, position + len(line)))
            position += len(line) + 1
    else:
        spans.append((None, 0, prefixes[0].start()))
        for i, match in enumerate(prefixes):
            end = prefixes[i + 1].start() if i + 1 < len(prefixes) else len(text)
            spans.append((match.group("label") or match.group("bracketed"), match.end(), end))

    turns: list[Turn] = []
    for label, raw_start, raw_end in spans:
        start, end = strip_span(text, raw_start, raw_end)
        if start == end:
            continue
        turns.append(
            Turn(
                number=len(turns) + 1,
                speaker=_speaker_for(label, roles),
                label=label,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
    if not turns:
        raise InputError("transcript has no turns with text")
    return Transcript(consultation_id=consultation_id, turns=tuple(turns), text=text)


# JSON


class _JsonTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    speaker: StrictStr | None = None
    text: StrictStr = Field(min_length=1)


class _JsonTranscript(BaseModel):
    model_config = ConfigDict(extra="ignore")

    consultation_id: StrictStr = Field(min_length=1)
    modality: StrictStr | None = None
    turns: list[_JsonTurn] = Field(min_length=1)


def _describe(error: ValidationError) -> str:
    problems = []
    for item in error.errors():
        location = ""
        for part in item["loc"]:
            if isinstance(part, int):
                location += f"[{part}]"
            else:
                location += f".{part}" if location else part
        problems.append(f"{location or 'transcript'}: {item['msg']}")
    return "invalid transcript JSON: " + "; ".join(problems)


def _parse_json(text: str, roles: Mapping[str, str]) -> Transcript:
    if not text.strip():
        raise InputError("transcript is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"invalid transcript JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except RecursionError as exc:
        raise InputError("invalid transcript JSON: nested too deeply") from exc
    if not isinstance(data, dict):
        raise InputError("transcript JSON must be an object with consultation_id and turns")
    try:
        parsed = _JsonTranscript.model_validate(data)
    except ValidationError as exc:
        raise InputError(_describe(exc)) from exc

    if not parsed.consultation_id.strip():
        raise InputError("invalid transcript JSON: consultation_id is blank")
    modality: Modality | None = None
    if parsed.modality is not None:
        value = parsed.modality.strip().lower()
        if value not in MODALITIES:
            allowed = ", ".join(MODALITIES)
            raise InputError(f"invalid modality {parsed.modality!r}; expected one of: {allowed}")
        modality = cast(Modality, value)

    turns = []
    for number, turn in enumerate(parsed.turns, start=1):
        if not turn.text.strip():
            raise InputError(f"invalid transcript JSON: turns[{number - 1}].text is blank")
        label = turn.speaker if turn.speaker and turn.speaker.strip() else None
        turns.append(
            Turn(number=number, speaker=_speaker_for(label, roles), label=label, text=turn.text)
        )
    return Transcript(consultation_id=parsed.consultation_id, modality=modality, turns=tuple(turns))
