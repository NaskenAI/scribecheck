"""The checker's entry points: one consultation, a folder of them, or the bundled demo.

Rules mode runs Pass A's structured omission candidates and Pass B's changed values.
While rules-mode certain flags are disabled (rules/changed.py), a Pass B item whose only
reason is "certain_disabled_pending_allowlist" (one that would have been a certain flag)
is reported as a changed-fact flag at possible confidence, with Pass B's rule ID and
severity. Every other item stays unchecked with its reasons, except an unaligned drug that
Pass A already reports, as a missing medicine or an unchecked one. In full mode both
kinds of flag go to the model's quote-verified second check (M3).

Flag IDs are numbered after merging, omissions first, each type in transcript order, so
a report is the same on every run. Every quote is verified against its source before it
enters a report.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import cast

from scribecheck.config import Config
from scribecheck.decision import apply_standard, decide
from scribecheck.errors import InputError
from scribecheck.flags import Flag, UncheckedItem, verify_evidence
from scribecheck.note import Note, parse_note, parse_note_file
from scribecheck.report import Case, Report, count
from scribecheck.rules import changed
from scribecheck.rules.omission import structured_omissions
from scribecheck.transcript import Transcript, parse_transcript, parse_transcript_file

_TYPE_ORDER = {"omission": 0, "changed_fact": 1, "unsupported": 2}
_TRANSCRIPT_SUFFIXES = (".json", ".txt")
_NOTE_SUFFIXES = (".md", ".txt")


def check(
    transcript: str,
    note: str,
    config: Config | None = None,
    *,
    consultation_id: str | None = None,
) -> Report:
    """Check a note against its transcript, both given as text.

    `consultation_id` names a plain-text transcript; a JSON transcript carries its own.
    Raises `InputError` for malformed input or a mode that is not available.
    """
    config = config or Config()
    _require_rules_mode(config)
    parsed_t = parse_transcript(
        transcript, speaker_map=config.speaker_map(), consultation_id=consultation_id
    )
    parsed_n = parse_note(note, heading_map=config.heading_map())
    return check_parsed(parsed_t, parsed_n, config)


def check_files(
    transcript_path: str | Path, note_path: str | Path, config: Config | None = None
) -> Report:
    """Check files on disk: a `.json` or plain-text transcript, and a text or Markdown note.

    A plain-text transcript is named by its file name without the extension, or without
    ".transcript" too (`case1.transcript.txt` is "case1").
    """
    config = config or Config()
    _require_rules_mode(config)
    parsed_t = parse_transcript_file(transcript_path, speaker_map=config.speaker_map())
    stem = parsed_t.consultation_id
    if Path(transcript_path).suffix.lower() != ".json" and stem and _marked(stem, ".transcript"):
        parsed_t = parsed_t.model_copy(update={"consultation_id": stem[: -len(".transcript")]})
    parsed_n = parse_note_file(note_path, heading_map=config.heading_map())
    return check_parsed(parsed_t, parsed_n, config)


def check_parsed(transcript: Transcript, note: Note, config: Config | None = None) -> Report:
    """Check an already parsed transcript and note."""
    config = config or Config()
    _require_rules_mode(config)
    brands = config.brands or None
    omissions = structured_omissions(transcript, note, brands=brands)
    changes = changed.check_changed_facts(
        transcript, note, brands=brands, symptom_extra=config.symptoms or None
    )
    found = [*omissions.candidates, *changes.flags]
    unchecked: list[UncheckedItem] = list(omissions.unchecked)
    pass_a_medicines = {
        f.transcript_evidence
        for f in omissions.candidates
        if f.category == "medicine" and f.transcript_evidence is not None
    }
    for skipped in omissions.unchecked:
        if skipped.category == "medicine":
            pass_a_medicines.update(skipped.transcript_evidence)
    for item in changes.unchecked:
        if (
            item.reason == "unaligned"
            and not item.note_evidence
            and pass_a_medicines.intersection(item.transcript_evidence)
        ):
            continue  # a drug in the transcript only is Pass A's, and reported there already
        if item.reasons == (changed.CERTAIN_DISABLED,):
            found.append(_possible_change(item))
        else:
            unchecked.append(item)

    flags = _numbered(apply_standard(found, config.standard))
    entries: list[Flag | UncheckedItem] = [*flags, *unchecked]
    for entry in entries:
        verify_evidence(entry, transcript, note)
    return Report(
        consultation_id=transcript.consultation_id,
        mode=config.mode,
        standard=config.standard,
        decision=decide(
            flags,
            mode=config.mode,
            fail_on=config.fail_on,
            fail_confidence=config.fail_confidence,
        ),
        counts=count(flags, unchecked),
        flags=flags,
        unchecked=tuple(unchecked),
    )


def _require_rules_mode(config: Config) -> None:
    if config.mode != "rules":
        raise InputError("full mode is not available yet (milestone M3); use --mode rules")


def _possible_change(item: UncheckedItem) -> Flag:
    """A would-be certain Pass B item as a changed-fact warning at possible confidence."""
    if item.category not in changed.RULE_ID:
        raise ValueError(f"no Pass B rule for category {item.category!r}")
    category = cast(changed.Category, item.category)
    return Flag(
        id="F0",
        type="changed_fact",
        category=category,
        verdict="changed",
        severity=changed.SEVERITY[category],
        confidence="possible",
        origin="rules",
        rule_id=changed.RULE_ID[category],
        transcript_value=item.transcript_value,
        note_value=item.note_value,
        transcript_evidence=item.transcript_evidence[0] if item.transcript_evidence else None,
        note_evidence=item.note_evidence[0] if item.note_evidence else None,
    )


def _numbered(flags: Iterable[Flag]) -> tuple[Flag, ...]:
    def order(flag: Flag) -> tuple[int, int, int]:
        quote = flag.transcript_evidence
        where = (quote.turn, quote.start) if quote else (0, 0)
        return (_TYPE_ORDER[flag.type], *where)

    ordered = sorted(flags, key=order)
    return tuple(f.model_copy(update={"id": f"F{i}"}) for i, f in enumerate(ordered, start=1))


# Folders


@dataclass(frozen=True)
class Pair:
    """A transcript and note matched by file name."""

    id: str
    transcript: Path
    note: Path


@dataclass(frozen=True)
class BatchResult:
    cases: tuple[Case, ...]
    problems: tuple[str, ...]
    """Unmatched, duplicate or unreadable files, and pairs that could not be checked."""

    @property
    def reports(self) -> tuple[Report, ...]:
        return tuple(c.report for c in self.cases)


def find_pairs(folder: str | Path) -> tuple[list[Pair], list[str]]:
    """Match transcripts to notes by file name, sorted by id, with any problems.

    In the folder itself, `case1.transcript.json` (or `.txt`) matches `case1.note.md` (or
    `.txt`). With `transcripts/` and `notes/` subfolders, files there match by name without
    extension, with or without the marker (`transcripts/case1.json` or
    `transcripts/case1.transcript.json` with `notes/case1.md`). Both layouts are read, ids
    match case-insensitively, and other files are ignored. A file found twice for one id,
    a file with no partner, and a pair-shaped name that is not a readable file are
    problems. Raises `InputError` if `folder` is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise InputError(f"{folder}: not a folder")
    problems: list[str] = []
    transcripts = _by_id(folder, _TRANSCRIPT_SUFFIXES, ".transcript", True, folder, problems)
    notes = _by_id(folder, _NOTE_SUFFIXES, ".note", True, folder, problems)
    t_dir, n_dir = folder / "transcripts", folder / "notes"
    if t_dir.is_dir() and n_dir.is_dir():
        for key, paths in _by_id(
            t_dir, _TRANSCRIPT_SUFFIXES, ".transcript", False, folder, problems
        ).items():
            transcripts.setdefault(key, []).extend(paths)
        for key, paths in _by_id(n_dir, _NOTE_SUFFIXES, ".note", False, folder, problems).items():
            notes.setdefault(key, []).extend(paths)
    pairs = []
    for key in sorted(transcripts.keys() | notes.keys()):
        t, n = transcripts.get(key, []), notes.get(key, [])
        if len(t) > 1 or len(n) > 1:
            extra = [*(t if len(t) > 1 else []), *(n if len(n) > 1 else [])]
            names = ", ".join(_rel(p, folder) for p in extra)
            problems.append(f"{key}: more than one file ({names}); skipped")
        elif t and n:
            pairs.append(Pair(_case_id(t[0]), t[0], n[0]))
        else:
            missing = "note" if not n else "transcript"
            problems.append(f"{_rel((t or n)[0], folder)}: no matching {missing}")
    return pairs, sorted(problems)


def _by_id(
    directory: Path,
    suffixes: tuple[str, ...],
    marker: str,
    required: bool,
    root: Path,
    problems: list[str],
) -> dict[str, list[Path]]:
    """Files in `directory` by case-folded id. `marker` is stripped, and if `required`,
    a file without it is ignored."""
    found: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in suffixes:
            continue
        stem = path.name[: -len(path.suffix)]
        if _marked(stem, marker):
            stem = stem[: -len(marker)]
        elif required:
            continue
        if not stem:
            continue
        if not path.is_file():
            problems.append(f"{_rel(path, root)}: not a readable file; skipped")
            continue
        found.setdefault(stem.casefold(), []).append(path)
    return found


def _marked(stem: str, marker: str) -> bool:
    return stem.lower().endswith(marker) and len(stem) > len(marker)


def _case_id(transcript: Path) -> str:
    stem = transcript.name[: -len(transcript.suffix)]
    return stem[: -len(".transcript")] if _marked(stem, ".transcript") else stem


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def check_folder(folder: str | Path, config: Config | None = None) -> BatchResult:
    """Check every matched pair in `folder`; a pair that fails is reported, not raised."""
    config = config or Config()
    _require_rules_mode(config)
    folder = Path(folder)
    pairs, problems = find_pairs(folder)
    cases = []
    for pair in pairs:
        try:
            report = check_files(pair.transcript, pair.note, config)
        except InputError as exc:
            problems.append(f"{pair.id}: {exc}")
            continue
        except Exception as exc:  # a bug in one pair should not hide the others' reports
            problems.append(f"{pair.id}: internal error: {type(exc).__name__}: {exc}")
            continue
        cases.append(Case(pair.id, _rel(pair.transcript, folder), _rel(pair.note, folder), report))
    return BatchResult(tuple(cases), tuple(problems))


# The bundled demo


DEMO_PREAMBLE = (
    "Demo: synthetic consultations bundled with scribecheck, checked in rules mode. "
    "They describe no real patient and have not yet been reviewed by a clinician."
)


def demo_cases() -> tuple[Case, ...]:
    """Rules-mode reports for the bundled synthetic demo consultations, in manifest order."""
    folder = resources.files("scribecheck") / "demo"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = Config()
    cases = []
    for entry in manifest["cases"]:
        t_name, n_name = f"{entry['id']}.transcript.json", f"{entry['id']}.note.md"
        report = check(
            (folder / t_name).read_text(encoding="utf-8"),
            (folder / n_name).read_text(encoding="utf-8"),
            config,
        )
        cases.append(Case(entry["id"], t_name, n_name, report))
    return tuple(cases)


def demo_reports() -> tuple[Report, ...]:
    """The demo cases' reports."""
    return tuple(c.report for c in demo_cases())
