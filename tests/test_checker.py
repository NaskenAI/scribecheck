from __future__ import annotations

import json
from pathlib import Path

import pytest

from scribecheck import Config, Report, check, check_files, checker
from scribecheck.checker import check_folder, find_pairs
from scribecheck.errors import InputError
from scribecheck.flags import verify_evidence
from scribecheck.note import parse_note
from scribecheck.report import render_markdown
from scribecheck.transcript import parse_transcript

KNEE = (
    "Doctor: What seems to be the problem?\n"
    "Patient: My left knee has been swollen for two weeks.\n"
    "Doctor: Are you taking anything for it?\n"
    "Patient: I take ibuprofen 400 mg three times a day.\n"
)
KNEE_NOTE = "History:\nKnee swollen for 2 weeks.\n\nMedications:\nIbuprofen 400 mg twice daily.\n"


def test_would_be_certain_change_becomes_a_possible_warning() -> None:
    report = check(KNEE, KNEE_NOTE)
    changes = [f for f in report.flags if f.type == "changed_fact"]
    assert len(changes) == 1
    flag = changes[0]
    assert (flag.category, flag.verdict, flag.confidence, flag.origin) == (
        "frequency",
        "changed",
        "possible",
        "rules",
    )
    assert (flag.rule_id, flag.severity) == ("frequency.mismatch", "critical")
    assert (flag.transcript_value, flag.note_value) == ("three times a day", "twice daily")
    assert flag.transcript_evidence is not None and flag.note_evidence is not None
    assert flag.note_evidence.section == "Medications"
    assert report.counts.changed_fact == 1
    assert report.decision == "warn"


def test_items_with_other_reasons_stay_unchecked() -> None:
    transcript = "Doctor: I'll prescribe doxycycline 100 mg once a day.\nPatient: Thank you.\n"
    note = "Plan:\nDoxycycline 100 mg twice daily.\n"
    report = check(transcript, note)
    assert not [f for f in report.flags if f.type == "changed_fact"]
    reasons = {u.reasons for u in report.unchecked}
    assert ("planned_or_past", "certain_disabled_pending_allowlist") in reasons


def test_unaligned_drug_already_reported_by_pass_a_is_not_repeated() -> None:
    transcript = "Doctor: What do you take?\nPatient: I'm on atorvastatin 20 mg at night.\n"
    report = check(transcript, "Plan:\nReview in a year.\n")
    assert [f.rule_id for f in report.flags] == ["medicine.missing"]
    assert not [u for u in report.unchecked if u.item == "atorvastatin"]


def test_unaligned_drug_unchecked_by_pass_a_is_not_repeated() -> None:
    transcript = (
        "Doctor: What do you take?\nPatient: I used to take amlodipine, but I stopped it.\n"
    )
    report = check(transcript, "Plan:\nReview in a year.\n")
    assert [(u.category, u.reason) for u in report.unchecked] == [("medicine", "planned_or_past")]


def test_flags_are_numbered_omissions_first_in_transcript_order() -> None:
    report = check(KNEE, KNEE_NOTE)
    assert [f.id for f in report.flags] == [f"F{i}" for i in range(1, len(report.flags) + 1)]
    types = [f.type for f in report.flags]
    assert types == sorted(types, key=lambda t: t != "omission")
    assert check(KNEE, KNEE_NOTE).to_json() == report.to_json()


def test_every_quote_verifies_against_its_source() -> None:
    transcript, note = parse_transcript(KNEE), parse_note(KNEE_NOTE)
    report = check(KNEE, KNEE_NOTE)
    for entry in (*report.flags, *report.unchecked):
        verify_evidence(entry, transcript, note)


def test_report_never_holds_the_full_transcript_or_note() -> None:
    sentinel = "The parrot recited Shakespeare in the waiting room"
    transcript = KNEE + f"Patient: {sentinel}.\n"
    note = KNEE_NOTE + f"\nSocial:\n{sentinel} again.\n"
    report = check(transcript, note)
    for text in (report.to_json(), render_markdown(report)):
        assert sentinel not in text
        assert KNEE_NOTE not in text


@pytest.mark.parametrize("mode", ["full"])
def test_full_mode_is_not_available_yet(tmp_path: Path, mode: str) -> None:
    config = Config(mode=mode)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="full mode"):
        check(KNEE, KNEE_NOTE, config)
    with pytest.raises(InputError, match="full mode"):
        check_files(tmp_path / "t.txt", tmp_path / "n.md", config)
    with pytest.raises(InputError, match="full mode"):
        check_folder(tmp_path, config)


def test_config_speakers_and_brands_are_used() -> None:
    transcript = "GP: What do you take?\nPt: I take Examplol 500 mg twice a day.\n"
    note = "Drug history:\n- Ibuprofen\n"
    assert check(transcript, note).flags == ()  # "Pt" is not the patient; Examplol unknown
    config = Config(
        speakers={"GP": "clinician", "Pt": "patient"},
        headings={"Drug history": "Medications"},
        brands={"Examplol": ["paracetamol"]},
    )
    report = check(transcript, note, config)
    assert [(f.rule_id, f.fact) for f in report.flags] == [("medicine.missing", "acetaminophen")]


def test_standard_is_reported() -> None:
    assert check(KNEE, KNEE_NOTE, Config(standard="strict")).standard == "strict"


def test_consultation_id() -> None:
    assert check(KNEE, KNEE_NOTE).consultation_id is None
    assert check(KNEE, KNEE_NOTE, consultation_id="case-7").consultation_id == "case-7"
    data = {"consultation_id": "json-1", "turns": [{"speaker": "Patient", "text": "Hello."}]}
    assert check(json.dumps(data), KNEE_NOTE).consultation_id == "json-1"


# Folders


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(KNEE_NOTE if "note" in name else KNEE, encoding="utf-8")


def test_pairs_by_suffix(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "b.transcript.txt",
        "b.note.md",
        "a.transcript.json",
        "a.note.txt",
        "c.transcript.txt",
        "d.note.md",
        "README.md",
        ".hidden.transcript.txt",
    )
    pairs, problems = find_pairs(tmp_path)
    assert [(p.id, p.transcript.name, p.note.name) for p in pairs] == [
        ("a", "a.transcript.json", "a.note.txt"),
        ("b", "b.transcript.txt", "b.note.md"),
    ]
    assert problems == [
        "c.transcript.txt: no matching note",
        "d.note.md: no matching transcript",
    ]


def test_pairs_by_subfolder(tmp_path: Path) -> None:
    _touch(tmp_path, "transcripts/x.txt", "notes/x.md", "transcripts/y.json", "notes/z.md")
    pairs, problems = find_pairs(tmp_path)
    assert [(p.id, p.transcript.name, p.note.name) for p in pairs] == [("x", "x.txt", "x.md")]
    assert problems == [
        "notes/z.md: no matching transcript",
        "transcripts/y.json: no matching note",
    ]


def test_duplicate_files_for_one_case_are_skipped(tmp_path: Path) -> None:
    _touch(tmp_path, "a.transcript.txt", "a.transcript.json", "a.note.md")
    pairs, problems = find_pairs(tmp_path)
    assert pairs == []
    assert problems == ["a: more than one file (a.transcript.json, a.transcript.txt); skipped"]


def test_plain_text_transcript_is_named_without_its_marker(tmp_path: Path) -> None:
    _touch(tmp_path, "case1.transcript.txt", "case1.note.md")
    report = check_files(tmp_path / "case1.transcript.txt", tmp_path / "case1.note.md")
    assert report.consultation_id == "case1"


def test_not_a_folder(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="not a folder"):
        find_pairs(tmp_path / "absent")


def test_a_broken_pair_is_reported_and_the_rest_still_run(tmp_path: Path) -> None:
    _touch(tmp_path, "good.transcript.txt", "good.note.md", "bad.transcript.json")
    (tmp_path / "bad.transcript.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "bad.note.md").write_text(KNEE_NOTE, encoding="utf-8")
    result = check_folder(tmp_path)
    assert [r.consultation_id for r in result.reports] == ["good"]
    assert len(result.problems) == 1 and result.problems[0].startswith("bad: ")


def test_both_layouts_are_read_together(tmp_path: Path) -> None:
    _touch(tmp_path, "transcripts/c1.txt", "notes/c1.md", "c2.transcript.txt", "c2.note.md")
    pairs, problems = find_pairs(tmp_path)
    assert [(p.id, p.transcript.relative_to(tmp_path).as_posix()) for p in pairs] == [
        ("c1", "transcripts/c1.txt"),
        ("c2", "c2.transcript.txt"),
    ]
    assert problems == []


def test_subfolders_accept_the_markers(tmp_path: Path) -> None:
    _touch(tmp_path, "transcripts/c1.transcript.txt", "notes/c1.note.md")
    pairs, problems = find_pairs(tmp_path)
    assert [p.id for p in pairs] == ["c1"] and problems == []


def test_the_same_case_in_both_layouts_is_a_duplicate(tmp_path: Path) -> None:
    _touch(tmp_path, "transcripts/c1.txt", "notes/c1.md", "c1.transcript.txt")
    pairs, problems = find_pairs(tmp_path)
    assert pairs == []
    assert problems == ["c1: more than one file (c1.transcript.txt, transcripts/c1.txt); skipped"]


def test_ids_match_case_insensitively(tmp_path: Path) -> None:
    _touch(tmp_path, "Case1.transcript.txt", "case1.note.md")
    pairs, problems = find_pairs(tmp_path)
    assert [(p.id, p.note.name) for p in pairs] == [("Case1", "case1.note.md")]
    assert problems == []


def test_a_pair_shaped_name_that_is_not_a_file_is_a_problem(tmp_path: Path) -> None:
    _touch(tmp_path, "x.note.md")
    (tmp_path / "x.transcript.txt").mkdir()
    pairs, problems = find_pairs(tmp_path)
    assert pairs == []
    assert problems == [
        "x.note.md: no matching transcript",
        "x.transcript.txt: not a readable file; skipped",
    ]


def test_an_unexpected_error_in_one_pair_spares_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _touch(tmp_path, "a.transcript.txt", "a.note.md", "b.transcript.txt", "b.note.md")
    real = checker.check_files

    def flaky(t: Path, n: Path, config: Config | None = None) -> Report:
        if t.name.startswith("a."):
            raise ValueError("boom")
        return real(t, n, config)

    monkeypatch.setattr(checker, "check_files", flaky)
    result = check_folder(tmp_path)
    assert [c.id for c in result.cases] == ["b"]
    assert result.problems == ("a: internal error: ValueError: boom",)


def test_cases_keep_their_files(tmp_path: Path) -> None:
    _touch(tmp_path, "transcripts/c1.txt", "notes/c1.md")
    (case,) = check_folder(tmp_path).cases
    assert (case.id, case.transcript, case.note) == ("c1", "transcripts/c1.txt", "notes/c1.md")
