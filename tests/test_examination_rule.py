"""The examination rule: remote consultations cannot support examination findings."""

from __future__ import annotations

import json

import pytest

from scribecheck import check
from scribecheck.note import parse_note
from scribecheck.rules.examination import examination_flags
from scribecheck.transcript import parse_transcript

NOTE = (
    "History:\nCough for a week.\n\n"
    "Examination:\nChest clear. Throat red.\nNot examined further.\n\n"
    "Plan:\nO/E abdomen soft and non-tender. Review in a week.\n\n"
    "Vitals:\nBP 120/80\n"
)


def _transcript(modality: str | None) -> str:
    data: dict[str, object] = {
        "consultation_id": "exam",
        "turns": [
            {"speaker": "Doctor", "text": "How is the cough?"},
            {"speaker": "Patient", "text": "Still there after a week."},
        ],
    }
    if modality:
        data["modality"] = modality
    return json.dumps(data)


def _found(modality: str | None) -> list[str]:
    flags = examination_flags(parse_transcript(_transcript(modality)), parse_note(NOTE))
    return [f.note_evidence.text for f in flags if f.note_evidence is not None]


def test_telephone_flags_every_finding() -> None:
    assert _found("telephone") == ["Chest clear", "Throat red", "abdomen soft and non-tender"]


def test_video_flags_only_hands_on_findings() -> None:
    assert _found("video") == ["Chest clear", "abdomen soft and non-tender"]


@pytest.mark.parametrize("modality", ["in-person", None])
def test_face_to_face_or_unknown_modality_does_not_run(modality: str | None) -> None:
    assert _found(modality) == []


def test_flags_are_possible_critical_warnings_whose_evidence_slices_back() -> None:
    note = parse_note(NOTE)
    for flag in examination_flags(parse_transcript(_transcript("telephone")), note):
        assert (flag.type, flag.verdict, flag.severity, flag.confidence) == (
            "unsupported",
            "unsupported",
            "critical",
            "possible",
        )
        assert flag.rule_id == "examination.remote"
        evidence = flag.note_evidence
        assert evidence is not None and note.text[evidence.start : evidence.end] == evidence.text


def test_rules_mode_reports_them() -> None:
    report = check(_transcript("telephone"), NOTE)
    assert report.counts.unsupported == 3 and report.decision == "warn"
