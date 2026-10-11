from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scribecheck.errors import InputError
from scribecheck.transcript import (
    DEFAULT_SPEAKER_MAP,
    Transcript,
    parse_transcript,
    parse_transcript_file,
)

LABELLED = """Doctor: Hello, what brings you in today?
Patient: My left knee has been swelling
for about two weeks now.
It hurts on the stairs.
Doctor: Any allergies to medicines?
Patient: Penicillin, it gives me a rash.
"""


def assert_round_trip(transcript: Transcript) -> None:
    assert transcript.text is not None
    for turn in transcript.turns:
        assert turn.start is not None and turn.end is not None
        assert transcript.text[turn.start : turn.end] == turn.text


def as_json(payload: Any) -> str:
    return json.dumps(payload)


# Plain text


@pytest.mark.parametrize("text", ["", "   ", "\n\n  \t\n"])
def test_empty_plain_text_is_an_input_error(text: str) -> None:
    with pytest.raises(InputError, match="empty"):
        parse_transcript(text, format="text")


def test_labelled_turns_are_numbered_from_one() -> None:
    transcript = parse_transcript(LABELLED)
    assert [t.number for t in transcript.turns] == [1, 2, 3, 4]
    assert [t.speaker for t in transcript.turns] == [
        "clinician",
        "patient",
        "clinician",
        "patient",
    ]
    assert transcript.has_speaker_labels
    assert transcript.modality is None
    assert_round_trip(transcript)


def test_multi_line_turn_keeps_its_newlines() -> None:
    transcript = parse_transcript(LABELLED)
    assert transcript.turns[1].text == (
        "My left knee has been swelling\nfor about two weeks now.\nIt hurts on the stairs."
    )


@pytest.mark.parametrize(
    ("label", "role"),
    [
        ("Doctor", "clinician"),
        ("DOCTOR", "clinician"),
        ("Dr", "clinician"),
        ("DR", "clinician"),
        ("dr.", "clinician"),
        ("Clinician", "clinician"),
        ("patient", "patient"),
        ("PATIENT", "patient"),
    ],
)
def test_speaker_label_variants(label: str, role: str) -> None:
    transcript = parse_transcript(f"{label}: Any chest pain?\n")
    (turn,) = transcript.turns
    assert turn.speaker == role
    assert turn.label == label
    assert turn.text == "Any chest pain?"
    assert_round_trip(transcript)


def test_spaces_around_label_and_colon() -> None:
    transcript = parse_transcript("  Doctor :   Any chest pain?\n Patient:No.\n")
    assert [(t.speaker, t.text) for t in transcript.turns] == [
        ("clinician", "Any chest pain?"),
        ("patient", "No."),
    ]
    assert_round_trip(transcript)


def test_turn_text_can_start_on_the_next_line() -> None:
    transcript = parse_transcript("Doctor:\n  Any chest pain?\nPatient:\nNo.")
    assert [t.text for t in transcript.turns] == ["Any chest pain?", "No."]
    assert_round_trip(transcript)


def test_prefix_with_no_text_adds_no_turn() -> None:
    transcript = parse_transcript("Doctor:\nPatient: Hello.\n")
    assert [(t.number, t.speaker) for t in transcript.turns] == [(1, "patient")]


def test_colon_lines_outside_the_map_stay_in_the_turn() -> None:
    transcript = parse_transcript("Patient: She said:\nMother: he has had a fever.\n")
    (turn,) = transcript.turns
    assert turn.text == "She said:\nMother: he has had a fever."


def test_custom_speaker_map_replaces_the_defaults() -> None:
    text = "Nurse: How are you?\nDoctor: Not a label here.\nPt: Fine.\n"
    transcript = parse_transcript(text, speaker_map={"nurse": "clinician", "pt": "patient"})
    assert [(t.speaker, t.text) for t in transcript.turns] == [
        ("clinician", "How are you?\nDoctor: Not a label here."),
        ("patient", "Fine."),
    ]
    assert_round_trip(transcript)


def test_speaker_map_keys_are_case_insensitive() -> None:
    transcript = parse_transcript("NURSE: Hello.\n", speaker_map={"Nurse": "clinician"})
    assert transcript.turns[0].speaker == "clinician"


def test_default_speaker_map() -> None:
    assert DEFAULT_SPEAKER_MAP == {
        "doctor": "clinician",
        "dr": "clinician",
        "clinician": "clinician",
        "patient": "patient",
    }


def test_no_speaker_labels_gives_one_unknown_turn_per_line() -> None:
    text = "Any chest pain?\n\nNo, none at all.\n  Any allergies?  \n"
    transcript = parse_transcript(text)
    assert [(t.number, t.speaker, t.label, t.text) for t in transcript.turns] == [
        (1, "unknown", None, "Any chest pain?"),
        (2, "unknown", None, "No, none at all."),
        (3, "unknown", None, "Any allergies?"),
    ]
    assert not transcript.has_speaker_labels
    assert_round_trip(transcript)


def test_text_before_the_first_label_is_an_unknown_turn() -> None:
    text = "Recording starts.\nConsent given.\nDoctor: Hello.\n"
    transcript = parse_transcript(text)
    assert [(t.speaker, t.text) for t in transcript.turns] == [
        ("unknown", "Recording starts.\nConsent given."),
        ("clinician", "Hello."),
    ]
    assert transcript.has_speaker_labels
    assert_round_trip(transcript)


def test_quote_slices_within_a_turn() -> None:
    transcript = parse_transcript(LABELLED)
    assert transcript.quote(4, 0, 10) == "Penicillin"
    assert transcript.quote(2, 8, 12) == "knee"
    with pytest.raises(IndexError):
        transcript.quote(5, 0, 1)
    with pytest.raises(IndexError):
        transcript.quote(0, 0, 1)


def test_consultation_id_can_be_given_for_plain_text() -> None:
    transcript = parse_transcript("Doctor: Hello.", consultation_id="demo-002")
    assert transcript.consultation_id == "demo-002"


def test_windows_line_endings_are_normalised() -> None:
    lf = parse_transcript(LABELLED)
    crlf = parse_transcript(LABELLED.replace("\n", "\r\n"))
    assert crlf == lf
    assert crlf.text is not None and "\r" not in crlf.text


def test_models_are_immutable() -> None:
    transcript = parse_transcript(LABELLED)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        transcript.turns[0].text = "changed"  # type: ignore[misc]
    with pytest.raises(Exception):  # noqa: B017
        transcript.consultation_id = "x"  # type: ignore[misc]


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_transcript("Doctor: hi", format="xml")  # type: ignore[arg-type]


# JSON

VALID: dict[str, Any] = {
    "consultation_id": "demo-001",
    "modality": "telephone",
    "turns": [
        {"speaker": "clinician", "text": "Any allergies to medicines?"},
        {"speaker": "patient", "text": "Penicillin, it gives me a rash."},
    ],
}


def test_valid_json() -> None:
    transcript = parse_transcript(as_json(VALID))
    assert transcript.consultation_id == "demo-001"
    assert transcript.modality == "telephone"
    assert transcript.text is None
    assert [(t.number, t.speaker, t.label) for t in transcript.turns] == [
        (1, "clinician", "clinician"),
        (2, "patient", "patient"),
    ]
    assert all(t.start is None and t.end is None for t in transcript.turns)
    assert transcript.has_speaker_labels
    assert transcript.quote(2, 0, 10) == "Penicillin"


def test_json_turn_text_is_kept_exactly() -> None:
    payload = {**VALID, "turns": [{"speaker": "patient", "text": "  No.\n  Never.  "}]}
    transcript = parse_transcript(as_json(payload))
    assert transcript.turns[0].text == "  No.\n  Never.  "


def test_json_speakers_go_through_the_speaker_map() -> None:
    payload = {
        **VALID,
        "turns": [
            {"speaker": "Doctor", "text": "Hello."},
            {"speaker": "Nurse", "text": "Hi."},
            {"speaker": None, "text": "Mm."},
            {"text": "Yes."},
        ],
    }
    transcript = parse_transcript(as_json(payload))
    assert [(t.speaker, t.label) for t in transcript.turns] == [
        ("clinician", "Doctor"),
        ("nurse", "Nurse"),
        ("unknown", None),
        ("unknown", None),
    ]
    assert transcript.has_speaker_labels


def test_json_without_any_speakers_has_no_labels() -> None:
    payload = {"consultation_id": "demo-003", "turns": [{"text": "Hello."}]}
    transcript = parse_transcript(as_json(payload))
    assert not transcript.has_speaker_labels
    assert transcript.modality is None


@pytest.mark.parametrize(
    ("modality", "expected"),
    [("in-person", "in-person"), ("video", "video"), (" Telephone ", "telephone")],
)
def test_valid_modalities(modality: str, expected: str) -> None:
    transcript = parse_transcript(as_json({**VALID, "modality": modality}))
    assert transcript.modality == expected


def test_null_modality_is_none() -> None:
    assert parse_transcript(as_json({**VALID, "modality": None})).modality is None


@pytest.mark.parametrize("modality", ["phone", "in person", "", 3, ["video"]])
def test_invalid_modality_is_an_input_error(modality: Any) -> None:
    with pytest.raises(InputError, match="modality"):
        parse_transcript(as_json({**VALID, "modality": modality}))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"turns": VALID["turns"]}, "consultation_id"),
        ({**VALID, "consultation_id": ""}, "consultation_id"),
        ({**VALID, "consultation_id": 7}, "consultation_id"),
        ({"consultation_id": "x"}, "turns"),
        ({**VALID, "turns": []}, "turns"),
        ({**VALID, "turns": "Doctor: hi"}, "turns"),
        ({**VALID, "turns": [{"speaker": "patient"}]}, "text"),
        ({**VALID, "turns": [{"speaker": "patient", "text": ""}]}, "text"),
        ({**VALID, "turns": [{"speaker": "patient", "text": 5}]}, "text"),
        ({**VALID, "turns": [{"speaker": "patient", "text": "  \n "}]}, "text"),
        ({**VALID, "consultation_id": "   "}, "consultation_id"),
        ({**VALID, "turns": [{"speaker": 5, "text": "hi"}]}, "speaker"),
        ({**VALID, "turns": ["hi"]}, "turns"),
    ],
)
def test_missing_or_invalid_json_fields(payload: dict[str, Any], match: str) -> None:
    with pytest.raises(InputError, match=match):
        parse_transcript(as_json(payload))


def test_json_top_level_must_be_an_object() -> None:
    with pytest.raises(InputError, match="object"):
        parse_transcript("[1, 2]", format="json")


def test_malformed_json_is_an_input_error() -> None:
    with pytest.raises(InputError, match="JSON"):
        parse_transcript('{"consultation_id": "x", ')


def test_empty_json_input_is_an_input_error() -> None:
    with pytest.raises(InputError, match="empty"):
        parse_transcript("  ", format="json")


def test_extra_json_fields_are_ignored() -> None:
    payload = {**VALID, "source": "synthetic", "turns": [{"text": "Hi.", "ts": 1.5}]}
    assert parse_transcript(as_json(payload)).turns[0].text == "Hi."


# Files


def test_json_file(tmp_path: Path) -> None:
    path = tmp_path / "demo-001.json"
    path.write_text(as_json(VALID), encoding="utf-8")
    assert parse_transcript_file(path).consultation_id == "demo-001"


def test_text_file_uses_the_stem_as_id_and_normalises_crlf(tmp_path: Path) -> None:
    path = tmp_path / "demo-004.txt"
    path.write_bytes(LABELLED.replace("\n", "\r\n").encode("utf-8"))
    transcript = parse_transcript_file(path)
    assert transcript.consultation_id == "demo-004"
    assert transcript.turns == parse_transcript(LABELLED).turns


def test_text_file_with_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfDoctor: Hello.\n")
    transcript = parse_transcript_file(path)
    assert transcript.turns[0].speaker == "clinician"
    assert_round_trip(transcript)


def test_text_file_with_non_ascii_text(tmp_path: Path) -> None:
    path = tmp_path / "dose.txt"
    path.write_text("Doctor: Take 500 µg — once daily.\n", encoding="utf-8")
    assert parse_transcript_file(path).turns[0].text == "Take 500 µg — once daily."


def test_invalid_utf8_is_an_input_error(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("Doctor: café".encode("latin-1"))
    with pytest.raises(InputError, match="UTF-8") as exc:
        parse_transcript_file(path)
    assert str(path) in str(exc.value)


def test_missing_file_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="missing.txt"):
        parse_transcript_file(tmp_path / "missing.txt")


def test_json_file_errors_name_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(as_json({**VALID, "modality": "phone"}), encoding="utf-8")
    with pytest.raises(InputError, match="bad.json"):
        parse_transcript_file(path)


def test_deeply_nested_json_is_an_input_error() -> None:
    with pytest.raises(InputError):
        parse_transcript("[" * 100_000 + "]" * 100_000, format="json")
    with pytest.raises(InputError):
        parse_transcript('{"a": ' + "[" * 100_000 + "]" * 100_000 + "}")


def test_auto_format_rejects_a_json_array() -> None:
    with pytest.raises(InputError, match="object"):
        parse_transcript('[{"speaker": "Doctor", "text": "Hello."}]')


def test_auto_format_keeps_bracketed_plain_text() -> None:
    transcript = parse_transcript("[00:01] Recording starts.\nDoctor: Hello.\n")
    assert [t.speaker for t in transcript.turns] == ["unknown", "clinician"]


def test_auto_format_detects_json_after_a_bom() -> None:
    transcript = parse_transcript("\ufeff" + as_json(VALID))
    assert transcript.consultation_id == "demo-001"


def test_leading_bom_in_plain_text_is_dropped() -> None:
    transcript = parse_transcript("\ufeffDoctor: Hello.")
    assert transcript.text == "Doctor: Hello."
    assert transcript.turns[0].speaker == "clinician"
