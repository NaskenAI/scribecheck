"""Regression tests for the Pass A review's findings (mocked HTTP only)."""

from __future__ import annotations

import json
from typing import Any

from _mock_http import Recorder, json_response, openai_completion
from _mock_model import MockClaude, text_of
from pydantic import BaseModel
from test_pass_a_pipeline import NOTE, TRANSCRIPT, run, scripts
from test_pass_a_route_one import _scripts

from scribecheck.config import Config
from scribecheck.flags import Flag, TranscriptEvidence
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import Block, Message, ModelFailure, ModelRequest
from scribecheck.model.budget import Budget
from scribecheck.model.endpoint import Secret
from scribecheck.model.openai_compat import OpenAICompatClient
from scribecheck.model.structured import complete
from scribecheck.note import parse_note
from scribecheck.pass_a import Case, run_pass_a
from scribecheck.pass_a.blind import assert_blind
from scribecheck.transcript import parse_transcript


def test_short_notes_do_not_trip_the_blindness_rail() -> None:
    note = "I have had a dry cough for two weeks."
    assert_blind(f"TRANSCRIPT: Patient: {note}", [note], "extract")  # under 80: skipped


def test_dedupe_matches_whole_words_and_only_medicines_and_allergies() -> None:
    def extract(_: Any, body: dict[str, Any]) -> Any:
        return {
            "facts": [
                {
                    "fact": "Home environment is damp",
                    "category": "family_social",
                    "attributes": [],
                    "evidence": "I've had a cough for two weeks",
                },
            ]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        rows = [
            {"id": i, "verdict": "absent", "note_quote": ""}
            for i, t in listed
            if "environment" in t
        ]
        rows += [
            {"id": i, "verdict": "present", "note_quote": "Metformin 500 mg BD."}
            for i, t in listed
            if "environment" not in t
        ]
        return {"verdicts": rows}

    mock = MockClaude(scripts(extract=extract, check=check))
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    t, n = parse_transcript(TRANSCRIPT, consultation_id="x"), parse_note(NOTE)
    iron = Flag(
        id="F0",
        type="omission",
        category="medicine",
        verdict="absent",
        severity="supporting",
        confidence="possible",
        origin="rules",
        rule_id="medicine.missing",
        fact="iron",
        transcript_evidence=TranscriptEvidence(text="I've had a cough", turn=2, start=0, end=16),
    )
    case = Case("c0", t, n, (iron,))
    config = Config(mode="full", provider="anthropic", model="m")
    outcome = run_pass_a([case], config, client, Budget())["c0"]
    facts = {f.fact for f in outcome.flags}
    assert "Home environment is damp" in facts  # "iron" is not a word in "environment"


def test_failures_count_every_call_made() -> None:
    recorder = Recorder(*(json_response(openai_completion("not json")) for _ in range(2)))
    client = OpenAICompatClient("m", base_url="http://localhost:1/v1", transport=recorder.transport)

    class Out(BaseModel):
        ok: bool

    request = ModelRequest(system="s", messages=(Message("user", (Block("q"),)),), output=Out)
    failure = complete(client, request, Budget())
    assert isinstance(failure, ModelFailure) and failure.attempts == 2


def test_prompt_only_retry_resends_the_identical_prompt() -> None:
    recorder = Recorder(
        json_response(openai_completion("no json here")),
        json_response(openai_completion('{"ok": true}')),
    )
    client = OpenAICompatClient("m", base_url="http://localhost:1/v1", transport=recorder.transport)

    class Out(BaseModel):
        ok: bool

    request = ModelRequest(
        system="", messages=(Message("user", (Block("q"),)),), output=Out, json_mode="prompt"
    )
    complete(client, request, Budget())
    assert recorder.body(0) == recorder.body(1)


def test_unparseable_route_one_check_gets_the_targeted_retry() -> None:
    scripts_ = _scripts([True])  # no missing id on the first answer
    calls: list[int] = []

    def check(listed: list[tuple[str, str]], body: dict[str, Any]) -> Any:
        calls.append(1)
        if len(calls) <= 2:
            return "I cannot answer in JSON today."  # both attempts unparseable
        return {"verdicts": {i: "full" for i, _ in listed}}

    scripts_["r1_check"] = check
    mock = MockClaude(scripts_)
    client = AnthropicClient("claude-sonnet-5-5", api_key=Secret("k"), transport=mock.transport)
    config = Config(mode="full", provider="anthropic", model="m", pipeline="route_one")
    case = Case("c0", parse_transcript(TRANSCRIPT, consultation_id="x"), parse_note(NOTE))
    outcome = run_pass_a([case], config, client, Budget())["c0"]
    assert mock.stages().count("r1_check") == 3  # one call, its retry, the targeted retry
    assert not [u for u in outcome.unchecked if u.item == "presence check"]
    assert outcome.flags == []


def test_route_one_on_openai_sends_effort_and_the_harness_temperature() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        body = json.loads(request.content)
        sent.append(body)
        return json_response(openai_completion('{"facts": []}'))

    import httpx

    client = OpenAICompatClient(
        "m", base_url="http://localhost:1/v1", transport=httpx.MockTransport(handler)
    )
    config = Config(
        mode="full",
        provider="openai",
        model="m",
        base_url="http://localhost:1/v1",
        pipeline="route_one",
    )
    case = Case("c0", parse_transcript(TRANSCRIPT, consultation_id="x"), parse_note(NOTE))
    run_pass_a([case], config, client, Budget())
    assert sent[0]["reasoning_effort"] == "medium" and sent[0]["temperature"] == 1.0
    assert "response_format" not in sent[0]


def test_a_failed_chunk_keeps_the_other_chunks_facts() -> None:
    turns = (
        "\n".join(
            f"{'Doctor' if i % 2 else 'Patient'}: Synthetic line number {i} about a dry cough."
            for i in range(1, 41)
        )
        + "\n"
    )
    seen: list[str] = []

    def extract(_: Any, body: dict[str, Any]) -> Any:
        seen.append(text_of(body))
        if len(seen) <= 2:  # the first chunk fails twice
            return "garbled"
        return {
            "facts": [
                {
                    "fact": "Dry cough",
                    "category": "symptom",
                    "attributes": [],
                    "evidence": "about a dry cough",
                }
            ]
        }

    def check(listed: list[tuple[str, str]], _: dict[str, Any]) -> Any:
        return {
            "verdicts": [
                {"id": i, "verdict": "present", "note_quote": "Cough for 2 weeks."}
                for i, _ in listed
            ]
        }

    mock = MockClaude(scripts(extract=extract, check=check))
    outcome, _ = run(mock, transcript=turns, candidates=False, chunk_chars=1200)
    assert [u.item for u in outcome.unchecked if u.category == "pass_a"] == ["extraction"]
    assert [f.text for f in outcome.facts] == ["Dry cough"]
