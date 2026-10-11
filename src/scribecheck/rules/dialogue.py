"""Dialogue negation: "Any chest pain?" then "No." negates chest pain.

Sentence-level rules miss this, because the denial sits in the patient's turn, not
in the question's sentence. In v1 a binding needs all of:

- speaker labels in the transcript, a clinician turn, and a patient turn straight
  after it (unknown or other speakers never bind);
- a clinician turn that ends with a presence question ("Any ...?", "Do you have
  ...?", "Have you had ...?", "How about ...?") naming exactly one target, with no
  other target in the turn's questions, no alternative ("or"), no condition ("... if
  you walk fast?", "on exertion", "at night"), no family or time qualifier ("in your
  family", "before", "last week", "ever"), and the target not itself negated,
  possible, historical, hypothetical or about family;
- a patient turn that starts with a direct negative (no, nope, nah, never, none) and
  says nothing else beyond a restatement ("No, I haven't", "Nah, no fever"). Hedges
  ("not really", "sometimes", "only when ..."), other targets, extra detail ("No,
  it's worse at night") and question marks mean no binding.

Questions about several targets go to the model. As in rules/context.py, the caller
supplies the target spans.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from scribecheck.rules._context import sentence_span
from scribecheck.rules._match import Match
from scribecheck.rules.context import classify
from scribecheck.transcript import Transcript, Turn

_LEAD_IN = r"(?:(?:and|so|ok|okay|right|now|also|then|well)\b[\s,]*)*"
_PRESENCE_QUESTION = re.compile(
    rf"^\s*{_LEAD_IN}(?:any|got\s+any|how\s+about|what\s+about|"
    r"do\s+you\s+(?:have|get)|"
    r"have\s+you\s+(?:had|got|noticed|been\s+having|been\s+getting)|"
    r"are\s+you\s+(?:having|getting)|did\s+you\s+(?:have|get|notice)|"
    r"is\s+there|has\s+there\s+been)(?![A-Za-z])",
    re.IGNORECASE,
)
_QUALIFIER = re.compile(
    r"(?<![A-Za-z])(?:if|when|whenever|after|during|while|unless|or|family|relatives?|"
    r"history|before|ago|last|yesterday|previously|ever|past|with|exertion|exercise|"
    r"night|morning|evening|rest|walking|lying|eating)(?![A-Za-z])",
    re.IGNORECASE,
)
_NEGATIVE_ANSWER = re.compile(r"^\W*(?P<cue>nope|none|never|nah|no)(?![A-Za-z'’])", re.IGNORECASE)
_ANSWER_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
# Words a bare negative answer may contain besides a restatement of the target.
_ANSWER_WORDS = frozenset(
    {
        "no", "nope", "nah", "never", "none", "not", "at", "all", "i", "i've", "ive",
        "haven't", "havent", "have", "don't", "dont", "do", "didn't", "didnt", "did",
        "hasn't", "had", "got", "any", "that", "it", "one", "thanks", "thank", "you",
        "doctor", "doc",
    }
)  # fmt: skip


class DialogueNegation(BaseModel):
    """A target in a clinician's question, negated by the patient's answer."""

    model_config = ConfigDict(frozen=True)

    question_turn: int
    answer_turn: int
    target: Match
    """The target, with offsets into the question turn's text."""
    answer: Match
    """The negative word, with offsets into the answer turn's text."""


def bind_negations(
    transcript: Transcript,
    targets: Mapping[int, Sequence[tuple[int, int]]],
) -> tuple[DialogueNegation, ...]:
    """Bind single-target questions to the patient's direct negative answers.

    `targets` maps turn numbers (from 1) to target spans within that turn's text.
    Raises ValueError for an unknown turn number or a span outside its turn.
    """
    for number, spans in targets.items():
        if not 1 <= number <= len(transcript.turns):
            raise ValueError(f"turn {number} out of range 1..{len(transcript.turns)}")
        length = len(transcript.turns[number - 1].text)
        for start, end in spans:
            if not 0 <= start < end <= length:
                raise ValueError(f"target span ({start}, {end}) is not inside turn {number}")
    if not transcript.has_speaker_labels:
        return ()
    results = []
    for question, answer in zip(transcript.turns, transcript.turns[1:], strict=False):
        if question.speaker != "clinician" or answer.speaker != "patient":
            continue
        found = _bind(
            question, targets.get(question.number, ()), answer, targets.get(answer.number, ())
        )
        if found is not None:
            results.append(found)
    return tuple(results)


def _bind(
    question: Turn,
    question_targets: Sequence[tuple[int, int]],
    answer: Turn,
    answer_targets: Sequence[tuple[int, int]],
) -> DialogueNegation | None:
    sentences = _sentences(question.text)
    if not sentences or not sentences[-1][2]:
        return None
    in_questions = [
        span
        for span in question_targets
        if any(start <= span[0] < end for start, end, is_question in sentences if is_question)
    ]
    if len(in_questions) != 1:
        return None
    target_start, target_end = in_questions[0]
    last_start, last_end, _ = sentences[-1]
    if not last_start <= target_start < last_end:
        return None
    asked = question.text[last_start:last_end]
    if not _PRESENCE_QUESTION.match(asked) or _QUALIFIER.search(asked):
        return None
    (context,) = classify(question.text, [(target_start, target_end)])
    if context.attributes:
        return None

    negative = _NEGATIVE_ANSWER.match(answer.text)
    if negative is None or "?" in answer.text:
        return None
    target_text = _key(context.text)
    if any(_key(answer.text[start:end]) != target_text for start, end in answer_targets):
        return None
    rest = list(answer.text)
    for start, end in answer_targets:
        rest[start:end] = " " * (end - start)
    words = _ANSWER_WORD.findall("".join(rest))
    if any(word.lower().replace("’", "'") not in _ANSWER_WORDS for word in words):
        return None
    cue_start, cue_end = negative.span("cue")
    return DialogueNegation(
        question_turn=question.number,
        answer_turn=answer.number,
        target=Match(start=target_start, end=target_end, text=context.text),
        answer=Match(start=cue_start, end=cue_end, text=negative.group("cue")),
    )


def _sentences(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, ends_with_question_mark) for each non-blank sentence."""
    sentences = []
    position = 0
    while position < len(text):
        start, end = sentence_span(text, position, position)
        if text[start:end].strip():
            sentences.append((start, end, text[end : end + 1] == "?"))
        position = end + 1
    return sentences


def _key(text: str) -> str:
    return " ".join(text.split()).casefold()
