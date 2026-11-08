from __future__ import annotations

from typing import Any

import pytest

from scribecheck.decision import apply_standard, decide
from scribecheck.flags import Flag


def _flag(**changes: Any) -> Flag:
    fields: dict[str, Any] = {
        "id": "F1",
        "type": "omission",
        "category": "medicine",
        "verdict": "absent",
        "severity": "supporting",
        "confidence": "possible",
        "origin": "rules",
    }
    fields.update(changes)
    return Flag(**fields)


def test_no_flags_pass() -> None:
    assert decide([], mode="rules") == "pass"
    assert decide([], mode="full", fail_on=["critical"]) == "pass"


@pytest.mark.parametrize("severity", ["critical", "supporting"])
def test_critical_and_supporting_warn(severity: str) -> None:
    assert decide([_flag(severity=severity)], mode="rules") == "warn"


def test_peripheral_is_report_only() -> None:
    assert decide([_flag(severity="peripheral")], mode="rules") == "pass"


def test_partial_verdict_warns_even_when_peripheral() -> None:
    assert decide([_flag(severity="peripheral", verdict="partial")], mode="full") == "warn"


@pytest.mark.parametrize("confidence", ["certain", "confirmed", "possible"])
def test_rules_mode_never_fails(confidence: str) -> None:
    flags = [_flag(severity="critical", confidence=confidence)]
    assert (
        decide(
            flags,
            mode="rules",
            fail_on=["critical", "supporting"],
            fail_confidence=["certain", "confirmed", "possible"],
        )
        == "warn"
    )


def test_full_mode_fails_only_when_opted_in() -> None:
    flags = [_flag(severity="critical", confidence="confirmed")]
    assert decide(flags, mode="full") == "warn"
    assert decide(flags, mode="full", fail_on=["critical"]) == "fail"


def test_full_mode_fail_needs_severity_and_confidence() -> None:
    critical_possible = [_flag(severity="critical", confidence="possible")]
    supporting_confirmed = [_flag(severity="supporting", confidence="confirmed")]
    assert decide(critical_possible, mode="full", fail_on=["critical"]) == "warn"
    assert decide(supporting_confirmed, mode="full", fail_on=["critical"]) == "warn"
    assert decide(supporting_confirmed, mode="full", fail_on=["critical", "supporting"]) == "fail"
    assert (
        decide(
            critical_possible,
            mode="full",
            fail_on=["critical"],
            fail_confidence=["possible"],
        )
        == "fail"
    )


def test_clinical_standard_drops_peripheral_omissions_only() -> None:
    peripheral_omission = _flag(id="F1", severity="peripheral")
    peripheral_change = _flag(
        id="F2", type="changed_fact", verdict="changed", severity="peripheral"
    )
    supporting = _flag(id="F3")
    flags = [peripheral_omission, peripheral_change, supporting]
    assert apply_standard(flags, "clinical") == (peripheral_change, supporting)
    assert apply_standard(flags, "strict") == tuple(flags)
