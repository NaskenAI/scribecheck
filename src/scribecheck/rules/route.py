"""Route of administration.

The two-letter abbreviations IM, SC, SL and PR count only in uppercase, because
lowercase "im", "sc", "sl" and "pr" are ambiguous in speech transcripts; "PR interval"
is not a route. "IV" after a staging word ("stage IV", "type iv") is a Roman numeral.
"applied"/"apply" counts only when followed by "to".
"""

from __future__ import annotations

import re
from enum import Enum

from scribecheck.rules._match import Match, non_overlapping


class Route(str, Enum):
    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    SUBLINGUAL = "sublingual"
    RECTAL = "rectal"
    INHALED = "inhaled"
    TOPICAL = "topical"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"


class RouteMatch(Match):
    route: Route


_B = r"(?<![A-Za-z0-9])"
_E = r"(?![A-Za-z0-9])"

# (pattern, route, case-sensitive)
_RULES: tuple[tuple[str, Route, bool], ...] = (
    (r"oral(?:ly)?|PO|by\s+mouth", Route.ORAL, False),
    (r"IV|intravenous(?:ly)?", Route.INTRAVENOUS, False),
    (r"IM", Route.INTRAMUSCULAR, True),
    (r"intramuscular(?:ly)?", Route.INTRAMUSCULAR, False),
    (r"SC", Route.SUBCUTANEOUS, True),
    (r"subcutaneous(?:ly)?|sub-?cut", Route.SUBCUTANEOUS, False),
    (r"SL", Route.SUBLINGUAL, True),
    (r"sublingual(?:ly)?", Route.SUBLINGUAL, False),
    (r"PR(?!\s+interval)", Route.RECTAL, True),
    (r"rectal(?:ly)?", Route.RECTAL, False),
    (r"inhaled|inhaler|nebuli[sz]ed", Route.INHALED, False),
    (r"topical(?:ly)?|appl(?:y|ied)(?=\s+to" + _E + ")", Route.TOPICAL, False),
    (r"eye\s+drops?", Route.OPHTHALMIC, False),
    (r"ear\s+drops?", Route.OTIC, False),
)

# Words before a Roman numeral; "IV" after one of these is not a route.
_NUMERAL_CONTEXT = re.compile(
    r"(?:stage|grade|type|class|phase|level|category|tier|part|chapter|group|nyha)\s+$",
    re.IGNORECASE,
)

_PATTERNS = tuple(
    (re.compile(_B + "(?:" + rule + ")" + _E, 0 if exact else re.IGNORECASE), route)
    for rule, route, exact in _RULES
)


def find_routes(text: str) -> tuple[RouteMatch, ...]:
    """Find every route of administration in `text`, left to right."""
    candidates = [
        RouteMatch(start=m.start(), end=m.end(), text=m.group(), route=route)
        for pattern, route in _PATTERNS
        for m in pattern.finditer(text)
        if not (m.group().lower() == "iv" and _NUMERAL_CONTEXT.search(text, 0, m.start()))
    ]
    return non_overlapping(candidates)
