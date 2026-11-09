"""Quote verification with one re-check, shared by both Pass A pipelines.

Each item's quote is looked for with `quotes.find_in_transcript` or `find_in_note`. Items
whose quote is not found are re-asked together, once per note (one call, or one batch
request), with the `quote_recheck` prompt; an item still not found has no evidence, and the
pipeline decides what that means (scribecheck: unchecked "quote_unverified"; route one: the
flag stands, marked unverified).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from scribecheck import prompts
from scribecheck.config import Config
from scribecheck.flags import NoteEvidence, TranscriptEvidence
from scribecheck.model.base import Block, Completion, Message, ModelRequest
from scribecheck.pass_a.base import Case, Ledger, Runner, facts_block, render_transcript
from scribecheck.quotes import find_in_note, find_in_transcript

PROMPTS = "scribecheck_v5"
Source = Literal["transcript", "note"]


class _Quote(BaseModel):
    id: str
    quote: str


class Requoted(BaseModel):
    quotes: list[_Quote]


@dataclass
class _Item:
    case: Case
    key: str
    text: str
    quote: str
    owners: tuple[str, ...] = ()
    found: TranscriptEvidence | NoteEvidence | None = None


@dataclass
class QuoteFixer:
    """Finds quotes, re-asking once for those not found."""

    runner: Runner
    ledger: Ledger
    config: Config
    source: Source = "transcript"
    _items: dict[tuple[str, str], _Item] = field(default_factory=dict)

    def want(
        self, case: Case, key: str, text: str, quote: str, owners: tuple[str, ...] = ()
    ) -> None:
        """Look for `quote`; `owners` are the notes the item serves (default: `case`)."""
        item = _Item(case, key, text, quote, owners or (case.id,))
        item.found = self._find(case, quote)
        self._items[(case.id, key)] = item

    def resolve(self) -> None:
        missing: dict[str, list[_Item]] = {}
        for item in self._items.values():
            if item.found is None:
                missing.setdefault(item.case.id, []).append(item)
        if not missing:
            return
        requests, owners = [], {}
        for items in missing.values():
            case = items[0].case
            ids = {f"q{i + 1}": item for i, item in enumerate(items)}
            task = prompts.render(
                prompts.load(PROMPTS, "quote_recheck"),
                source=self.source.upper(),
                unit="turn" if self.source == "transcript" else "passage",
                facts_block=facts_block((qid, item.text) for qid, item in ids.items()),
            )
            blocks = [Block("TRANSCRIPT:\n" + render_transcript(case.transcript), cache=True)]
            if self.source == "note":
                blocks.append(Block("NOTE:\n" + case.note.text, cache=True))
            blocks.append(Block(task))
            request = ModelRequest(
                system=prompts.load(PROMPTS, "system"),
                messages=(Message("user", tuple(blocks)),),
                output=Requoted,
                custom_id=self.ledger.custom_id("requote"),
                max_tokens=self._max_tokens(),
            )
            requests.append(request)
            owners[request.custom_id] = (items[0].owners, ids)
        for cid, result in self.runner.run(requests).items():
            served, ids = owners[cid]
            self.ledger.record(result, list(served))
            if not isinstance(result, Completion):
                continue
            for entry in result.parsed.quotes:
                wanted = ids.get(entry.id.strip().lower())
                if wanted is not None and wanted.found is None and entry.quote.strip():
                    wanted.found = self._find(wanted.case, entry.quote)

    def evidence(self, case: Case, key: str) -> TranscriptEvidence | NoteEvidence | None:
        item = self._items.get((case.id, key))
        return None if item is None else item.found

    def _find(self, case: Case, quote: str) -> TranscriptEvidence | NoteEvidence | None:
        if not quote.strip():
            return None
        if self.source == "transcript":
            return find_in_transcript(case.transcript, quote)
        return find_in_note(case.note, quote)

    def _max_tokens(self) -> int:
        settings = self.config.stages.get("scribecheck.quote_recheck")
        return settings.max_tokens if settings and settings.max_tokens else 4000
