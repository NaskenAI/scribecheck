"""Token usage, cost from configured prices, and the per-run cost cap.

No price is built in: cost is computed only from `Prices` set in config, and is None
without them. A cap needs prices, because it cannot be enforced otherwise. Calls run
one at a time and the cap is checked before each, so a run can overshoot it by at most
one call; a batch is checked before submission against a worst-case estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scribecheck.errors import InputError
from scribecheck.model.base import Usage

_PER = 1_000_000


@dataclass(frozen=True)
class Prices:
    """USD per million tokens, from config."""

    input: float
    output: float
    cache_write_5m: float | None = None
    """Defaults to the input price when unset."""
    cache_write_1h: float | None = None
    cache_read: float | None = None
    batch_multiplier: float | None = None
    """What batch usage costs relative to live calls; needed to price batches."""

    def cost(self, usage: Usage, *, batch: bool = False) -> float:
        rate_5m = self.input if self.cache_write_5m is None else self.cache_write_5m
        rate_1h = self.input if self.cache_write_1h is None else self.cache_write_1h
        rate_read = self.input if self.cache_read is None else self.cache_read
        total = (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_write_5m_tokens * rate_5m
            + usage.cache_write_1h_tokens * rate_1h
            + usage.cache_read_tokens * rate_read
        ) / _PER
        if batch:
            if self.batch_multiplier is None:
                raise InputError("batch prices need [model.prices] batch_multiplier")
            total *= self.batch_multiplier
        return total


@dataclass
class Budget:
    """Usage and spend for one run, and its cap."""

    prices: Prices | None = None
    cap_usd: float | None = None
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    _spent: float = 0.0

    def __post_init__(self) -> None:
        if self.cap_usd is not None and self.prices is None:
            raise InputError("a cost cap needs prices: set [model.prices]")
        if self.cap_usd is not None and self.cap_usd < 0:
            raise InputError("the cost cap must not be negative")

    @property
    def spent_usd(self) -> float | None:
        """Estimated spend so far, or None without prices."""
        return None if self.prices is None else self._spent

    def record(self, usage: Usage, *, batch: bool = False, calls: int = 1) -> None:
        self.usage = self.usage + usage
        self.calls += calls
        if self.prices is not None:
            self._spent += self.prices.cost(usage, batch=batch)

    def can_spend(self, estimate_usd: float = 0.0) -> bool:
        """Whether a call costing up to `estimate_usd` may start without passing the cap."""
        if self.cap_usd is None:
            return True
        if self._spent >= self.cap_usd:
            return False
        return self._spent + estimate_usd <= self.cap_usd

    def estimate(self, input_chars: int, max_tokens: int, *, batch: bool = False) -> float:
        """A worst-case cost for one call: two characters per input token, every input
        token at the dearest input rate (a cache write may cost more than plain input), and
        all `max_tokens` used."""
        if self.prices is None:
            return 0.0
        p = self.prices
        rate = max(r for r in (p.input, p.cache_write_5m, p.cache_write_1h) if r is not None)
        total = ((input_chars // 2 + 1) * rate + max_tokens * p.output) / _PER
        if batch:
            if p.batch_multiplier is None:
                raise InputError("batch prices need [model.prices] batch_multiplier")
            total *= p.batch_multiplier
        return total
