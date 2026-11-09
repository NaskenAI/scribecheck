"""Full mode's model passes, run one after another over a list of cases.

Pass A runs in the configured pipeline (scribecheck's or route one). Pass B's model path
and Pass C run only in the scribecheck pipeline: route one stays omission-only, as
published. `config.passes` chooses which passes run (benchmark runs use Pass A only).
A single check calls the model live; a folder run sends each stage as one Message Batch
(Anthropic) unless `live` is set.
"""

from __future__ import annotations

from collections.abc import Sequence

from scribecheck import pass_b, pass_c
from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import ModelClient
from scribecheck.model.budget import Budget
from scribecheck.pass_a import pipeline, route_one
from scribecheck.pass_a.base import BatchRunner, Case, Ledger, LiveRunner, Outcome, Runner


def pass_modes(config: Config) -> dict[str, str]:
    """What each pass ran as: the pipeline or "model", "rules" only, or "off"."""
    scribecheck = config.pipeline == "scribecheck"
    return {
        "a": config.pipeline if "a" in config.passes else "off",
        "b": ("model" if scribecheck else "rules") if "b" in config.passes else "off",
        "c": ("model" if scribecheck else "rules") if "c" in config.passes else "off",
    }


def run_passes(
    cases: Sequence[Case],
    config: Config,
    client: ModelClient,
    budget: Budget,
    *,
    live: bool = True,
) -> dict[str, Outcome]:
    """Each case's findings from the model passes `config.passes` selects."""
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("cases need distinct ids")
    runner: Runner
    if not live and isinstance(client, AnthropicClient):
        runner = BatchRunner(client, budget)
    else:
        runner = LiveRunner(client, budget)
    outcomes = {c.id: Outcome() for c in cases}
    ledger = Ledger(outcomes, budget, batch=runner.batch)
    modes = pass_modes(config)
    if modes["a"] == "route_one":
        route_one.run(cases, config, runner, ledger, outcomes)
    elif modes["a"] == "scribecheck":
        pipeline.run(cases, config, runner, ledger, outcomes)
    if modes["b"] == "model":
        pass_b.run(cases, config, runner, ledger, outcomes)
    if modes["c"] == "model":
        pass_c.run(cases, config, runner, ledger, outcomes)
    return outcomes
