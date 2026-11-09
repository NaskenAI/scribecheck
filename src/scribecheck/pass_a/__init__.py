"""Pass A with the model: two configurations behind one interface.

`run_pass_a` runs `config.pipeline`, either scribecheck's own pipeline (`pipeline.py`) or
Fox et al.'s route one reproduced from the OmissionBench harness (`route_one.py`), over a
list of cases, live or through Message Batches, and returns each note's findings and usage.
"""

from __future__ import annotations

from collections.abc import Sequence

from scribecheck.config import Config
from scribecheck.model.anthropic import AnthropicClient
from scribecheck.model.base import ModelClient
from scribecheck.model.budget import Budget
from scribecheck.pass_a import pipeline, route_one
from scribecheck.pass_a.base import BatchRunner, Case, Ledger, LiveRunner, Outcome, Runner
from scribecheck.pass_a.blind import BlindnessError

__all__ = ["BlindnessError", "Case", "Outcome", "prompt_set", "run_pass_a"]


def prompt_set(config: Config) -> str:
    """The prompt folder (and version) the configured pipeline uses."""
    return route_one.PROMPTS if config.pipeline == "route_one" else pipeline.PROMPTS


def run_pass_a(
    cases: Sequence[Case],
    config: Config,
    client: ModelClient,
    budget: Budget,
    *,
    live: bool = True,
) -> dict[str, Outcome]:
    """Each case's Pass A findings. Folder runs pass `live=False` to use Message Batches
    when the provider offers them (Anthropic); otherwise calls are made one by one."""
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Pass A cases need distinct ids")
    runner: Runner
    if not live and isinstance(client, AnthropicClient):
        runner = BatchRunner(client, budget)
    else:
        runner = LiveRunner(client, budget)
    outcomes = {c.id: Outcome() for c in cases}
    ledger = Ledger(outcomes, budget, batch=runner.batch)
    if config.pipeline == "route_one":
        route_one.run(cases, config, runner, ledger, outcomes)
    else:
        pipeline.run(cases, config, runner, ledger, outcomes)
    return outcomes
