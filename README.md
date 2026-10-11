# scribecheck

Checks an AI-written clinical note against the consultation transcript, and reports omissions,
changed facts and unsupported statements, each with the passage that supports the flag.

> **Status: pre-alpha.** Only the package skeleton exists. Checking arrives milestone by
> milestone; see `docs/build-plan.md`. Nothing here is ready for use yet.

## What it checks, and what it doesn't

- It checks whether the note is faithful to the transcript, not whether the transcript is correct.
  A misheard drug name copied faithfully into the note will pass.
- It is an evaluation tool for testing scribe systems, not a sign-off check on individual
  patients' notes in live care.
- v1 supports English only.

## Privacy

- It runs on your machine. Rules mode makes no network calls; full mode talks only to the model
  endpoint you configure.
- There is no telemetry, analytics or update checking.

## Attribution

The omission-checking method follows Fox S, Markham L, Lail R, Karotsieris M,
[*LLM Judges Verify Presence, Not Absence: Omission Blindness in AI Clinical Notes and What
Recovers It*](https://arxiv.org/abs/2608.31016), arXiv:2608.31016 (2026), which builds on
MED-OMIT (Schumacher et al., 2023). Evaluation uses OmissionBench by Composo (data CC BY 4.0,
harness MIT), which is built partly on PriMock57 and ACI-Bench.

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check . && python -m ruff format --check .
python -m mypy
```

Instructions for AI coding agents are in `CLAUDE.md`.

## Licence

Apache-2.0. See `LICENSE`.
