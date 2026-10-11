# scribecheck

Checks AI-written clinical notes against the consultation transcript for omissions,
changed facts and unsupported statements. Spec: `docs/spec.md`. Build plan:
`docs/build-plan.md`. Read only the section the current task needs.

## Commands

- Install: `python -m pip install -e ".[dev]"`
- Test: `python -m pytest -q`
- Lint and format: `python -m ruff check .` and `python -m ruff format .`
- Types: `python -m mypy`

## Non-negotiable rules

- The fact extractor never sees the note.
- Every flag carries an evidence quote that the code verifies in the transcript or note it quotes.
- Rules mode makes no network calls and never calls the model client.
- Tests run behind the network guard in `tests/conftest.py`. Never weaken, skip or bypass it.
- No telemetry, analytics or update checks anywhere.
- Core dependencies are `httpx` and `pydantic` only. Anything else is an optional extra; ask before adding one.
- Synthetic data only in tests, fixtures, examples and logs. Never real patient data.
- IMPORTANT: never read, copy, list or reference held-out benchmark data. Two folders hold it, both
  outside this repo: the OmissionBench held-out subset (default `~/scribecheck-heldout`, override
  `SCRIBECHECK_HELDOUT_DIR`) and the raw OmissionBench download, which contains held-out material
  (default `~/data/omissionbench-raw`, override `SCRIBECHECK_RAW_DIR`). Only the frozen Gate 2 run,
  started by a person, may use them. Read deny rules in `.claude/settings.json` cover the default
  paths; the guard hook also covers overridden ones. Tune on development data only.
- The omission method is Fox et al. (arXiv:2608.31016), building on MED-OMIT. Keep the attribution
  in the README and docs.

## Workflow

- One build-plan task per session. Use plan mode for anything touching more than one module.
- Write or update tests first. A task is done when ruff, mypy and pytest pass and the change is committed.
- Python 3.10+, type hints on all public functions, `src/` layout.
