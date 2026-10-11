# First Claude Code sessions

Start each session fresh. Begin in plan mode, approve the plan, then let Claude implement.

## Session 1, Person B track: M2 week 1, input parsers

> Read CLAUDE.md and the M2 section of @docs/build-plan.md, plus the "Data model and report
> format" section of @docs/spec.md. In plan mode, propose the modules and tests for: plain-text
> transcripts with speaker prefixes, JSON transcripts, and note section detection with a
> user-supplied heading map. Speaker labels are optional. Don't write code until I approve.

After approval:

> Implement the plan. Write the tests first, then the code. Use synthetic examples only. Run
> ruff, mypy and pytest, fix any failures, then commit with a descriptive message.

## Session 2, Person A track: M1, check the scoring code against the paper

Before this session, a person downloads OmissionBench and the harness, and moves the held-out
consultations to `~/scribecheck-heldout`. Clone the harness outside this repository.

> Read the M1 section of @docs/build-plan.md. Write `scripts/recompute_published.py`, which
> recomputes the paper's published detection and false-alarm figures from the released
> judgements. Write a test that runs it on a small synthetic judgements file. Make no API
> calls. Don't open anything under the held-out folder; a person will run the script on the
> full evaluation set.

## Before every commit

> Use a subagent to review this diff against the task in @docs/build-plan.md. Report only gaps
> that affect correctness or the stated requirements.
