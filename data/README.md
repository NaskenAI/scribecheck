# Data

Nothing in this folder is committed except this file.

- `data/omissionbench/` holds the development portion of OmissionBench, downloaded during M1.
  It is CC BY 4.0, so keep the attribution in `README.md`.
- The held-out subset is never stored here. Before any Claude Code session touches the benchmark,
  a person moves the held-out consultations to `~/scribecheck-heldout`, or to the folder named in
  `SCRIBECHECK_HELDOUT_DIR`. Claude Code is blocked from that folder by `.claude/settings.json`
  and `.claude/hooks/guard_heldout.py`. Only the frozen Gate 2 run, started by a person, reads it.
