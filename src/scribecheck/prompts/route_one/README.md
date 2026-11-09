# route_one prompts

These five prompts are Fox et al.'s per-fact pipeline ("route one"), copied byte for byte from
the OmissionBench harness, so scribecheck's `route_one` configuration runs the published method.

- Source: the OmissionBench harness by Composo, `w2_prompts/` at commit
  `bd64b4e09fc5e84d6fb920c53ecda9f9874a45b9` (release v1.0.4), github.com/composo-ai/omission-bench,
  archived at Zenodo DOI 10.5281/zenodo.22160954.
- Method: Fox S, Markham L, Lail R, Karotsieris M. *LLM Judges Verify Presence, Not Absence:
  Omission Blindness in AI Clinical Notes and What Recovers It.* arXiv:2608.31016 (2026), building
  on MED-OMIT (Schumacher et al., 2023).
- Licence: MIT, Copyright (c) 2026 Composo Limited; the full text is in `LICENSE` in this folder.

| File | Stage |
| --- | --- |
| `pipeline_extract.txt` | 1: facts from the transcript only |
| `pipeline_audit.txt` | 1b: drop unsupported, duplicate or not-note-bearing facts; grade severity |
| `pipeline_check_tri.txt` | 2: full / partial / absent per fact, keyed by id |
| `pipeline_check_binary.txt` | 2 (binary tier; kept for completeness, not used by default) |
| `pipeline_refute.txt` | 3: quote-checked rescue of facts judged absent |

The files are unmodified. `tests/test_prompts.py` checks each against the SHA-256 recorded in the
harness's `PROMPTS.sha256`; do not edit them.
