# scribecheck_v5 prompts

The prompts for scribecheck's own Pass A pipeline (docs/spec.md, "Pass A: omissions").

- `system.txt` is the same in every call, and the transcript (then the note, where a stage may
  see it) is sent first in its own cache-marked block, so a consultation's calls share a cached
  prefix. The files here are the task text that follows.
- `extract.txt` and `audit.txt` never receive the note; the pipeline also checks every
  extraction and audit request for note text before sending it.
- The severity rubric in `audit.txt`, and the shape of the check, refute-style second look and
  extraction instructions, follow Fox et al. (arXiv:2608.31016) and their OmissionBench harness
  (Composo, MIT; see `../route_one/LICENSE`), adapted to quote-verified output.

Changing any file here changes results: bump the folder version (`scribecheck_v6`) rather than
editing it after a measured run.

Earlier sets are kept unchanged, as measured runs used them:

- `scribecheck_v2` (from v1): `candidate.txt` says that a drug named in a recounted past
  episode is not a current medicine.
- `scribecheck_v3` (from v2): facts are atomic. `extract.txt` makes each medicine, denied
  symptom, plan decision (including decisions not to treat), safety-net instruction, numeric
  threshold and allergy status its own fact; `audit.txt` drops only true duplicates;
  `check.txt` judges each fact as its own statement (a missing statement is absent, not
  partial); `second_look.txt` rescues only a fact the note carries, not one it gestures at.
- `scribecheck_v4` (from v3): `audit.txt` makes current medicines (including occasional or
  as-needed use), allergies and allergy status, plan decisions (including decisions not to
  treat), safety-net instructions and numeric thresholds always core, never contextual;
  `second_look.txt` rescues a fact the note carries in any wording, including a more specific
  restatement, but never by a vaguer statement when the fact gives a number, threshold, dose,
  drug, side or specific instruction, and never by a different item of the same kind.
- `scribecheck_v5` (from v4): `audit.txt` leaves attendance and appointment logistics (whether
  to attend in person, booking, when a call will come, transport) out of the always-core plan
  rule; they are core or contextual by the ordinary rule. The pipeline's backstop never makes
  them core either (`is_logistics` in `pass_a/pipeline.py`).

Pass B (changed facts) uses `changes.txt` and `changes_second.txt`; Pass C (unsupported
statements) uses `split.txt`, `statements.txt` and `statements_second.txt`. Their severity
wording follows the same rubric.
