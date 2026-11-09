# scribecheck_v1 prompts

The prompts for scribecheck's own Pass A pipeline (docs/spec.md, "Pass A: omissions").

- `system.txt` is the same in every call, and the transcript (then the note, where a stage may
  see it) is sent first in its own cache-marked block, so a consultation's calls share a cached
  prefix. The files here are the task text that follows.
- `extract.txt` and `audit.txt` never receive the note; the pipeline also checks every
  extraction and audit request for note text before sending it.
- The severity rubric in `audit.txt`, and the shape of the check, refute-style second look and
  extraction instructions, follow Fox et al. (arXiv:2608.31016) and their OmissionBench harness
  (Composo, MIT; see `../route_one/LICENSE`), adapted to quote-verified output.

Changing any file here changes results: bump the folder version (`scribecheck_v2`) rather than
editing it after a measured run.

Pass B (changed facts) uses `changes.txt` and `changes_second.txt`; Pass C (unsupported
statements) uses `split.txt`, `statements.txt` and `statements_second.txt`. Their severity
wording follows the same rubric.
