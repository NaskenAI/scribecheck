# Correct-fact matching: words and a model judge (claude-medium-v4-full)

Correct detection asks whether a flag on an omission note names the benchmark's removed fact.
Until now a word matcher decided that (`scripts/score_run.py`, `similarity` and `comparable`,
threshold 0.5). On the full development run it leaves 31 removed facts as "weak match", several
plainly the same fact in other words, so it understates correct detection.

## Method

- **Judge:** `scripts/fact_judge.py`, called by `score_run.py judge`: one structured call per
  omission note (148), Claude Sonnet 5.5 at effort low. It answers yes or no, with the ids, for
  whether any extracted fact, and separately any Pass A omission flag or partial verdict, states
  the same clinical fact as the removed fact.
- **The judge is the same model family as the system under test** (Claude Sonnet 5.5). It could
  share the system's blind spots, so its matches are not taken on trust:
  - **It sees fact texts only:** the removed fact, our extracted facts and our flags' facts.
    Never the transcript, the note or any verdict.
  - **The word matcher is kept as a cross-check.** Every note where the two disagree is listed
    (`score --judgements`, `misses --judgements`).
  - **A person spot-checks 30 sampled pairs**, drawn with a fixed seed (20261109) by
    `score_run.py sample` into `docs/review/correct_fact_sample.csv`. The agreement rate below is
    reported alongside every model-assisted figure.
- **Cache:** `runs/m3s4/claude-medium-v4-full/fact_judgements.json`, keyed by the prompt version,
  the model and every text sent, so re-scoring makes no calls.

Runbook: `docs/runbooks/m3s4/run_judge.sh`. Development data only.

## Results

_Pending: the user runs `run_judge.sh`._

| Correct detection (148 omission notes) | Words | Model |
|---|---|---|
| Decision rule | 46/148 | _pending_ |
| Route one's rule | 37/148 | _pending_ |
| Detected notes where words and model differ | | _pending_ |

## Human spot-check (30 pairs)

_Pending: a person fills `reviewer_same` and `reviewer_comment` in
`docs/review/correct_fact_sample.csv`._

| Agreement with the reviewer | Rate |
|---|---|
| Model judge | _pending_ /30 |
| Word matcher | _pending_ /30 |
