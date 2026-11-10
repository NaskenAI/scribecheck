# v5: a negative result

**Result.** `scribecheck_v5` was worse than `scribecheck_v4` at every matched false-alarm
rate. It is kept in the repository, unused. The default prompt set is `scribecheck_v4`, frozen
as the M4 candidate (git tag `m4-candidate-v4`); under it the v5 code (the partial look and the
audit backstops) does not run.

**Run.** Both runs used the full development set: 65 consultations, 148 omission notes and 65
clean notes, with Claude Sonnet 5.5 at effort medium, Pass A only and the strict standard. v4 is
`runs/m3s4/claude-medium-v4-full` and v5 is `runs/m3s4/claude-medium-v5-full`. The scores are in
`runs/m3s4/score_v5_full.txt`, which is git-ignored because it holds benchmark fact text. This
file holds counts only.

**Correct detection** uses the word matcher (match threshold 0.5) for both runs. v5 was not
judged by the model, so its numbers compare with v4's word-matcher numbers, not with v4's judged
61/148.

## What v5 changed

v5 made four changes; the first three are tagged so one run can show each one's effect.

| Change | Tag |
|---|---|
| Partial look: for each partial verdict on a core fact with a specific element, ask whether that element is missing from the note | `pass_a.partial_element` |
| Audit: pertinent negatives and examination findings are always core | `core_kind` |
| Audit: plan, follow-up, safety-net and instruction facts are never dropped (logistics excepted) | `never_drop` |
| Second look: each confirmed absence is graded `clear` or `probable`, and that grade is the graded score's first key | `absence` |

## At the decision rule

| | v4 | v5 |
|---|---|---|
| Detection on omission notes | 70/148 | 116/148 |
| Correct detection (words) | 46/148 | 55/148 |
| Detected notes that name the removed fact | 46/70 (65.7%) | 55/116 (47.4%) |
| False alarms on clean notes | 6/65 | 38/65 |
| Route one's rule: detection | 54/148 | 85/148 |
| Route one's rule: correct detection (words) | 37/148 | 37/148 |
| Route one's rule: false alarms | 4/65 | 23/65 |
| Add notes with a Pass A flag (information) | 4/54 | 36/54 |
| Change notes with a Pass A flag (information) | 21/63 | 51/63 |
| Partial verdicts listed as information | 565 | 251 |

v5 found 9 more correct detections at a cost of 32 more false alarms on clean notes.

## At matched false-alarm rates (graded note score)

At each released reference's false-alarm rate, this is the most lenient graded-score threshold
that does not pass it.

| Matched to | v4: detection | v5: detection | v4: correct (words) | v5: correct (words) |
|---|---|---|---|---|
| B3 r1, 2/65 | 32/148 | 26/148 | 18/148 | 10/148 |
| power-gepa-high r3, 5/65 | 62/148 | 33/148 | 42/148 | 16/148 |
| power-gepa-high r1, 7/65 | 73/148 | 46/148 | 48/148 | 26/148 |

## Each change's flags (v5 run, decision rule)

"With" counts notes that have at least one of the change's flags. "Only" counts notes that
nothing else flags, which is what the change adds and what it costs.

| Change | View | Detection | Correct (words) | False alarms |
|---|---|---|---|---|
| partial_element | with | 91/148 | 12/148 | 34/65 |
| partial_element | only | 33/148 | 9/148 | 24/65 |
| core_kind | with | 0/148 | 0/148 | 0/65 |
| never_drop | with | 0/148 | 0/148 | 0/65 |
| clear | with | 55/148 | 19/148 | 12/65 |
| clear | only | 14/148 | 8/148 | 5/65 |
| probable | with | 101/148 | 36/148 | 33/65 |
| probable | only | 56/148 | 25/148 | 26/65 |

## Findings

- **Element-level checking of partial verdicts adds far more false alarms than correct
  detections.** On its own it flags 24 of the 65 clean notes and adds 9 correct detections.
  Only 12 of the 91 omission notes carrying a partial_element flag are named correctly by
  it.
- **The second look's confidence grade did not separate true flags from false ones.** Notes
  flagged only by "clear" absences give 14 detections at 5 false alarms. With "clear" as the
  graded score's first key, v5 is below v4 at every matched rate.
- **The audit backstops never fired.** No decision-rule flag carries a `core_kind` or
  `never_drop` tag, so the code never overrode the audit. Whatever effect v5's audit had came
  from the prompt wording, which one run cannot separate from the other changes.
- **The partial look also flagged add and change notes.** Add notes with a Pass A flag rose
  from 4/54 to 36/54. That fits a check that finds a "missing element" in notes whose
  omission is not what was changed.
