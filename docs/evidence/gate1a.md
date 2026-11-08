# Gate 1(a): recomputing the published OmissionBench figures

Gate 1(a) checks that `scripts/recompute_published.py` reproduces the figures the
OmissionBench dataset card publishes, from the released scored records alone, with no API
calls. The card is saved at `docs/references/omissionbench_dataset_card.md` (fetched
2026-11-08). The full run was made by a person on the released records. Its outputs are
`docs/evidence/recompute_published.txt` (full report) and
`docs/evidence/recompute_summary.txt` (`--summary`).

**Result: passed.** Every figure that can be recomputed matches the card. B3 replicate 1
cannot be fully recomputed, because some of its rows were not released (see below).

## Checks

| Check | Card | Recomputed | Match |
|---|---|---|---|
| FC-score-k8 (faithfulness and completeness, scored, eight-call ensemble), paired omissions, 495-pair evaluation set, mean of 3 replicates | 0.634 | 0.634 (r1 0.655, r2 0.645, r3 0.601) | yes |
| Reasoning-budget faithfulness-and-completeness judge (`power-FC-high`), paired omissions, 151-pair held-out subset, mean of 3 replicates | 0.670 | 0.670 (r1 0.687, r2 0.656, r3 0.668) | yes |
| Route two (`power-gepa-high`), `route-two-below-10`, held-out subset, strict majority of 3 replicates | 32.1% detection at 2.1% false alarms | 42/131 = 32.1% at 1/47 = 2.1% | yes |
| Route one (B3), `pipeline-critical-absent`, held-out subset, replicate 2 | 22.9% at 2.1% | 30/131 = 22.9% at 1/47 | yes |
| Route one (B3), replicate 3 | 21.4% at 2.1% | 28/131 = 21.4% at 1/47 | yes |
| Route one (B3), strict majority of 3 replicates | 21.4% | 28/131 = 21.4% at 1/47 | yes |
| Route one (B3), replicate 1 | 20.6% at 2.1% | not recomputable (see below) | — |
| RAGAS-style coverage recipe (ragas), paired omissions, 495-pair evaluation set, mean of 2 replicates | 0.809 | 0.808 (r1 0.824, r2 0.792) | yes, within the harness's tolerance |

For reference, with no card figure to compare against: route two's paired discrimination on
the held-out subset is 0.669 (r1 0.668, r2 0.664, r3 0.676). The card's paired 0.670 is the
`power-FC-high` row above, not route two.

## Notes on individual checks

**Rules.** The flag rules are the harness's own, found in the OmissionBench harness
(v1.0.4):

- Route two flags a note when its aggregate is below 10 (`FLAG_BELOW = 10.0`,
  `judges/w2_evalfull_analyze.py:44`).
- Route one flags a note when any fact graded critical is judged absent
  (`judges/w2_pipeline_replicates_analyze.py:99–108`).
- Replicates are combined by strict majority over the group's replicates. A replicate in
  which a note failed to parse counts as not flagged.

**B3 replicate 1.** The released records for this replicate cover 129 of the 151 errored
notes and 41 of the 47 clean notes, so 22 errored and 6 clean rows are absent. Over the rows
present, the rule gives 22/113 omission notes at 0/41 false alarms, which cannot be compared
with the card's 27/131 at 1/47. Replicates 2 and 3 cover all 198 notes and match, and so does
the strict majority of the three.

**ragas.** 0.808 against the card's 0.809. The harness's own figure check accepts 0.809
within ±0.0015 (`record/figures/extract_data.py:265–266`). Paired discrimination drops a pair
when either side failed to parse, the convention the harness uses for this arm
(`extract_data.py:80–87`), and the arm's parse failures are counted in its own group.

## Method

- Rows are grouped across a run's shard folders by family, model, arm, cell, criterion,
  format, k, tier, reasoning effort and dataset version, and scored per replicate.
- Paired discrimination compares each errored note with its own clean twin. A win is a
  strictly lower errored score, and a tie counts half.
- The detection and false-alarm figures apply the named rules above, with parse failures
  counted as not flagged on the full denominator.

Attribution: OmissionBench, Fox et al. (2026), CC BY 4.0, https://doi.org/10.57967/hf/10258.
The omission method is Fox et al. (arXiv:2608.31016), building on MED-OMIT.
