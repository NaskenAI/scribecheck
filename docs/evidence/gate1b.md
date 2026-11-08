# Gate 1(b): re-running the two published methods

Gate 1(b) checks that the published OmissionBench harness, run fresh on development data,
reproduces the released judgements. A person ran it on 2026-11-08, following
`docs/runbooks/gate1b.md`:

- **Harness:** v1.0.4, commit `bd64b4e`. Every run manifest records a clean harness tree.
- **Transport:** OpenRouter, as published.
- **Data:** development data only. The subset was the 51 notes of the runbook's 10
  consultations: 41 errored notes (23 omissions) and 10 clean notes.

The comparison was made with `scripts/compare_gate1b.py`. Its full output is
`runs/gate1b/compare.txt`, and the harness outputs are copied under
`runs/gate1b/harness-results/` (both gitignored).

**Result: passed.** Both methods pass the runbook's criteria. `compare_gate1b.py` exited 0.

## Route two: power-gepa-high (gpt-5.4, reasoning effort high), the below-10 flag

Per-note agreement on the `route-two-below-10` flag:

| Comparison | Agreement | Notes that disagree |
|---|---|---|
| New run vs released r1 | 46/51 (90.2%) | aci\|D2N156\|omit\|err; primock\|day1_consultation01\|change\|err; primock\|day1_consultation01\|omit\|err; primock\|day5_consultation07\|omit\|err; primock\|day5_consultation11\|clean\|60255ee169b5 |
| New run vs released r3 | 49/51 (96.1%) | aci\|D2N156\|omit\|err; primock\|day4_consultation10\|clean\|8fdce6b6faec |
| Released r1 vs r3 (reference) | 46/51 (90.2%) | primock\|day1_consultation01\|change\|err; primock\|day1_consultation01\|omit\|err; primock\|day4_consultation10\|clean\|8fdce6b6faec; primock\|day5_consultation07\|omit\|err; primock\|day5_consultation11\|clean\|60255ee169b5 |

**Pass mark:** 90%, because the released r1-versus-r3 agreement (90.2%) is not lower. The new
run reaches it against both r1 (90.2%) and r3 (96.1%). **Pass.**

The new run agrees with r1 exactly as well as r3 does, so it passes at the bar. A single
further disagreement with r1 would have failed it. Where the three runs differ:

- On four of the five notes where the new run disagrees with r1, r3 disagrees with r1 too.
  In each case the new run sides with r3.
- On one note (`primock|day4_consultation10|clean`), the new run sides with r1 against r3.
- On only one note (`aci|D2N156|omit|err`) does the new run disagree with both released
  replicates.

**Aggregate, new run vs r1:** identical on 35 of 51 notes, mean absolute difference 0.431
on the 0–10 scale, largest difference 3.

## Route one: B3 pipeline, the critical-absent flag

The run uses fresh fact extraction and audit, because the released fact lists were not
released, so per-note agreement does not gate.

| Criterion | New run | Released r1 | Margin | Result |
|---|---|---|---|---|
| Valid records (present, not a parse failure) | 51/51 | — | all 51 | pass |
| Detection on the 23 omission notes (`pipeline-critical-absent`) | 6/23 | 7/23 | ±4 | pass (difference 1) |
| False alarms on the 10 clean notes | 0/10 | 0/10 | ±1 | pass (difference 0) |

**Route one: pass.**

For information:

- **Per-note agreement with r1:** 48/51 (94.1%), above the 80% level that would call for
  investigation before M4. The notes that disagree are aci\|D2N177\|omitc\|trap6\|err,
  primock\|day5_consultation11\|omitpw\|mc27\|err and
  primock\|day5_consultation11\|omit\|err.
- **`n_absent_critical`, new run vs r1:** equal on 48 of 51 notes, mean absolute difference
  0.059, largest difference 1.

## Spend

From the harness's cost ledger (`runs/gate1b/harness-results/cost_ledger.jsonl`), in
OpenRouter credits:

| Run | Status | Calls | Cost | Wall time |
|---|---|---|---|---|
| Route two, `gate1b-power` `20261108-211735-bd64b4e-r1` | complete | 51 | $2.34 | 303 s |
| Route one, `gate1b-pipeline` `20261108-212342-bd64b4e-r1` (8 consultations) | complete | 88 | $5.32 | 829 s |
| Route one, `gate1b-pipeline` `20261108-213731-bd64b4e-r1` (2 consultations) | complete | 23 | $1.46 | 353 s |
| **Total** | | **162** | **$9.12** | |

That is under half the runbook's typical estimate (about $18), and far below its $60 credit
limit. Route one ran as two run folders because the harness opens one per 8 consultations
(`--chunk 8`).

## Notes

- **Seeds.** Per-note seeds differ from the released runs, because they come from each
  note's position among the loaded notes, and this run loaded the 265 development pairs, not
  495. The card notes seeds do not reproduce output token for token in any case.
- **Rules and margins.** The rules are the ones Gate 1(a) validated
  (`docs/evidence/gate1a.md`), applied by `scripts/compare_gate1b.py`. The margins and pass
  marks are those in `docs/runbooks/gate1b.md`, fixed before the run.

Attribution: OmissionBench, Fox et al. (2026), CC BY 4.0, https://doi.org/10.57967/hf/10258.
The omission method is Fox et al. (arXiv:2608.31016), building on MED-OMIT.
