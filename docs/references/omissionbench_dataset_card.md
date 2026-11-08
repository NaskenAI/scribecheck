<!--
Source: https://huggingface.co/datasets/ComposoAI/OmissionBench (the dataset card shown on
that page, saved as its published Markdown source, unchanged below this comment).
Fetched: 2026-11-08.
Licence: CC BY 4.0, Fox, S., Markham, L., Lail, R. and Karotsieris, M., OmissionBench,
https://doi.org/10.57967/hf/10258. The card's own "Licence and attribution" section is kept in full.
-->

---
license: cc-by-4.0
pretty_name: OmissionBench
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
tags:
  - clinical
  - evaluation
  - llm-as-a-judge
  - omission-detection
  - ambient-scribe
  - benchmark
configs:
  - config_name: pairs
    data_files: pairs/dataset_v2.json
    field: pairs
  - config_name: fact_sites
    data_files: pairs/fact_sites.json
    field: consultations
  - config_name: note_structure
    data_files: notes/structure/note_structure.json
    field: notes
dataset_info:
  dataset_version: factorial-v1
---

# OmissionBench

A benchmark for measuring whether an automated judge can detect that a clinical note has
left something out.

The data is for evaluating LLM judges and ambient scribe systems. It is not for clinical
use of any kind, and it is not training data for clinical systems. Nothing in it is a real
patient encounter or a clinical record.

Each consultation in the benchmark carries a fact sheet stating what a correct note must
and must not say, a verified clean note, and single-error pairs built from that clean note
by injecting exactly one defect of a known type. A judge is scored on whether it ranks the
errored note below its own clean twin, and on how often it flags the clean twin. Omissions
are graded on two axes kept separate throughout: how much the missing fact matters
clinically, and how much of it survives elsewhere in the note.

The benchmark exists because public corpora could not supply the ground truth. Every
clinician reference note we audited is materially discrepant with its own transcript, at 7
to 11 discrepancies per note, so "what the note should have said" had to be constructed
and verified rather than assumed.

Two companion papers, released concurrently, report on this data. The **judge paper** -
*LLM Judges Verify Presence, Not Absence: Omission Blindness in AI Clinical Notes and
What Recovers It* - is the benchmark's home: the presence/absence asymmetry in LLM
judges, its mechanism, and the designs that recover omission detection. The **census
paper** - *One note in three: a verified census of three deployed AI scribes, and the
instrument that counted it* - reports the audit of the three products whose verified
findings ship in `judgements/findings/`.

- **Harness and prompts**: `github.com/composo-ai/omission-bench` (MIT). The per-run manifests are in
  this repository rather than that one, beside each run under `judgements/judges/` and
  `judgements/census/`.
- **Dataset version**: `factorial-v1`

## Using the benchmark

Three configs load from the Hub, one row per pair, per consultation and per note:

```python
from datasets import load_dataset

pairs = load_dataset("ComposoAI/OmissionBench", "pairs", split="train")            # 500 rows
sites = load_dataset("ComposoAI/OmissionBench", "fact_sites", split="train")       # one row per mapped consultation
structure = load_dataset("ComposoAI/OmissionBench", "note_structure", split="train")  # 565 rows
```

The severity grades (`pairs/factorial_severity.json`) are not row-shaped, so read that file
directly. Its top level holds the build's own metadata (`_attribution`, `created`) alongside a
`facts` object, and it is `facts` that is keyed by fact id, with 880 entries.
The key is a pair record's `fact_uid`, so
`json.load(open(path))["facts"][pair["fact_uid"]]` gives both graders' verdicts behind that
pair's consensus grade, for the 185 pairs that carry a `fact_uid`. Every file is plain
JSON, and for the pair set that is usually easier anyway, because the records sit under a
top-level `pairs` key beside the build's own counts. One scope note on that `counts`
block: `omission_by_residual_level` and `omission_grid_all_sources` are computed over the
495 evaluation pairs, while its other tables cover all 500 records:

```python
import json, urllib.request

url = "https://huggingface.co/datasets/ComposoAI/OmissionBench/resolve/main/pairs/dataset_v2.json"
doc = json.load(urllib.request.urlopen(url))
evaluation_set = [p for p in doc["pairs"] if p["eval_set"]]
len(evaluation_set)   # 495
```

### Fields on a pair record

| field | type | values |
|---|---|---|
| `pair_id` | string | `stratum\|consultation\|edit\|fact`, e.g. `aci\|D2N072\|omitc\|mc6` |
| `key` | string | `stratum\|consultation` - the consultation this pair belongs to |
| `id` | string | the consultation's own id inside its stratum, e.g. `D2N072`, `day1_consultation01` |
| `stratum` | string | `aci`, `primock`, `authored`, `trapblind` |
| `eval_set` | bool | `true` for the 495 evaluation pairs. **Filter on this.** |
| `split` | string | `eval`, or `gepa_dev` for the 5 pairs the prompt optimiser trained on |
| `held_out_reason` | string or null | `trained_by_prompt_optimizer` on those 5, null otherwise |
| `type` | string | `omit`, `add`, `change` |
| `class` | string | `omit-complete`, `omit-partial`, `add`, `change` |
| `residual_level` | string or null | `complete`; `partial-weak`, a fragment trace, where a fragment of the removed statement or its immediate neighbour survives; `partial-strong`, a restatement trace, where a second independently stated mention of the same fact survives. Null on commissions |
| `severity` | string | `critical`, `supporting`, `peripheral` |
| `severity_source` | string | how the grade was reached: `rubric_2arm_consensus`, `trap_importance`, `rubric_graded_at_freeze`, ... |
| `severity_rationale` | string | the grader's reason, in prose |
| `cell` | string or null | `residual_level\|severity`, the factorial cell; null on commissions |
| `source` | string | the construction cohort: `factorial` (185), `frozen_add_change` (202), `frozen_omit` (79), `partial_seed` (34) |
| `construction.method` | string | five values across the 500 records. `programmatic_span_delete` (88): span deletion off the site map, no model call at all. `constructor_tidy` (53): the same deletion plus a grammar repair. `constructor_cleanup` (42): the deletion orphaned a neighbouring clause and the model dropped it, each edit logged in the record. `programmatic_span_delete_cleanup_rejected` (2): the validator rejected that cleanup, so the raw deletion stood. `v1_model_rewrite` (315): the carried-over cohort, built by model rewrite |
| `clean` | string | the verified clean note. **Included in full.** |
| `errored` | string | the same note with exactly one defect injected. **Included in full.** |
| `change` | string | what was done to make `errored`, in prose |
| `fact` / `what` | string | the fact that was removed, added or altered |
| `target` | object or null | the fact-sheet entry that was targeted. Always `kind` and `index`. The 124 `must_contain` targets add `fact` and `load_bearing`; the 303 `trap` targets add `trap`, `correct_handling`, `mode` and `importance` instead; 73 records have no target |
| `residual` | object or null | what survives the omission: `n_surviving`, `max_strength` and `sites[]` on the 219 omission pairs that have a residual reading. Null on the 202 commissions and the 79 carried-over complete omissions. The 114 verified factorial partial omissions also carry the verification panel's own `verified_n` and `verified_sites[]`, which can differ from the mapped count |
| `verification` | object | the cross-family panel's verdict, per criterion |
| `provenance` | object | varies by construction cohort: span-deleted pairs carry `builder`, `site_map` and `deterministic`; model-built pairs carry `model`, `effort` and `clean_note_source`; relabelled seed pairs carry `relabelled_by` and `relabelled_utc` |
| `verified` | bool | the pair passed its own construction check: `true` on 495, `false` on 5 carried-over partial seeds where one of the four semantic checks did not come back true. Not the same thing as `eval_set` - 3 of those 5 are in the evaluation set anyway |
| `dataset_version` | string | `factorial-v1` on every record |
| `severity_flagged`, `severity_arms`, `fact_uid`, `fact_kind`, `bucket`, `n_sites_total`, `removed_site_ids`, `kept_site_ids`, `primary_site_id`, `matched_pair_ids` | mixed | present on the 185 factorial pairs only. `severity_arms` is each grader's own grade, under `opus5` and `gpt55`; `severity_flagged` is `true` on the 36 pairs where the two disagreed and `severity` took the lower grade |

The note text of every pair is in the file. The pairs are built from our own repaired
reference notes, not from any scribe system's output, so nothing about them is withheld.

### The metric

**Paired discrimination.** For each pair, ask the judge to score the errored note and its
own clean twin, and count the pair as a win if the errored note scores strictly lower.
Ties count half. Chance is 0.500. The measure is relative and needs the clean reference,
which is why the clean twin ships with every pair.

```python
def paired_discrimination(scores):
    # scores: an iterable of (errored_score, clean_score)
    scores = list(scores)
    wins = sum(1 for e, k in scores if e < k)
    ties = sum(1 for e, k in scores if e == k)
    return (wins + 0.5 * ties) / len(scores)
```

**Clean-twin flag rate.** A judge's **flag rule** is the fixed threshold or verdict that
turns its output into a flag. The clean-twin flag rate is the share of the 112
clean notes a judge flags under its own flag rule. Quote it beside any absolute detection
rate; a detection number without its false-alarm rate is not a result. The 112
clean twins are the distinct `clean` notes of the evaluation set, one per consultation.

**Protocol.** A judge sees the consultation transcript and one note, and scores that
note. It never sees the clean and errored notes together - the eight single-prompt designs
in the table below open with the transcript, then the note, and nothing else - so detecting
an omission means noticing that something in the transcript is missing from the note.
Pairing happens afterwards, in scoring. The one deliberate exception is a fact-list
ablation, which inserts a fact list extracted from the transcript before the note is seen;
it uses its own prompt (`grid_FC_score_factlist.txt`) and is reported separately. The
transcripts are in `transcripts/`, keyed by `stratum/consultation.txt`.

### Baselines to beat

On the 495-pair evaluation set - 293 omission pairs and 202 commission pairs -
paired and tie-adjusted, judge `gpt-5.4`. The first two rows hold eight single-prompt
designs, one for each combination of three choices: whether the judge is told to check
faithfulness alone or faithfulness and completeness, whether it answers yes/no or on a 0-10
scale, and whether it is asked once or as an eight-call ensemble combined by a winsorised
mean, the extreme scores replaced by the nearest kept values before averaging.

| judge design | paired omissions | paired commissions |
|---|---|---|
| best of the eight (faithfulness and completeness, scored, eight-call ensemble) | **0.634** | 0.939 |
| the other seven of the eight | 0.500 to 0.591 | 0.792 to 0.944 |
| naive RAGAS-style coverage recipe (ours) | 0.809 | 0.712 |
| checklist judge (ours, about 25 items generated per transcript) | 0.647 | 0.744 |
| G-Eval-style judge, two dimensions (ours) | 0.568 | 0.890 |

The last three rows are our own implementations in the style of published methods, with
their own prompts in the harness under `w2_prompts/`. None of them is a run of the RAGAS
library or of any other shipped tool, and each one's flag rule is our runner's convention
rather than anything the method itself specifies. The coverage recipe is deliberately
naive, checking roughly 62 unaudited transcript facts per consultation, and its two paired
figures rest on 289 omission and 200 commission pairs per replicate rather than the full
293 and 202, because parse failures cost that arm 14 of its 1,214 records. Every other row
is on the full evaluation set.

Across the eight single-prompt designs, commissions run 0.792 to 0.944 while omissions run
from exactly chance to 0.634 with every prompt-side lever engaged, at the settings those
eight share: no reasoning budget and a 1,024-token answer cap, chosen because latency and
cost per note are the budgets real evaluation teams work to. The two designs that enumerate
facts invert that ordering. The coverage recipe and the checklist are the only rows above
0.634 on omissions, and both sit below all eight on commissions, which is mechanically what
a coverage score should do, since a fabrication does not reduce coverage. The G-Eval-style
judge splits the judgement into criteria without enumerating facts, and it behaves like the
eight on both axes, so what recovers omissions is enumeration rather than decomposition in
general.

Further results come from a 151-pair held-out subset rather than the full evaluation set,
and are not directly comparable with the table above. The subset is not a separate file;
it is recoverable as the distinct pair ids in any released `w2-power` or `w2-pipeline`
confirmation run's scored records:

- a two-stage decomposition pipeline - extract the facts from the transcript, then make one
  keyed present-or-absent call per note - reaches **0.786** paired on omissions, the mean of
  three runs spanning 0.771 to 0.801. Adding a third stage that audits the fact list and
  grades each fact's severity before any note is judged makes a note-level rule expressible:
  flag the note if any fact the audit graded critical is judged absent. That rule gives
  **20.6% detection at a 2.1% false-alarm rate** (1 of 47 clean notes) on the first of its
  three runs, which is the lowest of the three - runs two and three read 22.9% and 21.4%,
  a strict majority of the three lands at 21.4%, and the false alarm is the same single
  clean note every time. The rule cannot be stated on the two-stage tier at all, because
  that tier's missing facts carry no severity grade;
- giving those same single-prompt judges - each one reads the whole note in a single prompt
  and returns one holistic judgement - a reasoning budget (high reasoning effort, a
  12,000-token cap, prompt text byte-identical to the unpowered run) lifts the
  faithfulness-and-completeness judge asked once from 0.570 to **0.670** as a mean of three
  runs, and its fact-list-in-prompt variant to **0.699** in a single run - about half the
  remaining distance to the pipeline, at 2.9 times the two-stage pipeline's per-note judge
  cost;
- a prompt-optimised judge run with that reasoning budget scores almost every clean note at
  the top of its scale, so "flag anything scored below 10" becomes a flag rule: **32.1%
  detection at a 2.1% false-alarm rate** (strict majority over three replicates; 55%
  on omissions of critical facts), the study's best single-call operating point.

The judge paper carries the full tables, the conditioning by residual level and severity,
and the significance tests.

## What is in the bundle

| path | what |
|---|---|
| `pairs/dataset_v2.json` | the benchmark: 500 graded pairs, 495 in the evaluation set |
| `pairs/fact_sites.json` | per consultation, every located statement site for each candidate fact |
| `pairs/factorial_severity.json` | per-fact severity grades, both graders' verdicts and the consensus |
| `transcripts/` | 141 consultation transcripts across four strata, plus their attribution |
| `fact_sheets/` | per-consultation fact sheets, full and core views, including the authored answer key |
| `judgements/findings/` | every verified finding of the census over the scribe notes, with transcript-side evidence; `dedup_groups.json`, the hand-checked duplicate groupings behind the 618-findings-to-265-distinct-errors estimate; and `census_sensitivity_noprefill.json`: the census recomputed with the two failure classes a patient record would have prefilled removed, all four rungs with the intervals the census paper's appendix points at; and, from v1.0.4, `cluster_assignments.json`: which of the census paper's 17 subcategories each of the 618 findings belongs to (55 are marked unassigned), with the cluster labels, the clustering recipe and the paper's two cautions on using them |
| `judgements/judges/` | every benchmark judge run, one directory per run: `manifest.json`, the run's scored per-note records in a JSON file named after its runner, the raw per-call log (`per_call.jsonl.gz`) and the prompts it used (`prompts.json`). The prompt-optimiser campaigns (`gepa-*`) are optimisation runs: they carry no scored-records file, and most carry no `prompts.json` either, because the prompt under test changes every generation and is recorded in the campaign's own lineage file in the code repository; the reasoning-budget re-run family (`w2-power/`) adds its own analysis read-outs. The run that pointed three of these judges at the three products' real notes is not here, because its records judge the withheld note text and are withheld with it |
| `judgements/census/` | the census pipeline's own runs: one `manifest.json` per complete run (discovery, verification, clustering, analysis, and the four-standards second panel), the second panel's two review instructions verbatim as templates (`prompts.json` - placeholders, never filled), its `analysis.json` and `classmix.json` read-outs, and the lenient arm's 1,299-record store (`lenient-r1.jsonl`), one verdict per drawn candidate. The census per-call logs are not here: unlike the benchmark runs', their prompts carry the products' real notes and are withheld with them |
| `notes/structure/` | length features over the withheld notes, so the census rates can be recomputed; `excluded_notes.json` names the seven captured notes that are outside the corpus, with the reason for each |
| `validation/` | the four clinician sittings - the judge-adjudication sitting, the severity sitting, the census precision sitting (`precision-sitting/`) and the sitting by a clinician who is not an author (`second-clinician-sitting/`): scored results, per-item keys and the answers as returned |
| `LICENSE`, `ATTRIBUTION.md`, `NOTICE` | CC BY 4.0 and the upstream attribution chain |

To join a judge run back to the benchmark, read the run's scored-records file rather than
its `per_call.jsonl.gz`. Each entry in that file's `records` array carries `note_key`,
`note_role` (`clean` or `errored`), `clean_key`, `stratum`, the consultation id, the score
in `aggregate` and the flag in `flagged`. An errored row also carries `pair_id`, which joins
straight to `pairs/dataset_v2.json`; a clean row leaves `pair_id` null and is identified by
`clean_key`, whose last field is the first 12 hex characters of the SHA-256 of the clean
note text. `per_call.jsonl.gz` is the receipt rather than the join table: it holds prompt
text, completions, tokens and cost, and its only link back is the generation ids, which
match the `gen` values in a record's `receipts`. One naming trap: `consultation` in a judge
record is the bare id (`D2N072`), while in `notes/structure/` and `judgements/findings/` it
is `stratum/id` (`aci/D2N072`), which is also the transcript path.

### Fields on a verified finding

`judgements/findings/verified_findings.json` holds its 618 rows under a top-level
`findings` key. Each row is one finding the verification panel upheld against one note.

| field | type | values |
|---|---|---|
| `finding_id` | string | `system\|stratum\|consultation\|template#pass#n`, e.g. `A\|aci\|D2N077\|detailed#attribution#0`; the pass is the discovery instruction that raised the candidate |
| `note_id` | string | `system\|stratum\|consultation\|template` - joins to `notes/structure/note_structure.json` |
| `system` | string | `A`, `B`, `C` |
| `stratum` | string | `primock`, `aci`, `authored`, `trapblind` |
| `consultation` | string | `stratum/id`, e.g. `aci/D2N077`, which is also the transcript path |
| `template` | string | `audio` for Scribes B and C; `detailed` or `short` for Scribe A, which wrote two notes per consultation |
| `mode` | string | the failure mode the finding was raised under: one of the hunted categories for the targeted passes, or the discovery model's own short label for the open pass |
| `frame_tier1`, `frame_tier2` | string | the hunted category's place under the published two-tier scheme (`omission`, `addition`, `wrong_output`, `misplaced_or_irrelevant`; then the category identifier). Open-pass findings carry `null` and `open`: the paper places them at the top tier with a vocabulary matcher applied afterwards (census paper Appendix A.9), not from these fields |
| `salience` | string | `high`, `medium`, `low` - the verification panel's own importance rating |
| `severity_rubric` | string | `critical`, `supporting`, `peripheral` - **the severity grade the papers use**: the written clinical rubric (verbatim in `validation/sitting_severity_items.json`) applied to each upheld finding by one model call. One row is `null` |
| `severity` | string | `critical`, `moderate`, `trivial` (and one stray `minor`) - the discovery model's own guess at the moment it raised the candidate, never regraded or reviewed, and used by no published number. It ships for completeness only; **do not compute rates from it** |
| `description` | string | what the note got wrong, in the panel's words; around fifty of the 618 quote a fragment of the note - a name, a value, a heading or a phrase of a few words - where the finding cannot be stated without it, as the file's own `what` says |
| `source_quote` | string | the transcript passage that settles the finding |
| `panel_votes` | object | the two panel roles' votes (`constructor`, `auditor`) |
| `decided_by` | string | `unanimous keep` or `tiebreak` |

Two things about the join. `note_structure.json` lists 565 notes over 142 consultations, and
141 of those consultations carry all four notes (Scribe A's two templates, B and C); the
142nd is a trap-blind consultation for which only Scribe C's note was captured, as the census
paper's Section 3 discloses. And the severity grades describe survivors: a candidate reaches
the rubric only after the discovery importance filter and the panel, both of which remove
low-consequence material, so the split (476 critical, 140 supporting, 1 peripheral) is not
the severity mix of scribe errors at large. The rubric was checked in three blinded
clinician sittings (`validation/`), which agree with it to within a grade but not on
direction, so neither paper claims the rubric leans high or low. Which of the paper's 17
subcategories a finding sits in, or whether the clusterer left it unassigned, is in
`cluster_assignments.json` beside the findings file.

The sittings differ in how they were blinded, and the label belongs with any rate taken
from them. The severity, precision and second-clinician sittings were blinded, with every
status label concealed from the rater until after the sitting. The judge-adjudication
sitting is blinded in three of its seven sections and an endorsement in the other four,
where the item had to print the machine verdict to be answerable at all, and its keys were
recovered afterwards by re-running the pack builder at the same seed rather than held
alongside the answers. Most of the item packs the raters read print the withheld scribe
notes and are not released. Two files that carry no withheld note text do ship: the
judge-adjudication sitting's twenty severity items as the grader was given them, with the
written rubric verbatim (`sitting_severity_items.json`), and the 177-item
construction-adjudication queue from the pair build (`sitting_wa_items.json`).

`fact_sheets/` holds two different things and the difference matters. `authored.json` is
the answer key: the fact sheet each authored transcript was written to satisfy, ground
truth by construction. `authored_extracted.json` is what the blind extraction instrument
recovered from those same transcripts, which is the fidelity measurement rather than the
key.

## Composition

The corpus is 145 consultations across four strata: 30 authored, 10 trap-blind, 57
PriMock57, 48 ACI-Bench. A trap is a moment in a consultation where a scribe could
plausibly go wrong; every fact sheet lists them under `salience_traps`, and the severity
grades attach to those entries. The 30 authored scenarios were written to contain traps
deliberately, so that part of the corpus would have ground truth we controlled. The 10
trap-blind scenarios were written with no knowledge of the trap schema, as the check on
whether planting traps inflates the failure rates measured on material we wrote ourselves.

141 of the 145 have released transcripts: all 57 PriMock57 and all 30 authored,
plus the 45 ACI-Bench and 9 trap-blind consultations that kept a fact sheet. Eight
consultations lost their fact sheets at the quality gate - 4 PriMock57, 3 ACI-Bench and 1
trap-blind - and none of the eight yields a fact sheet, a clean note or a pair. The four
PriMock57 among them still ship their transcripts, so that stratum has four more transcripts
than fact sheets; the only consultations with no transcript at all are the 3 ACI-Bench and 1
trap-blind.

The benchmark is **495 evaluation pairs - 293 omissions (150 complete, 86
fragment trace, 57 restatement trace) and 202 commissions (95 additions, 107 alterations) -
plus 112 clean twin notes**, one per evaluation consultation. The omission
levels are named for what survives, and the judge paper uses the same names: a fragment
trace leaves a fragment of the removed statement or its immediate neighbour, and a
restatement trace leaves a second, independently stated mention of the same fact elsewhere
in the note. The `residual_level` field labels the three `complete`, `partial-weak` and
`partial-strong`. Every pair is scored against its own clean twin, so the twins are not a
third category of pair and must not be added to the pair count.

Five further pairs sit in the file tagged `split: gepa_dev` and `eval_set: false`. A
prompt optimiser trained on them, so they are out of every judge's evaluation set - all
judges score the identical item set, which is the point. Those five come from 22
consultations the optimiser saw; the evaluation set contains no pair from any of the 22,
and the other 17 contribute nothing to this file at all. Filter on `eval_set` and none of
this needs thinking about.

Construction runs transcript-first through nine steps: deterministic subsample (seed
20260728, with 67 previously-used ACI-Bench encounters excluded), blind extraction,
three-critic audit, consolidation, reference-note repair, site mapping, single-error
injection, the optimisation-split carve, and cross-family verification with failures
excluded rather than patched.

## Verification standards

Nothing here is described as clean by construction. Each layer has an instrument, a
measured rate on a stated denominator, and a published failure list.

| layer | standard | measured |
|---|---|---|
| fact sheets | three critics, one revision cycle, drop if material issues remain | dropped: PriMock57 4/57 (7.0%), ACI-Bench 3/48 (6%), trap-blind 1/10, authored re-extraction 3/30 |
| fact-sheet fidelity | blind extraction against the authored sheet, over the 27 transcripts whose re-extraction passed the gate | 646/650 authored facts recovered, 99.4% [98.8, 99.9], per-consultation minimum 93.8% |
| reference notes | audited against transcript and core sheet, repaired, re-verified | 53/53 PriMock57 and 45/45 ACI-Bench notes carry at least one material discrepancy with their own transcript, at means of 10.7 and 7.1 per note |
| clean notes | four-section checklist audit by a different model family | first-pass defect rate - **a model auditor's majority verdict with no human adjudication, so an upper bound on a human-confirmed rate** - authored 11/30, trap-blind 3/9, PriMock57 49/53, ACI-Bench 36/45; 88 repaired, 3 excluded |
| pairs, graded-omission build | cross-family panel of two families, third call on any split | 71/126 complete-removal attempts verified (56.3%), 114/151 partial removals (75.5%); 92 dropped, itemised |
| severity grades, benchmark pairs | written rubric, two independent cross-family graders, conservative tie-break | Cohen's kappa 0.662 [0.59, 0.73] over the 683 traps regraded under the rubric, against 0.177 [0.04, 0.31] over 171 matched traps graded without it, intervals non-overlapping; 92 of the 683 (13.5%) resolved to the lower grade. `pairs/factorial_severity.json` carries both graders' verdicts for the facts behind the released pairs, a subset of that 683-trap population |
| severity grades, census findings (`severity_rubric` in `judgements/findings/`) | the same written rubric, one grader, applied to each finding the panel upheld | two blinded clinician sittings over sampled findings (`validation/`): 25 of 32 rubric-versus-clinician grades exact, none more than one grade apart, and the disagreements run in opposite directions (the author-clinician graded 4 of 20 above the rubric, the independent clinician 3 of 12 below), so no direction is claimed; census paper Appendix A.4 |

The clean-note audit runs four checks on each note: that every fact the sheet requires is
present and stated accurately, that every statement the sheet forbids is absent, that every
trap is handled, and, open-endedly, that the note asserts nothing the transcript does not
support. One seed screened all four, and only a check that flagged something was then run at
three seeds and its answers compared, which happened on 222 of the 1,024 checks audited
across the first pass and the repair rounds. Agreement between those seeds averaged 0.794
over the three closed checks and 0.725 once the open-ended one is included. The other 802
checks were cleared on a single seed, so their agreement is unmeasured rather than perfect,
which makes both figures a lower bound over the disputed checks rather than a whole-corpus
number.

## Known biases, gaps and exclusions

**Population and setting.** English-language UK primary care and US ambulatory care only.
Two of four strata are consultations we wrote. No secondary care, no specialty outside
general practice, and no note carries an electronic-record structured field, so the
benchmark cannot speak to structured-field omissions at all.

**No real patients anywhere.** Actors, simulation and fiction throughout. An ethical
strength and an ecological limitation at once: acted consultations are cleaner and more
purposeful than real ones.

**Audio provenance is mixed.** Only PriMock57 is genuine recordings, and those recordings
predate the products by several years; the other three strata were rendered to synthetic
speech from their transcripts. Capture ran at native speed, with no tempo or
silence-compression transform on any audio, after a speed-up test was rejected when silence
trimming produced a critical medication substitution. No audio is in this release: the
synthetic speech is regenerable from the released transcripts with the harness's
text-to-speech recipe and its voice pins, and PriMock57's recordings belong to PriMock57's
own release.

**Selection asymmetry between strata.** First-pass pair failure was 18.9% on the authored
stratum against 52.9% on PriMock57 and 52.3% on ACI-Bench, because the authored stratum
is preserved verbatim and its failing pairs are excluded rather than regenerated, while
public-stratum notes were repaired and pairs regenerated up to three times. A surviving
public-stratum pair has passed a harsher filter, so judge and scribe scores should not be
compared across strata.

**Two construction cohorts are pooled, and recorded per row.** The 202 commission controls
and 79 complete omissions carried over from an earlier build were made by model rewrite
rather than by span deletion off the site map. `source` and `construction.method` record
which cohort each pair came from, so an analysis can be restricted to the 185 pairs with
`source` `factorial`, the internally uniform cohort. Two things to know before doing that.
All 202 commission controls sit in the carried-over cohort, so the uniform cohort is
omissions only and the omission-against-commission comparison cannot be made inside it. And
uniform means the deletion was programmatic off the site map throughout, not that no model
touched the note: 88 of the 185 are a raw span deletion with no model call at all, 53 needed
a grammar tidy and 42 a collateral cleanup afterwards, and on 2 the validator rejected the
cleanup so the raw deletion stood.

**Residual count is a property of the map, not of the note.** Construction left exactly one
mapped residual site on every newly built partial pair, but across the 151 partial-removal
attempts the verification panel's own reading found more than one surviving mention in 71 of
them. In the released file the 114 factorial partial pairs carry both counts:
`residual.n_surviving` is 1 on every one of them, and the panel's `residual.verified_n` is
above 1 on 51. The 34 carried-over partial seeds predate the site map and have no map count
at all; their `residual.n_surviving` is an auditor's own count of surviving mentions, which
runs from 1 to 9 and is above 1 on 16 of them.

**Thin and unpopulated cells.** The restatement-trace peripheral cell holds 2 pairs and
should be treated as unpopulated; the fragment-trace peripheral cell is thin at 8.

**Overlap with an external benchmark.** Fourteen of the 48 ACI-Bench encounters we drew are
also source dialogues in MedVAL's `dialogue2note` task, and 13 of those survive to this
release: D2N072, D2N077, D2N078, D2N081, D2N084, D2N086, D2N096, D2N113, D2N115, D2N117,
D2N121, D2N122 and D2N125, carrying 61 of the 495 evaluation pairs. They stay in
the benchmark, because the clean notes and the injected errors are ours and no MedVAL label
is used anywhere here. Drop them from any analysis that also uses MedVAL's `dialogue2note`
task, where the shared source dialogues mean scores on the two sets are not independent.

**Everything is model-produced except the transcripts.** Fact sheets, ideal notes,
injections, repairs, site maps and audits are all model outputs, checked by models from a
different family. Human clinical review is by an author-clinician, disclosed as such, plus
one sitting by a clinician who is not an author and had no involvement in the study; the
two are reported side by side and never pooled.

## What is withheld, and why

The text of the notes the three scribe systems wrote is not in this release. The reason is
contractual rather than clinical: the corpus contains no patient data at all, and it is the
products' terms of service, which differ from one another, that decide whether their output
can be republished. All three are treated the same way, partly because a release carrying
one product's notes and not another's would make products identifiable by their absence.

What travels instead is the evidence a reader can check. Every verified finding in
`judgements/findings/` names the consultation, says what the note got wrong, and quotes
the **transcript** passage that settles it - and the transcripts are here. One narrow
carve-out, disclosed in the findings file itself: around fifty of the 618 descriptions
(fewer than one in ten) quote a fragment of the note - a name, a value, a heading or a
phrase of a few words - where the finding cannot be stated without it, some of them
internal contradictions inside a single note. Nothing longer, and nothing else from any
product's note text, appears anywhere in this release. Length features per note are in `notes/structure/`,
which is what the census rates are computed against, and the per-finding cluster
assignments behind the census paper's subcategory table are in
`judgements/findings/cluster_assignments.json`. Researchers who need more should
contact the authors at seb@composo.ai.

## Anonymisation

The three scribe systems are Scribe A, B and C. No product name appears in any released
file, including prompts, configurations and run manifests; the packaging step enforces
that mechanically over every file in both repositories, checking contents and paths, with
no exemptions. The letter-to-product mapping stays in the private repository.

There is no patient data to de-identify, and no de-identification claim is made about the
upstream corpora beyond what their own releases state.

## Evaluation window

Most of the scribe notes were captured 9 to 11 August 2026; 121 of the 565 (84 of
Scribe A's, 22 of Scribe B's and 15 of Scribe C's) were carried over unchanged from a
June 2026 pilot capture, so the corpus holds two product snapshots about six weeks apart
rather than one. The judge runs were made between July and August 2026. Every figure here
is a measurement of the product and model versions those systems served in those windows.
These are moving targets; a rerun on a later version is a different measurement, not a
replication.

## Competing interests

The authors build evaluation tooling commercially, and two disclosures follow. The
reference judges include the authors' own deployed faithfulness judge, run as shipped and
in two variants that differ only in how they treat omissions. The decomposition pipeline
whose omission results this card reports above is the authors' own design, and it is the
method the judge paper recommends. Both ship here in full - prompts, run manifests and
per-call judgements under `judgements/judges/`, with the prompt text and its hashes in
`w2_prompts/` in the code repository - so either can be re-run and disputed. On the census
side, no commercial product of the authors sits anywhere in the measurement path that
produced the findings in `judgements/findings/`. Both papers carry these statements too.

## Reproducibility

Every run carries a manifest with its resolved model, provider route, generation
identifiers, seeds, dataset digest and cost, and the raw completions ship with it. The
completions are the real reproducibility guarantee, because the serving routes are not
token-level deterministic: a seed is accepted but does not reproduce output token for
token, though per-item verdicts held across seeds where we checked.

## Licence and attribution

Released under CC BY 4.0. Both public corpora are published under CC BY 4.0 and are
redistributed here in derived form. Reproduce these two strings with the data:

> PriMock57, Babylon Health, licensed under CC BY 4.0, github.com/babylonhealth/primock57. Changes made: transcripts normalised; the clinician reference notes were audited against their own transcripts and repaired; single-error variants were generated from the repaired notes.

> ACI-Bench, Yim et al. 2023, licensed under CC BY 4.0 on its figshare record, doi.org/10.6084/m9.figshare.22494601 (code at github.com/wyim/aci-bench, which carries no licence file of its own). Changes made: a 48-encounter subsample drawn at seed 20260728 from the 140 encounters remaining after 67 exclusions; transcripts normalised and rendered to synthetic speech; the dictation-derived reference notes were audited against their own transcripts and repaired; single-error variants were generated from the repaired notes.

`LICENSE` is the unmodified CC BY 4.0 legal code; the attribution above is also in
`ATTRIBUTION.md`, in `transcripts/README.md`, and in the `_attribution` block at the top of
every derived JSON artefact.

## Revisions

- **v1.0.4, 28 September 2026.** Adds `judgements/findings/cluster_assignments.json`,
  the per-finding cluster assignments behind the census paper's subcategory table, on an
  outside researcher's request. Documents the fields of `verified_findings.json` on this
  card for the first time, including that its `severity` column is the discovery model's
  own guess and `severity_rubric` the grade the papers use. Scopes the severity-grades
  verification row to the benchmark pairs and adds the census-findings row beside it. No
  benchmark number, pair, transcript or finding changes; the findings file's rows are
  byte-identical to v1.0.3. Only its `what` header changed, to carry the same explanation
  and to state the note-fragment carve-out accurately: v1.0.3 said ten descriptions quote
  a short note phrase, and the count is around fifty, none longer than a few words.
- **v1.0.3, 1 September 2026.** The release the two papers announced with: arXiv
  identifiers in the citation surfaces, the census run manifests, the second panel's
  templates and lenient store, the deduplication groupings and the excluded-notes list.

## Citation

Cite the judge paper for the benchmark and the judge results, the census paper for the
failure census over the three deployed scribes, and this dataset record for the data.
The code repository's Zenodo archive is citable at DOI 10.5281/zenodo.22160954, which
always resolves to the newest tagged release.

> Fox, S., Markham, L., Lail, R. and Karotsieris, M. (2026) *LLM Judges Verify Presence,
> Not Absence: Omission Blindness in AI Clinical Notes and What Recovers It*.
> arXiv:2608.31016. https://doi.org/10.48550/arXiv.2608.31016

> Fox, S., Markham, L., Lail, R. and Karotsieris, M. (2026) *One note in three: a verified
> census of three deployed AI scribes, and the instrument that counted it*.
> arXiv:2608.31017. https://doi.org/10.48550/arXiv.2608.31017

> Fox, S., Markham, L., Lail, R. and Karotsieris, M. (2026) *OmissionBench*, version
> `factorial-v1` [data set]. Hugging Face, `ComposoAI/OmissionBench`.
> https://doi.org/10.57967/hf/10258

That DOI was minted against the v1.0.3 revision of 1 September 2026, and it stays correct
for this one: v1.0.4 adds one file and documentation and changes no benchmark number (see
Revisions). Quote the dataset version alongside any number computed over the benchmark -
a revision that changes the data gets its own DOI, and the two are not interchangeable.
