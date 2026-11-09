# AI Scribe Note Checker — Build Plan

Oct 11, 2026 · @Sandesh Gubbi Virupaksha

Six weeks from the start date to a public v0.1 release of the checker specified in `docs/spec.md`, built by two people and gated twice: on reproducing the published methods' per-note flags at the end of week 1, and on the held-out result in week 5.

## Assumptions

All durations in this plan are my estimates, not commitments. Replace them with the owners' own estimates before week 1, because the spec makes the choice between a 4-week and a 6-week launch on those numbers.

- **Team:** two people, both full time. Person A owns evaluation and the full-mode pipeline. Person B owns the rules layer, integrations and the release. These are proposed roles; owners are named in the week 0 checklist. If either person is part time, durations scale: two half-time people would take about 12 weeks.
- **Part-time help:** a second person, not the rules author, writes the synthetic demo set from week 1 and the mutation-suite phrasings in week 2. A clinician reviews the demo set and rules-mode severity table in week 2, the severity rubric in week 3 if it is our own, and the calibration guidance in week 5.
- **Effort:** about 12 person-weeks over weeks 1–6, plus a short week 0 for decisions and setup, plus the part-time help above.
- **Compute:** The headline API model is Claude Sonnet 5.5, run through the Batch API with prompt caching (transcript first in every prompt) and one fixed effort level used everywhere. The open-weight model (a mid-size model that fits one 24–48 GB GPU, chosen in M3 session 4) is used for tuning, through OpenRouter with the provider pinned and the quantisation recorded; those results are indicative only. API budget, measured where a run has been made: Gate 1(b) cost $9.12 through OpenRouter, under a dedicated key with a $60 credit limit: route two $2.34 for 51 notes (about $0.046 per note) and route one $6.78 for 51 notes over 10 consultations (about $0.13 per note, including fresh extraction); see [docs/evidence/gate1b.md](evidence/gate1b.md). Estimates scaled from those measured costs, to be replaced by measured costs after M3: the Claude baselines in M4 (both published methods run once on Claude Sonnet 5.5 over the 198 held-out notes, through the harness's OpenRouter transport with model overrides only) about $20–40, paid in OpenRouter credits, where the low end applies Sonnet 5.5's prices ($2 in, $10 out per million tokens) to Gate 1(b)'s measured token counts and the high end keeps its measured cost per note. scribecheck's own Claude runs: about $0.09 per note, or about $0.04 with the Batch API, from route one's measured tokens at Sonnet 5.5 prices (about $0.06 per note: extraction and audit about $0.15 per consultation, the per-note check and second look about $0.03 per note) plus an allowance for Pass C of one more per-note check and re-check (about $0.03 per note); Pass B is mostly rules and is not costed separately. Over the planned runs (two development runs of 330 notes, three stability runs of 100 notes and the 198-note held-out run, 1,158 notes) that is about $100, or about $50 with the Batch API, from monthly API credits; about $107 or $54 if Gate 2 adds the 79 supplementary pairs. The budget is split by milestone after M1 measures cost per note, and spend is checked at each gate. The local-model headline (both published methods' baselines and scribecheck's Gate 2 local run) runs in one session on one rented 48 GB GPU (for example RunPod or Vast.ai) with vLLM: about 3–6 GPU-hours, roughly $3–6 (an estimate). Week 4's GPU hours are worked out in M3, and a second GPU is rented if they don't fit.

**Critical path:** M3, the full-mode pipeline, is now the critical path. Person A's chain runs from the week 0 decisions through reproduction (Gate 1), the full-mode pipeline, development evaluation and the held-out run (Gate 2) to launch. Person B's chain runs through the integrations, which launch needs by the end of week 5. The rules layer is no longer on either chain: M3 builds on the alignment and guards already in place, and M2's remaining rules work (structured omission candidates and the examination rule) feeds M3 without gating it, because rules-mode certain flags are disabled ([docs/evidence/allowlist_ceiling.md](evidence/allowlist_ceiling.md)). A one-week slip on either chain moves launch by a week. Person A's week 5 has room after the held-out run, so A also takes the Inspect task, the `bench` command and the preprint draft then.

## Before week 1

Week 0 ends when every item below is done; the three blockers come first, because the rules layer and the schedule depend on them.

- [ ] **Blocker:** choose the drug lexicon and confirm its licence allows redistribution (launch uses generic names plus a user-supplied brand map)
- [ ] **Blocker:** choose the common-symptom lexicon and confirm its licence
- [ ] **Blocker:** name an owner for every milestone, collect their estimates, and decide between 4 and 6 weeks
- [x] Choose the API model: Claude Sonnet 5.5, through the Batch API with prompt caching and one fixed effort level
- [x] Approve the API budget and set the per-run cost cap (measured: Gate 1(b) $9.12 through OpenRouter under a $60 key limit; estimated from it until M3 measures them: Claude baselines about $20–40 in OpenRouter credits; scribecheck's own Claude runs about $100, or $50 with the Batch API, from monthly API credits)
- [ ] Choose the open-weight model (moved to M3 session 4): a mid-size model that fits one 24–48 GB GPU, from OpenRouter's current list, with open weights under a licence that allows commercial use, reliable structured output and a context window of at least 32k tokens. Development runs use it through OpenRouter; the headline runs it on one rented 48 GB GPU with vLLM
- [x] Identify the paper's pinned model: gpt-5.4, the dataset card's baseline judge (availability still to check)
- [ ] Check API access to gpt-5.4, needed only for Gate 1(b); if it is unavailable, choose and record a substitute
- [ ] Check that Fox et al.'s harness can call Claude
- [ ] Check whether Fox et al.'s harness accepts an OpenAI-compatible endpoint, and whether the open-weight model supports a reasoning budget; estimate any adaptation work for M4
- [x] Route one's published 95% interval: no longer needed, because Gate 1 compares per-note flags with the released scored records
- [ ] Check whether Fox et al.'s severity rubric ships in their MIT-licensed harness
- [ ] Check arXiv's endorsement rules for the chosen category, and find an endorser if one is needed
- [ ] Decide the GitHub organisation (Nasken AI or personal) and create the repository under Apache-2.0
- [ ] Set up a PyPI Trusted Publisher for `scribecheck`
- [ ] Set up CI on Linux, macOS and Windows for Python 3.11 and later
- [ ] Download OmissionBench (CC BY 4.0) and clone the harness (MIT), recording their versions
- [ ] Name the second person, and confirm the clinician's availability for reviews in weeks 2, 3 and 5

## Roadmap

Both chains end without slack: Person A's reproduction, pipeline, development evaluation and held-out run, and Person B's rules layer and integrations. A slip on either moves launch.

_Diagram not included in this export (build roadmap · 6 milestones, 3 checkpoints); see the Claude Doc._

Each bar is an estimate in weeks from the start date. Diamonds are the checkpoints described under Gates and stop rules.

## Milestones at a glance

Six milestones; each is done only when its exit criterion is met, not when its tasks are ticked.

| Milestone | Weeks (estimate) | Lead (proposed) | Exit criterion | Status |
| --- | --- | --- | --- | --- |
| M1 Reproduction and scoring harness | 1 | Person A | Gate 1 passed (published numbers recomputed; route two agrees per note with the released records at the runbook's mark, and route one's detection and false-alarm counts fall within its margins), and the minimum detectable gain recorded | Gate 1 passed on 2026-11-08 ([docs/evidence/gate1a.md](evidence/gate1a.md), [docs/evidence/gate1b.md](evidence/gate1b.md)); provisional minimum detectable gain recorded ([docs/evidence/mdg.md](evidence/mdg.md)); the remaining M1 tasks are open |
| M2 Rules layer and rules-only mode | 1–2 | Person B | `scribecheck demo` runs offline in under a minute; rules precision and recall measured against the 95% and 90% targets | Alignment, guards and reason codes built; rules-mode certain flags disabled ([docs/evidence/allowlist_ceiling.md](evidence/allowlist_ceiling.md)); remaining rules work: structured omission candidates and the examination rule |
| M3 Full-mode pipeline | 2–3 | Person A | All three passes run end to end on both models; every flag carries a verified quote; calls and cost per note measured | Not started |
| M4 Development evaluation | 4 | Person A | Results on the 65 development consultations (265 pairs) written up; local-model baselines recorded; settings frozen and tagged | Not started |
| M5 Integrations, calibration and held-out run | 3–5 | Person B; Person A for the held-out run, Inspect task, bench command and preprint draft | Plugins, Action, calibrate and stability commands pass their tests; held-out run done once (Gate 2) | Not started |
| M6 Preprint, release and launch | 6 | Persons A and B | v0.1 released through Trusted Publishing; preprint posted; release checklist passed | Not started |

## M1 Reproduction and scoring harness (week 1)

The goal is to prove the evaluation setup matches the published results before building anything that will be measured with it. Lead: Person A. Depends on week 0 for model access, the budget and the downloaded benchmark. Only the first group of tasks gates week 2; the second must be ready by the start of week 4.

For Gate 1:

Gate 1(a) touches encrypted data (the raw download and the held-out subset), so it is run by a person in a normal terminal, never by Claude Code.

- [x] Gate 1(a), free: recompute the published numbers from the benchmark's released scored records, with no API calls, to validate the scoring code. Passed: [docs/evidence/gate1a.md](evidence/gate1a.md)
- [x] Gate 1(b), per-note check: run Fox et al.'s harness for both methods on gpt-5.4 for about 10 development consultations (about 55 notes), compare each note's flag with the released record for that note, and record cost per note, model calls per note and run time. Passed on 51 notes over 10 consultations: [docs/evidence/gate1b.md](evidence/gate1b.md). Route one is judged on its detection and false-alarm counts rather than per-note agreement, because its fact lists were not released ([docs/runbooks/gate1b.md](runbooks/gate1b.md))
- [ ] Record flags per note from the per-fact pipeline, which sizes the calibration workload
- [ ] Split the API budget by milestone from the measured cost per note
- [x] Write down the minimum detectable gain for the API-model comparison: done provisionally against the published gpt-5.4 results ([docs/evidence/mdg.md](evidence/mdg.md)); M4 recomputes it against the Claude baselines

Ready by the start of week 4:

- [ ] Scoring harness for scribecheck's output: the *k* and *c* sweep, counts on the 47 clean notes, breakdown by severity and trace, bootstrap over consultations, and the exact McNemar test
- [ ] Correct-fact matching: how a flagged fact is matched to the benchmark's named removed fact, using model-assisted matching with a manual check of a sample

**Exit (Gate 1):** the published numbers are recomputed, route two agrees per note with its released records at the runbook's mark and route one's counts fall within its margins, and the minimum detectable gain is recorded. **Passed on 2026-11-08:** [docs/evidence/gate1a.md](evidence/gate1a.md), [docs/evidence/gate1b.md](evidence/gate1b.md), and a provisional minimum detectable gain in [docs/evidence/mdg.md](evidence/mdg.md). The Claude baselines moved to M4, beside the route-one single-model variant they depend on.

## M2 Rules layer and rules-only mode (weeks 1–2)

The goal is a working rules-only checker that installs and runs offline, and the alignment that the full-mode pipeline builds on. Lead: Person B, with the second person writing the demo set and mutation phrasings. Depends on the week 0 lexicon decisions. M3 builds on the alignment already in place and no longer waits for the rest of M2. The rules layer nominates structured candidates for the model's check, guards against false certain flags, and gives a no-model screen whose findings are warnings; its remaining work is the structured omission candidates and the examination rule.

Week 1:

- [x] Package skeleton, config file format and `scribecheck init` (`config.py`: TOML read with the standard library's tomllib, after dropping Python 3.10; the user's speaker map, heading map, brand map and extra symptoms extend the built-in ones; unknown keys are errors)
- [ ] Publish a first development release to PyPI as soon as the package installs, to secure the name
- [ ] Input parsers for plain-text and JSON transcripts, and note section detection with the user's heading map
- [ ] Normalisers for numbers, units, frequency, route and side
- [ ] Drug and symptom lexicons, with the user-supplied brand map
- [ ] Built-in ConText rules, dialogue negation for single-symptom questions, and allergy rules

Week 2:

- [x] Alignment with the one-consistent-value rule, the Pass B guards and reason codes, and the deterministic negation check. Rules-mode certain flags are disabled: a deny-list could not rule out false ones, and the measured allow-list ceiling on the development data is zero ([docs/evidence/allowlist_ceiling.md](evidence/allowlist_ceiling.md))
- [x] Rules-mode Pass A structured omission candidates (medicines, doses, frequencies, allergies, sides) at possible confidence, for the model's second check in full mode and as warnings in rules mode (`rules/omission.py`). Measured on 2026-11-09: recall 6 of the 15 development structured omissions; 6 candidates on the 65 clean notes, held as a ratchet by `tests/test_dev_clean_notes.py`. Follow-ups: lexicon gaps (ASA, HCTZ, PCN, sulfa, plain "insulin"), dotted frequencies ("b.i.d.", "2x/day"), "OD" written alone for the right eye, and values after a comma, which are dropped
- [x] The examination rule, reporting at possible confidence (`rules/examination.py`; telephone: every examination finding, video: hands-on findings only, in-person or no modality: off; critical warnings)
- [x] Severity table, confidence levels, both standards and the decision rule (`decision.py`). Rules mode never fails; in full mode failing needs both a `fail_on` severity and a `fail_confidence` level. Follow-up: user overrides of the rules-mode severity table
- [x] JSON report, Markdown renderer, exit codes, and the `demo`, `check` and `batch` commands (`report.py`, `checker.py`, `cli.py`), with three synthetic demo consultations in `src/scribecheck/demo/`, marked `"reviewed": false` for the clinician. In rules mode a Pass B item that would have been a certain flag is reported as a changed-fact warning at possible confidence. Follow-ups: HTML output; the GitHub Action's job summary; values after a comma are dropped ("5 mg once a day, sometimes 10 mg" warns of a dose change), which now shows as a warning; in Pass A, an answer naming two allergens where one is in the drug lexicon ("Penicillin and codeine.") reports that one as a missing medicine rather than an allergy; a `.txt` transcript containing JSON is read as plain text
- [x] Golden-file tests for the JSON and Markdown reports (`tests/golden/`, regenerated with `SCRIBECHECK_UPDATE_GOLDEN=1`)
- [ ] Network guard: a socket-level block in tests on all three operating systems, plus a no-network container test on Linux
- [ ] Check whether OmissionBench labels the type of each altered fact; if not, label the structured-fact pairs by hand
- [ ] Precision and recall per change type, measured on the second person's phrasings and on the 117 development pairs that add or change a fact

Second person, weeks 1–2:

- [ ] Synthetic demo set of about 20 consultations, ready for clinician review by the end of week 2
- [ ] Mutation-suite phrasings for every change type

**Exit:** `pip install` then `scribecheck demo` runs offline in under a minute, the network guard tests pass, and precision and recall are reported against the 95% and 90% targets. Missing a target is reported, not hidden, and does not block M3.

## M3 Full-mode pipeline (weeks 2–3)

The goal is all three passes working with a language model, on both the API model and the open-weight model. Lead: Person A. Depends on M1's harness and on M2's alignment by the end of week 2.

Week 2:

- [x] Model client for OpenAI-compatible endpoints over `httpx`, plus the Anthropic adapter, which is required because Claude Sonnet 5.5 is the headline API model: Batch API, prompt caching (transcript first in every prompt) and one fixed effort level (`src/scribecheck/model/`; tested against mocked HTTP only; `scripts/smoke_model.py` makes one tiny real call per adapter when a person runs it with their own key). The fixed effort level is still to choose, by measurement on development data, before Gate 2
- [x] JSON schema validation with one retry, temperature 0 and a fixed seed, and the model name and version written to every report. The Anthropic API (version `2026-10-21`) rejects `temperature`, `top_p` and `top_k` on every model, so the Anthropic adapter sends none and the report records `temperature: null`; the OpenAI-compatible adapter sends temperature 0 and a fixed seed. Writing the model info into full-mode reports happens when the passes are wired up
- [x] Cost tracking and the per-run cost cap: usage is always recorded; cost comes only from `[model.prices]` in config; once spend reaches `[budget] cost_cap_usd` no further call is made and the remaining items are unchecked with `cost_cap_reached` (a batch is checked before submission against a worst-case estimate)
- [ ] Transcript chunking with overlap, and merging of facts across chunks
- [ ] Batch mode extracts facts once per transcript and reuses them across that transcript's notes
- [x] Pass A: extraction from the transcript only, audit with severity grading, per-fact presence check with the whole note, and the second look (`src/scribecheck/pass_a/pipeline.py`, prompts in `src/scribecheck/prompts/scribecheck_v1/`; extraction once per distinct transcript, chunked with overlap when long; about 10 facts per check call; quotes verified by `quotes.py`, re-asked once, else unchecked `quote_unverified`). Tested on mocked HTTP only; no real run yet

Week 3:

- [x] Merge rules facts into Pass A, and send rules "absent" verdicts for structured facts to the second check (a confirmed candidate takes the model's severity grade; a rejected one is dropped)
- [x] Send rules-mode changed-value warnings (Pass B items that would have been certain flags, reported at possible confidence by `checker.py`) to the model's quote-verified second check, like the omission candidates
- [x] Pass B model path, certainty changes and its second check (`pass_b.py`; scribecheck pipeline only; a verdict the second check does not uphold is unchecked `second_check_rejected` unless `[report] show_rejected_as_possible`)
- [x] Pass C: statement splitting, checking, the second check and the default exemptions (`pass_c.py`; record history is exempt only where the note marks it as from the record)
- [x] Deduplication across the three passes (`dedupe.py`: Pass B over Pass C, Pass A gives way to Pass B; `--passes` chooses the passes, default all)
- [x] Fox et al.'s route-one prompts as a selectable configuration inside scribecheck, so Gate 2's fallback is a config switch (`[pass_a] pipeline = "route_one"`, `src/scribecheck/pass_a/route_one.py`; the five prompts copied byte for byte from the MIT harness and hash-checked; prompt-only JSON and the harness's parsers, as published; per-stage model and effort in config, defaulting to the harness's efforts). Quote checks never change route one's verdicts: a flag whose transcript quote cannot be found stays, at possible confidence, with `evidence_verified: false`, counted in the report; scoring of route one must use its verdicts unchanged (for the M1 scoring harness)
- [x] Severity rubric: Fox et al.'s if the harness licence covers it, otherwise our own, reviewed by the clinician. It ships in the harness's audit prompt (MIT), and scribecheck_v1's audit prompt reuses it with attribution; the clinician review is still to do
- [ ] M3 session 4: choose the open-weight model from OpenRouter's current list against the criteria above (open weights with a commercial-use licence; fits one 24–48 GB GPU; reliable structured output; at least 32k context), then run it through OpenRouter with the OpenAI-compatible adapter, provider pinned and quantisation recorded, and record calls, cost and time per note for both models
- [ ] Prove the local path on an 8 GB Apple M3 with Ollama and a 3–4B model: the model smoke test plus `scribecheck check` on the demo cases; the README states that small laptop models are for trying the tool, not for accuracy

**Exit:** all three passes run end to end on the demo set with both models, every flag carries a verified quote, and the estimates of about 10 calls per note (spec) and about $0.09 per note (budget, scaled from Gate 1(b)) are replaced with measured figures, which feed a revised budget, replacing the estimates, and an estimate of week 4's GPU hours.

## M4 Development evaluation (week 4)

The goal is to tune on development data only and freeze the settings before the held-out run. Lead: Person A. Depends on M3.

- [ ] Before starting, check that the rented-GPU session (about 3–6 GPU-hours on one 48 GB GPU, estimated) fits the seconds per note M3 measured
- [ ] Re-run both published methods on the held-out subset with the open-weight model on one rented 48 GB GPU running vLLM, with the exact weights and precision recorded, in the same session as scribecheck's Gate 2 local run; adapt the harness if week 0 showed it is needed. Held-out data is copied to the pod only for that session and the pod is deleted afterwards; real patient data never goes on a rented machine. This run touches the encrypted held-out data, so it is run by a person in a normal terminal, never by Claude Code. If the local model cannot run route two, record that baseline as not applicable rather than reporting a misleading number.
- [ ] Route one (the three-stage pipeline, tier B3) uses three model families: Claude Opus 5 for fact extraction (stage 1), GPT-5.5 for the audit and the refute pass (stages 1b and 3), and GPT-5.4 for the per-note check (stage 2). The Claude baseline is therefore a single-model variant with every stage on Claude Sonnet 5.5. That needs model overrides for stages 1, 1b and 3 (the harness's `--check-role` covers stage 2 only), and the preprint must describe it as a variant of the published method. The published multi-model numbers remain the reference.
- [ ] Claude baselines: run both published methods once on Claude Sonnet 5.5 on the official 151-pair held-out subset (a single run, stated as such), so scribecheck is compared on the same model, with route one as the single-model variant above. They run through the harness's existing OpenRouter transport with model overrides only (estimated $20–40, in OpenRouter credits). The released gpt-5.4 results remain the reference and are not re-run. This run touches the encrypted held-out data, so it is run by a person in a normal terminal, never by Claude Code. Then recompute the minimum detectable gain against these baselines ([docs/evidence/mdg.md](evidence/mdg.md))
- [ ] State the minimum detectable gain for the local-model comparison before any held-out run of scribecheck
- [ ] Tune prompts and settings on the open-weight model through OpenRouter (provider pinned, quantisation recorded; indicative only), using a fixed 20-consultation development subset
- [ ] Run scribecheck on the 65 development consultations with the open-weight model through OpenRouter
- [ ] Run Claude Sonnet 5.5 twice on development data: once mid-way through tuning and once on the frozen settings
- [ ] Measure the rules layer's effect by running with and without it
- [ ] Score Pass B and Pass C on the 117 development pairs that add or change a fact
- Scoring note: OmissionBench detection and false alarms use Pass A runs only (`--passes a`); Pass B and Pass C are evaluated separately, on the 117 development pairs that add or change a fact
- [ ] Measure severity agreement with OmissionBench's per-fact grades (Cohen's kappa) for each model
- [ ] Stability: run each model 3 times on a fixed 100-note development subset
- [ ] Publish minimum model guidance, including the minimum context length
- [ ] Freeze the settings and tag the commit before any held-out run

**Exit:** development results are written up, the settings are frozen and tagged, the local-model baselines are recorded, and the decision on whether the open-weight model meets the minimum is recorded.

## M5 Integrations, calibration and held-out run (weeks 3–5)

The goal is every launch channel working, and the one held-out run that decides the launch claim. Person B leads the tools and integrations. Person A runs the held-out evaluation, then uses the rest of week 5 for the benchmark tooling and the preprint draft. Depends on M2 and M3 for the tools, and on M4's frozen settings for the held-out run.

Person B, week 3 (built against rules mode, then tested on full mode in week 4):

- [ ] `calibrate export` with `--sample N`, and `calibrate apply` with `--max-rejected`
- [ ] `--inject` for value changes
- [ ] `stability` command
- [ ] HTML report renderer

Person B, weeks 4–5:

- [ ] Test `calibrate` and `stability` on full mode, now that M3 is done
- [ ] promptfoo assertion, DeepEval metric and Inspect scorer, each scoring 1.0, 0.5 or 0.0, with tests pinned to each framework's version; the Inspect scorer ready by the end of week 4 for Person A's benchmark task
- [ ] GitHub Action: job summary, JSON report artifact, and by default only critical flags at certain confidence fail the build
- [ ] Colab notebook, Hugging Face datasets (demo set and mutation suite) and a Hugging Face Space
- [ ] README with the scope statement and attribution, plus docs on turning off host-framework telemetry and on self-hosted runners
- [ ] Clinician review of the calibration guidance

Person A, week 5:

- [ ] Held-out run, once, with the frozen settings: Claude Sonnet 5.5, and the open-weight model on the rented 48 GB GPU with vLLM (same session as the local-model baselines; pod deleted afterwards), on the official 151-pair subset; the 79 supplementary pairs only if the headline result is borderline
- [ ] Record the Gate 2 outcome
- [ ] Inspect task that runs OmissionBench
- [ ] `bench` command, printing the CC BY attribution
- [ ] Draft the preprint's method, setup and results sections

**Exit:** every integration passes its tests in CI, and the Gate 2 outcome is recorded as a clear gain, no detectable difference, or worse.

## M6 Preprint, release and launch (week 6)

The goal is a public v0.1 release whose launch claim matches the Gate 2 outcome. Persons A and B share it. Depends on M5.

- [ ] Preprint: method credited to Fox et al. and MED-OMIT, results as Gate 2 found them, including a null or negative result. Finished from the week 5 draft and submitted at the start of week 6, so it is public before the launch posts; the endorser question was settled in week 0.
- [ ] Release checklist (below) passed
- [ ] v0.1.0 published to PyPI through Trusted Publishing, with build attestations and the software bill of materials
- [ ] Pull requests asking promptfoo, DeepEval and Inspect to list the plugins on their integration pages
- [ ] Launch posts once the preprint is public: Hacker News, r/MachineLearning, LinkedIn health-AI circles, AMIA and HIMSS groups
- [ ] Outreach to two or three scribe vendors and hospital AI evaluation teams, offering help setting up the GitHub Action
- [ ] Monthly tracking set up: PyPI downloads, Hugging Face downloads, GitHub stars and dependent repositories, integration-page listings

Release checklist:

- [ ] Network guard tests pass on all three operating systems, and the no-network container test passes on Linux
- [ ] No telemetry, analytics or update checks anywhere in the package
- [ ] README states the scope (faithfulness to the transcript, not correctness), the evaluation-tool positioning, and the attribution to Fox et al. and MED-OMIT
- [ ] Demo set reviewed by the clinician
- [ ] `bench` prints the CC BY attribution
- [ ] Every launch claim matches the Gate 2 outcome

**Exit:** v0.1.0 is on PyPI, the preprint is posted, the release checklist has passed, and tracking is running.

## Gates and stop rules

Five checkpoints release the next stage of work only when their evidence is in. Sandesh decides at each one.

| Gate | When (estimate) | Evidence needed | Response |
| --- | --- | --- | --- |
| Gate 0 | End of week 0 | Both lexicons chosen, owners and estimates in, budget approved, GPU and clinician availability confirmed | Met: start week 1. Not met: delay the start; M2's lexicon work cannot begin. |
| Gate 1 | End of week 1 | Published numbers recomputed from the released scored records; route two agrees per note at the runbook's mark and route one's counts fall within its margins; minimum detectable gain recorded. **Passed on 2026-11-08** ([docs/evidence/gate1a.md](evidence/gate1a.md), [docs/evidence/gate1b.md](evidence/gate1b.md), [docs/evidence/mdg.md](evidence/mdg.md)); the Claude baselines moved to M4 | Met: proceed. Not met: fix the setup first; rules work continues, but no full-mode number is trusted until the gate passes. |
| Mid-point check | End of week 3 | Full mode runs end to end on both models | Met: proceed. Not met: move the four-week version's cuts to v1.1, or move launch by a week. |
| Gate 2 | End of week 5 | One held-out run with frozen settings: Claude Sonnet 5.5 and the open-weight model (on the rented GPU with vLLM) on the official 151-pair subset; the other 79 held-out pairs only if the headline result is borderline | Clear gain: launch claiming it. No detectable difference: launch as an open, local implementation of the published method with open-model results. Worse: make the route-one configuration the default and report this variant as a negative result. Outcomes are judged per model, and the launch claim names the model each one holds for. |
| Release gate | End of week 6 | Release checklist passed | Met: publish v0.1.0. Not met: fix and re-check; nothing is published until it passes. |

At any point, a run that exceeds its cost cap stops for review before anything else is spent.

## Fitting four weeks

A four-week launch means shortening Person A's chain, because cutting Person B's work alone does not move the launch date. It works only with Pass A as the only model-based pass at launch. This schedule is an estimate:

1. **Week 1.** Person A: M1 Gate 1 checks. Person B: parsers, normalisers, lexicons and ConText rules.
2. **Week 2.** Person A: model client, including the Anthropic adapter, and Pass A, with extraction once per transcript. Person B: alignment, Pass B rules, rules-mode Pass A, report and CLI.
3. **Week 3.** Person A: development evaluation on both models, the Claude and local-model baselines, settings frozen. Person B: promptfoo plugin, GitHub Action, `calibrate export` and `apply`, Colab notebook.
4. **Week 4.** Person A: held-out run (Gate 2), then the release. Person B: README, release checklist, launch posts.

Moved to v1.1 in this version, with what each costs:

- **Pass B's model path and certainty changes:** changed facts are checked by the rules only.
- **Pass C beyond the examination rule:** no general check for unsupported statements.
- **DeepEval and Inspect plugins, the HTML report and the Hugging Face Space:** fewer launch channels.
- **`--inject` and the stability command:** users can't estimate detection on their own notes, or measure stability before letting model flags fail builds.
- **Preprint:** posted after the release, so the launch rests on the README's results table.

The Gate 1 checks, the Anthropic adapter (Claude Sonnet 5.5 is the headline API model), the single held-out run with frozen settings and the release checklist stay, because they protect the launch claim. This version has no slack: any slip on either chain moves launch to week 5.

## After launch: v1.1

v1.1 picks up what the spec deferred, in no fixed order until launch feedback and the first month's tracking are in:

- Sound-alike drug check
- Section-aware check for restated facts
- Dialogue negation for questions about several symptoms
- MCP server
- Medication-extraction model on Hugging Face (Build 2)
- Fox et al.'s evolved single prompt as a clearly labelled note-level screen
- Anything cut to fit four weeks
