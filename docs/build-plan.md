# AI Scribe Note Checker — Build Plan

Oct 11, 2026 · @Sandesh Gubbi Virupaksha

Six weeks from the start date to a public v0.1 release of the checker specified in `docs/spec.md`, built by two people and gated twice: on reproducing the published methods at the end of week 1, and on the held-out result in week 5.

## Assumptions

All durations in this plan are my estimates, not commitments. Replace them with the owners' own estimates before week 1, because the spec makes the choice between a 4-week and a 6-week launch on those numbers.

- **Team:** two people, both full time. Person A owns evaluation and the full-mode pipeline. Person B owns the rules layer, integrations and the release. These are proposed roles; owners are named in the week 0 checklist. If either person is part time, durations scale: two half-time people would take about 12 weeks.
- **Part-time help:** a second person, not the rules author, writes the synthetic demo set from week 1 and the mutation-suite phrasings in week 2. A clinician reviews the demo set and rules-mode severity table in week 2, the severity rubric in week 3 if it is our own, and the calibration guidance in week 5.
- **Effort:** about 12 person-weeks over weeks 1–6, plus a short week 0 for decisions and setup, plus the part-time help above.
- **Compute:** the spec's estimated $1,000–2,000 in API costs, probably lower once fact extraction runs once per transcript. The budget is split by milestone after M1 measures cost per note, and spend is checked at each gate. One GPU runs the local model; if none is available, rent one for benchmark runs, which use benchmark data only. Week 4's GPU hours are worked out in M3, and a second GPU is rented if they don't fit.

**Critical path:** two chains end with no slack. Person A's runs from the week 0 decisions through reproduction (Gate 1), the full-mode pipeline, development evaluation and the held-out run (Gate 2) to launch. Person B's runs through the rules layer, which the pipeline needs by the end of week 2, and the integrations, which launch needs by the end of week 5. A one-week slip on either chain moves launch by a week. Person A's week 5 has room after the held-out run, so A also takes the Inspect task, the `bench` command and the preprint draft then.

## Before week 1

Week 0 ends when every item below is done; the three blockers come first, because the rules layer and the schedule depend on them.

- [ ] **Blocker:** choose the drug lexicon and confirm its licence allows redistribution (launch uses generic names plus a user-supplied brand map)
- [ ] **Blocker:** choose the common-symptom lexicon and confirm its licence
- [ ] **Blocker:** name an owner for every milestone, collect their estimates, and decide between 4 and 6 weeks
- [ ] Approve the API budget and set the per-run cost cap
- [ ] Choose the local open-weight model and confirm a GPU is available that runs it with enough context for a long transcript; if none is, arrange a rented one
- [ ] Check whether the paper's pinned model is still available; if not, choose and record a substitute
- [ ] Check whether Fox et al.'s harness accepts an OpenAI-compatible endpoint, and whether the local model supports a reasoning budget; estimate any adaptation work for M4
- [ ] Fill in route one's published 95% interval for Gate 1 from the paper
- [ ] Check whether Fox et al.'s severity rubric ships in their MIT-licensed harness
- [ ] Check arXiv's endorsement rules for the chosen category, and find an endorser if one is needed
- [ ] Decide the GitHub organisation (Nasken AI or personal) and create the repository under Apache-2.0
- [ ] Set up a PyPI Trusted Publisher for `scribecheck`
- [ ] Set up CI on Linux, macOS and Windows for Python 3.10 and later
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
| M1 Reproduction and scoring harness | 1 | Person A | Both published methods reproduced inside their intervals, and baselines on the held-out subset recorded (Gate 1) | Not started |
| M2 Rules layer and rules-only mode | 1–2 | Person B | `scribecheck demo` runs offline in under a minute; rules precision and recall measured against the 95% and 90% targets | Not started |
| M3 Full-mode pipeline | 2–3 | Person A | All three passes run end to end on both models; every flag carries a verified quote; calls and cost per note measured | Not started |
| M4 Development evaluation | 4 | Person A | Results on the 65 development consultations written up; local-model baselines recorded; settings frozen and tagged | Not started |
| M5 Integrations, calibration and held-out run | 3–5 | Person B; Person A for the held-out run, Inspect task, bench command and preprint draft | Plugins, Action, calibrate and stability commands pass their tests; held-out run done once (Gate 2) | Not started |
| M6 Preprint, release and launch | 6 | Persons A and B | v0.1 released through Trusted Publishing; preprint posted; release checklist passed | Not started |

## M1 Reproduction and scoring harness (week 1)

The goal is to prove the evaluation setup matches the published results before building anything that will be measured with it. Lead: Person A. Depends on week 0 for model access, the budget and the downloaded benchmark. Only the first group of tasks gates week 2; the second must be ready by the start of week 4.

For Gate 1:

- [ ] Recompute the paper's published numbers from its released judgements, with no API calls, to validate the scoring code
- [ ] Run Fox et al.'s harness for both methods on the full evaluation set with the API model, recording cost per note, model calls per note and run time
- [ ] Re-run both methods on the held-out subset with the API model to fix those baselines; local-model baselines follow in M4
- [ ] Record flags per note from the per-fact pipeline, which sizes the calibration workload
- [ ] Split the API budget by milestone from the measured cost per note
- [ ] Write down the minimum detectable gain for the API-model comparison

Ready by the start of week 4:

- [ ] Scoring harness for scribecheck's output: the *k* and *c* sweep, counts on the 47 clean notes, breakdown by severity and trace, bootstrap over consultations, and the exact McNemar test
- [ ] Correct-fact matching: how a flagged fact is matched to the benchmark's named removed fact, using model-assisted matching with a manual check of a sample

**Exit (Gate 1):** both reproductions land inside their published 95% intervals, and the held-out baselines and minimum detectable gain are recorded.

## M2 Rules layer and rules-only mode (weeks 1–2)

The goal is a working rules-only checker that installs and runs offline, and the alignment that the full-mode pipeline builds on. Lead: Person B, with the second person writing the demo set and mutation phrasings. Depends on the week 0 lexicon decisions; M3 depends on this by the end of week 2.

Week 1:

- [ ] Package skeleton, config file format and `scribecheck init`
- [ ] Publish a first development release to PyPI as soon as the package installs, to secure the name
- [ ] Input parsers for plain-text and JSON transcripts, and note section detection with the user's heading map
- [ ] Normalisers for numbers, units, frequency, route and side
- [ ] Drug and symptom lexicons, with the user-supplied brand map
- [ ] Built-in ConText rules, dialogue negation for single-symptom questions, and allergy rules

Week 2:

- [ ] Alignment with the one-consistent-value rule, Pass B rules flags and the deterministic negation check
- [ ] Rules-mode Pass A for medicines, doses, allergies and sides, plus the examination rule
- [ ] Severity table, confidence levels, both standards and the decision rule
- [ ] JSON report, Markdown renderer, exit codes, and the `demo`, `check` and `batch` commands
- [ ] Golden-file tests for the JSON and Markdown reports
- [ ] Network guard: a socket-level block in tests on all three operating systems, plus a no-network container test on Linux
- [ ] Check whether OmissionBench labels the type of each altered fact; if not, label the structured-fact pairs by hand
- [ ] Precision and recall per change type, measured on the second person's phrasings and on those alteration pairs outside the held-out subset

Second person, weeks 1–2:

- [ ] Synthetic demo set of about 20 consultations, ready for clinician review by the end of week 2
- [ ] Mutation-suite phrasings for every change type

**Exit:** `pip install` then `scribecheck demo` runs offline in under a minute, the network guard tests pass, and precision and recall are reported against the 95% and 90% targets. Missing a target is reported, not hidden, and does not block M3.

## M3 Full-mode pipeline (weeks 2–3)

The goal is all three passes working with a language model, on both the API model and the local model. Lead: Person A. Depends on M1's harness and on M2's alignment by the end of week 2.

Week 2:

- [ ] Model client for OpenAI-compatible endpoints over `httpx`, plus the Anthropic adapter
- [ ] JSON schema validation with one retry, temperature 0 and a fixed seed, and the model name and version written to every report
- [ ] Cost tracking and the per-run cost cap
- [ ] Transcript chunking with overlap, and merging of facts across chunks
- [ ] Batch mode extracts facts once per transcript and reuses them across that transcript's notes
- [ ] Pass A: extraction from the transcript only, audit with severity grading, per-fact presence check with the whole note, and the second look

Week 3:

- [ ] Merge rules facts into Pass A, and send rules "absent" verdicts for structured facts to the second check
- [ ] Pass B model path, certainty changes and its second check
- [ ] Pass C: statement splitting, checking, the second check and the default exemptions
- [ ] Deduplication across the three passes
- [ ] Fox et al.'s route-one prompts as a selectable configuration inside scribecheck, so Gate 2's fallback is a config switch
- [ ] Severity rubric: Fox et al.'s if the harness licence covers it, otherwise our own, reviewed by the clinician
- [ ] Run with the chosen local model through Ollama, and record calls, cost and time per note for both models, including seconds per note on the GPU

**Exit:** all three passes run end to end on the demo set with both models, every flag carries a verified quote, and the spec's estimates of about 10 calls and $0.46 per note are replaced with measured figures, which feed a revised budget and an estimate of week 4's GPU hours.

## M4 Development evaluation (week 4)

The goal is to tune on development data only and freeze the settings before the held-out run. Lead: Person A. Depends on M3.

- [ ] Before starting, check the week's GPU hours against M3's seconds per note, and rent a second GPU if they don't fit (benchmark data only)
- [ ] Re-run both published methods on the held-out subset with the local model, adapting the harness if week 0 showed it is needed. If the local model cannot run route two, record that baseline as not applicable rather than reporting a misleading number.
- [ ] State the minimum detectable gain for the local-model comparison before any held-out run of scribecheck
- [ ] Run scribecheck on the 65 development consultations with the local model and the API model
- [ ] Choose a fixed development subset, then tune prompts and settings on it with the local model first, using the API model to confirm
- [ ] Measure the rules layer's effect by running with and without it
- [ ] Score Pass B and Pass C on the 182 added and altered pairs outside the held-out subset
- [ ] Measure severity agreement with OmissionBench's per-fact grades (Cohen's kappa) for each model
- [ ] Stability: run each model 3 times on the fixed development subset
- [ ] Publish minimum model guidance, including the minimum context length
- [ ] Freeze the settings and tag the commit before any held-out run

**Exit:** development results are written up, the settings are frozen and tagged, the local-model baselines are recorded, and the decision on whether the local model meets the minimum is recorded.

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

- [ ] Held-out run, once, with the frozen settings, on both models
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
| Gate 1 | End of week 1 | Both reproductions inside their published intervals; held-out baselines recorded | Met: proceed. Not met: fix the setup first; rules work continues, but no full-mode number is trusted until the gate passes. |
| Mid-point check | End of week 3 | Full mode runs end to end on both models | Met: proceed. Not met: move the four-week version's cuts to v1.1, or move launch by a week. |
| Gate 2 | End of week 5 | One held-out run with frozen settings | Clear gain: launch claiming it. No detectable difference: launch as an open, local implementation of the published method with open-model results. Worse: make the route-one configuration the default and report this variant as a negative result. Outcomes are judged per model, and the launch claim names the model each one holds for. |
| Release gate | End of week 6 | Release checklist passed | Met: publish v0.1.0. Not met: fix and re-check; nothing is published until it passes. |

At any point, a run that exceeds its cost cap stops for review before anything else is spent.

## Fitting four weeks

A four-week launch means shortening Person A's chain, because cutting Person B's work alone does not move the launch date. It works only with Pass A as the only model-based pass at launch. This schedule is an estimate:

1. **Week 1.** Person A: M1 reproduction (Gate 1). Person B: parsers, normalisers, lexicons and ConText rules.
2. **Week 2.** Person A: model client and Pass A, with extraction once per transcript. Person B: alignment, Pass B rules, rules-mode Pass A, report and CLI.
3. **Week 3.** Person A: development evaluation on both models, local-model baselines, settings frozen. Person B: promptfoo plugin, GitHub Action, `calibrate export` and `apply`, Colab notebook.
4. **Week 4.** Person A: held-out run (Gate 2), then the release. Person B: README, release checklist, launch posts.

Moved to v1.1 in this version, with what each costs:

- **Pass B's model path and certainty changes:** changed facts are checked by the rules only.
- **Pass C beyond the examination rule:** no general check for unsupported statements.
- **Anthropic adapter:** users of Anthropic models wait.
- **DeepEval and Inspect plugins, the HTML report and the Hugging Face Space:** fewer launch channels.
- **`--inject` and the stability command:** users can't estimate detection on their own notes, or measure stability before letting model flags fail builds.
- **Preprint:** posted after the release, so the launch rests on the README's results table.

The Gate 1 reproduction, the single held-out run with frozen settings and the release checklist stay, because they protect the launch claim. This version has no slack: any slip on either chain moves launch to week 5.

## After launch: v1.1

v1.1 picks up what the spec deferred, in no fixed order until launch feedback and the first month's tracking are in:

- Sound-alike drug check
- Section-aware check for restated facts
- Dialogue negation for questions about several symptoms
- MCP server
- Medication-extraction model on Hugging Face (Build 2)
- Fox et al.'s evolved single prompt as a clearly labelled note-level screen
- Anything cut to fit four weeks
