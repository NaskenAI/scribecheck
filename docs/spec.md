# AI Scribe Note Checker — Technical Specification v1

Oct 11, 2026 · @Sandesh Gubbi Virupaksha

## Summary and goals

The checker compares an AI-written clinical note with its consultation transcript and reports omissions, changed facts and unsupported statements, each with the transcript passage as evidence. It runs entirely on the user's machine, either with the user's own AI model or with no model at all in rules-only mode.

Users are AI scribe vendors testing model and prompt changes, hospital AI teams evaluating scribes, and researchers. Omissions are the main target, because published audits find them to be the most common scribe error.

Goals for v1:

- A first report within 5 minutes of `pip install`, with no model, API key or account.
- Omission detection on OmissionBench's held-out subset, compared with both published methods re-run on the same subset, using one local open-weight model and one API model.
- Ships as a Python package, a CLI, a GitHub Action, and plugins for promptfoo, DeepEval and Inspect.
- Patient data never leaves the user's environment except to a model endpoint the user configures.

Success is measured 6 months after launch by PyPI and Hugging Face downloads, the number of public repositories running the checker in CI, and listings on framework integration pages. The expected case is 1,000–3,000 cumulative PyPI downloads by month 6, an estimate from comparable packages.

## Scope and non-goals

v1 checks English-language notes against their transcripts, one consultation at a time or in batches, and is positioned as an evaluation tool for testing scribe systems.

In scope:

- **Inputs:** transcript as plain text or JSON, with speaker labels optional; note as plain text or Markdown with section headings.
- **Error classes:** omissions; changed facts (negation, certainty, side, dose, frequency, route, drug, allergy); unsupported statements.
- **Modes:** rules-only (no model) and full (rules plus a language model).
- **Data:** synthetic demo data in the package; evaluation on OmissionBench, downloaded by the user rather than bundled.

Not in v1:

- **Transcript accuracy.** The checker tests whether the note matches the transcript, not whether either is correct, so a misheard drug name copied faithfully will pass. A sound-alike drug check is planned for v1.1.
- **Audio input** or speech recognition.
- **Live clinical use** as a sign-off check on individual patients' notes.
- **Clinical appropriateness** of the assessment or plan.
- **Languages** other than English.
- **Hosted service** or any storage of user data.
- **EHR integration.**
- **The patient-message and translation checkers,** which are later builds on the same core.

## Prior work and attribution

The omission method in this spec is Fox et al.'s (Composo, August 2026): list the facts the transcript establishes, then check each against the note as a closed present-or-absent question. The README, docs and any paper must credit it and MED-OMIT (Schumacher et al., 2023), which they name as its direct ancestor.

Their findings that shape this design:

- LLM judges reading the whole note score 0.79–0.94 on added or altered content but 0.50–0.63 on omissions, where 0.5 is a coin flip.
- Their per-fact pipeline detects 24.6% of omission notes at 2.7% false alarms; their single-call evolved prompt detects 36.9% at 6.2%, both on the full evaluation set.
- Omissions where the fact survives as a restatement elsewhere in the note, usually under another heading, defeat both methods.
- No threshold set on the benchmark transferred to real vendor notes; the single call had to be re-calibrated.
- A naive per-fact coverage recipe flagged 98.7% of clean notes.
- The largest real-world failure group in their companion audit was missing or wrongly recorded allergy and medication information.

Published results, each measured on its own set. Rates from different sets are not comparable, and the paper itself never compares them.

| Method | Set | Omission detection | False alarms |
| --- | --- | --- | --- |
| Per-fact pipeline (route one) | Full evaluation set; flag rule set after the fact | 24.6% | 2.7% |
| Evolved single prompt (route two) | Full evaluation set; flag rule set after the fact | 36.9% (95% CI 31.5–42.5%) | 6.2% |
| Per-fact pipeline (route one), decision rule | Held-out subset (47 clean notes) | 20.6% | 2.1% |

The baseline for this project's headline result is both methods re-run by us on the held-out subset, not the published figures above.

Licences: the OmissionBench data is CC BY 4.0 and the harness is MIT, so both can be reused with attribution. The benchmark is built partly on PriMock57 and ACI-Bench.

What this project adds, each measured against those re-run baselines: deterministic checks for structured facts, results on local open-weight models, and calibration on clinician-adjudicated flags. Section-aware checking for the restatement case follows in v1.1.

## Architecture

The core is one Python library: every entry point calls the same three checking passes, which share a rules layer and a model client.

_Diagram not included in this export (checker architecture · 3 passes, 2 shared services); see the Claude Doc._

In rules mode the model client is never called, Pass A covers only medicines, doses, allergies and sides, and Pass C runs only its examination rule.

## Checking pipeline

Three separate passes run on each consultation, because one detector cannot find both missing and added content. Every flag must carry an evidence quote that the code finds in the transcript or note it quotes, after normalising spacing and punctuation and allowing a fuzzy-match threshold. One discrepancy produces one flag: when Pass B flags an aligned pair, Pass A does not also report that fact as an omission, and Pass C does not report the note's side of it as unsupported.

### Fact categories

Every extracted fact belongs to one category. The categories drive the rules-mode severity table and, in v1.1, the section-aware check.

| Category | Example | Usual note section |
| --- | --- | --- |
| Presenting complaint | knee pain for 2 weeks | History |
| Symptoms, present or denied | no chest pain | History |
| Past medical and surgical history | appendicectomy in 2015 | History |
| Family and social history | smokes 10 a day | History |
| Medicines: current, started, stopped or changed | metformin 500 mg BD | Medications or Plan |
| Allergies, or an explicit none | penicillin, rash | Allergies |
| Examination findings | swollen left knee | Examination |
| Investigations and results | HbA1c 58 mmol/mol | Results |
| Assessment or diagnosis | likely osteoarthritis | Assessment |
| Plan: tests, referrals, prescriptions | refer to physiotherapy | Plan |
| Follow-up and safety-netting | come back if fever develops | Plan |

Note templates name their sections differently, so users map their own headings to these sections in the config file.

### Pass A: omissions (transcript to note)

1. **Segment** the transcript into turns with character offsets. Speaker labels are optional; unlabelled turns are marked as unknown speaker.
2. **Extract facts from the transcript only.** The extractor never sees the note. A transcript longer than the model's context window is split into overlapping chunks, and facts found in more than one chunk are merged. Each fact has a category, attributes (negated, family member, hypothetical, historical) and an evidence span. The rules layer adds structured facts for medicines, doses, allergies and sides, merged with the model's facts and deduplicated.
3. **Audit the list.** Drop facts whose quote is not in the transcript, merge duplicates, mark each fact core or contextual, and grade severity as critical, supporting or peripheral against a written rubric.
4. **Check each fact against the note,** with the whole note in every call, as a closed question: present, partial or absent. A "present" verdict must quote the note; if the quote is not found, the fact is re-checked. Structured facts are checked by the rules layer on normalised values. In full mode, a structured fact the rules find absent also goes to the model's quote-verified second check, which can confirm the omission.
5. **Second look** at facts judged absent, asking for a supporting quote before the flag is raised.

Fox et al.'s evolved single prompt is not in v1. It judges the note as a whole without naming the missing fact, so it cannot meet the evidence requirement, and it only became a working detector with a reasoning budget that most local models lack. It may return in v1.1 as a clearly labelled note-level screen. The section-aware check for restated facts also moves to v1.1.

### Pass B: changed facts

Structured facts from the transcript and note are aligned by drug, body site or symptom, then compared after normalisation. Flag types are negation reversed, certainty changed (a tentative statement in the transcript written as definite in the note, or the reverse), side changed, dose changed, frequency changed, route changed, drug substituted, and allergy lost or altered.

Pairs the rules cannot align go to the model. Certainty changes are always judged by the model, with the ConText "possible" attribute as a hint. Every model verdict in Pass B gets a second, quote-verified check before it can be marked confirmed.

### Pass C: unsupported statements (note to transcript)

1. Split the note into atomic statements.
2. Check each against the transcript as supported, contradicted, unsupported or not verifiable, with the transcript quote where one exists.
3. **Second check.** Each contradicted or unsupported verdict is re-checked against the transcript, with a quote requirement, before it can be marked confirmed.
4. **Default exemptions.** Content clinicians normally add without saying it aloud is reported as "not verifiable" and never fails a note: examination findings in face-to-face visits, vitals, test results, history taken from the record, and prefilled demographics. Users can change this list.
5. **Examination rule.** When the modality is telephone, any examination finding is flagged as unsupported. For video, only hands-on findings such as palpation, auscultation and reflexes are flagged. Without a modality, the rule does not run. Findings are detected in the note's Examination section and after phrases such as "O/E" and "on examination". A finding inside an Examination section is a certain flag; one detected only by phrase is possible.
6. Grade each flag for severity, as in Pass A.
7. In rules mode only the examination rule runs.

In full mode the pipeline makes roughly 10 model calls per note, with presence checks batched about 10 facts per call. This is an estimate to confirm in milestone 1.

## Deterministic rules layer

The rules layer gives the same answer every run, needs no model, and powers rules-only mode. It favours precision: when it cannot align two facts with confidence, it passes them to the model, or in rules-only mode reports them as unchecked rather than guessing. A certain flag also needs one consistent value or status for the item in the transcript, matched on temporality and experiencer. When the transcript mentions the item more than once with different values (a dose changed during the consultation, a symptom present last week and absent today, both knees discussed), the rules emit no certain flag and send the item to the model.

| Fact type | Normalised to | Examples handled |
| --- | --- | --- |
| Numbers | Digits | "five hundred" = 500; "one and a half" = 1.5; "half a tablet" |
| Units | Canonical unit, with conversion | mg, mcg/µg, g, ml, units, IU, puffs, drops; 0.5 mg = 500 mcg |
| Frequency | Times per day, interval, as-needed flag | OD, BD/BID, TDS/TID, QDS/QID, nocte, mane, PRN, "every 8 hours", weekly |
| Route | Canonical route | oral/PO, IV, IM, SC, SL, PR, inhaled, topical |
| Side | Left, right or both, plus body site | left/right/bilateral, L/R, "left-sided", LLQ/RUQ; "that's right" ignored unless near a body part |
| Symptoms | Lexicon term | "short of breath" and "breathless" = shortness of breath; common symptoms only |
| Context | Negated, possible, historical, family, hypothetical | Built-in implementation of the ConText rules; "if the pain gets worse, come back" is hypothetical |
| Dialogue negation | Symptom bound to its answer | "Any chest pain?" then "No" = chest pain negated; single-symptom questions only in v1 |
| Allergies | Allergen, reaction, or explicit none | "allergic to penicillin", "amoxicillin gives me a rash", "no allergies", NKDA |
| Drug names | Generic name | Generic names at launch; brand names through a user-supplied map; spelling-tolerant matching |

Dialogue negation needs its own rule because sentence-level negation tools miss it: the denial sits in the patient's turn, not in the question's sentence. In v1 the rule handles only a question about one symptom followed by a direct answer. Questions about several symptoms go to the model, and the rule is off when speaker labels are missing.

**Deterministic negation check.** When a lexicon symptom appears in both the transcript and the note, the rules compare its negation status, using ConText and dialogue negation on the transcript side. A mismatch, such as "no chest pain" in the transcript and "chest pain" in the note, is a certain negation-reversed flag. Symptoms outside the lexicon go to the model.

In rules-only mode the layer reports changed values (dose, frequency, side, route), negation reversals of lexicon symptoms, and omissions of medicines, doses, allergies and sides. Pass C runs only its examination rule, and all other checks need the model. Rules-extracted facts count as core facts, because rules mode has no model audit. Confidence and severity in rules mode follow the tables in the next section.

## Severity, decision rule and calibration

The checker acts on per-fact verdicts and severity, not on a single note score, because Fox et al. found that combining per-fact evidence into one note-level score destroys the omission signal.

**Confidence** describes the evidence, not where the flag came from:

- *Certain:* a deterministic mismatch found by the rules layer, when the transcript gives one consistent value or status for the item: a value mismatch on an aligned pair (dose, frequency, side or route), a negation reversal of a lexicon symptom, or an examination finding inside the Examination section of a telephone consultation's note.
- *Confirmed:* a model verdict in any pass upheld by its quote-verified second check.
- *Possible:* everything else, including rules-based omissions in rules mode, where no model can confirm them (a medicine may have been stopped, declined or a relative's), partial verdicts, and model verdicts the second check did not uphold.

**Severity** is critical, supporting or peripheral, the same three grades as OmissionBench. In full mode the model grades it against a written rubric, reusing Fox et al.'s rubric if their MIT-licensed harness includes it. Rules mode has no grader, so it uses this default table, which users can change:

| Rules-mode finding | Severity |
| --- | --- |
| Allergy lost or changed | Critical |
| Dose, frequency or side changed | Critical |
| Negation reversed for a lexicon symptom | Critical |
| Examination finding in a telephone consultation | Critical |
| Route changed | Supporting |
| Medicine missing from the note | Supporting |

**Standard.** A `standard` setting decides what counts as an omission:

- *Clinical* (default): critical and supporting core facts must appear in the note. Peripheral and contextual facts are not checked.
- *Strict:* every core fact must appear, including peripheral ones, as in OmissionBench. Use it for benchmarking and calibration runs, not for routine checks of real notes, because a concise note can legitimately leave transcript facts out.

**Default decision rule:**

- **Fail** on any critical flag at certain or confirmed confidence.
- **Warn** on supporting flags, partial verdicts, and critical flags at possible confidence.
- **Report only** peripheral flags (strict standard only) and statements marked not verifiable.

The GitHub Action is stricter about stability: by default only critical flags at certain confidence fail the build, and model-derived flags warn, until the user has measured run-to-run stability on their own test set with the stability command. Users change all of these rules in the config file.

**Calibration.** Clinician-approved notes are not clean under the tool's standard: Fox et al. found material discrepancies in every reference note they audited in two public corpora. So `calibrate` works from adjudicated flags, not from notes labelled clean:

1. `calibrate export` runs the checker on at least 30 notes from the user's own scribe (an estimate to confirm in testing) at the most inclusive settings: strict standard and every confidence level. Every candidate setting's flags are therefore already in the export. It writes them to a CSV with an empty label column. With `--sample N`, it exports a random sample of N flags from each severity and confidence combination instead of all of them. The expected labelling workload is stated once milestone 1 has measured flags per note.
2. A clinician labels each flag as a real error, an acceptable omission, or wrong.
3. `calibrate apply labels.csv --max-rejected 0.2` picks the setting (standard, failing severities and confidence levels) that raises the most flags while keeping the share labelled acceptable omission or wrong under the target. The 0.2 is an example, not a default.
4. With `--inject`, it also copies the notes with injected value changes (flipped negations, swapped sides, changed doses) to estimate detection. Injected omissions are not used, because deleting a fact cleanly is unreliable: 44% of the benchmark's complete-removal attempts failed verification.
5. It writes a config with the chosen settings and the counts behind them. With 30 notes these rates are rough, and the output says so.

## Data model and report format

Inputs are a transcript and a note; output is a versioned JSON report, with Markdown and HTML renderings of the same content.

**Transcript input.** Plain text with speaker prefixes ("Doctor:", "Patient:") or JSON. The optional `modality` field (in-person, video or telephone) drives the examination rule. Speaker labels are optional; without them, dialogue negation is switched off. The example below is an excerpt, which is why the report example cites later turns.

```json
{
  "consultation_id": "demo-001",
  "modality": "telephone",
  "turns": [
    {"speaker": "clinician", "text": "Any allergies to medicines?"},
    {"speaker": "patient", "text": "Penicillin, it gives me a rash."}
  ]
}
```

**Note input.** Plain text or Markdown. Sections are detected from headings or lines such as "Allergies:".

**Report output.**

```json
{
  "schema_version": "1.0",
  "consultation_id": "demo-001",
  "mode": "full",
  "standard": "clinical",
  "model": {"name": "<user's model>", "version": "<pinned version>", "endpoint_host": "localhost"},
  "decision": "fail",
  "counts": {"omission": 2, "changed_fact": 1, "unsupported": 0, "not_verifiable": 3, "unchecked": 1},
  "flags": [
    {
      "id": "F1",
      "type": "omission",
      "category": "allergy",
      "verdict": "absent",
      "severity": "critical",
      "confidence": "confirmed",
      "origin": "rules",
      "rule_id": "allergy.missing",
      "second_check": "model",
      "model_version": "<pinned version>",
      "fact": "Allergic to penicillin (rash)",
      "transcript_evidence": {"text": "Penicillin, it gives me a rash.", "turn": 2, "start": 0, "end": 31},
      "note_evidence": null,
      "expected_section": "Allergies"
    },
    {
      "id": "F2",
      "type": "changed_fact",
      "category": "side",
      "verdict": "changed",
      "severity": "critical",
      "confidence": "certain",
      "origin": "rules",
      "rule_id": "side.mismatch",
      "transcript_value": "left knee",
      "note_value": "right knee",
      "transcript_evidence": {"text": "my left knee has been swelling", "turn": 5, "start": 12, "end": 42},
      "note_evidence": {"text": "Right knee swelling for 2 weeks", "section": "History"}
    },
    {
      "id": "F3",
      "type": "omission",
      "category": "follow_up",
      "verdict": "absent",
      "severity": "supporting",
      "confidence": "confirmed",
      "origin": "model",
      "model_version": "<pinned version>",
      "fact": "Return if the swelling spreads or a fever develops",
      "transcript_evidence": {"text": "if it spreads or you get a fever, come straight back", "turn": 14, "start": 0, "end": 52},
      "note_evidence": null,
      "expected_section": "Plan"
    }
  ]
}
```

The report never repeats the full transcript or note; only the evidence quotes. `--include-facts` adds the full fact list for debugging.

**Exit codes:** 0 = pass or warn, 1 = fail under the decision rule, 2 = tool error. In GitHub Actions the Markdown report is also written to the job summary.

## Interfaces

One core Python API sits under every interface; the CLI, plugins and GitHub Action are thin wrappers around it. The working name is `scribecheck`, which was free on PyPI on 11 October 2026 and should be registered now.

### Python API

```python
from scribecheck import check, check_files, Config

config = Config(
    mode="full",                           # or "rules"
    standard="clinical",                   # or "strict"
    base_url="http://localhost:11434/v1",  # any OpenAI-compatible endpoint
    model="<model name>",
)

# Text already in memory
report = check(transcript=transcript_text, note=note_text, config=config)

# Files on disk
report = check_files("transcript.json", "note.md", config=config)
print(report.decision, len(report.flags))
```

### CLI

| Command | What it does |
| --- | --- |
| `scribecheck demo` | Runs bundled synthetic examples in rules mode; no setup |
| `scribecheck check -t transcript.json -n note.md` | Checks one consultation |
| `scribecheck batch cases/` | Checks every transcript–note pair in a folder, matched by file name |
| `scribecheck stability cases/ --runs 3` | Runs the checker several times on the same cases and reports how often each note's decision stays the same |
| `scribecheck calibrate export notes/ --sample 20` | Runs the checker at the most inclusive settings and writes flags (here a sample of 20 per severity and confidence combination) to a CSV for clinician labelling |
| `scribecheck calibrate apply labels.csv --max-rejected 0.2` | Chooses settings from the labelled flags and writes a config file |
| `scribecheck bench omissionbench` | Downloads OmissionBench, prints its CC BY attribution, and reports detection, false alarms and correct-fact rate |
| `scribecheck init` | Writes a starter config file |

Shared flags are `--mode rules|full`, `--standard clinical|strict`, `--format json|md|html`, `--fail-on critical|supporting`, `--repeats N` (majority vote over N runs) and `--config`.

### Plugins

Each plugin is an optional extra, such as `pip install scribecheck[deepeval]`, so the core stays light.

| Framework | Integration point | Mapping |
| --- | --- | --- |
| promptfoo | Python assertion file with `get_assert(output, context)` | Transcript from test variables; note = model output; returns pass, score and reason |
| DeepEval | `ScribeCheckMetric`, a `BaseMetric` subclass | Transcript in the test case's context; note = `actual_output` |
| Inspect | `scribecheck_scorer()` built with `@scorer`, plus a task that runs OmissionBench | Transcript in the sample input; note = model output |

Plugins report the decision as a score of 1.0 for pass, 0.5 for warn and 0.0 for fail, with the flag counts in the reason text.

The docs for each plugin explain how to turn off that framework's own telemetry and cloud sync.

### GitHub Action

A composite action installs the package, runs `batch` on a folder of test consultations, writes the Markdown report to the job summary, uploads the JSON report as an artifact, and fails the job under the decision rule. By default only critical flags at certain confidence fail the build, and model-derived flags warn. The model endpoint comes from a repository secret; rules mode needs none.

GitHub-hosted runners cannot run a local model at useful speed, so there the Action runs in rules mode or calls an API endpoint. Teams testing on real patient data need self-hosted runners, and the docs say so.

An MCP server is deferred to v1.1.

## Model backends, privacy and security

The checker talks only to a model endpoint the user configures, sends nothing anywhere else, and keeps its dependency list short enough to pass a hospital security review.

**Backends.** Any OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio, llama.cpp server, Azure OpenAI, OpenAI), plus an Anthropic adapter. Calls go through `httpx` directly rather than an LLM framework, to keep the dependency tree small.

**Model output handling.** Every call asks for JSON, which is validated against a schema; one retry on a parse failure, after which the item is reported as unchecked. Calls use temperature 0 and a fixed seed where the backend supports one, and every report records the model name and version. Run-to-run stability is measured for each supported model and published as a release metric. Minimum model guidance, including the minimum context length, is published from milestone 1 results, not guessed.

**Privacy rules:**

- No telemetry, analytics or update checks.
- Rules mode makes no network calls; a test runs it with the network blocked to prove this.
- A warning is printed when the model endpoint is not on the local machine or local network.
- Logs contain no transcript or note text unless `--debug-log` is set, which prints a warning.
- The response cache is off by default, because it would store patient text; turning it on requires a directory path, and it is meant for synthetic test fixtures in CI.
- Reports contain evidence quotes, so the docs tell users to treat reports as patient data.

**Dependencies and releases:**

- Core dependencies are `httpx` and `pydantic`. The ConText rules are a small built-in implementation, which keeps spaCy out of the core install; medspaCy is an optional extra for users who want its fuller rule set.
- Plugins are optional extras.
- Releases go to PyPI through Trusted Publishing from GitHub Actions, with build attestations and a CycloneDX software bill of materials.
- Dependabot keeps dependencies current.
- The project is licensed Apache-2.0, supports Python 3.10 and later, and runs on Linux, macOS and Windows.

## Test data and evaluation plan

Headline results come from one run on OmissionBench's held-out subset, compared with both of Fox et al.'s methods re-run by us on the same subset. All tuning happens on other data.

| Dataset | Contents | Used for |
| --- | --- | --- |
| Synthetic demo set (bundled) | About 20 authored consultations across general practice, emergency, orthopaedics, cardiology and paediatrics; a clean note for each, plus variants with known errors. No real patient data; clinician-reviewed before release; CC BY 4.0. | Demo, docs, unit tests |
| Mutation suite (generated) | Synthetic notes with one injected change each (negation flip, side swap, dose doubled or halved, unit swap, frequency change, allergy deleted, drug substituted), with phrasings written by a second person, not the rules author | Rules-layer development |
| OmissionBench added and altered pairs | The 182 of the 202 pairs (95 additions and 107 alterations in total) that sit outside the held-out subset; authored outside this project | Pass B and Pass C accuracy, and checking that Pass A does not reduce detection of these |
| OmissionBench development portion | The 65 evaluation-set consultations outside the held-out subset | Development and ablations |
| OmissionBench held-out subset | 151 pairs over 47 consultations: 131 omissions, 20 added or altered, 47 clean twins | Headline results, run once |

Targets for the rules layer are at least 95% precision and 90% recall per change type, measured on the second author's phrasings and on OmissionBench's alteration pairs outside the held-out subset, not only on mutations the rules author generated. These are goals, not measured figures.

Protocol:

1. **Reproduce** both of Fox et al.'s methods with their released harness on the full evaluation set. Gate 1 passes when each lands inside its published 95% interval: 31.5–42.5% for route two, and the interval the paper reports for route one. Use their pinned model if it is still available; otherwise record the substitute and treat the gate as approximate.
2. **Re-run both methods on the held-out subset.** These are the baselines.
3. **State the minimum detectable gain** before any held-out run. With 131 omission pairs over 47 consultations, a rough estimate is about 10 percentage points; smaller gains are unlikely to be distinguishable.
4. **Develop** on the 65 development consultations: run scribecheck in full mode on the reproduction model and on one local open-weight model that runs on a single consumer GPU, then add the rules layer and measure its effect separately. Develop Pass B and Pass C on the 182 added and altered pairs outside the held-out subset.
5. **Confirm once** on the held-out subset with settings fixed beforehand. Fox et al. found that gains chosen on a small validation set did not survive a held-out test.
6. **Score and report.** Benchmark runs use the strict standard. Detection and false alarms count only Pass A omission flags: a note counts as flagged when it has at least *k* omission flags at confidence *c* or higher. Sweeping *k* from 1 to 5 and *c* across possible and confirmed gives the detection curve. Report the points nearest 1, 2 and 3 false alarms on the 47 clean notes (steps of about 2.1 points), broken down by severity and by trace (complete, fragment, restatement). Pass B and Pass C are scored separately on the added and altered pairs. Also report the correct-fact rate: the share of detected notes where a flag names the removed fact. The outcome is one of three: a clear gain, no detectable difference, or worse.
7. **Measure severity agreement.** Compare each model's severity grades with OmissionBench's per-fact grades and report Cohen's kappa. For reference, two frontier models agreed at 0.177 without a rubric and 0.662 with one.
8. **Measure stability.** Run each supported model 3 times on the development portion and publish the share of notes whose decision does not change.
9. **Publish** every judgement, config and model version.

Statistics follow the benchmark paper: confidence intervals by bootstrap over consultations, and an exact McNemar test for paired comparisons.

**Compute budget.** A published summary of Fox et al.'s results puts the single call at about $0.046 per note, and the paper puts it at roughly a tenth of the pipeline's cost, so the pipeline costs about $0.46 per note on an API model. One pass over the evaluation set's 607 notes then costs about $280, and the full plan (both reproductions, development runs, three stability runs and the held-out run) an estimated $1,000–2,000 in API costs, plus local GPU time. These are estimates. Each run has a cost cap in the config, and work stops for review if a run exceeds it; development runs use the local model first, so the API model is needed mainly for reproduction and the held-out run.

## Distribution and adoption

We expect the largest download driver to be the checker running in other teams' automated tests, an assumption rather than a measurement, so CI and framework integrations ship at launch, not later.

Channels, in order of expected impact:

1. **CI use.** The GitHub Action, plus a pull request to each of promptfoo, DeepEval and Inspect asking them to list the plugin on their integrations page.
2. **Zero-setup first run.** `pip install scribecheck`, then `scribecheck demo`, works offline in rules mode in under a minute. The README opens with those two commands.
3. **Hugging Face.** Publish the synthetic demo set and the mutation suite as datasets, a Space running the demo in the browser, and later the medication-extraction model (Build 2).
4. **Colab notebook** that installs the package and runs the demo, linked from the top of the README.
5. **Launch with the benchmark result.** A short preprint on arXiv, then posts the same week on Hacker News, r/MachineLearning, LinkedIn health-AI circles, and AMIA and HIMSS groups.
6. **Direct outreach** to two or three scribe vendors and hospital AI evaluation teams at launch, offering help setting up the GitHub Action.
7. **Findability.** Keywords in the package description (AI scribe, clinical note, omission, hallucination), PyPI classifiers, GitHub topics, and a citation file so the preprint is cited correctly.

Tracked monthly: PyPI downloads (pypistats), Hugging Face downloads, GitHub stars and dependent repositories, and integration-page listings.

## Milestones and stop rule

Launch is planned for week 4, with two gates before it: the setup must reproduce the published numbers, and the held-out result decides what the launch claims. To fit four weeks, the section-aware check moves to v1.1 and dialogue negation covers single-symptom questions only. Before week 1, every item gets an owner and an estimate, and the choice between 4 and 6 weeks is made from those numbers.

_Diagram not included in this export (milestones · 4 phases, 2 gates); see the Claude Doc._

If Gate 1 fails, fix the setup before building anything else, because no later number can be trusted. If the held-out run shows no detectable gain over both re-run baselines, launch on schedule as an open, local implementation of the published method with open-model results, rather than spending more weeks tuning. If it is worse, make the better re-run baseline the default and report this project's variant as a negative result.

## Risks and open questions

The largest risk is that the additions show no detectable gain over the re-run baselines, which weakens the launch story but leaves a usable tool.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| No detectable gain over the re-run baselines | Weaker preprint and launch | Ship as an open, local implementation of the published method with open-model results; report the result honestly |
| Held-out subset too small to show a modest gain | Gate 2 ends inconclusive | State the minimum detectable gain beforehand; report "no detectable difference", not a win |
| Local open-weight models perform far worse | Full mode unusable on local hardware | Publish per-model results and a minimum model; rules mode still works |
| Composo's MIT-licensed harness is already open | An alternative exists today | Differentiate as a product: CLI, rules mode, evidence reports, calibration and the GitHub Action; launch within 4 weeks |
| Flags on good real notes | Vendors remove the tool from CI | Clinical standard by default, calibration on adjudicated flags, Pass C exemptions |
| Model verdicts change between runs | Builds fail at random | Model-derived flags only warn in the Action by default; `--repeats`; stability published per model; a stability command for users' own test sets |
| Transcription errors pass through | False confidence in a clean report | Scope stated at the top of the README; sound-alike drug check in v1.1 |
| Hospital security reviews | Slow hospital adoption | Target scribe vendors first; no telemetry, short dependency list, software bill of materials |
| Used as a live sign-off check on patients' notes | May count as regulated medical software in some markets | Positioned and documented as an evaluation tool; a regulatory adviser should be asked before anyone uses it per patient |
| Schedule overrun | Late launch | Section-aware check and broader dialogue negation moved to v1.1; a named owner for each item |
| API costs exceed the budget | Fewer runs and weaker evidence | Cost cap per run; develop on the local model first; API model mainly for reproduction and the held-out run |

Open questions:

- [ ] **Blocker, decide before week 1:** drug lexicon source and whether its licence allows redistribution. Launch plan is generic names plus a user-supplied brand map.
- [ ] **Blocker, decide before week 1:** source for the common-symptom lexicon and its licence.
- [ ] Owner and estimate for each milestone item, which decide between 4 and 6 weeks
- [ ] Fill in route one's published 95% interval for Gate 1 from the paper
- [ ] Which local open-weight model to standardise on for published results
- [ ] Whether the benchmark paper's pinned model is still available for the reproduction step
- [ ] Whether Fox et al.'s severity rubric ships in their MIT-licensed harness or only in the paper
- [ ] Whether to publish under the Nasken AI GitHub organisation or a personal account
- [ ] Which clinician reviews the synthetic demo set, the rules-mode severity table and the calibration guidance

## References

- Fox S, Markham L, Lail R, Karotsieris M. [LLM Judges Verify Presence, Not Absence: Omission Blindness in AI Clinical Notes and What Recovers It](https://arxiv.org/abs/2608.31016). arXiv:2608.31016, 31 August 2026.
- OmissionBench data: huggingface.co/datasets/ComposoAI/OmissionBench (CC BY 4.0), as named in the paper's release appendix.
- OmissionBench harness: github.com/composo-ai/omission-bench (MIT), archived at Zenodo DOI 10.5281/zenodo.22160954.
- Schumacher et al., 2023. MED-OMIT, cited by Fox et al. as the direct ancestor of fact enumeration for omission measurement; not yet read directly.
- PriMock57 and ACI-Bench, the public corpora OmissionBench is partly built on.
