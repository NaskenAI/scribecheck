# Gate 1(b) runbook: re-running the two published methods

Gate 1(b) checks that the published OmissionBench harness, run fresh on development data,
reproduces the released judgements:

- Route two: `power-gepa-high`, the prompt-optimised single call at high reasoning effort.
- Route one: the three-stage pipeline, tier B3.

**A person runs every command here, in their own terminal, with their own OpenRouter key.
Claude Code never runs them.** They call paid APIs. They read only from `data/omissionbench`
(development data) and the harness's own code and prompts. Nothing here touches held-out
data.

## What runs

| | Route two | Route one |
|---|---|---|
| Harness runner | `judges/w2_power.py --arms power-gepa-high` | `judges/w2_pipeline.py --tiers B3` |
| Models | `openai/gpt-5.4` at reasoning effort high, 12,000-token cap, temperature 1.0 | stage 1 extraction `anthropic/claude-opus-5` (medium); stage 1b audit and stage 3 refute `openai/gpt-5.5` (high); stage 2 check `openai/gpt-5.4` (medium) |
| Replicates | 1 (seed 11) | 1 |
| Transport | OpenRouter, as published (`models.lock.json`) | OpenRouter, as published |

Both run on the same 51 development notes: 10 consultations, 41 errored notes (23
omissions, 8 additions, 10 changes) and 10 clean notes. The consultations were chosen across
the four strata in proportion to their size (largest remainder), from the 65 development
consultations, all of which have complete released records for both methods, with
`random.Random(20261108)`:

| Consultation | Errored (omit / add / change) | Clean | Notes |
|---|---|---|---|
| aci\|D2N090 | 6 (4/1/1) | 1 | 7 |
| aci\|D2N156 | 5 (3/1/1) | 1 | 6 |
| aci\|D2N177 | 5 (3/1/1) | 1 | 6 |
| authored\|family_history_attribution | 3 (1/1/1) | 1 | 4 |
| authored\|ha_sudden_thunderclap | 3 (1/1/1) | 1 | 4 |
| primock\|day1_consultation01 | 4 (3/0/1) | 1 | 5 |
| primock\|day4_consultation10 | 3 (2/0/1) | 1 | 4 |
| primock\|day5_consultation07 | 3 (1/1/1) | 1 | 4 |
| primock\|day5_consultation11 | 4 (2/1/1) | 1 | 5 |
| trapblind\|tb_lipids_results_review | 5 (3/1/1) | 1 | 6 |

The released references, all on `factorial-v1` and `openai/gpt-5.4`, are:

- Route two: `power-gepa-high` replicates 1 and 3.
- Route one: B3 replicate 1.

## Pass criteria

**Route two.** Per-note agreement on the `route-two-below-10` flag (aggregate below 10) must
reach the pass mark against released r1 **and** against released r3. The pass mark is 90%,
or the released r1-versus-r3 agreement on these notes if that is lower. A note the new run
failed to parse, or lacks, counts as a disagreement.

On these 51 notes, released r1 and r3 agree on 46 (90.2%), so the pass mark is 90%. That
means 46 of 51 passes and 45 fails, and the new run must agree with each released replicate
about as well as the two agree with each other.

**Route one.** This run uses fresh fact extraction and audit. The released runs'
per-consultation fact lists (`results/w2-pipeline/_cache/`) were never released, so per-note
agreement also measures fact-list variation, and it does not gate. The run passes when:

1. every one of the 51 notes has a valid record (present, not a parse failure); and
2. on the `pipeline-critical-absent` rule (any fact graded critical judged absent):
   - its detection count on the 23 omission notes is within **4** of released r1 on the same
     notes; and
   - its false-alarm count on the 10 clean notes is within **1**.

Why these margins:

- The released B3 replicates differ by up to 3 detections out of 131 (27, 30 and 28 on the
  held-out subset). They shared one cached fact list, so that spread is the per-note check's
  variation alone.
- Fresh extraction adds variation that has not been measured. On 23 notes, one run's
  binomial standard deviation is about 2 notes.
- A margin of 4 (17 percentage points) is wide enough not to fail on noise and narrow enough
  to catch a broken run.

Per-note agreement is reported for information. **If it is below 80%, investigate before
M4**: compare the new fact lists with what the released records imply, and check the
extraction and audit stages before building on route one.

## Cost and spend caps

These estimates use the pinned prices (per million tokens: gpt-5.4 $2.50 in / $15 out;
gpt-5.5 $5 / $30; claude-opus-5 $5 / $25) and the harness's token caps. Input sizes were
measured from these 10 transcripts (about 2,100 tokens each) and 51 notes (about 830 tokens
each).

| Run | Typical | Worst case (every call at its cap) |
|---|---|---|
| Route two | about $4.50 | about $10 |
| Route one | about $13 | about $39 |
| **Total** | **about $18** | **about $50** |

Caps:

1. **Primary:** a dedicated OpenRouter key with a credit limit of **$60**, kept in your own
   shell only. Don't put it in a file in either repository, and don't use the harness's
   `secrets.env`.
2. **Route two:** `--warn-usd 8 --stop-usd 12`, scoped to the run. Do not pass
   `--authorize-spend`, which disables the stop (`w2_common.py:770`).
3. **Route one:** `--warn-usd 25 --stop-usd 40 --hard-usd 45`, scoped to the experiment.
   Past the stop, it finishes the current chunk and closes cleanly (`w2_pipeline.py:429–441`).

## Steps

The harness is at `~/data/omission-bench-harness`, commit `bd64b4e` (v1.0.4). Run every step
from the scribecheck repository root.

### 0. Re-run the split

The development copy was written before the split kept `n_absent_critical`, which route
one's released counts need. Re-run `scripts/split_omissionbench.py` with the same arguments
as before. Both output folders must be empty, so move the previous outputs aside first.
Check that `data/omissionbench/judge_scores` B3 rows now carry `n_absent_critical`.

### 1. Set up

```sh
cd ~/myproject/scribecheck
H=~/data/omission-bench-harness
git -C "$H" rev-parse --short HEAD          # expect bd64b4e
git -C "$H" status --porcelain              # expect nothing; the harness refuses a dirty tree
mkdir -p runs/gate1b
python3 -m venv runs/gate1b/venv
runs/gate1b/venv/bin/pip install -r "$H/requirements.txt"
```

Keep the venv outside the harness, because a folder inside it would make its tree dirty. The
harness writes its outputs to `$H/results/`, which its dirty check ignores
(`common.py:245–246`). Don't replace that folder with a symlink: a symlink counts as dirty.
Step 6 copies the outputs into `runs/gate1b/`.

If the full requirements file fails to install (it includes clustering libraries these
runners don't use), install `httpx==0.28.1 python-dotenv==1.2.2 openai==2.44.0
numpy==2.4.6`, then confirm with step 3's dry runs.

### 2. Write the subset spec

```sh
.venv/bin/python scripts/compare_gate1b.py spec \
  --released data/omissionbench/judge_scores --out runs/gate1b/subset_spec.json
```

Expect `51 note(s) over 10 consultation(s)`.

### 3. Dry runs (no API calls)

```sh
export OMISSIONBENCH_TRANSCRIPTS="$PWD/data/omissionbench/transcripts"
D="$PWD/data/omissionbench/pairs/dataset_v2.json"
SPEC="$PWD/runs/gate1b/subset_spec.json"
PY="$PWD/runs/gate1b/venv/bin/python"

"$PY" "$H/judges/w2_power.py" --plan --arms power-gepa-high --replicate 1 \
  --dataset "$D" --subset-spec "$SPEC" --experiment gate1b-power --tag power-gepa-high-r1

"$PY" "$H/judges/w2_pipeline.py" --plan --tiers B3 \
  --dataset "$D" --subset-spec "$SPEC" --experiment gate1b-pipeline --tag gate1b
```

Check both report 51 notes, gpt-5.4 at effort high with a 12,000-token cap for route two,
and extraction and audit cached for 0 of 10 consultations for route one. Stop if either
names a different dataset or note count.

### 4. Route two (paid)

```sh
export OPENROUTER_API_KEY=...               # the dedicated $60-limit key
"$PY" "$H/judges/w2_power.py" --arms power-gepa-high --replicate 1 \
  --dataset "$D" --subset-spec "$SPEC" --experiment gate1b-power --tag power-gepa-high-r1 \
  --warn-usd 8 --stop-usd 12 -y
```

### 5. Route one (paid)

```sh
"$PY" "$H/judges/w2_pipeline.py" --tiers B3 \
  --dataset "$D" --subset-spec "$SPEC" --experiment gate1b-pipeline --tag gate1b \
  --warn-usd 25 --stop-usd 40 --hard-usd 45 -y
unset OPENROUTER_API_KEY
```

If a run stops partway, the same command resumes: completed notes are skipped.

### 6. Copy the outputs and compare

```sh
mkdir -p runs/gate1b/harness-results
cp -R "$H/results/." runs/gate1b/harness-results/
.venv/bin/python scripts/compare_gate1b.py compare \
  --released data/omissionbench/judge_scores --spec runs/gate1b/subset_spec.json \
  --new-power runs/gate1b/harness-results/gate1b-power/_state/power-gepa-high-r1.jsonl \
  --new-b3 runs/gate1b/harness-results/gate1b-pipeline/_state/gate1b-B3.jsonl \
  | tee runs/gate1b/compare.txt
```

The exit status is 0 only if every compared method passes. The output holds counts,
percentages and note keys only. Once the copy is checked, `$H/results/` can be removed.

## Outputs (all under `runs/gate1b/`, gitignored)

| Path | What |
|---|---|
| `subset_spec.json` | the 51 note keys, passed to both runners |
| `harness-results/gate1b-power/` | route two's run folders (manifests, per-call logs) and `_state/power-gepa-high-r1.jsonl` |
| `harness-results/gate1b-pipeline/` | route one's run folders and `_state/gate1b-B3.jsonl`, plus the stage log |
| `harness-results/w2-pipeline/_cache/` | the fresh fact lists and audits, per consultation |
| `harness-results/cost_ledger.jsonl` | spend per call |
| `compare.txt` | the comparison and the pass or fail lines |

## Known differences from the released runs

- **Seeds.** Per-note seeds come from each note's position among all loaded notes
  (`w2_common.py:569`). The released runs loaded all 495 evaluation pairs, and this run
  loads the 265 development pairs, so seeds differ. The card notes seeds do not reproduce
  output token for token anyway.
- **Route one's fact lists.** These are re-extracted and re-audited (see the pass criteria).
- **Dataset label.** New records carry the development file's own `dataset_version`. The
  comparison matches notes by `note_key`, not by version.
