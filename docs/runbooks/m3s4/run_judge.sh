#!/usr/bin/env bash
# M3 session 4: model-assisted correct-fact matching on a full development run, for our
# flags and route one's (released B3 r1) flagged facts, whose text the development split
# keeps in judge_scores for development rows only. Route two (power-gepa-high) gives a score
# and no named omissions, so its correct detection is not applicable and nothing of it is
# judged. judge stops before any call when no development row carries baseline_facts (the
# split has not been re-run). One call per item to Claude Sonnet 5.5 at effort low, fact
# texts only (never the transcript or note); capped at $3. Paid API calls: run it yourself,
# never from an agent. Development data only. See docs/evidence/fact_matching_v4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_judge.sh [RUN]     # RUN defaults to runs/m3s4/claude-medium-v4-full
#
# Writes RUN/fact_judgements.json (the cache: running again makes no calls for items already
# judged), runs/m3s4/score_<name>_judged.txt, docs/evidence/severity_agreement_<version>.md
# (words and model) and, unless it is already there (a reviewer may be filling it in),
# docs/review/correct_fact_sample[_<version>].csv (30 pairs for a person to check;
# git-ignored, as it holds benchmark fact text).
set -euo pipefail

cd "$(dirname "$0")/../../.."

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set. Load it first:" >&2
    echo "  read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY" >&2
    exit 1
fi

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

RUNS=runs/m3s4
RUN=${1:-$RUNS/claude-medium-v4-full}
RUN=${RUN%/}                         # tab completion adds a trailing slash
if [[ ${RUN##*/} != claude-medium-v*-full ]]; then
    echo "expected a full development run folder named claude-medium-v<N>-full: $RUN" >&2
    exit 1
fi
NAME=${RUN##*/claude-medium-}        # v4-full, v5-full
VERSION=${NAME%%-*}                  # v4, v5
CACHE=$RUN/fact_judgements.json
OUT=$RUNS/score_${NAME//-/_}_judged.txt
SEVERITY=docs/evidence/severity_agreement_$VERSION.md
SAMPLE=docs/review/correct_fact_sample.csv
[[ $VERSION == v4 ]] || SAMPLE=docs/review/correct_fact_sample_$VERSION.csv
RELEASED=data/omissionbench/judge_scores
DATA=(--pairs data/omissionbench/pairs/dataset_v2.json --spec $RUNS/full_spec.json
      --manifest $RUNS/cases-full/manifest.json)

if [[ ! -f $RUN/batch.json ]]; then
    echo "$RUN/batch.json is missing: make the run first." >&2
    exit 1
fi

# judge exits 2 when an item is left without a valid judgement (a refusal, say, or the cost
# cap): score what was judged anyway, then exit with judge's status.
status=0
python scripts/score_run.py judge --run "$RUN" "${DATA[@]}" \
    --config docs/runbooks/m3s4/judge.toml --cache "$CACHE" --released $RELEASED || status=$?
if [[ ! -f $CACHE ]]; then
    echo "judge wrote no cache (exit $status)." >&2
    exit "$status"
fi

{
    echo "######## ${RUN##*/}: score, with the judge's matches"
    python scripts/score_run.py score --run "$RUN" "${DATA[@]}" \
        --released $RELEASED --judgements "$CACHE"
    echo
    echo "######## ${RUN##*/}: misses, with the judge's matches"
    python scripts/score_run.py misses --run "$RUN" "${DATA[@]}" --judgements "$CACHE" \
        --show-facts
} 2>&1 | tee "$OUT"

python scripts/score_run.py severity --run "$RUN" "${DATA[@]}" --judgements "$CACHE" \
    --out "$SEVERITY" > /dev/null
if [[ -e $SAMPLE ]]; then
    echo "$SAMPLE is already there; left as it is."
else
    python scripts/score_run.py sample --run "$RUN" "${DATA[@]}" --judgements "$CACHE" \
        --out "$SAMPLE"
fi

echo "Wrote $OUT and $SEVERITY"
exit "$status"
