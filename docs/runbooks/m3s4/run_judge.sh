#!/usr/bin/env bash
# M3 session 4: model-assisted correct-fact matching on the full development run. One call
# per omission note (148) to Claude Sonnet 5.5 at effort low, fact texts only (never the
# transcript or note); about $1, capped at $3. Paid API calls: run it yourself, never from an
# agent. Development data only. See docs/evidence/fact_matching_v4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_judge.sh
#
# Writes runs/m3s4/claude-medium-v4-full/fact_judgements.json (the cache: running again makes
# no calls for notes already judged), runs/m3s4/score_v4_full_judged.txt,
# docs/evidence/severity_agreement_v4.md (words and model) and
# docs/review/correct_fact_sample.csv (30 pairs for a person to check; git-ignored, as it holds
# benchmark fact text).
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
RUN=$RUNS/claude-medium-v4-full
CACHE=$RUN/fact_judgements.json
OUT=$RUNS/score_v4_full_judged.txt
DATA=(--pairs data/omissionbench/pairs/dataset_v2.json --spec $RUNS/full_spec.json
      --manifest $RUNS/cases-full/manifest.json)

if [[ ! -f $RUN/batch.json ]]; then
    echo "$RUN/batch.json is missing: run run_v4_full.sh first." >&2
    exit 1
fi

# judge exits 2 when a note is left without a valid judgement (a refusal, say, or the cost
# cap): score what was judged anyway, then exit with judge's status.
status=0
python scripts/score_run.py judge --run $RUN "${DATA[@]}" \
    --config docs/runbooks/m3s4/judge.toml --cache $CACHE || status=$?
if [[ ! -f $CACHE ]]; then
    echo "judge wrote no cache (exit $status)." >&2
    exit "$status"
fi

{
    echo "######## claude-medium-v4-full: score, with the judge's matches"
    python scripts/score_run.py score --run $RUN "${DATA[@]}" \
        --released data/omissionbench/judge_scores --judgements $CACHE
    echo
    echo "######## claude-medium-v4-full: misses, with the judge's matches"
    python scripts/score_run.py misses --run $RUN "${DATA[@]}" --judgements $CACHE --show-facts
} 2>&1 | tee "$OUT"

python scripts/score_run.py severity --run $RUN "${DATA[@]}" --judgements $CACHE \
    --out docs/evidence/severity_agreement_v4.md > /dev/null
python scripts/score_run.py sample --run $RUN "${DATA[@]}" --judgements $CACHE \
    --out docs/review/correct_fact_sample.csv

echo "Wrote $OUT, docs/evidence/severity_agreement_v4.md and docs/review/correct_fact_sample.csv"
exit "$status"
