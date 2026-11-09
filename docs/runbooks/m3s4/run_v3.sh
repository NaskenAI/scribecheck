#!/usr/bin/env bash
# M3 session 4: Claude Sonnet 5.5 at effort medium with the scribecheck_v3 prompts (atomic
# facts) and the no-allergy candidate, then score it beside claude-medium-v2 and trace both
# runs' misses. Paid API calls: run it yourself, never from an agent. Development data
# only. See docs/runbooks/m3s4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_v3.sh
#
# Writes runs/m3s4/claude-medium-v3/ and runs/m3s4/compare_v3.txt (which holds benchmark
# fact text: keep it with the runs).
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
OUT=$RUNS/compare_v3.txt
DATA=(--pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json
      --manifest runs/m3s4/cases/manifest.json)
REFS=(--released data/omissionbench/judge_scores --gate1b runs/gate1b/harness-results)

# The v2 run to compare against must be there; never overwrite a paid run's results.
if [[ ! -f $RUNS/claude-medium-v2/batch.json ]]; then
    echo "$RUNS/claude-medium-v2/batch.json is missing: it is the run v3 is compared with." >&2
    exit 1
fi
if [[ -e $RUNS/claude-medium-v3/batch.json ]]; then
    echo "$RUNS/claude-medium-v3/batch.json already exists; move it aside first." >&2
    exit 1
fi

scribecheck batch runs/m3s4/cases --live --config docs/runbooks/m3s4/claude-medium.toml \
    --format json --output $RUNS/claude-medium-v3/batch.json

{
    for run in claude-medium-v2 claude-medium-v3; do
        echo "######## $run: score"
        python scripts/score_run.py score --run $RUNS/$run "${DATA[@]}" "${REFS[@]}"
        echo
        echo "######## $run: misses"
        python scripts/score_run.py misses --run $RUNS/$run "${DATA[@]}" --show-facts
        echo
    done
} 2>&1 | tee "$OUT"

echo "Wrote $OUT"
