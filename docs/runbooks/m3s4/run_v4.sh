#!/usr/bin/env bash
# M3 session 4: Claude Sonnet 5.5 at effort medium with the scribecheck_v4 prompts (audit:
# always-core kinds; second look: rescues a more specific restatement, never a vaguer one) on
# the 51-note subset, scored beside claude-medium-v3 with the stricter matcher, plus both
# runs' misses. Paid API calls: run it yourself, never from an agent. Development data only.
# See docs/runbooks/m3s4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_v4.sh
#
# Writes runs/m3s4/claude-medium-v4/ and runs/m3s4/compare_v4.txt (which holds benchmark
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
OUT=$RUNS/compare_v4.txt
DATA=(--pairs data/omissionbench/pairs/dataset_v2.json --spec runs/gate1b/subset_spec.json
      --manifest runs/m3s4/cases/manifest.json)
REFS=(--released data/omissionbench/judge_scores --gate1b runs/gate1b/harness-results)

# The v3 run to compare against must be there; never overwrite a paid run's results.
if [[ ! -f $RUNS/claude-medium-v3/batch.json ]]; then
    echo "$RUNS/claude-medium-v3/batch.json is missing: it is the run v4 is compared with." >&2
    exit 1
fi
if [[ -e $RUNS/claude-medium-v4/batch.json ]]; then
    echo "$RUNS/claude-medium-v4/batch.json already exists; move it aside first." >&2
    exit 1
fi

# batch exits 2 when it reports a problem (an unreadable pair, say) but still writes its
# reports: score them anyway, then exit with batch's status.
status=0
scribecheck batch runs/m3s4/cases --live --config docs/runbooks/m3s4/claude-medium.toml \
    --format json --output $RUNS/claude-medium-v4/batch.json || status=$?
if [[ ! -f $RUNS/claude-medium-v4/batch.json ]]; then
    echo "batch wrote no reports (exit $status)." >&2
    exit "$status"
fi

{
    for run in claude-medium-v3 claude-medium-v4; do
        echo "######## $run: score"
        python scripts/score_run.py score --run $RUNS/$run "${DATA[@]}" "${REFS[@]}"
        echo
        echo "######## $run: misses"
        python scripts/score_run.py misses --run $RUNS/$run "${DATA[@]}" --show-facts
        echo
    done
} 2>&1 | tee "$OUT"

echo "Wrote $OUT"
exit "$status"
