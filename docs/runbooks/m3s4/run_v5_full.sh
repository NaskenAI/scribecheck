#!/usr/bin/env bash
# M3 session 4: scribecheck_v5 at effort medium on all 65 development consultations (265
# errored and 65 clean notes), with a $25 cost cap, scored beside the v4 full run against the
# released B3 r1 and power-gepa-high r1/r3 records. v5's three changes are tagged, so the
# score's "v5 mechanisms" section shows what each adds and costs: the partial look
# (pass_a.partial_element), the audit's core kinds and never-dropped plan facts (core_kind,
# never_drop), and the second look's clear or probable absence. Paid API calls: run it
# yourself, never from an agent. Development data only. About 330 notes x $0.04, about 1.5
# hours of live calls. See docs/runbooks/m3s4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_v5_full.sh
#   docs/runbooks/m3s4/run_judge.sh runs/m3s4/claude-medium-v5-full    # then, about $1
#
# Writes runs/m3s4/claude-medium-v5-full/ and runs/m3s4/score_v5_full.txt (which holds
# benchmark fact text: keep it with the runs).
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
PAIRS=data/omissionbench/pairs/dataset_v2.json
SPEC=$RUNS/full_spec.json
CASES=$RUNS/cases-full
RUN=$RUNS/claude-medium-v5-full
OUT=$RUNS/score_v5_full.txt
DATA=(--pairs $PAIRS --spec $SPEC --manifest $CASES/manifest.json)

# Never overwrite a paid run's results.
if [[ -e $RUN/batch.json ]]; then
    echo "$RUN/batch.json already exists; move it aside first." >&2
    exit 1
fi

# The v4 full run's cases, prepared once; prepared again only if missing.
if [[ ! -f $CASES/manifest.json ]]; then
    python scripts/score_run.py spec --pairs $PAIRS --out $SPEC
    python scripts/score_run.py prepare --pairs $PAIRS --transcripts data/omissionbench/transcripts \
        --spec $SPEC --out $CASES
fi

# batch exits 2 when it reports a problem (an unreadable pair, say) but still writes its
# reports: score them anyway, then exit with batch's status.
status=0
scribecheck batch $CASES --live --config docs/runbooks/m3s4/claude-medium-full.toml \
    --format json --output $RUN/batch.json || status=$?
if [[ ! -f $RUN/batch.json ]]; then
    echo "batch wrote no reports (exit $status)." >&2
    exit "$status"
fi

{
    for run in claude-medium-v4-full claude-medium-v5-full; do
        echo "######## $run: score"
        python scripts/score_run.py score --run $RUNS/$run "${DATA[@]}" \
            --released data/omissionbench/judge_scores
        echo
        echo "######## $run: misses"
        python scripts/score_run.py misses --run $RUNS/$run "${DATA[@]}" --show-facts
        echo
    done
} 2>&1 | tee "$OUT"

echo "Wrote $OUT"
exit "$status"
