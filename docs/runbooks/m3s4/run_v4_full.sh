#!/usr/bin/env bash
# M3 session 4: scribecheck_v4 at effort medium on all 65 development consultations (265
# errored and 65 clean notes), with a $25 cost cap, scored against the released B3 r1 and
# power-gepa-high r1/r3 records: detection and false alarms for all of them, correct
# detection for scribecheck (the released records hold no fact text). Paid API calls: run it
# yourself, never from an agent. Development data only. About 330 notes x $0.04 and about
# 1.5 hours of live calls. See docs/runbooks/m3s4.md.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_v4_full.sh
#
# Writes runs/m3s4/full_spec.json, runs/m3s4/cases-full/, runs/m3s4/claude-medium-v4-full/
# and runs/m3s4/score_v4_full.txt (which holds benchmark fact text: keep it with the runs).
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
RUN=$RUNS/claude-medium-v4-full
OUT=$RUNS/score_v4_full.txt

# Never overwrite a paid run's results.
if [[ -e $RUN/batch.json ]]; then
    echo "$RUN/batch.json already exists; move it aside first." >&2
    exit 1
fi

# Every development note, prepared once (preparing again rewrites the same files).
python scripts/score_run.py spec --pairs $PAIRS --out $SPEC
python scripts/score_run.py prepare --pairs $PAIRS --transcripts data/omissionbench/transcripts \
    --spec $SPEC --out $CASES

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
    echo "######## claude-medium-v4-full: score"
    python scripts/score_run.py score --run $RUN --pairs $PAIRS --spec $SPEC \
        --manifest $CASES/manifest.json --released data/omissionbench/judge_scores
    echo
    echo "######## claude-medium-v4-full: misses"
    python scripts/score_run.py misses --run $RUN --pairs $PAIRS --spec $SPEC \
        --manifest $CASES/manifest.json --show-facts
} 2>&1 | tee "$OUT"

echo "Wrote $OUT"
exit "$status"
