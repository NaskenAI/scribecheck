#!/usr/bin/env bash
# M4 stability: three runs of the frozen M4 candidate (scribecheck_v4, tag m4-candidate-v4)
# at effort medium on a fixed 100-note development subset, scored together for per-note
# agreement between the runs and the spread of detection, correct detection and false
# alarms. The subset (docs/runbooks/m4/stability_spec.json, note ids only) is a stratified
# sample with seed 20261110: 80 errored notes by kind, source stratum and severity, and 20
# clean notes by stratum, in proportion to the 330 development notes. Each run is capped at
# $5 (docs/runbooks/m4/stability.toml), so the three cannot pass $15; about $3.80 each is
# expected. Paid API calls: run it yourself, never from an agent. Development data only.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m4_stability.sh
#
# A run whose batch.json is already there is kept, never run again, so after an
# interruption the script picks up at the next run. A run stopped by its cost cap also keeps
# its batch.json: its unfinished notes are left out of every run's rates (the score lists
# them); move that run's folder aside to make it again. Writes runs/m4/cases-stability/,
# runs/m4/stability-r1/, -r2/, -r3/ and runs/m4/stability.txt (counts and note ids only).
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ANTHROPIC_API_KEY is not set. Load it first:" >&2
    echo "  read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY" >&2
    exit 1
fi

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# The frozen candidate only: refuse if the default prompt set is not scribecheck_v4.
default=$(python -c "from scribecheck import prompts; print(prompts.DEFAULT)")
if [[ $default != scribecheck_v4 ]]; then
    echo "the default prompt set is $default, not the frozen scribecheck_v4." >&2
    exit 1
fi

RUNS=runs/m4
PAIRS=data/omissionbench/pairs/dataset_v2.json
SPEC=docs/runbooks/m4/stability_spec.json
CONFIG=docs/runbooks/m4/stability.toml
CASES=$RUNS/cases-stability
OUT=$RUNS/stability.txt

python scripts/score_run.py prepare --pairs $PAIRS --transcripts data/omissionbench/transcripts \
    --spec $SPEC --out $CASES

# batch exits 2 when it reports a problem (an unreadable pair, say) but still writes its
# reports: carry on, and exit with the worst status at the end.
status=0
for r in 1 2 3; do
    RUN=$RUNS/stability-r$r
    if [[ -e $RUN/batch.json ]]; then
        echo "$RUN/batch.json is already there; kept."
        continue
    fi
    echo "######## run $r of 3"
    scribecheck batch $CASES --live --config $CONFIG --format json \
        --output $RUN/batch.json || status=$?
    if [[ ! -f $RUN/batch.json ]]; then
        echo "batch wrote no reports for run $r (exit $status)." >&2
        exit $((status == 0 ? 1 : status))
    fi
done

python scripts/score_run.py stability --runs $RUNS/stability-r1 $RUNS/stability-r2 \
    $RUNS/stability-r3 --pairs $PAIRS --spec $SPEC --manifest $CASES/manifest.json \
    2>&1 | tee "$OUT"

echo "Wrote $OUT"
exit "$status"
