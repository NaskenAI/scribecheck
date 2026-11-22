#!/usr/bin/env bash
# Gate 2 preparation: label the 148 development omission pairs' removed facts with the
# plan-action classifier (scripts/action_classify.py: Claude Sonnet 5.5 at effort low,
# removed-fact text only, capped at $1, about $0.50 expected), write a 20-item spot-check
# sheet, and score the v4 full development run on the plan-action subset (no further calls:
# its fact judgements are cached). Paid API calls: run it yourself, never from an agent.
# Development data only.
#
#   read -s ANTHROPIC_API_KEY
#   export ANTHROPIC_API_KEY
#   docs/runbooks/m3s4/run_action_classify.sh
#
# Writes runs/m3s4/action_classes.json (the cache: running again makes no calls for pairs
# already labelled), runs/m3s4/score_v4_full_actions.txt and, unless it is already there (a
# reviewer may be filling it in), docs/review/action_spot_check.csv (git-ignored: it holds
# benchmark fact text). Fill in its reviewer_label column.
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
CACHE=$RUNS/action_classes.json
RUN=$RUNS/claude-medium-v4-full
SAMPLE=docs/review/action_spot_check.csv
OUT=$RUNS/score_v4_full_actions.txt

# classify exits 2 when a pair is left without a label (a refusal, say, or the cost cap):
# carry on with what was labelled, then exit with its status.
status=0
python scripts/action_classify.py classify --pairs $PAIRS \
    --config docs/runbooks/action_classify.toml --cache $CACHE || status=$?
if [[ ! -f $CACHE ]]; then
    echo "classify wrote no cache (exit $status)." >&2
    exit "$status"
fi

if [[ -e $SAMPLE ]]; then
    echo "$SAMPLE is already there; left as it is."
else
    python scripts/action_classify.py sample --pairs $PAIRS --cache $CACHE --out $SAMPLE
fi

python scripts/score_run.py score --run $RUN --pairs $PAIRS --spec $RUNS/full_spec.json \
    --manifest $RUNS/cases-full/manifest.json --released data/omissionbench/judge_scores \
    --judgements $RUN/fact_judgements.json --actions $CACHE > "$OUT"
sed -n '/^== Plan-action subset/,/^== v5 mechanisms/p' "$OUT" | sed '$d'
echo "Wrote $OUT"
exit "$status"
