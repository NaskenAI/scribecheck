#!/usr/bin/env bash
# Gate 2, Claude arm: the frozen system on the held-out data, as
# docs/evidence/gate2_preregistration.md fixes it. Run by a person in a normal terminal,
# never by Claude Code (it refuses to run there). Paid API calls: three primary runs (cap $12
# each, about $7.40 each expected), one supplementary run (cap $12, about $4.70 expected),
# the plan-action classifier (cap $1) and the fact judge (cap $5).
#
#   export SCRIBECHECK_HELDOUT_DIR=<the held-out folder, as an absolute path>
#   export SCRIBECHECK_RAW_DIR=<the raw-download folder, as an absolute path>
#   read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
#   docs/runbooks/gate2_claude.sh
#
# The two folders must be the ones the archives were made from, under the home folder:
# each archive is the folder's path with .tar.gz.gpg added, made with tar czf from the home
# folder, so its members start with the folder's path relative to home. Before decrypting,
# the script checks:
# - HEAD is the pre-registration commit, or a descendant changing only docs/ outside
#   docs/runbooks/; nothing is uncommitted; src/ is as tagged m4-candidate-v4; the default
#   prompt set is scribecheck_v4 (docs/runbooks/gate2_guard.sh);
# - the classifier prompt's hash and both archives' SHA-256 are the pre-registered ones.
#
# Then, with umask 077: decrypts the held-out archive into its folder, and only the
# w2-pipeline and w2-power records from the raw archive into its folder; builds the specs
# and cases; makes runs r1-r3 on the primary set (151 errored and 47 clean notes) and one
# supplementary run (79 errored and the 47 clean notes), each checked as it finishes; labels
# the primary omission pairs with the plan-action classifier; judges the flagged facts (ours
# and route one's released ones); writes docs/evidence/gate2_claude.txt (counts only) and
# the 30-pair spot-check; seals the reports, caches, logs and spot-check into
# gate2_outputs.tar.gpg in the held-out folder (you choose its passphrase); and deletes the
# plaintext. On any exit the decrypted inputs and cases are deleted and gpg-agent is stopped.
#
# A run is discarded when it reached its cost cap or failed on more than 2% of its notes:
# the script moves it to gate2/discarded/ and stops. Run the script again to repeat that run
# in full; finished runs are kept (their reports stay, in plaintext, in the held-out
# folder's gate2/ until the script completes and seals them, and the script says so when it
# stops early). Model and batch output goes to gate2/gate2.log in the held-out folder, never
# to the terminal: nothing printed here holds transcript, note or fact text. A score whose
# recomputed baselines do not match the published figures says so at its top.
set -euo pipefail

cd "$(dirname "$0")/../.."
# shellcheck source=docs/runbooks/gate2_guard.sh
source docs/runbooks/gate2_guard.sh

gate2_refuse_agent

for var in ANTHROPIC_API_KEY SCRIBECHECK_HELDOUT_DIR SCRIBECHECK_RAW_DIR; do
    if [[ -z ${!var:-} ]]; then
        echo "$var is not set (see the usage at the top of this script)." >&2
        exit 1
    fi
done
HELD=${SCRIBECHECK_HELDOUT_DIR%/}
RAW=${SCRIBECHECK_RAW_DIR%/}
for dir in "$HELD" "$RAW"; do
    if [[ $dir != "$HOME"/* ]]; then
        echo "refused: $dir is not under $HOME (the archives were made from the home folder)." >&2
        exit 1
    fi
done
HELD_ARCHIVE=$HELD.tar.gz.gpg
RAW_ARCHIVE=$RAW.tar.gz.gpg
HELD_MEMBER=${HELD#"$HOME"/}
RAW_MEMBER=${RAW#"$HOME"/}
RAW_FAMILIES=("$RAW_MEMBER/judgements/judges/w2-pipeline" "$RAW_MEMBER/judgements/judges/w2-power")
HELD_FILES=(LICENSE ATTRIBUTION.md official_151_pair_ids.json pairs transcripts)

GATE2_PREREG_FILE=docs/evidence/gate2_preregistration.md
gate2_check_git "$GATE2_PREREG_FILE" m4-candidate-v4
gate2_check_tag m4-candidate-v4

if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

default=$(python -c "from scribecheck import prompts; print(prompts.DEFAULT)")
if [[ $default != scribecheck_v4 ]]; then
    echo "refused: the default prompt set is $default, not the frozen scribecheck_v4." >&2
    exit 1
fi
want=$(gate2_frozen action-classifier-prompt-sha256)
got=$(python scripts/action_classify.py hash)
if [[ $got != "$want" ]]; then
    echo "refused: the classifier prompt's hash is $got, not the pre-registered $want." >&2
    exit 1
fi
want=$(gate2_frozen development-dataset-sha256)
if [[ $(gate2_sha256 data/omissionbench/pairs/dataset_v2.json) != "$want" ]]; then
    echo "refused: the development pairs file is not the pre-registered one." >&2
    exit 1
fi
for archive in "$HELD_ARCHIVE" "$RAW_ARCHIVE"; do
    if [[ ! -f $archive ]]; then
        echo "refused: $archive is missing." >&2
        exit 1
    fi
done
gate2_check_digest "$HELD_ARCHIVE" held-out-archive-sha256
gate2_check_digest "$RAW_ARCHIVE" raw-archive-sha256

WORK=$HELD/gate2
SEALED=$HELD/gate2_outputs.tar.gpg
if [[ -e $SEALED ]]; then
    echo "refused: $SEALED already exists: Gate 2 has been run." >&2
    exit 1
fi
# Decrypted inputs left from an interrupted session would be deleted on exit with the new
# ones; refuse rather than delete something this run did not make.
for f in "${HELD_FILES[@]}"; do
    if [[ -e $HELD/$f ]]; then
        echo "refused: $HELD/$f already exists (plaintext from an earlier session?): remove it." >&2
        exit 1
    fi
done
for f in "${RAW_FAMILIES[@]}"; do
    if [[ -e $HOME/$f ]]; then
        echo "refused: $HOME/$f already exists (plaintext from an earlier session?): remove it." >&2
        exit 1
    fi
done

umask 077
# gpg's passphrase prompt needs the terminal (pinentry-curses and -tty read it from here).
GPG_TTY=$(tty) || true
export GPG_TTY
DONE=0
cleanup() {
    local f
    for f in "${HELD_FILES[@]}"; do
        rm -rf "${HELD:?}/$f"
    done
    for f in "${RAW_FAMILIES[@]}"; do
        rm -rf "${HOME:?}/$f"
    done
    rm -rf "${WORK:?}/cases-primary" "${WORK:?}/cases-supplementary" "${SEALED:?}.partial"
    gpgconf --kill gpg-agent || true
    if [[ $DONE -ne 1 && -d $WORK ]]; then
        echo "Not finished: the decrypted inputs are deleted, but the finished runs' reports" \
            "(which quote held-out text) stay unencrypted in $WORK until this script completes" \
            "and seals them. Run it again to carry on." >&2
    fi
}
trap cleanup EXIT
mkdir -p "$HELD" "$RAW" "$WORK"
LOG=$WORK/gate2.log
echo "######## $(date -u +%FT%TZ) gate2_claude.sh at $(git rev-parse HEAD)" >> "$LOG"

# gpg's own messages (the passphrase prompt, the cipher) hold no benchmark text.
echo "Decrypting the held-out archive (gpg asks for its passphrase)."
gpg --decrypt "$HELD_ARCHIVE" | tar -xzf - -C "$HOME" "${HELD_FILES[@]/#/$HELD_MEMBER/}"
echo "Decrypting the raw archive's w2-pipeline and w2-power records."
gpg --decrypt "$RAW_ARCHIVE" | tar -xzf - -C "$HOME" "${RAW_FAMILIES[@]}"

PAIRS=$HELD/pairs/dataset_v2.json
PRIMARY_SPEC=$WORK/primary_spec.json
SUPPLEMENTARY_SPEC=$WORK/supplementary_spec.json
PRIMARY_CASES=$WORK/cases-primary
SUPPLEMENTARY_CASES=$WORK/cases-supplementary
CONFIG=docs/runbooks/gate2/claude.toml

python scripts/gate2.py specs --pairs "$PAIRS" --ids "$HELD/official_151_pair_ids.json" \
    --out "$WORK"
python scripts/score_run.py prepare --pairs "$PAIRS" --transcripts "$HELD/transcripts" \
    --spec "$PRIMARY_SPEC" --out "$PRIMARY_CASES" >> "$LOG"
python scripts/score_run.py prepare --pairs "$PAIRS" --transcripts "$HELD/transcripts" \
    --spec "$SUPPLEMENTARY_SPEC" --out "$SUPPLEMENTARY_CASES" >> "$LOG"
echo "Prepared the primary and supplementary cases."

# run_one <name> <cases> <spec>: makes the run unless its batch.json is already there, then
# checks it; a discarded run is moved aside and the script stops.
run_one() {
    local name=$1 cases=$2 spec=$3 run=$WORK/$1 status=0
    if [[ -e $run/batch.json ]]; then
        echo "$name: already made; kept."
    else
        echo "######## $name"
        scribecheck batch "$cases" --live --config "$CONFIG" --format json \
            --output "$run/batch.json" >> "$LOG" 2>&1 || status=$?
        if [[ ! -f $run/batch.json ]]; then
            echo "$name: batch wrote no reports (exit $status); see the log in the held-out folder." >&2
            exit 1
        fi
    fi
    status=0
    python scripts/gate2.py check-run --run "$run" --pairs "$PAIRS" --spec "$spec" \
        --manifest "$cases/manifest.json" || status=$?
    if [[ $status -eq 3 ]]; then
        mkdir -p "$WORK/discarded"
        mv "$run" "$WORK/discarded/$name-$(date -u +%Y%m%dT%H%M%SZ)"
        echo "$name was discarded and moved to gate2/discarded/: run this script again to" \
            "repeat it in full, and record the discard in the pre-registration's Deviations." >&2
        exit 1
    elif [[ $status -ne 0 ]]; then
        exit "$status"
    fi
}

for r in 1 2 3; do
    run_one "r$r" "$PRIMARY_CASES" "$PRIMARY_SPEC"
done
run_one s1 "$SUPPLEMENTARY_CASES" "$SUPPLEMENTARY_SPEC"

RUNS=("$WORK/r1" "$WORK/r2" "$WORK/r3")
PRIMARY=(--pairs "$PAIRS" --spec "$PRIMARY_SPEC" --manifest "$PRIMARY_CASES/manifest.json")

# A pair the classifier leaves without a label, or a note the judge leaves without a
# judgement, is reported as such by score (unlabelled; counted as not named) rather than
# stopping Gate 2; running the script again retries them, from the caches.
echo "######## plan-action classifier"
python scripts/action_classify.py classify --pairs "$PAIRS" \
    --config docs/runbooks/action_classify.toml --cache "$WORK/action_classes.json" \
    --ids "$HELD/official_151_pair_ids.json" 2>> "$LOG" \
    || echo "WARNING: some pairs have no label (see the score's count)." >&2

echo "######## fact judge"
python scripts/gate2.py judge --runs "${RUNS[@]}" "${PRIMARY[@]}" --raw "$RAW" \
    --config docs/runbooks/gate2/judge.toml --cache "$WORK/fact_judgements.json" 2>> "$LOG" \
    || echo "WARNING: some items have no judgement (see the score's count)." >&2

echo "######## score"
python scripts/gate2.py score --runs "${RUNS[@]}" --supplementary-run "$WORK/s1" \
    --pairs "$PAIRS" --primary-spec "$PRIMARY_SPEC" --supplementary-spec "$SUPPLEMENTARY_SPEC" \
    --primary-manifest "$PRIMARY_CASES/manifest.json" \
    --supplementary-manifest "$SUPPLEMENTARY_CASES/manifest.json" --raw "$RAW" \
    --judgements "$WORK/fact_judgements.json" --actions "$WORK/action_classes.json" \
    --commit "$(git rev-parse HEAD)" --prereg "$GATE2_PREREG" \
    --out docs/evidence/gate2_claude.txt
python scripts/gate2.py sample --runs "${RUNS[@]}" "${PRIMARY[@]}" \
    --judgements "$WORK/fact_judgements.json" --out "$WORK/spot_check.csv"

echo "######## sealing the outputs (gpg asks for a new passphrase for gate2_outputs.tar.gpg)"
rm -rf "${WORK:?}/cases-primary" "${WORK:?}/cases-supplementary"
# Written under a temporary name and moved into place only once it decrypts and lists, so an
# interrupted seal never looks like a finished Gate 2.
rm -f "$SEALED.partial"
tar -czf - -C "$HELD" gate2 | gpg --symmetric --cipher-algo AES256 --output "$SEALED.partial"
gpg --decrypt "$SEALED.partial" 2> /dev/null | tar -tzf - > /dev/null
mv "$SEALED.partial" "$SEALED"
rm -rf "${WORK:?}"
DONE=1
echo "Sealed the outputs in $SEALED and deleted the plaintext."
echo "Wrote docs/evidence/gate2_claude.txt (counts only)."
