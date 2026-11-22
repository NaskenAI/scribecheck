# Checks for the Gate 2 runbook (docs/runbooks/gate2_claude.sh), sourced by it, and by
# tests/test_gate2_guard.py in a throwaway repository. Every function prints why it refuses
# on stderr and returns non-zero; none reads benchmark data.

# Never from an agent: Claude Code sets CLAUDECODE in the shells it starts.
gate2_refuse_agent() {
    if [[ -n ${CLAUDECODE:-} ]]; then
        echo "refused: this runbook is run by a person in a normal terminal, never by Claude Code." >&2
        return 1
    fi
}

# gate2_check_git <pre-registration file> <tag>
# Refuses unless HEAD is the pre-registration commit or a descendant that changes only
# documentation, nothing is uncommitted, no untracked or ignored file sits among the code
# (src/, scripts/, docs/runbooks/; Python caches aside), and the package is the tagged one
# (gate2_check_tag then pins the tag to the pre-registered commit). The
# pre-registration commit is the one that added the file; it must change nothing else.
# Documentation means docs/, except docs/runbooks/ (which holds this runbook and its
# configs, frozen with the code). Sets GATE2_PREREG to the commit.
gate2_check_git() {
    local file=$1 tag=$2 prereg changed path
    if ! git diff --quiet HEAD -- || ! git diff --cached --quiet; then
        echo "refused: there are uncommitted changes to tracked files." >&2
        return 1
    fi
    # An untracked module would be imported as if it were part of the frozen code.
    path=$(git ls-files --others -- src scripts docs/runbooks | grep -v -e '/__pycache__/' \
        -e '\.egg-info/' | head -n 1 || true)
    if [[ -n $path ]]; then
        echo "refused: $path is not committed (untracked or ignored files may not sit in" \
            "src/, scripts/ or docs/runbooks/)." >&2
        return 1
    fi
    prereg=$(git log --diff-filter=A --format=%H -- "$file" | tail -n 1)
    if [[ -z $prereg ]]; then
        echo "refused: no commit adds $file: commit the pre-registration first." >&2
        return 1
    fi
    if ! git merge-base --is-ancestor "$prereg" HEAD; then
        echo "refused: HEAD does not descend from the pre-registration commit $prereg." >&2
        return 1
    fi
    changed=$(git diff-tree --no-commit-id --name-only -r --root "$prereg")
    if [[ $changed != "$file" ]]; then
        echo "refused: the pre-registration commit $prereg changes more than $file." >&2
        return 1
    fi
    while IFS= read -r path; do
        [[ -z $path ]] && continue
        if [[ $path != docs/* || $path == docs/runbooks/* ]]; then
            echo "refused: $path changed after the pre-registration commit (only docs/," \
                "outside docs/runbooks/, may change)." >&2
            return 1
        fi
    done < <(git diff --name-only "$prereg" HEAD)
    if ! git rev-parse --verify --quiet "$tag^{commit}" > /dev/null; then
        echo "refused: no tag $tag." >&2
        return 1
    fi
    if ! git diff --quiet "$tag" HEAD -- src; then
        echo "refused: src/ differs from $tag." >&2
        return 1
    fi
    GATE2_PREREG=$prereg
}

# gate2_frozen <name>: a value from the pre-registration as committed (never a later edit),
# from its "<name>: <value>" line, where the value is a 40-character commit hash or a
# 64-character SHA-256, in lower-case hex.
gate2_frozen() {
    local name=$1 value
    value=$(git show "$GATE2_PREREG:$GATE2_PREREG_FILE" | tr -d '\r' \
        | sed -n "s/^$name: \([0-9a-f]\{40\}\([0-9a-f]\{24\}\)\{0,1\}\)\$/\1/p")
    if [[ $(printf '%s\n' "$value" | grep -c .) -ne 1 ]]; then
        echo "refused: the pre-registration has no single '$name: <hex>' line." >&2
        return 1
    fi
    printf '%s\n' "$value"
}

# gate2_check_tag <tag>: the tag must point at the pre-registered commit (a moved tag
# refuses).
gate2_check_tag() {
    local want got
    want=$(gate2_frozen "$1-commit") || return 1
    got=$(git rev-parse --verify --quiet "$1^{commit}") || true
    if [[ $got != "$want" ]]; then
        echo "refused: tag $1 points at ${got:-nothing}, not the pre-registered $want." >&2
        return 1
    fi
}

# gate2_sha256 <file>
gate2_sha256() {
    if command -v sha256sum > /dev/null; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

# gate2_check_digest <file> <name>: the file's SHA-256 must be the frozen value.
gate2_check_digest() {
    local want got
    want=$(gate2_frozen "$2") || return 1
    got=$(gate2_sha256 "$1")
    if [[ $got != "$want" ]]; then
        echo "refused: $2 is $got, not the pre-registered $want." >&2
        return 1
    fi
}
