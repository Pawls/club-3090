#!/usr/bin/env bash
# The WSL2 allocator override must land in the directory docker compose actually
# reads — the compose file's OWN directory (`<engine>/compose/<topology>/<quant>/`),
# not the engine-level compose root. A file at the root is silently never loaded, so
# the container boots with the very default the file exists to override.
#
# Also asserts the value is derived from each directory's composes (other alloc-conf
# knobs preserved), that existing user .env files are never rewritten, and that no
# LIVE vLLM compose hardcodes PYTORCH_CUDA_ALLOC_CONF — a hardcoded value makes the
# override unreachable no matter where the .env sits.
set -uo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/wsl2-env.sh
source "${ROOT_DIR}/scripts/lib/wsl2-env.sh"

fail=0
note() { echo "[wsl2-env] FAIL: $*" >&2; fail=1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

compose_with() {   # <path> <alloc-conf entry>
  mkdir -p "$(dirname "$1")"
  cat > "$1" <<EOF
services:
  vllm:
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=$2
EOF
}

CR="$TMP/models/m/vllm/compose"

# A: knob-preserving flip — the dir's own max_split_size_mb survives.
compose_with "$CR/dual/q1/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}'
# B: bare default → bare flip (no knob invented).
compose_with "$CR/single/q2/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
# C: no expandable_segments at all (lmcache-style) → no .env written.
compose_with "$CR/dual/q3/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6}'
# D: siblings disagree → bare fix, not one sibling's knobs imposed on the other.
compose_with "$CR/dual/q4/a.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
compose_with "$CR/dual/q4/b.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}'
# E: retired composes are not launchable → skipped.
compose_with "$CR/_archive/dual/q5/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
# F: user file with an active pin → untouched.
compose_with "$CR/dual/q6/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
printf 'GPU_MEMORY_UTILIZATION=0.86\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:False\n' > "$CR/dual/q6/.env"
# G: user file with no mention of the key → warned about, never edited.
compose_with "$CR/dual/q7/base.yml" '${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}'
printf 'GPU_MEMORY_UTILIZATION=0.90\n' > "$CR/dual/q7/.env"

# The dead parent file, byte-identical to what setup.sh used to write.
_wsl2_legacy_parent_env > "$CR/.env"

out="$(wsl2_sync_alloc_conf "$CR" 2>&1)"

want_value() {   # <dir> <expected value>
  local f="$CR/$1/.env" got
  if [[ ! -f "$f" ]]; then note "$1/.env not written"; return; fi
  got="$(grep -E '^PYTORCH_CUDA_ALLOC_CONF=' "$f" | tail -1)"
  [[ "$got" == "PYTORCH_CUDA_ALLOC_CONF=$2" ]] \
    || note "$1/.env has '$got', want 'PYTORCH_CUDA_ALLOC_CONF=$2'"
}

want_value dual/q1 'expandable_segments:False,max_split_size_mb:512'
want_value single/q2 'expandable_segments:False'
want_value dual/q4 'expandable_segments:False'

[[ -f "$CR/dual/q3/.env" ]] && note "wrote .env for a dir whose composes don't use expandable_segments"
[[ -f "$CR/_archive/dual/q5/.env" ]] && note "wrote .env under _archive/ (retired composes aren't launchable)"

# User files are theirs. Byte-compare, not just a key check.
[[ "$(cat "$CR/dual/q6/.env")" == "$(printf 'GPU_MEMORY_UTILIZATION=0.86\nPYTORCH_CUDA_ALLOC_CONF=expandable_segments:False\n')" ]] \
  || note "rewrote a user .env that already pinned the key"
[[ "$(cat "$CR/dual/q7/.env")" == "$(printf 'GPU_MEMORY_UTILIZATION=0.90\n')" ]] \
  || note "edited a user .env instead of warning"
grep -q "q7/.env exists without a PYTORCH_CUDA_ALLOC_CONF override" <<<"$out" \
  || note "no warning for a user .env missing the override"

# The dead parent file is retired, and its content isn't resurrected elsewhere.
[[ -f "$CR/.env" ]] && note "left the dead compose-root .env in place"

# A hand-edited parent file is reported, not deleted.
printf 'MODEL_DIR=/somewhere\n' > "$CR/.env"
out2="$(wsl2_sync_alloc_conf "$CR" 2>&1)"
[[ -f "$CR/.env" ]] || note "deleted a hand-edited compose-root .env"
grep -q "NEVER read by docker compose" <<<"$out2" || note "no warning for a hand-edited compose-root .env"

# Idempotent: a second run must not duplicate assignments.
[[ "$(grep -c '^PYTORCH_CUDA_ALLOC_CONF=' "$CR/dual/q1/.env")" == "1" ]] \
  || note "second run duplicated the assignment in dual/q1/.env"

# ---- repo-wide: no live vLLM compose may hardcode the value ----
while IFS= read -r hit; do
  note "hardcoded PYTORCH_CUDA_ALLOC_CONF (an .env cannot override it): ${hit%%:*}"
done < <(grep -rn --include='*.yml' -F 'PYTORCH_CUDA_ALLOC_CONF=' "$ROOT_DIR"/models/*/vllm*/compose 2>/dev/null \
  | grep -v '/_archive/' \
  | grep -E '^[^:]+:[0-9]+:[[:space:]]*-[[:space:]]*PYTORCH_CUDA_ALLOC_CONF=' \
  | grep -vF 'PYTORCH_CUDA_ALLOC_CONF=${')

# ---- repo-wide: no .env may sit at the never-read compose-root level ----
while IFS= read -r dead; do
  note "dead .env at compose root (never loaded by compose): ${dead#"$ROOT_DIR"/}"
done < <(find "$ROOT_DIR"/models/*/*/compose -maxdepth 1 -name .env 2>/dev/null)

if [[ "$fail" == "0" ]]; then
  echo "[wsl2-env] PASS"
else
  echo "[wsl2-env] FAILED" >&2
fi
exit "$fail"
