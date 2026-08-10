#!/usr/bin/env bash
#
# WSL2 allocator override — placed where docker compose will actually read it.
#
# WSL2 + recent drivers hit a `gptq_marlin_repack` boot crash (cudaErrorNotReady)
# with the vLLM composes' default `expandable_segments:True` (PR #84 / issue #60).
# The fix is a `.env` pinning `expandable_segments:False`.
#
# The placement is the whole point. Docker compose auto-loads `.env` ONLY from the
# directory it is invoked in, and switch.sh / launch.sh `cd` into the compose file's
# own directory (`<engine>/compose/<topology>/<quant>/`) before `docker compose up`.
# setup.sh used to write a single `.env` at `<engine>/compose/` — one level too high,
# so it was silently never read: no error, no warning, and the container booted with
# the very default the file existed to override. This writes one `.env` per quant
# directory instead, which is the level compose reads.
#
# The value is derived per directory from the composes that live there, not
# hardcoded: we take their own `${PYTORCH_CUDA_ALLOC_CONF:-<default>}` and flip only
# the `expandable_segments` token, preserving every other knob the compose author set
# (`max_split_size_mb`, `garbage_collection_threshold`, ...). Same principle as
# detect_nvlink.sh's alloc-conf strip. A directory whose composes don't opt into
# expandable_segments at all (e.g. vllm-lmcache, which omits it deliberately) is
# skipped — there is nothing to override there.

# _wsl2_alloc_defaults <dir>
#   The distinct PYTORCH_CUDA_ALLOC_CONF defaults declared by the composes in <dir>,
#   one per line. Reads `- PYTORCH_CUDA_ALLOC_CONF=<v>` environment entries and
#   unwraps `${PYTORCH_CUDA_ALLOC_CONF:-<v>}` to <v>. Comment lines never match:
#   the pattern is anchored to the YAML list dash.
_wsl2_alloc_defaults() {
  local dir="$1"
  grep -hoE '^[[:space:]]*-[[:space:]]*PYTORCH_CUDA_ALLOC_CONF=[^[:space:]]*' "$dir"/*.yml 2>/dev/null \
    | sed -E 's/^[[:space:]]*-[[:space:]]*PYTORCH_CUDA_ALLOC_CONF=//; s/^\$\{PYTORCH_CUDA_ALLOC_CONF:-//; s/\}$//' \
    | grep -v '^$' \
    | sort -u
}

# _wsl2_override_value <dir>
#   The value to pin in <dir>/.env, or empty when the directory needs no override.
_wsl2_override_value() {
  local dir="$1" defaults count
  defaults="$(_wsl2_alloc_defaults "$dir")"
  printf '%s' "$defaults" | grep -q 'expandable_segments:True' || return 0
  count="$(printf '%s\n' "$defaults" | grep -c .)"
  if [[ "$count" == "1" ]]; then
    # Flip only the expandable_segments token; keep the compose's other knobs.
    printf '%s' "${defaults/expandable_segments:True/expandable_segments:False}"
  else
    # Composes in one directory disagree on the default — pin the bare fix rather
    # than silently imposing one compose's extra knobs on its siblings.
    printf 'expandable_segments:False'
  fi
}

# _wsl2_env_state <env_file>
#   active   — an uncommented PYTORCH_CUDA_ALLOC_CONF assignment (user is covered,
#              whatever value they chose)
#   disabled — the key appears only commented out (a deliberate opt-out; don't nag)
#   absent   — no mention at all
_wsl2_env_state() {
  local env_file="$1"
  if grep -qE '^[[:space:]]*(export[[:space:]]+)?PYTORCH_CUDA_ALLOC_CONF[[:space:]]*=' "$env_file" 2>/dev/null; then
    printf 'active'
  elif grep -q 'PYTORCH_CUDA_ALLOC_CONF' "$env_file" 2>/dev/null; then
    printf 'disabled'
  else
    printf 'absent'
  fi
}

# The exact blocks setup.sh has historically auto-written to `<engine>/compose/.env`.
# Used to decide whether a stale parent file is ours to remove or the user's to keep.
_wsl2_legacy_parent_env() {
  cat <<'EOF'
# WSL2 boot-crash workaround — see PR #84 + issue #60.
# vLLM + WSL2 + driver 596.36 hit `gptq_marlin_repack` cudaErrorNotReady on boot
# with the default `expandable_segments:True`. This override fixes it.
# Auto-created by scripts/setup.sh on detected WSL2 systems. Safe to delete
# on bare-metal Linux (the compose default works there).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
EOF
}

# wsl2_sync_alloc_conf <compose_root>
#   Write the override into every quant directory under <compose_root> that ships a
#   compose using expandable_segments, and retire the dead `<compose_root>/.env`.
#   Skips `_archive/` (retired composes are not launchable). Never overwrites a
#   `.env` that already exists — those are user files.
wsl2_sync_alloc_conf() {
  local compose_root="$1"
  [[ -d "$compose_root" ]] || return 0

  local dir env_file value state wrote=0
  while IFS= read -r dir; do
    value="$(_wsl2_override_value "$dir")"
    [[ -n "$value" ]] || continue
    env_file="${dir}/.env"

    if [[ ! -f "$env_file" ]]; then
      cat > "$env_file" <<EOF
# WSL2 boot-crash workaround — see PR #84 + issue #60.
# vLLM + WSL2 + recent drivers hit \`gptq_marlin_repack\` cudaErrorNotReady on boot
# with the default \`expandable_segments:True\`. This override fixes it.
#
# Must live in THIS directory: docker compose auto-loads .env only from the dir it
# is invoked in, and switch.sh / launch.sh cd here before \`docker compose up\`.
# A copy one level up (compose/.env) is silently ignored.
#
# Auto-created by scripts/setup.sh on detected WSL2 systems. Safe to delete on
# bare-metal Linux (the compose default works there).
PYTORCH_CUDA_ALLOC_CONF=${value}
EOF
      echo "[wsl2] wrote ${env_file#"${ROOT_DIR:-}/"} (PYTORCH_CUDA_ALLOC_CONF=${value})"
      wrote=$((wrote + 1))
      continue
    fi

    state="$(_wsl2_env_state "$env_file")"
    case "$state" in
      active)
        echo "[wsl2] ${env_file#"${ROOT_DIR:-}/"} already pins PYTORCH_CUDA_ALLOC_CONF — left as-is. ✓" ;;
      disabled)
        echo "[wsl2] ${env_file#"${ROOT_DIR:-}/"} has PYTORCH_CUDA_ALLOC_CONF commented out — respecting that opt-out."
        echo "[wsl2]       If vLLM crashes at boot with cudaErrorNotReady, uncomment it as:"
        echo "[wsl2]         PYTORCH_CUDA_ALLOC_CONF=${value}" ;;
      *)
        echo "[wsl2] WARN: ${env_file#"${ROOT_DIR:-}/"} exists without a PYTORCH_CUDA_ALLOC_CONF override."
        echo "[wsl2]       Not editing your file. If vLLM fails to boot with cudaErrorNotReady, add:"
        echo "[wsl2]         PYTORCH_CUDA_ALLOC_CONF=${value}"
        echo "[wsl2]       See PR #84 / issue #60 for context." ;;
    esac
  done < <(find "$compose_root" -name '*.yml' -not -path '*/_archive/*' -printf '%h\n' 2>/dev/null | sort -u)

  [[ "$wrote" -gt 0 ]] && echo "[wsl2] this fixes the known gptq_marlin_repack boot crash on WSL2 + driver ≥596.36 (issue #60)."

  # Retire the parent-level file this script used to write. Compose never reads it,
  # so leaving it in place only advertises protection that isn't there. Removed only
  # when its content is byte-for-byte what setup.sh generated; a user-edited file is
  # reported and kept.
  local stale="${compose_root}/.env"
  [[ -f "$stale" ]] || return 0
  if [[ "$(cat "$stale")" == "$(_wsl2_legacy_parent_env)" ]]; then
    rm -f "$stale"
    echo "[wsl2] removed dead ${stale#"${ROOT_DIR:-}/"} (compose never read it — the override now lives next to each compose)."
  else
    echo "[wsl2] WARN: ${stale#"${ROOT_DIR:-}/"} is NEVER read by docker compose (wrong directory level)."
    echo "[wsl2]       It looks hand-edited, so it was kept. Move any overrides you rely on"
    echo "[wsl2]       into the .env next to the compose you launch, then delete it."
  fi
}
