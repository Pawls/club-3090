#!/usr/bin/env bash
# ===========================================================================
# preserve-state.sh — resolve the cross-turn <think> carryover mode for a
# compose and hand it to the LiteLLM re-inline hook.
#
# WHY THIS EXISTS
# ---------------
# `services/litellm/custom_hooks.py` §C re-inlines prior-turn <think> blocks on
# the way into the model.  That hook runs in a SEPARATE container and cannot see
# the GPU compose's environment, so the launcher has to hand it the resolved
# mode out-of-band — via `services/litellm/preserve_state.json`, which the
# litellm compose bind-mounts.
#
# Before this script, only `serve.sh` wrote that file.  Booting the same model
# through `switch.sh` / `launch.sh` / the serve-cockpit left the file holding
# whatever the LAST serve.sh boot wrote — a silent mismatch: nothing errors, the
# hook just replays <think> into a model whose compose says not to.  On the
# always-reasoning 35B-A3B lanes that carryover is what seeded thought-loops
# (which is why those composes ship PRESERVE_THINKING=false).
#
# So: ONE resolver, called by every launcher.  serve.sh delegates here; the
# cockpit runs it right after `switch.sh <slug>`.
#
# GPU-mutex (exactly one live model) is what makes a single global state file
# exact — if that ever stops holding, this file has to become per-model.
#
# USAGE
#   bash scripts/preserve-state.sh --slug  vllm/qwen-35b-a3b-dual
#   bash scripts/preserve-state.sh --compose models/…/fp8.yml
#   bash scripts/preserve-state.sh --compose <path> --mode window --window 2
#   bash scripts/preserve-state.sh --mode off          # force, no compose read
#
# Resolution order (mirrors what the chat template does):
#   explicit --mode/--window  →  compose-dir .env  →  compose ${VAR:-default}
#   PRESERVE_THINKING_WINDOW > 0 → window · PRESERVE_THINKING=true → full · else off
#
# Env:
#   NO_LITELLM=1   skip entirely (same switch serve.sh honours)
#
# Exit codes: 0 ok / skipped-by-design · 1 bad usage or unresolvable slug
# ===========================================================================
set -uo pipefail

# Python's UTF-8 mode (PEP 540) — the slug→compose lookup shells out to python3
# and this repo's sources are full of unicode.  Defaulted, not forced, so a user
# who deliberately sets PYTHONUTF8=0 keeps control.  See AGENTS.md "File encoding".
export PYTHONUTF8="${PYTHONUTF8:-1}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITELLM_DIR="services/litellm"
STATE_FILE="$REPO/$LITELLM_DIR/preserve_state.json"

COMPOSE="" SLUG="" MODE="" WINDOW="" QUIET=0

die() { echo "preserve-state: $*" >&2; exit 1; }
say() { [[ "$QUIET" == 1 ]] || echo "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compose)   shift; COMPOSE="${1:-}" ;;
    --compose=*) COMPOSE="${1#--compose=}" ;;
    --slug)      shift; SLUG="${1:-}" ;;
    --slug=*)    SLUG="${1#--slug=}" ;;
    --mode)      shift; MODE="${1:-}" ;;
    --mode=*)    MODE="${1#--mode=}" ;;
    --window)    shift; WINDOW="${1:-}" ;;
    --window=*)  WINDOW="${1#--window=}" ;;
    -q|--quiet)  QUIET=1 ;;
    -h|--help)   sed -n '2,41p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           die "unknown argument: $1  (see --help)" ;;
  esac
  shift
done

# Opt-out — same switch serve.sh honours, so `NO_LITELLM=1` disables the whole
# LiteLLM lane consistently across launchers.
if [[ "${NO_LITELLM:-0}" == 1 ]]; then
  say "Preserve carryover: skipped (NO_LITELLM=1)."
  exit 0
fi

# No LiteLLM checkout → nothing to hand off to.  Not an error: plenty of rigs
# hit the model's raw port directly and never run the proxy.
if [[ ! -d "$REPO/$LITELLM_DIR" ]]; then
  say "Preserve carryover: skipped ($LITELLM_DIR not present)."
  exit 0
fi

# ── slug → compose path (registry is the single source of truth) ───────────────
# sys.path is built from $REPO, not the caller's cwd, so this works from any
# directory.  stderr is NOT swallowed (AGENTS.md: don't blind-2>/dev/null derive
# paths) — an import failure or unknown slug says which, on stderr.
if [[ -n "$SLUG" && -z "$COMPOSE" ]]; then
  COMPOSE="$(python3 - "$SLUG" "$REPO" <<'PY'
import sys, os
slug, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "scripts", "lib"))
try:
    from profiles.compose_registry import COMPOSE_REGISTRY
except Exception as e:
    print(f"registry import failed ({e!r}) — looked in {repo}/scripts/lib", file=sys.stderr)
    sys.exit(1)
entry = COMPOSE_REGISTRY.get(slug)
if not entry:
    print(f"slug {slug!r} not in COMPOSE_REGISTRY", file=sys.stderr)
    sys.exit(1)
path = entry.get("compose_path") if isinstance(entry, dict) else getattr(entry, "compose_path", None)
if not path:
    print(f"registry entry for {slug!r} has no compose_path", file=sys.stderr)
    sys.exit(1)
print(path)
PY
)" || die "could not resolve compose_path for slug '$SLUG' (see error above)"
  [[ -n "$COMPOSE" ]] || die "could not resolve compose_path for slug '$SLUG' (empty lookup)"
  [[ "$COMPOSE" = /* ]] || COMPOSE="$REPO/$COMPOSE"
fi

# ── resolve the mode ──────────────────────────────────────────────────────────
# Ported verbatim from serve.sh so the two launchers can never disagree:
#   _compose_default  <compose> <VAR> -> value after ${VAR:-…}
#   _env_override     <compose> <VAR> -> value from the compose-dir .env (the
#                                        ONLY .env docker compose auto-loads)
# Comment lines are stripped FIRST -- serve.sh's twin does this via _cmdlines and
# this port originally lost it.  A compose whose prose mentions the OPPOSITE default
# (qwen3.8-27b mtp-vision.yml documents ${PRESERVE_THINKING:-false} above threading
# ${PRESERVE_THINKING:-true}) otherwise resolves off the COMMENT, because head -1
# takes the first match in file order.  Silent, and the wrong way round.
_cmdlines() { grep -v '^[[:space:]]*#' "$1"; }
_compose_default() { _cmdlines "$1" | grep -oE "\\\$\\{$2:-[^},\"' ]*" 2>/dev/null | head -1 | sed "s/.*:-//" || true; }
_env_override() {
  local e; e="$(dirname "$1")/.env"
  [[ -f "$e" ]] || return 0
  grep -oE "^[[:space:]]*$2=[^#[:space:]]*" "$e" 2>/dev/null | head -1 | sed "s/.*=//" || true
}
_effective() { local v; v="$(_env_override "$1" "$2")"; [[ -z "$v" ]] && v="$(_compose_default "$1" "$2")"; echo "$v"; }

ps_mode=off ps_window=0

if [[ -n "$MODE" ]]; then
  # Explicit override (serve.sh's --preserve / --no-preserve / --preserve-window).
  case "$MODE" in
    off)    ps_mode=off;  ps_window=0 ;;
    full)   ps_mode=full; ps_window=0 ;;
    window)
      [[ "$WINDOW" =~ ^[0-9]+$ ]] || die "--mode window requires --window <N> (got '${WINDOW:-}')"
      if [[ "$WINDOW" -gt 0 ]]; then ps_mode=window; ps_window="$WINDOW"; else ps_mode=off; ps_window=0; fi ;;
    *) die "--mode must be one of: off | full | window (got '$MODE')" ;;
  esac
elif [[ -n "$COMPOSE" ]]; then
  [[ -f "$COMPOSE" ]] || die "compose not found: $COMPOSE"
  # Only composes that actually interpolate ${PRESERVE_THINKING…} have a mode to
  # resolve; everything else is correctly 'off'.
  if _cmdlines "$COMPOSE" | grep -qE '\$\{PRESERVE_THINKING'; then
    _w="$(_effective "$COMPOSE" PRESERVE_THINKING_WINDOW)"
    if   [[ "$_w" =~ ^[0-9]+$ && "$_w" -gt 0 ]]; then ps_mode=window; ps_window="$_w"
    elif [[ "$(_effective "$COMPOSE" PRESERVE_THINKING)" == true ]]; then ps_mode=full
    fi
  fi
else
  die "need --compose <path>, --slug <slug>, or --mode <mode>"
fi

# ── write ─────────────────────────────────────────────────────────────────────
# Truncate-in-place (same inode) so the bind mount reflects it live — the litellm
# container does NOT need a restart.  Do NOT replace this with a temp-file +
# os.replace(): that swaps the inode and the running container keeps reading the
# old one.  (This is the deliberate exception to the write-atomically rule in
# AGENTS.md — the file is tiny, rewritten wholesale, and its reader is watching
# this exact inode.)
prev=""
[[ -f "$STATE_FILE" ]] && prev="$(cat "$STATE_FILE" 2>/dev/null || true)"

printf '{"mode":"%s","window":%s}\n' "$ps_mode" "$ps_window" > "$STATE_FILE" \
  || die "could not write $STATE_FILE"

new="$(cat "$STATE_FILE")"
if [[ -n "$prev" && "$prev" != "$new" ]]; then
  say "Preserve carryover: mode=$ps_mode window=$ps_window  (was: $(echo "$prev" | tr -d '\n')) — LiteLLM cross-turn <think> re-inline"
else
  say "Preserve carryover: mode=$ps_mode window=$ps_window  (LiteLLM cross-turn <think> re-inline)"
fi
