#!/usr/bin/env bash
#
# freeze-capture.sh — one-shot evidence capture for a wedged / crashed model.
#
# Different from health.sh: that one answers "is it healthy right now?".
# This one answers "it just froze — grab everything before I kill it."
# It fans out the three streams you need for a post-mortem into a single
# timestamped directory, so a freeze investigation is one command.
#
#   1. Engine log   — the container's stdout/stderr (errors, KV pool, asserts)
#   2. GPU state    — nvidia-smi snapshot + a short per-second dmon sample
#   3. Kernel ring  — dmesg lines for Xid / NVRM / OOM-killer (the root cause
#                     the engine log can't see); needs sudo, degrades if absent
#   +  docker inspect state (exit code / OOMKilled / restart count)
#
# THE RULE: run this BEFORE `docker compose down` / `docker rm`. Removing the
# container deletes its json-file logs and you lose the engine stream forever.
#
# Usage:
#   bash scripts/freeze-capture.sh                 # snapshot the auto-matched engine container
#   CONTAINER=vllm-qwen36-27b bash scripts/freeze-capture.sh
#   SINCE=30m bash scripts/freeze-capture.sh       # limit engine log to the last 30m
#   bash scripts/freeze-capture.sh --follow        # stream all three to files until Ctrl-C
#
# Env:
#   CONTAINER   Target a specific container instead of auto-matching.
#               Default: unset -> first running recognized-engine container.
#   OUT_DIR     Base dir for capture folders. Default: $HOME/club-3090-freezes
#   SINCE       docker logs --since window (snapshot mode). Default: unset (all).
#   DMON_SECS   seconds of nvidia-smi dmon to sample. Default: 5
#
set -uo pipefail

CONTAINER="${CONTAINER:-}"
OUT_DIR="${OUT_DIR:-$HOME/club-3090-freezes}"
SINCE="${SINCE:-}"
DMON_SECS="${DMON_SECS:-5}"
FOLLOW=0
[[ "${1:-}" == "--follow" ]] && FOLLOW=1

# Same engine-prefix convention as health.sh: any recognized inference engine.
ENGINE_PREFIX_RE='^(vllm-|llama-cpp-|ik-llama-|sglang-|beellama-)'

die() { printf 'freeze-capture: %s\n' "$1" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker not found on PATH"

# 1. Resolve the target container.
if [[ -z "$CONTAINER" ]]; then
  CONTAINER=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E "$ENGINE_PREFIX_RE" | head -1 || true)
  [[ -n "$CONTAINER" ]] || die "no running engine container matched ${ENGINE_PREFIX_RE}. Pass CONTAINER=<name> (see: docker ps)."
fi
docker inspect "$CONTAINER" >/dev/null 2>&1 || die "container '$CONTAINER' not found (already removed? capture the logs BEFORE 'docker rm')."

# 2. Timestamped output dir. Stamp comes from `date` (host clock), fine in a shell script.
STAMP=$(date '+%Y%m%d-%H%M%S')
DEST="${OUT_DIR}/${CONTAINER}-${STAMP}"
mkdir -p "$DEST" || die "cannot create $DEST"

echo "freeze-capture: container=$CONTAINER  ->  $DEST"

# --- container state (cheap, always) ---
docker inspect \
  --format 'State: {{.State.Status}}  ExitCode: {{.State.ExitCode}}  OOMKilled: {{.State.OOMKilled}}  Restarts: {{.RestartCount}}  StartedAt: {{.State.StartedAt}}' \
  "$CONTAINER" > "$DEST/state.txt" 2>&1
cat "$DEST/state.txt"

if [[ "$FOLLOW" -eq 1 ]]; then
  echo "freeze-capture: --follow (streaming; Ctrl-C to stop)"
  # Stream all three concurrently; trap cleans up children on Ctrl-C.
  pids=()
  docker logs -f --timestamps "$CONTAINER" > "$DEST/engine.log" 2>&1 & pids+=($!)
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -o DT >> "$DEST/gpu-dmon.log" 2>&1 & pids+=($!)
  fi
  if command -v dmesg >/dev/null 2>&1; then
    # -w may need root; if it fails it just exits and leaves what it had.
    { sudo -n dmesg -w 2>/dev/null || dmesg -w 2>/dev/null; } \
      | grep --line-buffered -iE 'xid|nvrm|out of memory|oom-kill|nvidia' >> "$DEST/dmesg.log" & pids+=($!)
  fi
  trap 'echo; echo "freeze-capture: stopping..."; kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null; echo "saved -> $DEST"; exit 0' INT TERM
  wait
  exit 0
fi

# --- snapshot mode (default) ---

# Engine log
if [[ -n "$SINCE" ]]; then
  docker logs --since "$SINCE" --timestamps "$CONTAINER" > "$DEST/engine.log" 2>&1
else
  docker logs --timestamps "$CONTAINER" > "$DEST/engine.log" 2>&1
fi
echo "  engine.log   $(wc -l < "$DEST/engine.log" 2>/dev/null || echo 0) lines"

# GPU: full snapshot + short dmon sample
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$DEST/gpu-snapshot.txt" 2>&1
  nvidia-smi > "$DEST/gpu-summary.txt" 2>&1
  # timeout keeps a wedged driver query from hanging the capture.
  timeout "$((DMON_SECS + 3))" nvidia-smi dmon -s pucm -o DT -c "$DMON_SECS" > "$DEST/gpu-dmon.log" 2>&1 || true
  echo "  gpu-*        captured (snapshot + ${DMON_SECS}s dmon)"
else
  echo "  gpu-*        SKIPPED (nvidia-smi not found)"
fi

# Kernel ring: Xid / NVRM / OOM. Needs root for full buffer; degrade gracefully.
if sudo -n dmesg >/dev/null 2>&1; then
  sudo -n dmesg -T 2>/dev/null | grep -iE 'xid|nvrm|out of memory|oom-kill|nvidia' | tail -200 > "$DEST/dmesg.log"
elif dmesg >/dev/null 2>&1; then
  dmesg -T 2>/dev/null | grep -iE 'xid|nvrm|out of memory|oom-kill|nvidia' | tail -200 > "$DEST/dmesg.log"
else
  echo "kernel ring buffer needs root: re-run 'sudo dmesg -T | grep -iE \"xid|nvrm|oom\"'" > "$DEST/dmesg.log"
fi
echo "  dmesg.log    $(wc -l < "$DEST/dmesg.log" 2>/dev/null || echo 0) lines"

echo
echo "freeze-capture: done -> $DEST"
echo "  Safe to restart now. Review with:  less $DEST/engine.log"
