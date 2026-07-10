#!/usr/bin/env bash
# serve.sh — GPU-mutex model switcher (project-root convenience wrapper).
#
# The rig runs ONE GPU-heavy model at a time (both 3090s, PCIe). This script
# evicts whatever GPU model container is currently up, then boots a target
# compose — so you never have to hand-`down` the old one first. LiteLLM (:4000)
# and Hermes are CPU-side and are left running.
#
# It exists because the cockpit (c3) launches by REGISTRY SLUG, and the
# experimental vision variants (apex-35b-vision, hauhau vision, deckard-40b-vision)
# are not in the registry — this boots them (or any compose) directly by path.
#
# Usage:
#   ./serve.sh <shortcut>            # e.g. ./serve.sh deckard-vision
#   ./serve.sh <path/to/compose.yml> # any compose file, relative to repo root or absolute
#   ./serve.sh --list                # show shortcuts
#   ./serve.sh --down                # just evict the current GPU model, boot nothing
#   ./serve.sh --status              # what's on the cards right now
#
# Env:
#   MODEL_DIR   passed through to the compose (default /home/pawl/models on this rig)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_DIR:=/home/pawl/models}"
export MODEL_DIR

# GPU model containers match these name prefixes; litellm / hermes never do,
# so they are never touched.
GPU_CONTAINER_RE='^(vllm|llama-cpp|ik-llama|sglang)-'

# --- Shortcut map: friendly name -> compose file (relative to repo root) -------
declare -A SHORTCUTS=(
  [deckard]="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/mtp.yml"
  [deckard-vision]="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/vision.yml"
  [hauhau]="models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/mtp.yml"
  [hauhau-vision]="models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/vision.yml"
  [apex]="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/mtp.yml"
  [apex-vision]="models/qwen3.6-35b-a3b/llama-cpp/compose/single/mudler-apex-compact/vision.yml"
  [carnice]="models/qwen3.6-27b/beellama/compose/dual/carnice-v2-q8/mtp-q8kv.yml"
  [27b-single]="models/qwen3.6-27b/vllm/compose/single/autoround-int4/fp8-mtp.yml"
)

running_gpu_containers() { docker ps --format '{{.Names}}' | grep -E "$GPU_CONTAINER_RE" || true; }

evict() {
  local names; names="$(running_gpu_containers)"
  if [[ -z "$names" ]]; then echo "  (no GPU model container running)"; return 0; fi
  while IFS= read -r c; do
    [[ -z "$c" ]] && continue
    echo "  evicting $c ..."
    # Prefer `compose down` via the container's own compose labels so its network
    # is cleaned up too; fall back to stop/rm if it wasn't started by compose.
    local wd cf
    wd="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.working_dir" }}' "$c" 2>/dev/null || true)"
    cf="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project.config_files" }}' "$c" 2>/dev/null || true)"
    if [[ -n "$wd" && -n "$cf" ]]; then
      ( cd "$wd" && docker compose -f "$cf" down >/dev/null 2>&1 ) || docker rm -f "$c" >/dev/null
    else
      docker rm -f "$c" >/dev/null
    fi
  done <<< "$names"
  # Give the driver a moment to reclaim VRAM before the next model grabs it.
  local i; for i in $(seq 1 10); do
    local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
    [[ "${used:-99999}" -lt 8000 ]] && break; sleep 1
  done
}

status() {
  echo "GPU:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "GPU model container:"; local n; n="$(running_gpu_containers)"; echo "  ${n:-<none>}"
}

case "${1:-}" in
  ""|--help|-h) sed -n '2,30p' "$0"; exit 0 ;;
  --list)
    echo "Shortcuts:"; for k in "${!SHORTCUTS[@]}"; do printf "  %-16s %s\n" "$k" "${SHORTCUTS[$k]}"; done | sort; exit 0 ;;
  --status) status; exit 0 ;;
  --down) echo "Evicting current GPU model:"; evict; echo "done."; exit 0 ;;
esac

# Resolve target -> compose file
target="$1"
if [[ -n "${SHORTCUTS[$target]:-}" ]]; then
  compose="$REPO/${SHORTCUTS[$target]}"
elif [[ -f "$target" ]]; then
  compose="$(cd "$(dirname "$target")" && pwd)/$(basename "$target")"
elif [[ -f "$REPO/$target" ]]; then
  compose="$REPO/$target"
else
  echo "ERROR: '$target' is neither a known shortcut nor a compose file." >&2
  echo "Try: ./serve.sh --list" >&2; exit 1
fi
[[ -f "$compose" ]] || { echo "ERROR: compose not found: $compose" >&2; exit 1; }

echo "Target: $compose"
echo "Evicting current GPU model:"; evict
echo "Booting target:"
( cd "$(dirname "$compose")" && docker compose -f "$compose" up -d ) 2>&1 | sed 's/^/  /'

# Report the port the target publishes + wait briefly for health.
port="$(cd "$(dirname "$compose")" && docker compose -f "$compose" config 2>/dev/null \
        | grep -oE 'published: "[0-9]+"' | grep -oE '[0-9]+' | head -1 || true)"
if [[ -n "$port" ]]; then
  echo "Waiting for :$port to answer ..."
  for i in $(seq 1 90); do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && { echo "  ready: http://localhost:$port  (model list: /v1/models)"; break; }
    sleep 2
  done
  [[ "${code:-}" != "200" ]] && echo "  still loading (last /health=$code) — large models take a few min; check: docker logs -f $(basename "$(dirname "$compose")")"
fi
