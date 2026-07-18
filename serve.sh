#!/usr/bin/env bash
# serve.sh — GPU-mutex model switcher + weights puller (project-root convenience wrapper).
#
# The rig runs ONE GPU-heavy model at a time (both 3090s, PCIe). This script
# evicts whatever GPU model container is currently up, then boots a target
# compose — so you never have to hand-`down` the old one first. It also makes sure
# the CPU-side LiteLLM proxy (:4000) is up — Hermes / OpenWebUI reach models THROUGH
# it, and a Docker/WSL restart can leave it exited despite `unless-stopped`.
#
# This is the ONE launcher: it lists every model worth running on this rig —
# canonical repo composes AND our pawl-custom ones — and can download missing
# weights (like the cockpit's Download UX, but `hf download`-based so it also
# works for models that aren't in the registry catalog).
#
# Which model for which task → MODEL_REFERENCE.md (capability + max-ctx tables).
#
# Usage:
#   ./serve.sh <name>                # e.g. ./serve.sh deckard-vision
#   ./serve.sh <name> [toggles]      # e.g. ./serve.sh 27b --no-think --no-preserve
#   ./serve.sh <path/to/compose.yml> # any compose file, relative to repo root or absolute
#   ./serve.sh --list                # all models w/ think/preserve state, ⬇ = weights missing
#   ./serve.sh --pull <name>         # download the missing weights for <name>
#   ./serve.sh --down                # just evict the current GPU model, boot nothing
#   ./serve.sh --status              # what's on the cards right now
#
# Thinking / preserve toggles (override the compose default at launch, NO .env edit —
# shell env beats the compose-dir .env in docker-compose interpolation). Neither costs
# VRAM (KV pool / concurrency / MTP / ctx / quant are fixed at boot) — they only shape
# the request, so they're flags, not separate composes. Ideal for A/B through Hermes:
#   --think | --no-think          reasoning on/off  (vLLM enable_thinking / llama --reasoning)
#   --preserve | --no-preserve    keep vs strip prior-turn <think> from context
# Not all models can toggle: agents-a1 + 27b-minimal hardcode thinking off; omni isn't a
# thinking model. --list shows each model's state; a bad toggle errors instead of no-op'ing.
#
# Context override (UNLIKE the toggles above, this IS a boot-time KV allocation — a bigger
# window costs VRAM and can OOM at load; a smaller one frees it):
#   --ctx <N>                     set the context window for this boot. Auto-targets the
#                                 compose's ctx var (VISION_CTX_SIZE / CTX_SIZE / MAX_MODEL_LEN).
#                                 e.g. ./serve.sh apex-vision-ik --ctx 262144
#
# Env:
#   MODEL_DIR    passed through to the compose (default /home/pawl/models on this rig)
#   NO_LITELLM=1 skip the LiteLLM-proxy readiness step on serve
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_DIR:=/home/pawl/models}"
export MODEL_DIR

# GPU model containers match these name prefixes; litellm / hermes never do,
# so they are never touched. (beellama added 2026-07-10 — carnice was
# previously invisible to eviction.)
GPU_CONTAINER_RE='^(vllm|llama-cpp|ik-llama|beellama|sglang)-'

# --- Model catalog ------------------------------------------------------------
# COMPOSE  name -> compose file (relative to repo root)
# GROUP    name -> repo (canonical, exists on master) | ours (pawl-custom only)
# INFO     name -> ":port  one-line description" (display only)
# WEIGHTS  name -> space-separated MODEL_DIR-relative weight paths.
#                  *.gguf = single file; anything else = HF snapshot dir
#                  (presence checked via <dir>/config.json).
# HFREPO   weight path -> HF repo to pull it from (gguf: file = basename;
#                  dir: whole-repo snapshot).

declare -A COMPOSE=(
  # -- canonical (repo compose; our .env tweaks may apply — see MODEL_REFERENCE §5)
  [apex]="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/long.yml"
  [apex-fit]="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/fit-mtp.yml"
  [27b]="models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml"
  [27b-minimal]="models/qwen3.6-27b/vllm/compose/single/autoround-int4/minimal.yml"
  [35b-a3b]="models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml"
  [carnice]="models/qwen3.6-27b/beellama/compose/dual/carnice-v2-q8/mtp-q8kv.yml"
  [deckard]="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/mtp.yml"
  [hauhau]="models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/mtp.yml"
  [omni]="models/qwen3-omni-30b-a3b/vllm-omni/compose/dual/autoround-int4/omni.yml"
  [agents-a1]="models/agents-a1/vllm/compose/dual/fp8-dynamic/fp8.yml"
  # -- ours (pawl-custom only; not on repo master)
  [apex-vision-ik]="models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/vision.yml"
  [apex-vision-mainline]="models/qwen3.6-35b-a3b/llama-cpp/compose/single/mudler-apex-compact/vision.yml"
  [deckard-vision]="models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/vision.yml"
  [hauhau-vision]="models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/vision.yml"
  [27b-single]="models/qwen3.6-27b/vllm/compose/single/autoround-int4/fp8-mtp.yml"
  [apex-yarn-1m]="models/qwen3.6-35b-a3b/ik-llama/compose/dual/mudler-apex-quality/yarn-1m.yml"
)

declare -A GROUP=(
  [apex]=repo [apex-fit]=repo [27b]=repo [27b-minimal]=repo [35b-a3b]=repo
  [carnice]=repo [deckard]=repo [hauhau]=repo [omni]=repo [agents-a1]=repo
  [apex-vision-ik]=ours [apex-vision-mainline]=ours [deckard-vision]=ours
  [hauhau-vision]=ours [27b-single]=ours [apex-yarn-1m]=ours
)

declare -A INFO=(
  [apex]=":8056  apex-35b-a3b (Q4_K_M) · single GPU1 · 262K · text · q8 KV · ~90 TPS · think ON — ⭐ daily driver"
  [apex-fit]=":8057  apex-35b-a3b (Q4_K_M) · single GPU1 · 262K · asym q8/q5 KV + no-mmap — alt apex lane"
  [27b]=":8010  dual vLLM · 262K · vision · MTP n=3 — ⭐ big-ctx image analysis"
  [27b-minimal]=":8020  single · 65K · text-only · no MTP · ~32 TPS — debug/fallback"
  [35b-a3b]=":8051  dual vLLM · 262K · vision · N=4 concurrency — ⭐ subagent fan-out"
  [carnice]=":8070  dual beellama · 262K · agentic-SFT Q8 — the 'agent brain'"
  [deckard]=":8199  dual · 131K · uncensored 40B · MTP n=2 — ⭐ hard reasoning"
  [hauhau]=":8073  dual · 262K · uncensored MoE · MTP n=3"
  [omni]=":8042  dual stage-parallel · 48K · image/audio/VIDEO in — use via :4000, not raw"
  [agents-a1]=":8072  dual vLLM · 262K · agentic (thinking ON only via LiteLLM hook)"
  [apex-vision-ik]=":8057  apex-35b-a3b (Q4_K_M) · single GPU1 · 262K · vision · asym q8/q5 KV · ~90 TPS — ⭐ screenshot driver"
  [apex-vision-mainline]=":8058  apex-35b-a3b (Q4_K_M) · single GPU1 · 200K · vision · mainline build — backup"
  [deckard-vision]=":8200  dual · 131K · uncensored · vision + MTP kept"
  [hauhau-vision]=":8073  dual · 262K · uncensored · vision + MTP kept"
  [27b-single]=":8021  single GPU0 · 28K · vision · MTP — fast solo, tiny ctx"
  [apex-yarn-1m]=":8057  apex-35b-a3b (Quality) · dual · 1M YaRN · quality unproven >262K — eval/park"
)

APEX_GGUF="qwen3.6-35b-a3b-gguf/mudler-apex-mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf"
APEX_Q_GGUF="qwen3.6-35b-a3b-gguf/mudler-apex-mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf"
QWEN_MMPROJ="qwen3.6-35b-a3b-gguf/mmproj/mmproj-BF16.gguf"
DECKARD_GGUF="qwen3.6-40b-deckard-gguf/piehsoft-q6k/Qwen3.6-40B-Deckard-MTP-Q6_K.gguf"
DECKARD_MMPROJ="qwen3.6-40b-deckard-gguf/mmproj/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking.mmproj-Q8_0.gguf"
HAUHAU_GGUF="qwen3.6-35b-a3b-uncensored-mtp-gguf/morikomorizz-q6kp/Qwen3.6-35B-A3B-Uncensored-HauhauCS-MTP-Q6_K_P.gguf"
CARNICE_GGUF="carnice-v2-27b-gguf/stuchapin-q8/Carnice-V2-27B-Q8_0-mtp.gguf"

declare -A WEIGHTS=(
  [apex]="$APEX_GGUF"
  [apex-fit]="$APEX_GGUF"
  [apex-vision-ik]="$APEX_GGUF $QWEN_MMPROJ"
  [apex-vision-mainline]="$APEX_GGUF $QWEN_MMPROJ"
  [apex-yarn-1m]="$APEX_Q_GGUF"
  [deckard]="$DECKARD_GGUF"
  [deckard-vision]="$DECKARD_GGUF $DECKARD_MMPROJ"
  [hauhau]="$HAUHAU_GGUF"
  [hauhau-vision]="$HAUHAU_GGUF $QWEN_MMPROJ"
  [carnice]="$CARNICE_GGUF"
  [27b]="qwen3.6-27b-autoround-int4"
  [27b-single]="qwen3.6-27b-autoround-int4"
  [27b-minimal]="qwen3.6-27b-autoround-int4"
  [35b-a3b]="qwen3.6-35b-a3b-autoround-int4"
  [omni]="qwen3-omni-30b-a3b-instruct-int4-autoround"
  [agents-a1]="Agents-A1-FP8-dynamic"
)

declare -A HFREPO=(
  ["$APEX_GGUF"]="mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF"
  ["$APEX_Q_GGUF"]="mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF"
  ["$QWEN_MMPROJ"]="unsloth/Qwen3.6-35B-A3B-GGUF"
  ["$DECKARD_GGUF"]="PiehSoft/Qwen3.6-40B-Deckard-MTP-Q6_K"
  ["$DECKARD_MMPROJ"]="mradermacher/Qwen3.5-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-GGUF"
  ["$HAUHAU_GGUF"]="morikomorizz/Qwen3.6-35B-A3B-Uncensored-HauhauCS-MTP"
  ["$CARNICE_GGUF"]="stuchapin/Carnice-V2-27B-MTP-GGUF"
  ["qwen3.6-27b-autoround-int4"]="Lorbus/Qwen3.6-27B-int4-AutoRound"
  ["qwen3.6-35b-a3b-autoround-int4"]="Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound"
  ["qwen3-omni-30b-a3b-instruct-int4-autoround"]="Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound"
  ["Agents-A1-FP8-dynamic"]="InternScience/Agents-A1-FP8-dynamic"
)

# --- Weights helpers ----------------------------------------------------------

missing_weights() {  # <name> -> print MODEL_DIR-relative paths that are absent
  local rel
  for rel in ${WEIGHTS[$1]:-}; do
    if [[ "$rel" == *.gguf ]]; then
      [[ -f "$MODEL_DIR/$rel" ]] || echo "$rel"
    else
      [[ -f "$MODEL_DIR/$rel/config.json" ]] || echo "$rel"
    fi
  done
}

pull_weights() {  # <name> -> hf-download everything missing_weights reports
  local name="$1" rel repo any=0
  command -v hf >/dev/null || { echo "ERROR: 'hf' CLI not found (pip install -U huggingface_hub)" >&2; return 1; }
  while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    repo="${HFREPO[$rel]:-}"
    if [[ -z "$repo" ]]; then echo "  no download source known for $rel — fetch by hand" >&2; continue; fi
    any=1
    if [[ "$rel" == *.gguf ]]; then
      echo "  pulling $(basename "$rel")  <-  $repo"
      hf download "$repo" "$(basename "$rel")" --local-dir "$MODEL_DIR/$(dirname "$rel")"
    else
      echo "  pulling full repo $repo  ->  $MODEL_DIR/$rel"
      hf download "$repo" --local-dir "$MODEL_DIR/$rel"
    fi
  done < <(missing_weights "$name")
  [[ $any -eq 0 ]] && echo "  nothing missing for '$name'."
  return 0
}

# --- Thinking / preserve toggles ----------------------------------------------
# Two request-shaping knobs, wired differently per engine but presented uniformly:
#   thinking  → vLLM composes interpolate ${ENABLE_THINKING} (true/false) into
#               --default-chat-template-kwargs; llama.cpp/ik/beellama interpolate
#               ${REASONING} (on/off) into --reasoning.
#   preserve  → every engine threads ${PRESERVE_THINKING} (true/false) into the
#               chat template's preserve_thinking kwarg (keep prior <think> in ctx).
# Neither changes boot-time VRAM — KV pool, concurrency, MTP depth, ctx, and quant
# are all fixed at launch — so they are FLAGS on an existing model, never separate
# composes. We AUTO-DETECT each compose's wiring (below) so the flags also work for
# raw compose-path targets, and so --list can't drift from what the file actually reads.

think_spec() {  # <compose> -> REASONING | ENABLE_THINKING | FIXED | NONE
  local c="$1"
  if   grep -qE '\$\{ENABLE_THINKING' "$c"; then echo ENABLE_THINKING
  elif grep -qE '\$\{REASONING'       "$c"; then echo REASONING
  elif grep -qiE 'enable_thinking|--reasoning[ =]' "$c"; then echo FIXED   # hardcoded, not env-togglable
  else echo NONE                                                            # not a thinking model
  fi
}
_compose_default() {  # <compose> <VAR> -> value after ${VAR:-…}, else ""
  grep -oE "\\\$\\{$2:-[^},\"' ]*" "$1" 2>/dev/null | head -1 | sed "s/.*:-//" || true
}
_env_override() {  # <compose> <VAR> -> value from the compose-dir .env (the one docker loads), else ""
  local e; e="$(dirname "$1")/.env"
  [[ -f "$e" ]] || return 0
  grep -oE "^[[:space:]]*$2=[^#[:space:]]*" "$e" 2>/dev/null | head -1 | sed "s/.*=//" || true
}
_effective() {  # <compose> <VAR> -> .env override, else compose ${VAR:-default}
  local v; v="$(_env_override "$1" "$2")"; [[ -z "$v" ]] && v="$(_compose_default "$1" "$2")"; echo "$v"
}
think_state() {  # <compose> -> on | off | on·fixed | off·fixed | n/a
  local c="$1" s; s="$(think_spec "$c")"
  case "$s" in
    NONE)  echo "n/a"; return ;;
    FIXED) grep -qiE 'enable_thinking"?:? *true' "$c" && echo "on·fixed" || echo "off·fixed"; return ;;
  esac
  case "$(_effective "$c" "$s")" in on|true) echo on ;; off|false) echo off ;; *) echo "?" ;; esac
}
preserve_state() {  # <compose> -> keep | strip | n/a
  local c="$1"
  grep -qE '\$\{PRESERVE_THINKING' "$c" || { echo "n/a"; return; }
  case "$(_effective "$c" PRESERVE_THINKING)" in true) echo keep ;; false) echo strip ;; *) echo "?" ;; esac
}
ctx_spec() {  # <compose> -> which env var sets the context window (for --ctx to target)
  local c="$1"
  if   grep -qE '\$\{VISION_CTX_SIZE' "$c"; then echo VISION_CTX_SIZE   # ik vision.yml (most specific first)
  elif grep -qE '\$\{CTX_SIZE'        "$c"; then echo CTX_SIZE          # llama.cpp / ik text
  elif grep -qE '\$\{MAX_MODEL_LEN'   "$c"; then echo MAX_MODEL_LEN     # vLLM
  else echo NONE
  fi
}

list_models() {
  local name mark c
  echo "Models (GPU-mutex: launching one evicts the current one).  Details: MODEL_REFERENCE.md"
  echo
  echo "Canonical (repo composes):"
  for name in $(for k in "${!COMPOSE[@]}"; do [[ "${GROUP[$k]}" == repo ]] && echo "$k"; done | sort); do
    c="$REPO/${COMPOSE[$name]}"; mark=""; [[ -n "$(missing_weights "$name")" ]] && mark="  ⬇ weights missing"
    printf "  %-22s %s  {think:%s preserve:%s}%s\n" \
      "$name" "${INFO[$name]}" "$(think_state "$c")" "$(preserve_state "$c")" "$mark"
  done
  echo
  echo "Ours (pawl-custom only):"
  for name in $(for k in "${!COMPOSE[@]}"; do [[ "${GROUP[$k]}" == ours ]] && echo "$k"; done | sort); do
    c="$REPO/${COMPOSE[$name]}"; mark=""; [[ -n "$(missing_weights "$name")" ]] && mark="  ⬇ weights missing"
    printf "  %-22s %s  {think:%s preserve:%s}%s\n" \
      "$name" "${INFO[$name]}" "$(think_state "$c")" "$(preserve_state "$c")" "$mark"
  done
  echo
  echo "⬇ = ./serve.sh --pull <name> downloads it (hf CLI); any compose path also works as a target."
  echo
  echo "{think:X preserve:Y} = current default (no VRAM cost; override at launch, no .env edit):"
  echo "  --think / --no-think        reasoning on/off   (vLLM enable_thinking · llama.cpp --reasoning)"
  echo "  --preserve / --no-preserve  keep vs strip prior-turn <think> from context (preserve_thinking)"
  echo "  A/B through Hermes:  ./serve.sh 27b --think   then   ./serve.sh 27b --no-think"
  echo "  ·fixed = hardcoded in the compose (agents-a1, 27b-minimal)   n/a = not a thinking model."
}

# --- GPU mutex ----------------------------------------------------------------

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

# --- LiteLLM proxy ------------------------------------------------------------
# GPU models are reached THROUGH the LiteLLM proxy on :4000 (it routes by
# served-model-name), NOT their raw vLLM/llama.cpp port. serve.sh swaps the GPU
# model underneath a stable proxy, so the proxy must stay up independently. It's
# `unless-stopped`, but a Docker/WSL restart can leave it exited without
# recovering — so every serve re-asserts it. Set NO_LITELLM=1 to skip.
LITELLM_DIR="services/litellm"
LITELLM_PORT=4000
LITELLM_KEY="sk-litellm-master-key"

litellm_ready() {  # 0 iff :4000 answers /v1/models with the master key
  curl -sf -m 3 -H "Authorization: Bearer $LITELLM_KEY" \
    "http://127.0.0.1:$LITELLM_PORT/v1/models" >/dev/null 2>&1
}

ensure_litellm() {  # bring the proxy up if it isn't already answering
  [[ "${NO_LITELLM:-0}" == 1 ]] && return 0
  echo "LiteLLM proxy (:$LITELLM_PORT):"
  if litellm_ready; then echo "  already up."; return 0; fi
  if [[ ! -f "$REPO/$LITELLM_DIR/docker-compose.yml" ]]; then
    echo "  ⚠ $LITELLM_DIR/docker-compose.yml not found — skipping."; return 0
  fi
  echo "  not responding — starting $LITELLM_DIR ..."
  ( cd "$REPO/$LITELLM_DIR" && docker compose up -d ) 2>&1 | sed 's/^/    /'
  local i
  for i in $(seq 1 30); do
    litellm_ready && { echo "  ready: http://localhost:$LITELLM_PORT/v1/models"; return 0; }
    sleep 1
  done
  # Non-fatal: a slow/broken proxy shouldn't fail an otherwise-good GPU serve.
  echo "  ⚠ still not answering after 30s — check: docker logs litellm"
  return 0
}

status() {
  echo "GPU:"; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "GPU model container:"; local n; n="$(running_gpu_containers)"; echo "  ${n:-<none>}"
  echo "LiteLLM proxy (:$LITELLM_PORT):"
  if litellm_ready; then echo "  up"
  else echo "  DOWN — starts automatically on next ./serve.sh <model> (or: cd $LITELLM_DIR && docker compose up -d)"; fi
}

# --- Extract thinking/preserve toggles (order-independent; leave the rest) -----
THINK_FLAG="" PRESERVE_FLAG="" CTX_FLAG="" _pos=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --think)       THINK_FLAG=on ;;
    --no-think)    THINK_FLAG=off ;;
    --preserve)    PRESERVE_FLAG=keep ;;
    --no-preserve) PRESERVE_FLAG=strip ;;
    --ctx)         shift; CTX_FLAG="${1:-}" ;;
    --ctx=*)       CTX_FLAG="${1#--ctx=}" ;;
    *)             _pos+=("$1") ;;
  esac
  shift
done
if [[ -n "$CTX_FLAG" && ! "$CTX_FLAG" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --ctx needs an integer token count (e.g. --ctx 262144)." >&2; exit 1
fi
set -- "${_pos[@]+"${_pos[@]}"}"

case "${1:-}" in
  ""|--help|-h) sed -n '2,43p' "$0"; exit 0 ;;
  --list) list_models; exit 0 ;;
  --status) status; exit 0 ;;
  --down) echo "Evicting current GPU model:"; evict; echo "done."; exit 0 ;;
  --pull)
    [[ -n "${2:-}" && -n "${COMPOSE[${2:-}]:-}" ]] || { echo "Usage: ./serve.sh --pull <name>   (names: ./serve.sh --list)" >&2; exit 1; }
    echo "Checking weights for '$2':"; pull_weights "$2"; exit $? ;;
esac

# Resolve target -> compose file
target="$1"
if [[ -n "${COMPOSE[$target]:-}" ]]; then
  compose="$REPO/${COMPOSE[$target]}"
elif [[ -f "$target" ]]; then
  compose="$(cd "$(dirname "$target")" && pwd)/$(basename "$target")"
elif [[ -f "$REPO/$target" ]]; then
  compose="$REPO/$target"
else
  echo "ERROR: '$target' is neither a known model name nor a compose file." >&2
  echo "Try: ./serve.sh --list" >&2; exit 1
fi
[[ -f "$compose" ]] || { echo "ERROR: compose not found: $compose" >&2; exit 1; }

# Weights present? (only checkable for cataloged names, not raw compose paths)
if [[ -n "${WEIGHTS[$target]:-}" ]]; then
  miss="$(missing_weights "$target")"
  if [[ -n "$miss" ]]; then
    echo "Missing weights for '$target':"; printf '  %s\n' $miss
    if [[ -t 0 ]]; then
      read -r -p "Download now with hf? [y/N] " ans
      [[ "$ans" == [yY]* ]] || exit 1
      pull_weights "$target"
      miss="$(missing_weights "$target")"
      [[ -n "$miss" ]] && { echo "ERROR: still missing after download:"; printf '  %s\n' $miss; exit 1; }
    else
      echo "Run: ./serve.sh --pull $target" >&2; exit 1
    fi
  fi
fi

# Apply --think / --preserve overrides. Exported shell vars beat the compose-dir
# .env in docker-compose interpolation, so this needs no file edit. Refuse (don't
# silently no-op) when the target's compose can't honor the toggle.
if [[ -n "$THINK_FLAG" ]]; then
  case "$(think_spec "$compose")" in
    REASONING)       export REASONING=$([[ "$THINK_FLAG" == on ]] && echo on || echo off)
                     echo "Override: REASONING=$REASONING (thinking $THINK_FLAG)" ;;
    ENABLE_THINKING) export ENABLE_THINKING=$([[ "$THINK_FLAG" == on ]] && echo true || echo false)
                     echo "Override: ENABLE_THINKING=$ENABLE_THINKING (thinking $THINK_FLAG)" ;;
    FIXED) echo "ERROR: '$target' hardcodes thinking in its compose — not env-togglable." >&2
           echo "       (agents-a1: thinking-ON is via the LiteLLM hook / per-request kwargs — MODEL_REFERENCE §1 n.8.)" >&2
           exit 1 ;;
    NONE)  echo "ERROR: '$target' is not a thinking model — --think/--no-think don't apply." >&2; exit 1 ;;
  esac
fi
if [[ -n "$PRESERVE_FLAG" ]]; then
  if grep -qE '\$\{PRESERVE_THINKING' "$compose"; then
    export PRESERVE_THINKING=$([[ "$PRESERVE_FLAG" == keep ]] && echo true || echo false)
    echo "Override: PRESERVE_THINKING=$PRESERVE_THINKING (prior <think> $PRESERVE_FLAG)"
  else
    echo "ERROR: '$target' doesn't thread preserve_thinking — nothing to override." >&2; exit 1
  fi
fi
# --ctx: boot-time context window. UNLIKE think/preserve this resizes the KV allocation,
# so too large can OOM at load. Auto-targets the var the compose actually reads.
if [[ -n "$CTX_FLAG" ]]; then
  ctx_var="$(ctx_spec "$compose")"
  if [[ "$ctx_var" == NONE ]]; then
    echo "ERROR: '$target' has no recognized context var (VISION_CTX_SIZE/CTX_SIZE/MAX_MODEL_LEN) to override." >&2; exit 1
  fi
  export "$ctx_var=$CTX_FLAG"
  echo "Override: $ctx_var=$CTX_FLAG (context window — boot-time KV alloc; if load OOMs, retry lower)"
fi

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

# The GPU model serves through the LiteLLM proxy (:4000), not its raw port — make
# sure the proxy is up so Hermes / OpenWebUI can actually reach the model.
ensure_litellm
