#!/usr/bin/env bash
# ===========================================================================
# preserve-probe.sh — prove whether prior-turn reasoning actually reaches the
# model, with no harness memory in the loop.
#
# WHY THIS EXISTS
# ---------------
# "Is preserve_thinking working?" is easy to ask and hard to answer from a chat
# client: Hermes (and any harness with memory) can answer a follow-up from its
# OWN store, so the model appearing to remember proves nothing.  This sends the
# two turns by hand, twice, against the SAME turn-1 answer — once carrying the
# assistant's reasoning_content, once with it stripped — so the only variable is
# the reasoning replay.
#
# TWO SIGNALS, deliberately.  The semantic one (can it name a number it only
# ever wrote in its reasoning?) proves the replay is USABLE.  The objective one
# (turn-2 prompt_tokens, with vs without) proves the tokens reached the template
# at all, and does not depend on the model cooperating.  A model that simply
# refuses the game still moves the token count.  See the measured shape in
# models/qwen3.8-27b/.../mtp-vision.yml (403 vs 38 tokens, delta 365).
#
# WHICH LAYER YOU ARE TESTING — point it at the right port
#   :PORT direct   the ENGINE + chat template (PRESERVE_THINKING kwarg).  This
#                  is the live layer for llama.cpp models whose template consumes
#                  the reasoning_content FIELD (qwen3.8-27b).
#   :4000 proxy    additionally exercises custom_hooks.py §C, the cross-turn
#                  re-inline that preserve-state.sh configures.  §C is GATED to
#                  QWEN_PRESERVE_MODELS (qwen3.6-27b / qwen3.6-35b-a3b /
#                  apex-35b) — for any other model the proxy adds nothing here
#                  and the direct port is the honest test.
#
# Reasoning parsing must be ON for the compose under test, or turn 1 returns no
# reasoning_content and both arms are trivially identical (the script says so).
#
# USAGE
#   URL=http://127.0.0.1:8117/v1 MODEL=qwen3.8-27b-vision KEY=EMPTY \
#     bash scripts/preserve-probe.sh
#   URL=http://127.0.0.1:4000/v1 MODEL=qwen3.6-27b-dual bash scripts/preserve-probe.sh
#
# Env: URL (default the :4000 proxy) · MODEL (autodetected from /v1/models when
#      unset, per the repo-wide convention) · KEY (default the proxy master key;
#      use EMPTY for a direct engine port) · EFFORT (unset = server default)
#
# Exit codes: 0 probe ran · 1 no model / turn 1 failed
# ===========================================================================
set -uo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

URL="${URL:-http://127.0.0.1:4000/v1}"
KEY="${KEY:-sk-litellm-master-key}"
MODEL="${MODEL:-$(curl -s -m 5 "$URL/models" -H "Authorization: Bearer $KEY" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)}"
[[ -n "$MODEL" ]] || { echo "preserve-probe: no MODEL and autodetect failed at $URL" >&2; exit 1; }
echo "[probe] url=$URL  model=$MODEL"

Q1='Think of two different 4-digit numbers. Work them both out inside your reasoning. Then reply with ONLY the first number and nothing else — do not write, hint at, or restate the second number anywhere in your visible answer.'
Q2='What was the SECOND 4-digit number you thought of? Reply with just that number, or the single word UNKNOWN if you do not have it.'

post() {  # <json-body-file> -> response json on stdout
  curl -s -m 900 "$URL/chat/completions" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    --data-binary "@$1"
}

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# ---- turn 1 -----------------------------------------------------------------
python3 - "$MODEL" "$Q1" "${EFFORT:-}" > "$TMP/t1.json" <<'PY'
import json, sys
model, q1, effort = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": model, "messages": [{"role": "user", "content": q1}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 2000}
if effort:
    body["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": effort}
json.dump(body, sys.stdout)
PY
post "$TMP/t1.json" > "$TMP/r1.json"

read -r RC_LEN SHOWN < <(python3 - "$TMP/r1.json" "$TMP/a1.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
if "choices" not in r:
    print("ERR", json.dumps(r)[:300]); raise SystemExit
m = r["choices"][0]["message"]
rc = (m.get("reasoning_content") or m.get("reasoning") or "")
content = m.get("content") or ""
json.dump({"role": "assistant", "content": content, "reasoning_content": rc},
          open(sys.argv[2], "w", encoding="utf-8"))
print(len(rc), (content.strip().replace("\n", " ") or "<empty>")[:40])
PY
)
[[ "$RC_LEN" == "ERR" ]] && { echo "[probe] turn 1 failed: $SHOWN" >&2; exit 1; }
echo "[probe] turn 1: visible answer = $SHOWN   (reasoning_content: $RC_LEN chars)"
[[ "$RC_LEN" -gt 0 ]] || echo "[probe] ⚠ no reasoning returned — is reasoning parsing ON for this compose?"
python3 -c 'import json,sys; rc=json.load(open(sys.argv[1],encoding="utf-8"))["reasoning_content"]; print("[probe] reasoning excerpt:", " ".join(rc.split())[:220])' "$TMP/a1.json"

# ---- turn 2, both arms ------------------------------------------------------
for arm in with without; do
  python3 - "$MODEL" "$Q1" "$Q2" "$TMP/a1.json" "$arm" > "$TMP/t2.json" <<'PY'
import json, sys
model, q1, q2, a1path, arm = sys.argv[1:6]
a1 = json.load(open(a1path, encoding="utf-8"))
if arm == "without":
    a1 = {"role": "assistant", "content": a1["content"]}
json.dump({"model": model,
           "messages": [{"role": "user", "content": q1}, a1, {"role": "user", "content": q2}],
           "temperature": 0.0, "max_tokens": 2000}, sys.stdout)
PY
  post "$TMP/t2.json" > "$TMP/r2-$arm.json"
  python3 - "$TMP/r2-$arm.json" "$arm" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
if "choices" not in r:
    print(f"[probe] {sys.argv[2]:>7} arm: ERROR {json.dumps(r)[:200]}"); raise SystemExit
ans = (r["choices"][0]["message"].get("content") or "").strip().replace("\n", " ")
pt = r.get("usage", {}).get("prompt_tokens", "?")
print(f"[probe] {sys.argv[2]:>7} reasoning: prompt_tokens={pt:<6} answer={ans[:60]!r}")
PY
done

echo
echo "[probe] VERDICT — carryover is working iff the 'with' arm names a number that appears"
echo "        in the reasoning excerpt above AND the 'without' arm says UNKNOWN/guesses."
echo "        A prompt_tokens gap of roughly the reasoning length is the objective half."
