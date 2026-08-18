"""LiteLLM proxy pre-call hooks — per-model request fixups Hermes can't express itself.

(A) Qwen3-Omni (2026-07-09): its real context is 65536 ≈ Hermes' hard 64K minimum. Hermes
refuses models with context_length < 64K, so we advertise 65536 — but Hermes has no separate
max_tokens knob and uses context_length AS max_tokens, asking for the whole 64K window as
OUTPUT. That leaves 0 tokens for input, so every request (especially image/video, which Hermes
can't token-count) 400s — and the bad request crashes the omni EngineCore. litellm_params.
max_tokens is only a DEFAULT (client wins), so the cap must happen here (pre-call, mutating the
body). We also force modalities:["text"] so requests skip the crash-prone Talker/Code2Wav stage.

(B) Agents-A1 (2026-07-10): its edge is agentic thinking-ON (its own data: 8-pack ON 110 vs
OFF 105, cli-40 23/40), but its GENERATED vLLM compose hardcodes enable_thinking:false and is
"never hand-repaired". Hermes never sends chat_template_kwargs, so it'd always run in the weaker
OFF mode. We inject enable_thinking/preserve_thinking here — request-level kwargs override the
server default. (Caveat: the hermes-20 pack REGRESSES with thinking ON, 12→9 — right for
CLI/agentic, mixed for hermes-style. Flip AGENTS_A1_THINKING below if that bites.)

(C) Qwen3.6 preserve_thinking replay fix (2026-07-13): the Qwen3.6 vLLM composes wire
`preserve_thinking` correctly (server default + reasoning-parser + template), BUT vLLM v0.24.0
SILENTLY DROPS the top-level `reasoning_content`/`reasoning` field on incoming assistant
messages (vllm#38488). Proven on :8051 /tokenize: replaying prior reasoning as a FIELD → gone
from the rendered prompt (+4 tok empty <think> scaffold); replaying it INLINE as <think>…</think>
in `content` → preserved (+63 tok, recalled). So the ONLY replay form that survives is inline.
This hook re-inlines any assistant reasoning field back into content as <think>…</think> before
forwarding, so whatever the harness echoes (Hermes' reasoning_content, or a raw field) actually
reaches the template's fallback extractor and preserve_thinking keeps it. We do NOT touch the
enable/preserve chat_template_kwargs here — the co-located .env server default already carries
them, and forcing them risks flipping 27b's intended thinking-OFF. See memory
preserve-thinking-vllm-field-drop + NousResearch/hermes-agent#56004 (Hermes must also STOP
stripping reasoning on replay — the primary, harness-side half of this fix).

(D) Muse-Glimmer-30B reasoning effort (2026-08-10): Muse's effort control is the chat-template
variable `reasoning_strength` (low/medium/high/xhigh, template-default 'high'), NOT --reasoning
or enable_thinking — so neither the (B) path nor any server flag reaches it, and out of the box
the level is frozen at whatever the compose booted with. Two levers are wired here: an
OpenAI-style `reasoning_effort` on the request is mapped through (so a client's global effort
knob works IF it sends one), and failing that a model-ALIAS suffix is honoured
(`muse-glimmer-30b-low` / `-xhigh` → same :8210 backend, different level). The alias route is
the one that needs no client support at all — it turns "switch reasoning level" into "pick a
different model in the dropdown", the same trick qwen3.8-max / qwen3.8-max-nothink already use.
An explicit client-sent chat_template_kwargs.reasoning_strength always wins over both.

Only these models are touched; the big-context text models keep their full output budget.
"""
import json
import os
import re
import sys
from litellm.integrations.custom_logger import CustomLogger

OMNI_MAX_TOKENS = 8192

# Set CLUB3090_REASONING_DEBUG=1 in the litellm container env to log, per matched request,
# what reasoning shape the harness actually replayed (field vs inline vs none). This is the
# probe for hermes-agent#56004 — it tells us whether the harness is stripping reasoning
# before it ever reaches us. One compact line to stdout (docker logs litellm).
_REASON_DEBUG = os.environ.get("CLUB3090_REASONING_DEBUG", "") not in ("", "0", "false")

# Durable copy of the diagnostic lines. `docker logs` is per-CONTAINER, and this stack needs
# `down` + `up -d` to pick up edits to its bind-mounted files (a plain `restart` exits 127 on
# Docker-Desktop-for-WSL2 once the mount cache is stale) — so every redeploy wipes the log,
# including the evidence of whatever a client just did. Bind-mounted to services/litellm/logs.
_HOOK_LOG = "/app/logs/muse-effort.log"


def _hook_log(line):
    """Emit to stdout (docker logs) AND append to the bind-mounted file. Never raises —
    a diagnostic must not be able to take down a request."""
    print(line, file=sys.stdout, flush=True)
    try:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(_HOOK_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {line}\n")
    except Exception:
        pass


def _debug_reasoning_shape(model, messages):
    if not _REASON_DEBUG or not isinstance(messages, list):
        return
    field = inline = plain = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if (m.get("reasoning_content") or m.get("reasoning")):
            field += 1
        elif isinstance(m.get("content"), str) and "<think>" in m["content"]:
            inline += 1
        else:
            plain += 1
    print(f"[club3090/reasoning] model={model} assistant_turns: "
          f"field={field} inline={inline} plain={plain}", file=sys.stdout, flush=True)
AGENTS_A1_THINKING = True   # set False to let Agents-A1 use its compose default (thinking off)

# Qwen3.6 endpoints whose chat_template implements preserve_thinking by reading reasoning
# inline-from-content (the reasoning_content FIELD is dropped on replay — vLLM v0.24.0 per
# vllm#38488, and empirically the ik_llama apex lanes too). Substring match against the
# litellm model_name. "apex-35b" covers apex-35b-compact / -vision / -vision-ik.
# 2026-07-20: this is the CAPABILITY gate, not a default. Re-inline no longer means
# "always carry ALL prior <think>" (that's what seeded the a3b thought-loops). It now honors
# the launch-time mode serve.sh writes to preserve_state.json — off (default, no carryover)
# / full (--preserve) / window N (--preserve-window). So a3b/apex are BACK in the set: their
# capability is restored, but off-by-default, flag-driven. See _read_preserve_state below +
# memory preserve-thinking-vllm-field-drop.
QWEN_PRESERVE_MODELS = ("qwen3.6-27b", "qwen3.6-35b-a3b", "apex-35b")

# (D) Muse-Glimmer reasoning effort (2026-08-10) — see module docstring §D.
# Muse's effort knob is a CHAT-TEMPLATE VARIABLE (`reasoning_strength`), not a thinking
# on/off gate: its template renders "Reasoning strength: <level>." into the system block,
# defaulting to 'high' when unset. It is NOT --reasoning / enable_thinking, so nothing in
# the existing (B) path reaches it. Verified this hop end-to-end through the proxy:
# chat_template_kwargs passes through LiteLLM untouched (low → 104 chars of
# reasoning_content / 55 completion tokens; xhigh → 398 / 137).
# (E) Qwen3.8-27B reasoning effort (2026-08-18) — the llama.cpp lane at :8101.
# Qwen3.8's effort knob is the chat-template variable `reasoning_effort`. MEASURED against
# the running lane (prompt_tokens on a fixed 1-message body, thinking forced on):
#     low -> 41 tok · medium -> 11 tok (injects NOTHING) · xhigh -> 53 · high -> 53
# ⚠️⚠️ TWO measured facts drive everything here:
#  (1) A TOP-LEVEL OpenAI `reasoning_effort` IS INERT on this engine. low/minimal/none all
#      returned 200 at 13 prompt_tokens — byte-identical to sending nothing. llama.cpp does
#      NOT forward it into the template. So Hermes' global effort knob, which it sends on
#      every request, currently does NOTHING on this lane. Translating it here is what makes
#      the dial work AT ALL — this hook is not defensive polish, it is the feature.
#  (2) The template RAISES on an out-of-set value reaching it via chat_template_kwargs:
#      `minimal` / `none` -> HTTP 500 (jinja raise_exception). So the translation MUST clamp;
#      forwarding a client's raw string would convert an inert no-op into a hard failure.
# ⚠️ `high` is NOT a distinct level here. The GGUF's embedded template silently remaps
# high -> xhigh (measured: both render 53 tok), whereas the vLLM path RAISES on it. We map it
# explicitly so the behaviour is visible in this table rather than surprising inside jinja.
# ⚠️ THE SAME INERT-TOP-LEVEL TRAP APPLIES TO `preserve_thinking` (2026-08-18).
# Hermes' local qwen-preserve-thinking patch sends it as a TOP-LEVEL body field via
# `extra_body` — an x-unsloth extension that Unsloth Studio reads natively. llama.cpp does
# NOT: it only reads chat_template_kwargs. MEASURED on this lane against a 3-turn body
# carrying reasoning_content (server default preserve_thinking=true):
#     no field                                  -> 403 prompt_tokens
#     TOP-LEVEL preserve_thinking:false         -> 403  (INERT — silently ignored)
#     chat_template_kwargs preserve_thinking:false -> 38 (honoured)
# So Hermes' new toggle is a no-op here until translated, exactly like its effort picker.
# Unlike reasoning_effort this one cannot 500 (the template does `is true` / `is undefined`
# checks, never raise_exception), so the risk is silence, not failure — which is worse to
# diagnose. Ref: hermes-agent agent/preserve_thinking.py (PRESERVE_THINKING_FIELD).
QWEN38_MODELS = ("qwen3.8-27b",)
QWEN38_LEVELS = ("low", "medium", "xhigh")   # what the template actually accepts post-remap

# Normalised (lowercased, non-alphanumerics stripped) so "extra high" / "extra_high" /
# "x-high" all land together. Collapses at BOTH ends, like the Muse table: Qwen3.8 has no
# reasoning-off, so none/minimal floor to `low` — a user picking "thinking off" gets low
# thinking, not none. Turning thinking OFF is a different control (enable_thinking), not an
# effort level, and is left to the client.
QWEN38_EFFORT_MAP = {
    "none": "low", "minimal": "low", "low": "low",
    "medium": "medium", "normal": "medium",
    "high": "xhigh",          # ⚠️ no distinct 'high' on this template — see above
    "extrahigh": "xhigh", "xhigh": "xhigh", "max": "xhigh", "ultra": "xhigh",
}


def _norm_effort(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _as_bool(v):
    """Coerce a wire value to bool, or None if it is not recognisably boolean.

    Hermes sends a real JSON bool, but curl/scripts/gateways stringify freely, and a
    stray "false" string is truthy in Python — which would silently invert the toggle.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    return None


def _apply_qwen38_effort(data, model):
    """Translate an OpenAI-style reasoning_effort into chat_template_kwargs.reasoning_effort.

    Precedence (first match wins) — SPECIFIC beats GLOBAL, same rule as Muse §D:
      1. explicit chat_template_kwargs.reasoning_effort from the client — VALIDATED, never
         silently overridden. An out-of-set value is clamped rather than forwarded, because
         forwarding it is a 500 (measured), and a 500 is a worse answer than a clamp.
      2. top-level OpenAI reasoning_effort — mapped, then POPPED so the raw value can never
         reach the engine as an unknown field.
      3. nothing — leave the body alone so the server default from
         LLAMA_ARG_CHAT_TEMPLATE_KWARGS (REASONING_EFFORT, currently 'low') applies.

    ⚠️ Server-side kwargs MERGE per-key with request kwargs on llama.cpp b10236 (measured:
    a request sending only {enable_thinking:true} still inherited the server's
    reasoning_effort='low' — 41 tok, not the template-default xhigh's 53). So setting only
    the key we resolved is safe; we do NOT need to restate the server's other defaults.
    """
    ck = dict(data.get("chat_template_kwargs") or {})
    raw_top = data.pop("reasoning_effort", None)
    # preserve_thinking: same inert-top-level trap (see the header). Hermes' patch puts it
    # top level via extra_body; llama.cpp only reads chat_template_kwargs. Translate it.
    raw_preserve = data.pop("preserve_thinking", None)
    # `think` is Ollama heritage that Hermes emits alongside reasoning_effort:"none".
    # llama.cpp ignores it; popped as hygiene so a future engine bump cannot start 400ing.
    data.pop("think", None)

    chosen = source = None
    client_ck = ck.get("reasoning_effort")
    if client_ck is not None:
        mapped = QWEN38_EFFORT_MAP.get(_norm_effort(client_ck))
        if mapped is None:
            print(f"[club3090/qwen38] WARN unrecognised chat_template_kwargs.reasoning_effort="
                  f"{client_ck!r}; clamping to 'low' (template raises on unknown values)",
                  file=sys.stdout, flush=True)
            mapped = "low"
        chosen, source = mapped, f"chat_template_kwargs={client_ck}"
    elif raw_top is not None:
        mapped = QWEN38_EFFORT_MAP.get(_norm_effort(raw_top))
        if mapped is None:
            print(f"[club3090/qwen38] WARN unrecognised reasoning_effort={raw_top!r}; "
                  f"clamping to 'low'", file=sys.stdout, flush=True)
            mapped = "low"
        chosen, source = mapped, f"reasoning_effort={raw_top}"

    if chosen is not None:
        ck["reasoning_effort"] = chosen

    # preserve_thinking — an explicit client chat_template_kwargs value always wins; a
    # top-level field is translated into it; otherwise the server default stands.
    preserve = psource = None
    if "preserve_thinking" in ck:
        coerced = _as_bool(ck["preserve_thinking"])
        if coerced is None:
            print(f"[club3090/qwen38] WARN non-boolean chat_template_kwargs.preserve_thinking="
                  f"{ck['preserve_thinking']!r}; dropping it so the server default applies",
                  file=sys.stdout, flush=True)
            ck.pop("preserve_thinking", None)
        else:
            preserve, psource = coerced, f"chat_template_kwargs={ck['preserve_thinking']}"
            ck["preserve_thinking"] = coerced
    elif raw_preserve is not None:
        coerced = _as_bool(raw_preserve)
        if coerced is None:
            print(f"[club3090/qwen38] WARN non-boolean top-level preserve_thinking="
                  f"{raw_preserve!r}; ignoring (server default applies)",
                  file=sys.stdout, flush=True)
        else:
            preserve, psource = coerced, f"top-level={raw_preserve}"
            ck["preserve_thinking"] = coerced

    if ck:
        data["chat_template_kwargs"] = ck
    print(f"[club3090/qwen38] model={model} effort={chosen or '(server default)'} "
          f"via={source or 'none'} | preserve={preserve if preserve is not None else '(server default)'} "
          f"via={psource or 'none'}", file=sys.stdout, flush=True)
    return data


MUSE_MODELS = ("muse-glimmer",)
MUSE_LEVELS = ("low", "medium", "high", "xhigh")   # ALL the template accepts — only four

# reasoning_effort → Muse level. Muse has exactly FOUR levels, while clients expose more:
# Hermes' picker offers minimal / low / medium / high / extra high / max / ultra (seven), and
# OpenAI ships none / minimal / low / medium / high. So the ladder necessarily COLLAPSES at
# both ends: everything below 'low' floors to low, and extra-high / max / ultra all mean the
# same thing here — xhigh, because Muse has nothing above it. Picking 'ultra' does NOT buy
# more thinking than 'extra high'.
# Keys are NORMALISED (lowercased, non-alphanumerics stripped) so "extra high", "extra_high"
# and "x-high" all land on the same entry — a client's exact spelling is not knowable up front.
# 2026-08-10: 'ultra' was MISSING here and fell through to the server default, i.e. silently
# became 'high' — exactly the failure the previous version of this comment claimed to prevent.
# Any value not in this table now logs a loud WARN (see _apply_muse_effort) instead of
# vanishing, because a silent fallback is indistinguishable from a working knob.
MUSE_EFFORT_MAP = {
    "none": "low", "minimal": "low", "min": "low", "low": "low",
    "medium": "medium", "med": "medium", "moderate": "medium",
    "high": "high",
    "xhigh": "xhigh", "extrahigh": "xhigh", "veryhigh": "xhigh",
    "max": "xhigh", "maximum": "xhigh", "ultra": "xhigh", "highest": "xhigh",
}


def _norm_effort(v):
    """'Extra High' / 'extra_high' / 'x-high' -> 'extrahigh' / 'extrahigh' / 'xhigh'."""
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _muse_level_from_alias(model):
    """`muse-glimmer-30b-xhigh` -> 'xhigh'; bare alias -> None (server default wins).

    Longest-suffix-first so '-xhigh' cannot be shadowed by a '-high' prefix match.
    """
    for lvl in sorted(MUSE_LEVELS, key=len, reverse=True):
        if model.endswith("-" + lvl):
            return lvl
    return None


# Hermes fires at least one AUXILIARY call per visible turn (title generation,
# summarisation — agent/auxiliary_client.py) that carries NO reasoning field, so it lands on
# Muse's server default of `high`: a full high-effort reasoning pass to write an 8-word title.
# MEASURED 2026-08-11 on the title-shaped call (msgs=1, max_tokens=1500, non-streaming):
#   no effort field (today)  4.27 s   1095c reasoning   406 completion tokens
#   reasoning_effort=low     1.78 s    543c reasoning   191 completion tokens
# ⇒ ~2.5 s of dead latency per aux call, on every turn.
#
# ⚠ THE OBVIOUS FIX — "no effort field ⇒ low" — IS A TRAP, DO NOT DO IT. benchlocal-cli,
# curl, LM Studio, cron and the messaging gateways ALL send no reasoning_effort. A blanket
# rule silently downgrades every one of them, which would (a) invalidate the committed
# 8-pack numbers in dflash-vision.yml while looking like a model regression, and (b) be
# exactly the request-rewriting class AGENTS.md forbids shipping default-on.
#
# Nor can the two populations be told apart by fingerprint, contra the Hermes-side agent's
# suggestion: this hook's own log has `via=none msgs=2 stream=True max_tokens=None` — a
# STREAMING call with no effort field, matching neither the visible-turn shape
# (stream=True, max_tokens=65536) nor the title shape. The populations overlap.
#
# So this gates on the NARROW title shape only, and deliberately does NOT touch the msgs>=2
# aux population: those may include SUMMARISATION, whose output is fed back into the agent's
# own context, so quietly thinning it would degrade context quality in a way that is very
# hard to trace later. A title is cosmetic; a summary is not. The bounds below cannot collide
# with benchlocal (--max-tokens 4096/8192) or a visible turn (65536 / unset).
# Disable with CLUB3090_MUSE_AUX_LOW=0.
_AUX_LOW = os.environ.get("CLUB3090_MUSE_AUX_LOW", "1").strip().lower() not in ("0", "false", "no")
_AUX_MAX_TOKENS_CEILING = 2048


def _image_stats(data):
    """(image_part_count, total_decoded_MB) for the request's inline images.

    Exists to answer one recurring question: DID THE CLIENT DOWNSCALE BEFORE SENDING?
    A 30 MB file that arrives as 1.2 MB has been resized by the app, which means any
    "hi-res vision" conclusion drawn from that request is about the CLIENT, not the model.
    Measured directly from the base64 payload because the pre-call hook cannot see
    usage.prompt_tokens (that is a response field, and there is no post-call hook here).
    base64 inflates by 4/3, hence the * 3 / 4.
    """
    n, b = 0, 0
    for m in data.get("messages") or []:
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for part in c:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = ((part.get("image_url") or {}).get("url")) or ""
            n += 1
            if "base64," in url:
                b += len(url.split("base64,", 1)[1]) * 3 // 4
    return n, round(b / 2**20, 2)


def _is_aux_title_shaped(data):
    """True for a small, single-message, non-streaming bookkeeping call (title generation).

    Bounds rather than an equality test on 1500: that is Hermes' current constant, not a
    contract. Anything larger is assumed to be real work and left at the server default.
    """
    if data.get("stream"):
        return False
    if len(data.get("messages") or []) != 1:
        return False
    mt = data.get("max_tokens")
    return isinstance(mt, int) and 0 < mt <= _AUX_MAX_TOKENS_CEILING


def _apply_muse_effort(data, model):
    """Resolve Muse reasoning effort into chat_template_kwargs.reasoning_strength.

    Precedence (first match wins) — SPECIFIC beats GLOBAL:
      1. explicit chat_template_kwargs.reasoning_strength from the client — never overridden
      2. the model ALIAS suffix (muse-glimmer-30b-low / -xhigh) — a deliberate per-request
         pick from the model list
      3. OpenAI-style reasoning_effort on the request — mapped, then POPPED so it can't
         reach llama.cpp as an unknown field. ⚠ THIS IS NO LONGER ALWAYS A GLOBAL KNOB —
         see the note below rule 4.
      4. nothing — leave the body alone so the server's own --chat-template-kwargs default
         (REASONING_STRENGTH in the compose .env, currently 'high') applies

    ⚠ 2 MUST OUTRANK 3, and originally it didn't (fixed 2026-08-10 the same day). Hermes
    sends `reasoning_effort: high` (its global config knob) on EVERY main chat request, so
    with reasoning_effort ranked higher the -low and -xhigh aliases were both silently
    rewritten to 'high' and produced byte-comparable output — the aliases looked broken
    while behaving exactly as ordered. Observed in this hook's own debug line:
      model=muse-glimmer-30b-low   strength=high via=reasoning_effort=high
      model=muse-glimmer-30b-xhigh strength=high via=reasoning_effort=high
    A global default must never beat an explicit per-request selection. Net effect now:
    bare `muse-glimmer-30b` follows the app's effort knob; an alias PINS its level and
    ignores the knob.

    ⚠⚠ DON'T COMBINE AN ALIAS WITH HERMES' IN-CHAT PICKER (changed 2026-08-11).
    The Hermes desktop picker used to be inert for this provider — its selection never left
    the UI, so every request carried the GLOBAL `agent.reasoning_effort: high` and ranking
    the alias above it (rule 2 > rule 3) was unambiguously right. The Hermes-side agent then
    FIXED the picker: the desktop model menu was comparing a catalog slug against a billing
    class (`'litellm' === 'custom'`, never true for a user-defined provider), so it never
    made the `config.set` RPC. Its selection is now session-scoped and reaches the wire.
    ⇒ `reasoning_effort` can now be a genuine PER-CONVERSATION pick, and rule 2 still beats
    it — so selecting `muse-glimmer-30b-xhigh` AND moving the picker gives you xhigh, and the
    picker will look broken again. The precedence is deliberately unchanged (an alias exists
    to PIN a level; that is its whole job), so pick ONE mechanism per conversation:
      · Hermes desktop  → bare `muse-glimmer-30b` + the picker.
      · Anything whose effort control does NOT work (LM Studio, scripts, cron, the
        messaging gateways) → the -low/-medium/-high/-xhigh aliases.
    The hook's own debug line disambiguates: `via=model-alias (alias pinned; ignored global
    reasoning_effort=...)` means an alias ate a picker value.

    ⚠ On my earlier reading of the log being wrong: I tallied 13 log lines as flat requests
    and concluded the effort field was ABSENT on 8 of them. They are PAIRS — one streaming
    visible turn (which carried reasoning_effort EVERY time) plus one non-streaming auxiliary
    call (title/summary, via `agent/auxiliary_client.py`, which attaches reasoning only when
    its caller passes a reasoning_config). The symptom was never "nothing arrives"; it was
    "`high` arrives, always, regardless of the picker". The debug line below already recorded
    the stream/max_tokens fingerprint needed to separate the two populations — the aggregate
    tally is what got it wrong. Fingerprint them, don't count them.
    """
    # Snapshot the reasoning-ish keys BEFORE any mutation — this line's whole job is to
    # reveal an unrecognised wire shape from a new client, so it must see the body as sent.
    _seen = sorted(k for k in data
                   if "reason" in k.lower() or "think" in k.lower() or "effort" in k.lower())
    ck = dict(data.get("chat_template_kwargs") or {})
    effort = data.get("reasoning_effort")
    # (2) always consume reasoning_effort even if we end up not using it — llama.cpp has no
    # use for it and forwarding unknown top-level fields is how 400s start.
    if effort is not None:
        data.pop("reasoning_effort", None)
    # Same treatment for `think`, which ONLY appears on a "thinking off" selection: Hermes'
    # `custom` provider profile emits reasoning_effort:"none" PLUS a top-level think:false
    # (Ollama heritage — ollama#14820, where /v1/chat/completions ignores `think` and only
    # /api/chat honours it). MEASURED 2026-08-11 against this build: llama.cpp ignores it
    # outright — think absent / false / true all return HTTP 200 with byte-identical
    # reasoning length — so this is hygiene, NOT a live bugfix. Popped anyway because it is
    # an unrecognised top-level field that only a future engine bump has to start
    # validating. Reported by the Hermes-side agent, who found the emit path we can't see.
    # ⚠ Muse has NO reasoning-off, so "none" maps to `low` — a user who picks "thinking
    # off" silently gets low. That is a real semantic mismatch between the two systems and
    # the only possible behaviour here, not a bug.
    thinking = data.pop("think", None)
    chosen = source = None
    alias = _muse_level_from_alias(model)
    if ck.get("reasoning_strength"):
        # VALIDATE, don't trust. The template does NO validation of this variable: it renders
        # "Reasoning strength: <whatever>." straight into the system prompt, so a client
        # sending "max" or "ultra" here would inject an UNTRAINED value the model never saw
        # in training. Only the four real levels may reach the prompt; anything else is
        # normalised through the same map, and if even that fails we drop it so the server
        # default applies. (Credit: r/LocalLLM 1vkr0vr, which flagged the no-validation
        # behaviour; our own map already normalised the reasoning_effort path but this
        # client-supplied path was passing through verbatim.)
        raw = str(ck["reasoning_strength"])
        if raw in MUSE_LEVELS:
            chosen, source = raw, "client-kwargs"
        elif _norm_effort(raw) in MUSE_EFFORT_MAP:
            chosen = MUSE_EFFORT_MAP[_norm_effort(raw)]
            source = f"client-kwargs={raw}->normalised"
        else:
            ck.pop("reasoning_strength", None)
            data["chat_template_kwargs"] = ck
            _hook_log(f"[club3090][muse-effort][WARN] client sent unknown "
                      f"reasoning_strength={raw!r} — DROPPED (would render an untrained "
                      f"value into the system prompt). Valid: {MUSE_LEVELS}")
    elif alias:
        # Explicit per-request pick — outranks any global effort knob (see docstring).
        chosen, source = alias, "model-alias"
        if isinstance(effort, str) and effort.strip():
            source += f" (alias pinned; ignored global reasoning_effort={effort})"
    elif isinstance(effort, str) and effort.strip():
        key = _norm_effort(effort)
        if key in MUSE_EFFORT_MAP:
            chosen, source = MUSE_EFFORT_MAP[key], f"reasoning_effort={effort}"
        else:
            # UNCONDITIONAL warn (not gated on _REASON_DEBUG): an effort level we cannot map
            # would otherwise fall through to the server default and look like a working knob.
            # If this fires, add the spelling to MUSE_EFFORT_MAP.
            _hook_log(f"[club3090][muse-effort][WARN] unmapped reasoning_effort={effort!r} "
                      f"(normalised {key!r}) — falling back to the server default. "
                      f"Muse accepts only {MUSE_LEVELS}; add this spelling to MUSE_EFFORT_MAP.")
    if chosen is None and _AUX_LOW and _is_aux_title_shaped(data):
        chosen, source = "low", "aux-title-default"
    if chosen:
        ck["reasoning_strength"] = chosen
        data["chat_template_kwargs"] = ck
    if _REASON_DEBUG:
        # Ground truth for "does the app send an effort field at all?" — the ONLY way to
        # learn a harness's wire shape is to look at a real request from it. If a client's
        # effort control does nothing and this logs request_keys=[], the client sent NOTHING
        # and no proxy-side mapping can help; if it logs an unhandled key, add it above.
        # msgs/stream/max_tokens fingerprint the CALL, not just the effort: a harness often
        # fires a secondary request per turn (title/summary) that carries different fields,
        # and "which of the two is the visible answer?" is unanswerable without this.
        # The visible chat turn is the streaming, many-message, big-max_tokens one.
        _msgs = data.get("messages") or []
        _nimg, _imgmb = _image_stats(data)
        _hook_log(f"[club3090][muse-effort] model={model} strength={chosen or '(server default)'} "
                  f"via={source or 'none'} request_keys={_seen} "
                  f"think={thinking!r} "
                  f"msgs={len(_msgs)} stream={bool(data.get('stream'))} "
                  f"max_tokens={data.get('max_tokens')} "
                  f"imgs={_nimg} img_mb={_imgmb}")

# serve.sh writes the ACTIVE model's cross-turn preserve mode here on every launch (GPU-mutex
# → one live model, so a single global file is unambiguous). Mounted read-only into the
# container by services/litellm/docker-compose.yml. Missing / malformed → off (loop-safe).
PRESERVE_STATE_PATH = os.environ.get("CLUB3090_PRESERVE_STATE", "/app/preserve_state.json")


def _read_preserve_state():
    """(mode, window) serve.sh recorded for the live model. Cheap enough to read per call."""
    try:
        with open(PRESERVE_STATE_PATH, encoding="utf-8") as f:
            s = json.load(f)
        mode = str(s.get("mode", "off")).lower()
        if mode not in ("off", "full", "window"):
            mode = "off"
        window = int(s.get("window", 0) or 0)
        return mode, window
    except Exception:
        return "off", 0


def _content_text(content):
    """Flatten str / multimodal-list content to text for role/tool-response detection."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _is_real_user(m):
    """A genuine user query — NOT a tool_response echoed back as a user turn. Mirrors the
    chat template's window pre-pass so hook-side and template-side windows count identically."""
    if not isinstance(m, dict) or m.get("role") != "user":
        return False
    t = _content_text(m.get("content")).strip()
    return not (t.startswith("<tool_response>") and t.endswith("</tool_response>"))


def _apply_preserve(messages, mode, window):
    """Cross-turn <think> carryover at the DURABLE (content) layer — the reasoning_content
    field is dropped on replay, so inlining into content is the only form that survives.
      off    → strip all prior reasoning (no carryover; the loop-safe default)
      full   → re-inline every assistant turn's reasoning as <think> in content
      window → re-inline only assistant turns AFTER the Nth-from-last real user query
    The field is always popped (it's discarded downstream anyway); mode only decides whether
    we first move it into content. Idempotent w.r.t. content that already carries <think>."""
    if not isinstance(messages, list):
        return
    # keep_from: inline reasoning only for assistant messages at index > keep_from.
    if mode == "full":
        keep_from = -1
    elif mode == "window" and window >= 1:
        real_users = [i for i, m in enumerate(messages) if _is_real_user(m)]
        keep_from = -1 if window >= len(real_users) else real_users[-window]
    else:  # off (or a degenerate window < 1)
        keep_from = len(messages)                       # nothing qualifies → strip all
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content") or m.get("reasoning")
        # Always strip the (about-to-be-dropped) field so it can't linger ambiguously.
        m.pop("reasoning_content", None)
        m.pop("reasoning", None)
        if i <= keep_from:
            continue                                    # out of window → drop, don't inline
        if not isinstance(rc, str) or not rc.strip():
            continue
        block = "<think>\n" + rc.strip() + "\n</think>\n\n"
        content = m.get("content")
        if content is None or content == "":
            m["content"] = block.rstrip()
        elif isinstance(content, str):
            if "<think>" not in content:      # don't double-wrap an already-inline turn
                m["content"] = block + content
        elif isinstance(content, list):
            # Multimodal assistant content: prepend a text part unless one already has <think>.
            has_think = any(
                isinstance(p, dict) and "<think>" in str(p.get("text", "")) for p in content
            )
            if not has_think:
                m["content"] = [{"type": "text", "text": block}] + content
        # any other content type: leave untouched (defensive)


class MaxTokensCap(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = str(data.get("model", ""))
        if "omni" in model:
            # (1) Cap output so there is room for image/video input (see module docstring).
            mt = data.get("max_tokens")
            if mt is None or mt > OMNI_MAX_TOKENS:
                data["max_tokens"] = OMNI_MAX_TOKENS
            # (2) Force text-only output — dodge the omni audio-stage bf16 crash.
            data["modalities"] = ["text"]
        elif "agents-a1" in model and AGENTS_A1_THINKING:
            # Flip Agents-A1 into its intended thinking-ON mode (compose hardcodes it off).
            # Merge, don't clobber, so an explicit client value still wins.
            # 2026-07-20: preserve_thinking NO LONGER forced on — replaying prior <think> on
            # this a3b MoE seeds thought-loops (Paul). enable_thinking stays (its agentic
            # edge); preserve defaults off like the other a3b lanes. A client can still opt in.
            ck = dict(data.get("chat_template_kwargs") or {})
            ck.setdefault("enable_thinking", True)
            ck.setdefault("preserve_thinking", False)
            data["chat_template_kwargs"] = ck
        elif any(tag in model for tag in QWEN38_MODELS):
            # (E) Qwen3.8's effort knob only responds to chat_template_kwargs; a top-level
            # reasoning_effort is inert on llama.cpp. Translate + clamp so the dial works.
            _apply_qwen38_effort(data, model)
        elif any(tag in model for tag in MUSE_MODELS):
            # (D) Resolve Muse's reasoning_strength template var per request, so effort is
            # switchable from the client instead of needing a compose reboot.
            _apply_muse_effort(data, model)
        if any(tag in model for tag in QWEN_PRESERVE_MODELS):
            # (C) Carry prior <think> across turns at the durable layer, honoring the launch-time
            # mode (off/full/window) serve.sh recorded — vLLM/ik drop the reasoning FIELD, so
            # inline-into-content is the only replay form that survives to the template.
            mode, window = _read_preserve_state()
            _debug_reasoning_shape(model, data.get("messages"))   # probe: what did the harness send?
            _apply_preserve(data.get("messages"), mode, window)
        return data


proxy_handler_instance = MaxTokensCap()
