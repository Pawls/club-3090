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

Only these models are touched; the big-context text models keep their full output budget.
"""
import os
import sys
from litellm.integrations.custom_logger import CustomLogger

OMNI_MAX_TOKENS = 8192

# Set CLUB3090_REASONING_DEBUG=1 in the litellm container env to log, per matched request,
# what reasoning shape the harness actually replayed (field vs inline vs none). This is the
# probe for hermes-agent#56004 — it tells us whether the harness is stripping reasoning
# before it ever reaches us. One compact line to stdout (docker logs litellm).
_REASON_DEBUG = os.environ.get("CLUB3090_REASONING_DEBUG", "") not in ("", "0", "false")


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

# Qwen3.6 vLLM endpoints whose embedded chat_template implements preserve_thinking by
# reading reasoning inline-from-content (the field is dropped by vLLM). Substring match
# against the litellm model_name (e.g. "qwen3.6-35b-a3b-autoround", "qwen3.6-27b").
QWEN_PRESERVE_MODELS = ("qwen3.6-27b", "qwen3.6-35b-a3b")


def _reinline_reasoning(messages):
    """Move any assistant `reasoning_content`/`reasoning` field into content as an inline
    <think>…</think> block, then drop the field (vLLM discards it anyway). Idempotent:
    skips messages whose content already contains a <think> block."""
    if not isinstance(messages, list):
        return
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        rc = m.get("reasoning_content") or m.get("reasoning")
        # Always strip the (about-to-be-dropped) field so it can't linger ambiguously.
        m.pop("reasoning_content", None)
        m.pop("reasoning", None)
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
            ck = dict(data.get("chat_template_kwargs") or {})
            ck.setdefault("enable_thinking", True)
            ck.setdefault("preserve_thinking", True)
            data["chat_template_kwargs"] = ck
        if any(tag in model for tag in QWEN_PRESERVE_MODELS):
            # (C) Re-inline replayed reasoning so vLLM's field-drop doesn't defeat preserve_thinking.
            _debug_reasoning_shape(model, data.get("messages"))   # probe: what did the harness send?
            _reinline_reasoning(data.get("messages"))
        return data


proxy_handler_instance = MaxTokensCap()
