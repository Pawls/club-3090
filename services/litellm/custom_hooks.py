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

Only these two models are touched; the big-context text models keep their full output budget.
"""
from litellm.integrations.custom_logger import CustomLogger

OMNI_MAX_TOKENS = 8192
AGENTS_A1_THINKING = True   # set False to let Agents-A1 use its compose default (thinking off)


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
        return data


proxy_handler_instance = MaxTokensCap()
