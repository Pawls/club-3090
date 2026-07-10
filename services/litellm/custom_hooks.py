"""LiteLLM proxy pre-call hook — cap max_tokens for Qwen3-Omni.

Why this exists (2026-07-09): Qwen3-Omni's real context is 65536, which is essentially
Hermes' hard 64K minimum. Hermes refuses any model with context_length < 64K, so we MUST
advertise 65536 — but Hermes has no separate max_tokens knob and uses context_length AS
max_tokens, asking for the whole 64K window as OUTPUT. That leaves 0 tokens for input, so
every request (especially image/video, which Hermes can't token-count) 400s — and the bad
request crashes the omni EngineCore.

LiteLLM's litellm_params.max_tokens is only a DEFAULT (client value wins), so the cap has to
happen here in a pre-call hook, which mutates the request body before it reaches the engine.
Omni analysis output is short, so 8192 is plenty and leaves ~57K for image/video input.

Only omni is capped — the big-context text models (apex/27b/gemma/deckard) keep their full
output budget.
"""
from litellm.integrations.custom_logger import CustomLogger

OMNI_MAX_TOKENS = 8192


class MaxTokensCap(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        model = str(data.get("model", ""))
        if "omni" in model:
            # (1) Cap output so there is room for image/video input (see module docstring).
            mt = data.get("max_tokens")
            if mt is None or mt > OMNI_MAX_TOKENS:
                data["max_tokens"] = OMNI_MAX_TOKENS
            # (2) Force text-only output. Hermes never sends `modalities`, so requests would
            # route through Qwen3-Omni's Talker/Code2Wav (audio) stage — which has a bf16/dtype
            # bug that CRASHES the EngineCore (needs down/up to recover). We want text out anyway
            # for analysis, so pin it here for every omni request.
            data["modalities"] = ["text"]
        return data


proxy_handler_instance = MaxTokensCap()
