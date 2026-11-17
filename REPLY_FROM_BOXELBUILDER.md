# Answers — tool-call repair questions from club-3090

TL;DR: BoxelBuilder's in-editor agent (`agent_llm.py` + `agent_session.py` + `window_agent.py`)
is a hand-rolled, non-SDK HTTP client (raw `urllib.request`, manual SSE parsing) supporting two
wire dialects — Anthropic `/v1/messages` and any OpenAI-compatible `/chat/completions` — always
streaming, with per-index tool-call fragment assembly. JSON parse failure on `arguments` is
isolated into a `(args, parse_error)` tuple, never raised, and fed back to the model as an
ordinary tool-result message so it can self-correct — same for unknown-tool names and any
tool-execution exception. There is no JSON-schema validation library and no
`<tool_call>`/`<function=...>`-style text-markup parsing anywhere; both dialects read only
provider-native structured tool-call fields. `boxel_mcp.py` (the MCP server) never itself calls
an LLM — it's a pure tool host for an *external* client (Claude Code/Desktop/Cursor) — so all of
this repair/loop-guard logic is specific to the in-app panel and irrelevant to that path.

### A. Connection to the model

**1. Client config.** Raw `urllib.request` (no SDK/requests/httpx). Provider chosen by
`agent_llm.make_provider(cfg)`: `AnthropicProvider` if `cfg["provider"]=="anthropic"` else
`OpenAICompatProvider` (agent_llm.py:817-832). Request built in `Provider._request`
(agent_llm.py:252-330, `urllib.request.Request(...)` at agent_llm.py:311-319). Base URL from
the active profile's `base_url` (Anthropic falls back to hardcoded `ANTHROPIC_URL`,
agent_llm.py:459). Model name `cfg.get("model","")`. API key + profile fields live in
gitignored `ai_config.json` at repo root (tools/ai_config.py:29 `CONFIG_PATH`), env-overridable
via `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`BOXEL_AI_BASE_URL`/`BOXEL_AI_MODEL`
(tools/ai_config.py:8-10). `PROVIDER_PRESETS` (tools/ai_config.py:57+) gives canned profiles
for anthropic/openai/openrouter/lmstudio/ollama.

**2. Endpoint / streaming.** Anthropic: POST `{base_url or ANTHROPIC_URL}/v1/messages`.
OpenAI-compat: POST `{base_url}/chat/completions`. **Streaming is always on, not
configurable** — both `stream_turn` implementations hardcode `"stream": True`. No
non-streaming path exists.

**3. Request fields.** `tools` sent provider-shaped, derived from one agnostic
`[{"name","description","schema"}]` list. `tool_choice` and `parallel_tool_calls`: **not
found** anywhere — never sent. `max_tokens`: `None` = "auto" → omitted on OpenAI-compat;
Anthropic requires it so falls back to `ANTHROPIC_AUTO_MAX_TOKENS` (agent_llm.py:492-494).
`temperature`: only sent when not `None` (Provider._sampling, agent_llm.py:337-342).
Reasoning controls exist only on OpenAI-compat (`OpenAICompatProvider._sampling`,
agent_llm.py:584+): top-level `reasoning_effort` (clamped), `chat_template_kwargs:
{"enable_thinking": false}` when off, `preserve_thinking` sent both places. Anthropic gets
none of these.

**4. Multiple paths — yes, no overlap.** The in-app panel is the only code path that makes
an outbound LLM HTTP call. `boxel_mcp.py` is a FastMCP stdio server for an **external** MCP
client (Claude Desktop/Code/Cursor); grep for `agent_llm|stream_turn|urlopen|requests.post|
httpx` in boxel_mcp.py → zero matches. It only executes tool calls handed to it
(boxel_mcp.py:204). **So everything below about parse-error isolation, loop guards,
unknown-tool handling and the max_tokens rescue is specific to the in-editor panel and has no
counterpart on the MCP path — that's the external agent's problem, not Boxel's.**

### B. How tool calls are received

**5. Parsing.** Both providers parse only **structured** tool-call fields — no text-markup
scanning anywhere. Repo-wide grep for `<tool_call>`, `<function=`, `function_call`, `<|tool`
→ zero matches in any .py file. Anthropic: SSE `content_block_start`/`_delta`
(`input_json_delta`)/`_stop` accumulate a `tool_use` block's `partial_json`, parsed at
block-stop (agent_llm.py:558-566). OpenAI-compat: `choice.delta.tool_calls[]` accumulated
per-`index` into `pending[idx]`, parsed at stream end. One unrelated text parser exists —
`ThinkSplitter` strips leaked `<think>...</think>` from OpenAI-compat content (reasoning
leakage, not tool calls).

**6. Fragment assembly.** By **index**, not id, in both dialects (`blocks`/`pending` dicts
keyed by SSE/delta index). Fragments append to a list and are joined only once the
block/stream ends, then parsed whole — never parsed mid-stream.

**7. String→object parsing.** `agent_llm._parse_args(raw)` (agent_llm.py:227-239): empty →
`({}, "")`; `json.loads` `ValueError` → `({}, f"tool arguments were not valid JSON ({e}):
{raw[:200]}")` — **never raises**; non-dict JSON also becomes a parse error. Stored on
`ToolCall.parse_error`. `AgentSession._marshal_calls` (agent_session.py:405-436) checks
`if call.parse_error:` first (agent_session.py:418-419) and short-circuits **before the
executor runs**, appending `{"ok": False, "error": call.parse_error}` as the tool result.
Test: tests/test_agent_session.py:175-181.

**8. finish_reason.** Read/normalized in both dialects (Anthropic `stop_reason`; OpenAI
`finish_reason` mapped, `"length"`→`"max_tokens"`). Truncation handled explicitly:
`AgentSession._run_steps` (agent_session.py:349-370) — on `stop == "max_tokens"` with no
tool calls and no prior resume this turn, keeps partial text, appends a note + synthetic
user message asking the model to finish without restarting, retries **once per turn**.

### C. Existing repair / validation

**11 (full inventory, all engine-aware, none generic wire-format repair):**
- `agent_llm._parse_args` — malformed JSON → non-raising `(args, parse_error)`.
- `AgentSession._marshal_calls` parse-error short-circuit (agent_session.py:418-419).
- `AgentSession._run_steps` max_tokens rescue (agent_session.py:349-370), one-shot/turn.
- `AgentSession._trim_history` context-error retry (agent_session.py:333-337), one-shot/turn.
- `window_agent._agent_execute` unknown-tool check (window_agent.py:710-714).
- `window_agent._agent_execute` withheld-tool check (window_agent.py:715-728) — distinct
  "switched off" error vs. unknown.
- `window_agent._agent_run_tool` bad-args/exception catch (window_agent.py:753-760) —
  `TypeError` → "bad arguments for {name}: {e}"; broad `Exception` → "{type}: {e}". Comment:
  "tool failures are DATA to the model, not crashes."
- `window_agent._agent_loop_guard` (window_agent.py:1121-1154) — refuses 3rd consecutive
  identical `(name, args)` call (polling tools exempt).
- `window_agent._agent_repeat_note` (window_agent.py:1156-1192) — flags a non-consecutive
  repeat with a byte-identical result.
- `_agent_encode` (window_agent.py:1194-1198) — fallback for non-JSON-encodable tool
  *return values* (sanitizes output, not model input).
- `gamekit_maps._script_hint` (gamekit_maps.py:114-139) — on TypeError/AttributeError in the
  sandboxed `run_script` tool, appends the real signature or a difflib-nearest name. Coaching,
  not JSON repair.
- No hits for `heal`, `jsonrepair`, or any schema-validation library.

**12. Schema validation before dispatch.** **Not found.** No `jsonschema`/`validate(...)`
tied to tool args anywhere. Correctness enforced only by (a) `_parse_args` requiring a JSON
object and (b) Python's own `TypeError` at `entry.fn(**call.args)` (window_agent.py:754).

**13. Unknown tool name.** `window_agent._agent_execute` (window_agent.py:706-714):
`TOOLS_BY_NAME.get(call.name)` is `None`/not in-editor →
`{"ok": False, "error": f"unknown tool {call.name!r} — use only the listed tools"}`, same
tool-result round-trip as any other error. Test: tests/test_agent_approvals.py:121-125.

### D. The turn loop

**14. Location.** `AgentSession._run_steps` (agent_session.py:289-403): drain steering
notes → compact history if needed → `provider.stream_turn(...)` → collect
text/reasoning/tool_calls/stop → `_marshal_calls` (agent_session.py:405-436) enqueues calls
for the main/GL thread via `window_agent._agent_execute` (through `pump_tools`,
agent_session.py:229-262), blocks for results, appends `tool_result_msgs(...)` to history,
loops back to the model.

**15. Failure → model, not just UI.** Always back to the model as
`{"ok": false, "error": ...}` tool-result JSON (window_agent.py:755-760) →
`tool_result_msgs` → history for the next call (agent_session.py:395). A single tool failure
never aborts the turn. Test: tests/test_agent_session.py:184-189.

**16. Cap / reset.** `MAX_STEPS_PER_TURN = 200` enforced by `for _step in range(max_steps)`;
hitting it ends the turn, history preserved. Loop-guard counter resets to 1 on any
differently-shaped call (window_agent.py:1139) — not specifically "on success." The
max_tokens/context-trim rescues are one-shot per turn, reset by starting a new turn.

**17. Tool defs.** Static: `agent_tools.py` registers ~91 functions into
`TOOLS_BY_NAME` (agent_tools.py:262), schemas derived via `inspect.signature` + docstrings,
never discovered dynamically. `agent_toolsets.py` layers a session/profile *selection* on top
of that same registry. `boxel_mcp.py` re-wraps the same entries (boxel_mcp.py:204) — one
source of truth for both surfaces.

**18. Tool-call ids.** Anthropic: always the API's own `content_block.id`. OpenAI-compat:
the API's `tc["id"]` when present, else a **locally generated `f"call_{idx}"`** (a
stream-local index counter, not uuid/hash) when the server omits it.

### E. Evidence

**19. Logging.** No raw request/response debug log in agent_llm.py. `_agent_save_chat`
persists history to gitignored `agent_chat.json`; `agent_corpus.py` writes per-conversation
run dirs, gated off by default. **No real captured leaked-`<tool_call>` or unknown-tool
example found** — only synthetic fixtures in unit tests (tests/test_agent_llm.py:337,
tests/test_agent_session.py:175-181, tests/test_agent_approvals.py:121-125). Corpus/
transcript files are gitignored and absent from this checkout, so real-world failure
frequency **cannot be answered from this repo** — treat as "not found," not "zero."

**20. Other notes.** All repair here is engine-aware (references live tool registry/session
state); none is generic wire-format repair, and a proxy sitting in front would only ever see
already-well-formed requests this code builds itself. On the response side, `_parse_args`'s
non-raising contract already gives the model a self-correction signal — if your proxy repairs
malformed local-model tool-call JSON before it reaches this harness, either (a) only promote
genuinely complete/parseable calls (matches your proposed split) and leave malformed ones
alone so `_parse_args` still reports them as today, or (b) if you do normalize, preserve an
equivalent "your last call had a problem" signal or the loop-guard/repeat-note logic loses
information. Both dialects are hand-rolled against raw SSE (`iter_sse`, agent_llm.py:166-189)
— a proxy must still emit one of the two exact `text/event-stream` shapes this parser
understands; there's no abstraction layer here for a third dialect. `tool_choice`/
`parallel_tool_calls` are never sent — proxy-side repair should not assume a forced
tool-choice mode is in play.

**Key files:** `agent_llm.py`, `agent_session.py`, `window_agent.py`, `agent_tools.py`,
`agent_toolsets.py`, `boxel_mcp.py` (MCP server, no LLM calls), `tools/ai_config.py`,
`agent_corpus.py`, `gamekit_maps.py:114-139`, `docs/architecture/editor-ai-agent.md`.
