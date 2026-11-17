# Tool call repair — how Hermes (hermes-agent) does it

Researched 2026-09-12 from `HERMES_HOME\hermes-agent` (the `local` branch checkout).
Goal: extract the pattern so it can be re-taught to the Club-3090 agent for repairing
tool calls on LiteLLM's side too. Unsloth's side is a separate follow-up doc.

Hermes treats "tool call repair" as **five independent layers**, each catching a
different class of malformation a local/open-weight model produces. They compose in
this order on the request/response path:

```
provider stream/response
   │
   ▼
1. Argument-JSON repair        (message_sanitization._repair_tool_call_arguments)
   ▼
2. Tool-name repair             (agent_runtime_helpers.repair_tool_call)
   ▼
3. Turn-level validation/retry  (turn_tool_validation.validate_tool_calls)
   ▼
4. Arg type coercion            (tools/arg_coercion.coerce_tool_args)   ← at dispatch time
   ▼
5. Message-sequence repair      (agent_runtime_helpers.repair_message_sequence) ← pre-send belt
```

## 1. Argument-JSON repair — `agent/message_sanitization.py:_repair_tool_call_arguments`

Fixes malformed JSON text in `tool_call.function.arguments` before it's parsed. Pure
function, `raw_args -> repaired_json_str`, never raises — worst case returns `"{}"`.
Pipeline, cheapest/most-common fix first:

1. Empty string or literal `"None"` (Python repr leaking through) → `"{}"`.
2. **Pass 0**: `json.loads(raw, strict=False)` then re-`dumps`. `strict=False` accepts
   literal control characters inside JSON strings — the most common local-model
   malformation (llama.cpp-family output). If this round-trips, done.
3. **Passes 1–3**: regex-strip trailing commas before `}`/`]`; auto-close unclosed
   `{`/`[` by counting brackets; if still unparseable, trim excess trailing closers one
   char at a time (bounded to 50 iterations) until it parses or gives up.
4. **Pass 4**: manually walk the string char-by-char tracking whether we're inside a
   JSON string, and escape any raw control char (`0x00`-`0x1F`) found inside a string as
   `\uXXXX` — handles control chars *combined with* other malformations (trailing
   commas etc.) that pass 0 alone can't fix. Retry parse.
5. Unrepairable → log the **full** raw content at WARNING (bounded to 100k chars — this
   is called out explicitly as "the last surviving copy of content that can hold real
   user data, e.g. a truncated `write_file`") and return `"{}"` so the request can still
   go out rather than crash.

Every repair step logs at WARNING with a before/after snippet, so this is fully
observable in logs without special instrumentation.

Called from two places:
- **Streaming assembly** (`agent/chat_completion_helpers.py:_assemble_tool_calls`,
  ~line 2884): after accumulating a streamed tool call, if `json.loads` on the raw
  arguments fails, repair is attempted; if it still comes back `"{}"` the call is
  flagged `has_truncated_tool_args` (see layer 3 for what that triggers downstream).
- **Send-path canonicalization** (`agent/conversation_loop.py:_canonicalize_api_tool_calls`,
  ~line 928): every time history is re-sent, tool-call args are canonicalized
  (`json.dumps(..., sort_keys=True)`) for stable prompt-cache prefixes; canonicalization
  raises on malformed input, and the `except` falls back to full repair. Repairs are
  applied to a copy-on-write send-path clone, never mutating the persisted transcript.

## 2. Tool-name repair — `agent/agent_runtime_helpers.py:repair_tool_call`

Fixes a hallucinated/mismatched tool **name** (not args) before rejecting the call.
Pure string function, `(agent, tool_name) -> repaired_name | None`. Tried in order,
first hit wins:

1. Trim at the first stray XML/quote delimiter (`"`, `'`, `<`, `>`) — some backends
   (VolcEngine noted specifically) leak XML attribute fragments into the name field,
   e.g. `terminal" parameter="command" string="true`. Deliberately does **not** split
   on whitespace, because a legitimate `"write file"` must still normalize to
   `write_file`.
2. Fast path: lowercase match, then normalize (`lower()`, `-`→`_`, space→`_`) match.
3. Build a candidate set from `{raw, lowered, normalized, CamelCase→snake_case}`, then
   for up to 2 rounds strip a trailing `_tool`/`-tool`/`tool` suffix and re-normalize
   each variant (handles doubled suffixes like `TodoTool_tool`).
4. Last resort: `difflib.get_close_matches(lowered, valid_tool_names, n=1, cutoff=0.7)`
   — fuzzy match.

Returns `None` if nothing clears the 0.7 cutoff, meaning the call is genuinely invalid
(see layer 3).

## 3. Turn-level validation/retry — `agent/turn_tool_validation.py:validate_tool_calls`

This is the orchestrator that decides what happens to a whole batch of tool calls in
one assistant turn, using layers 1 and 2 as primitives. Order of operations:

1. **Uniquify duplicate ids first** (`agent._uniquify_tool_call_ids`, backed by
   `message_sanitization.uniquify_tool_call_ids`) — some providers reuse one id across a
   batch; downstream pairing code keeps only the first call/result per id, so dedup must
   happen before anything else touches ids. Collisions get a deterministic `<id>_d<n>`
   suffix (never `uuid4` — ids feed prompt-cache prefixes, so they must stay
   byte-identical across replays of the same input).
2. **Repair tool names** (layer 2) for any name not in `agent.valid_tool_names`.
3. **Mixed-batch handling**: if a batch has *some* valid and *some* still-invalid names
   after repair, only the invalid calls get a tool-role error result
   (`_invalid_tool_name_error_content` — a blank name gets a terse "don't re-emit tool
   syntax as data" message to avoid feeding a priming loop; a nonempty wrong name gets
   the full tool catalog so the model can self-correct); the valid calls still execute.
   This does **not** advance the invalid-call strike counter — a real error is not the
   model being degenerate.
4. **All-invalid handling**: strike counter (`agent._invalid_tool_retries`) advances,
   capped at 3. Under 3, an error tool-result is appended per call and the turn is
   retried (`action: "continue"`). At 3, it's a terminal partial exit that also closes
   any dangling tool-result tail so the next turn isn't `tool → user` (role-alternation
   violation — strict providers like Gemini/Claude hallucinate a continuation to "fix"
   that themselves rather than erroring, which is worse).
5. **Argument JSON validation** (separate from repair — this runs even after layer 1 may
   have already fixed things in the streaming path, as a second checkpoint): dict/list
   args get re-serialized to string; empty/whitespace args become `"{}"`; anything else
   gets `json.loads` tried. On failure:
   - If the raw arguments don't end in `}`/`]` after stripping, it's treated as a
     **stream truncation** the router mis-labeled (some routers rewrite
     `finish_reason: "length"` to `"tool_calls"`, hiding the real truncation) — refused
     outright as a partial exit rather than retried, because retrying a truncated stream
     just wastes 3 attempts.
   - Otherwise, same 3-strike retry-then-recover pattern as invalid names: under 3,
     retry silently; at 3, inject a tool-role error result per call ("use `{}` for tools
     with no required parameters...") so the model can recover in-band instead of ending
     the turn.

Both strike counters (`_invalid_tool_retries`, `_invalid_json_retries`) reset to 0 on
any successful validation, so they track *consecutive* failures only.

## 4. Argument type coercion — `tools/arg_coercion.py:coerce_tool_args`

Distinct from layers 1–3: this runs once the arguments are already valid JSON, just the
**wrong JSON types** for the tool's declared schema (a very common local-model failure
mode — everything gets stringified). Runs at dispatch time
(`model_tools.py:871`, inside `handle_function_call`), schema-guided and conservative
(keeps the original value whenever a coercion isn't unambiguous):

- `"42"` → `42` for an `integer`/`number` schema (rejects `inf`/`nan`/non-numeric,
  keeps original).
- `"true"`/`"false"` (case-insensitive) → real `bool` for a `boolean` schema.
- A JSON-encoded string (`"[1,2,3]"`, `"{...}"`) → parsed list/dict when the schema
  expects `array`/`object`, including **recursively** inside already-native
  containers (so a list-of-JSON-encoded-dicts gets normalized element-by-element) —
  ported from an upstream Cline fix (cline/cline#11803).
- A bare scalar where an `array` schema is expected gets wrapped in a one-element list
  (with a WARNING if it looked like a botched JSON array string that failed to parse).
- `"null"` (any case) → real `None` only if the schema explicitly allows null
  (`type: null`, `nullable: true`, or a `null` branch in `anyOf`/`oneOf`).
- Also unrenames provider-sanitized property keys back to the registry's wire names
  first (`tools/schema_sanitizer.unrename_tool_args`) — the model saw a schema with
  provider-illegal keys renamed, so args must be mapped back before coercion.

## 5. Message-sequence repair — `agent/agent_runtime_helpers.py:repair_message_sequence`

Not per-call — this repairs **role alternation** in the whole message list before a
send, which is where orphaned/duplicate tool_call↔tool_result pairs get cleaned up
(distinct problem from malformed JSON, but the same "local/replayed history can be
messy" motivation). Four ordered passes, each returns `(new_messages, repair_count)`:

1. `_merge_consecutive_assistants` — must run first so the union of tool_call ids from
   a merged turn is known before orphan detection.
2. `_drop_stray_tool_results` — a `tool` message whose `tool_call_id` (checked across
   all alias spellings: `call_id`, `id`, `response_item_id`, or composite `call|item`)
   doesn't match any known call in the current assistant/tool-call run gets dropped.
3. `_prune_unanswered_tool_calls` — the inverse: a tool_call in an assistant message
   with no matching result in the immediately-following tool run gets pruned from that
   message (message dropped entirely if pruning empties it).
4. `_merge_consecutive_users` — plain-text-only, skips summary carriers and `/steer`
   rows.

Runs as a "pre-call belt" for "host-fed, resumed, or replayed histories" — i.e. this is
explicitly for cases where history didn't come cleanly out of the same turn loop (session
resume, external injection, etc.), on top of (not instead of) the per-turn validation in
layer 3.

## Key design principles worth carrying over to LiteLLM-side repair

- **Never raise, always degrade to something sendable.** Unrepairable args become
  `"{}"`, not an exception — the request still goes out and the *model* gets a chance to
  self-correct via a tool-role error result, rather than the whole turn crashing.
- **Log the discarded content, not just "repair failed."** The full-content WARNING
  (bounded to 100k chars) is explicitly kept because it's the last copy of data that
  might otherwise be silently lost (e.g. truncated file-write args).
- **IDs must stay deterministic**, never random — both `deterministic_call_id` (hash of
  `name:arguments:index`) and `uniquify_tool_call_ids`'s `_d<n>` suffix scheme are called
  out as hard invariants because they feed prompt-cache prefixes; a `uuid4` fallback
  would silently break caching on every replay.
- **Strike counters are per-consecutive-failure, not per-session**, and reset on any
  success — avoids permanently penalizing a model after one bad turn.
- **Repair happens on a copy on the send path, never mutates persisted history** — the
  `_clone_message_for_send` / copy-on-write pattern in `conversation_loop.py` exists
  specifically because an earlier bug (#80498) wrote a repaired-to-`"{}"` args string
  back into the stored turn.
- **Distinguish "truncated stream" from "malformed but complete"** — a truncated
  argument string (doesn't end in `}`/`]`) is refused outright rather than retried,
  because some routers mislabel `finish_reason` and retrying wastes attempts on
  something that will truncate identically again.
- **Mixed-validity batches only penalize the actually-invalid calls** — don't discard
  real work (valid tool calls) just because a sibling call in the same batch was bad.

## Source files referenced

- `hermes-agent/agent/message_sanitization.py` — arg-JSON repair, call-id policy, image/
  reasoning-content policy
- `hermes-agent/agent/agent_runtime_helpers.py` — tool-name repair, message-sequence
  repair passes
- `hermes-agent/agent/turn_tool_validation.py` — turn-level validation/retry orchestration
- `hermes-agent/agent/chat_completion_helpers.py` — streaming tool-call assembly (calls
  layer 1)
- `hermes-agent/agent/conversation_loop.py` — send-path canonicalization (calls layer 1
  as fallback)
- `hermes-agent/tools/arg_coercion.py` — schema-guided type coercion at dispatch
- `hermes-agent/model_tools.py:871` — dispatch call site for `coerce_tool_args`

Tests worth reading directly if reproducing behavior: `tests/run_agent/
test_repair_tool_call_arguments.py`, `test_streaming_tool_call_repair.py`,
`test_repair_tool_call_name.py`, `test_message_sequence_repair.py`, `test_tool_call_args_sanitizer.py`,
`tests/run_agent/test_tool_arg_coercion.py`.

---

# Tool call repair — how Unsloth Studio does it

Researched 2026-09-12 from the installed Studio backend under
`~/.unsloth/studio/unsloth_studio/Lib/site-packages/studio/backend/` (package name
on disk is just `studio`, not `unsloth_studio` — that directory name is the venv).
Same goal as the Hermes doc above: extract the pattern for Club-3090/LiteLLM-side
repair. Unsloth's shape is quite different from Hermes's — no single
"repair pipeline" module, and grammar-constrained decoding does most of the work
Hermes does with post-hoc JSON repair.

## Two request paths, two different repair postures

Unsloth Studio runs two structurally different flows, and "healing" only exists
on one of them:

1. **Server-side "enable-tools" loop** (`unsloth run` default, GGUF via
   llama.cpp/llama-server, or the safetensors/MLX agentic loop). The server
   advertises tools to llama-server and gets back **grammar-bounded**
   `delta.tool_calls` — llama.cpp constrains sampling with a GBNF grammar
   derived from the tool schemas, so most malformation (bad JSON, wrong tool
   names) is prevented at generation time rather than repaired afterward. Text
   still leaks sometimes (older/uninstructed models, narration-then-call
   turns, formats the grammar doesn't cover), so a text-based parser/promoter
   still runs as a fallback — see Layer 1.
2. **Client-tool passthrough** (`unsloth run --disable-tools`, which is what
   every `unsloth start <coding-agent>` uses). The client's own `tools` are
   relayed to/from llama-server **verbatim** — Unsloth's tool loop and grammar
   constraints never engage, because the coding agent (Claude Code, Codex,
   etc.) owns the loop. Small GGUF models on this path routinely emit their
   tool call as **text** instead of structured `tool_calls`, and without
   intervention that text reaches the agent as prose and the turn dies. This
   is what `core/inference/passthrough_healing.py` exists for — see Layer 2.

Both paths share the same underlying text parser and the same argument
coercion code, so a fix to one benefits both.

## Layer 1 — Text-based tool-call extraction — `core/tool_healing.py` + `core/inference/tool_call_parser.py`

Pure, dependency-light parsing of ~11 vendor-specific text serializations a
model can emit instead of (or leaking around) structured tool calls:

- `<tool_call>{json}</tool_call>` (Qwen/Hermes)
- `<|tool_call>call:name{k:<|"|>v<|"|>}<tool_call|>` (Gemma 4, custom quoting)
- `<function=name><parameter=k>v</parameter></function>` (Qwen3.5 XML)
- `<|python_tag|>NAME.call(k="v")` / `<|python_tag|>{"name":...,"parameters":...}` (Llama-3)
- `{"name":...,"parameters":...}` bare JSON (Llama-3.2 drift)
- `[TOOL_CALLS] [{...}, ...]` / `[TOOL_CALLS]name{json}` / `[TOOL_CALLS]name[ARGS]{json}` (Mistral v0.3 → v11/Magistral/Ministral)
- DeepSeek R1/V3/V3.1's `<｜tool▁calls▁begin｜>...` family (5 opener variants llama.cpp itself keeps)
- GLM 4.5/4.6/4.7's `<tool_call>NAME\n<arg_key>k</arg_key>...`
- Kimi K2's `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}`
- TML Inkling's `<|content_invoke_tool_json|>{...}<|end_message|>`
- bare reasoning-model "rehearsal": `name[ARGS]{json}` with no wrapper at all

`core/tool_healing.py` is explicitly the shared subset ("Qwen/Hermes, Qwen3.5
XML and Gemma 4 live in `core.tool_healing`; [`tool_call_parser.py`] adds the
rest") kept import-light enough that "external inference servers... import
this module without pulling in the inference orchestrator, structlog, httpx,"
so llama-swap/custom shims can reuse it. Notable mechanics:

- **Balanced-bracket scanning, not fixed-depth regex** — `_balanced_json_span`/
  `_balanced_brace_end`/`_balanced_bracket_end` walk char-by-char tracking
  string/escape state so arbitrarily nested JSON args parse correctly; a
  fixed-depth regex "left two-level args behind."
- **Truncation tolerance is a first-class parameter** (`allow_incomplete`) —
  the streaming healer parses with `allow_incomplete=False` mid-stream (an
  unclosed call stays held, not promoted early) and `True` only at
  `finalize()`, so a call is never promoted before its closing tag is known
  to have arrived or the stream is confirmed done.
- **Rehearsal vs. real call is gated two ways**: `enabled_tool_names` (a bare
  `name[ARGS]{...}` is only a call when `name` is a declared tool — an
  inactive name is prose) and markdown code-span exclusion (`_code_spans`/
  `_in_code`) — a model *documenting* the call syntax in a fenced block is not
  executed. `<think>`/`[THINK]` blocks get the same treatment in reverse:
  content inside them is rehearsal, not a call, unless the call's markers
  bracket the think-tag (then the tag is argument data).
- **Gemma's bespoke unquoted syntax** (`call:name{key:value, arr:[a,b]}`) gets
  a dedicated micro-parser (`_quote_gemma_object_keys`/`_quote_gemma_array_elements`/
  `_normalise_gemma_quoted_strings`) that walks the string quoting bare keys/
  values so `json.loads` succeeds, handling nested objects/arrays and
  comma-in-value cases (`location:New York, NY`) via a "next token look like a
  key?" lookahead rather than naive comma-splitting.
- **Mistral's comma-less multi-call arrays** (`[{...}{...}]`, no separating
  comma — an actual template quirk) are decoded element-by-element with
  `json.JSONDecoder().raw_decode` in a loop, because one `json.loads` on the
  whole array body would reject the malformed separator and drop every call
  in the batch, not just the bad one.
- **Perf-hardened by construction, not by exception**: several comments call
  out avoided quadratic blowups explicitly — a lazy `<tag>.*?</tag>` regex is
  never run without first confirming a close token exists (`_PAT_REQUIRED_TOKEN`
  bounds the scan to the last real closer), and the markdown code-span regex
  uses length-enumerated backtick runs instead of a backreferenced `(`+)..\1`
  because the naive version "turned 21 KB of unmatched runs into a 1.8s stall."

## Layer 2 — Passthrough healing — `core/inference/passthrough_healing.py`

This is the module the CLI flags `--enable-tool-call-healing` /
`--enable-tool-call-nudging` (`unsloth studio`/`unsloth run`, env vars
`UNSLOTH_DISABLE_TOOL_CALL_HEALING` / `UNSLOTH_TOOL_CALL_NUDGE`) actually gate.
It only ever touches the **response**, never the outbound request — "the
upstream request body is never touched, no extra generation is issued, so
llama-server slot/KV-cache reuse is byte-identical." Two entry points, one
shared core:

- **Non-streaming**: `heal_openai_message(msg, allowed_tools, tools)` — no-ops
  unless the message has no structured `tool_calls` already (grammar mode
  already worked in that case) and its content contains a signal string
  (`_HEAL_SIGNALS`, a narrower allowlist than the parser's full
  `TOOL_XML_SIGNALS` — only formats this healer's parser can actually
  promote). Parses with `parse_tool_calls_from_text(..., with_spans=True)`,
  keeps only calls whose name is in the **declared** `tools` allowlist, and
  removes **exactly** the promoted calls' markup spans from the content —
  undeclared calls, unparseable blocks, and unsupported formats are left
  byte-intact as prose. `content` becomes `None` (not `""`) on a pure-tool
  turn per the OpenAI spec.
- **Streaming**: `StreamToolCallHealer`, a buffer-and-drain state machine.
  `feed(text)` forwards ordinary prose immediately and only withholds (a) a
  trailing partial-signal suffix shorter than the longest signal string
  (`_partial_signal_suffix`) or (b) a suspected-but-unclosed tool block
  (bounded to 64 KiB via `_MAX_HOLD_CHARS`, past which it's declared a false
  alarm and flushed verbatim — bounds memory against a model rambling
  XML-lookalike text forever). `finalize()` does a last-chance heal of
  whatever's left in the buffer with `allow_incomplete=True`. It goes
  `dormant` the moment upstream sends a real structured `delta.tool_calls` —
  grammar mode worked, so healing steps aside entirely for the rest of that
  response.
- **Never invents a tool**: promotion is gated on `allowed_tools`, always the
  request's own declared tool-name set (further narrowed by `tool_choice`:
  `"none"` disables healing outright, a forced `{"name": N}` narrows
  promotion to just `N`). A parsed call naming an undeclared tool is left as
  text, full stop.
- **Nothing is silently discarded even on promotion**: the streaming healer
  keeps `_promoted_spans[call_id] = <original markup>` specifically so a
  caller that ends up having to discard a promoted call (e.g. the turn hit
  `finish_reason: "length"` mid-call and can't be executed) can hand the
  model's own original words back instead of losing them — "promotion is
  destructive... has no other way to give the model's words back."
- **Optional one-shot nudge retry** (opt-in — per-request `nudge_tool_calls`
  or process-wide `UNSLOTH_TOOL_CALL_NUDGE=1`): `nudge_should_retry` fires
  only when healing is enabled, the first response had zero structured calls,
  a heal signal is present in the text, AND healing still failed to promote
  anything — i.e. "the exact failure a single re-ask can fix." `nudge_messages`
  appends an assistant-echo + a terse "call the tool now, natively" user turn,
  built so **the prompt prefix stays byte-identical** to the original request
  (same cache-reuse motivation as the no-touch-the-request rule above).
  `response_has_promotable_calls` then checks the retry actually improved
  things before using it — a nudge that produces a hallucinated undeclared
  call is explicitly "not an improvement."
- **Kill switches**: process-wide `UNSLOTH_DISABLE_TOOL_CALL_HEALING=1` (an
  "absolute server kill-switch" per the CLI help — overrides everything, no
  per-request override), and per-request `auto_heal_tool_calls: false`.

## Layer 3 — Argument JSON repair — `core/inference/tool_loop_controller.py:_repair_json_value`

Analogous to Hermes's layer 1 but a different technique: instead of a
multi-pass "try stripping trailing commas, then close brackets, then escape
control chars" pipeline, Unsloth does a **single string-aware bracket-balance
rewrite**:

- `_JSON_STRING_RE` finds every JSON string token (open or still-unclosed —
  the trailing `\?$` alternative matches an unterminated string at EOF).
- Everything **between** string tokens goes through `_balanced()`, which walks
  char-by-char: an opening `[`/`{` pushes its expected closer onto a stack; a
  closer of **the wrong kind gets silently rewritten to the right one**
  ("the commonest slip: a closer of the wrong kind becomes the right one") —
  this is a repair Hermes's doc doesn't call out explicitly.
- At the end, any still-open string is closed with a bare `"` and every
  unclosed bracket on the stack is appended in reverse order, then the whole
  reassembled string is parsed with `json.loads(..., strict=False)` (the same
  `strict=False` control-character tolerance Hermes's pass 0 relies on).
  Returns `None` (never raises) if it still won't parse.

## Layer 4 — Schema-guided type coercion — `coerce_arguments_by_schema` / `_coerce_by_property`

Same problem Hermes's layer 4 solves (XML/text-parsed args arrive as strings
regardless of declared type — `replace_all` becomes `"false"`, and
`bool("false")` is `True`), solved with a recursive schema walk:

- `_read_schema` resolves one declared type per property, including a single
  non-null branch of `anyOf`/`oneOf` (refuses to guess across a real type
  union) and OpenAPI 3.0's `nullable: true`. An unrecognized keyword anywhere
  in the subschema (checked via an explicit allowlist, `_readable`) makes the
  whole property unreadable rather than guessed at — conservative by
  construction.
- Scalar coercion: `"true"`/`"yes"` → `True`, `"false"`/`"no"` → `False` for a
  `boolean` schema (deliberately excludes bare `"0"`/`"1"` — those arrive
  already typed from a native argument, so treating them as booleans would
  make one spelling mean two things); numeric strings → `int` (tried first,
  since `float()` would round a value like `9007199254740993`) then `float`,
  rejecting non-finite results (`nan`/`inf` parse in Python but `json.dumps`
  writes them unquoted, which the client would reject).
- Container coercion (`array`/`object`) tries, in order: `json.loads`, then
  `ast.literal_eval` (catches a model emitting a Python-literal-shaped value —
  tuples get converted to lists), then — **only when `repair=True`** — the
  layer-3 bracket-balance repair. Every candidate is checked by
  `_usable_container`, which **re-encodes the value with `json.dumps` and
  compares the round-trip** — a parse that doesn't survive re-serialization
  (an integer dict key silently becoming a string, a lone UTF-16 surrogate
  that dies on `.encode("utf-8")`) is rejected rather than accepted and
  corrupted downstream. This round-trip-verification idea doesn't appear in
  the Hermes writeup and is worth carrying over.
- Recursion is schema-driven (only descends into `object`/`array` properties
  that have `properties`/`items`/`prefixItems` declared) and depth-capped at
  8 (`_MAX_SCHEMA_DEPTH`) against adversarial/deeply-nested schemas.

## Layer 5 — Bare-string argument healing — `coerce_tool_arguments` (heal branch) + `_heal_arg_key` / `_looks_like_broken_json`

The one true "heal, don't just coerce" case: a model sends a single bare
string instead of a JSON object for a tool whose schema has exactly **one**
required string argument (`python(code=...)`, `terminal(command=...)`,
`render_html(code=...)`, `web_search(query=...)` — the last is hand-listed
because it has no *required* argument at all, so schema-derivation can't find
it). The interesting part is what it refuses to heal:

- `_looks_like_broken_json(raw)` distinguishes "this was never JSON" (heal it)
  from "this was JSON that got **cut off mid-stream**" (do NOT heal — the
  fragment is not the argument, it's wreckage). The test inspects **where**
  `json.loads` failed: an `Unterminated string` error, or a failure position
  at the very end of input, means input ran out — that's truncation. A bad
  property name right after `{` is excluded from that (still healable — a
  bona fide non-JSON string can start with `{`), and "Extra data" is excluded
  the other way (a *complete* document with trailing junk lost nothing). This
  was written to fix an observed real bug: a truncated `{"code":"html = ...`
  call to `python` got wrapped whole as the `code` argument, and "the model
  read its own file back as `{"code":"html = ...` and spent the rest of the
  turn convinced the sandbox had mangled it."
- `_heal_arg_key` derives the healable key **from the schema itself**
  (`_healable_keys_from_schemas`, cached) rather than a hand-maintained name→key
  map, specifically because the hand-kept version "silently went stale the
  moment a tool was added" (a 3-required-arg tool got auto-healed into a
  `"query"` key that only existed on search tools). MCP tools discovered at
  request time are resolved from the **request's own** tool schemas rather
  than cached process-wide — caching them would let one chat's MCP server
  decide another chat's healing.
- When neither condition applies (broken JSON, or no single-string-healable
  key), the raw text is **not** discarded and not guessed into a wrong key —
  it's tagged with a sentinel (`UNPARSED_ARGUMENTS_KEY =
  "__unsloth_unparsed_arguments__"`) that `execute_tool` recognizes and reports
  as "arguments were cut off / could not be read" rather than surfacing a
  confusing type error blaming the model for a required key it never sent.
  The sentinel is explicitly kept out of both user-facing surfaces it could
  leak into: the tool card (shown as `{"raw": fragment}` instead) and the
  replayed assistant turn (replayed as a short valid `{"error": ...}` object,
  never the raw unparseable fragment, because llama-server parses `arguments`
  as JSON while rendering the chat template and 500s on anything that
  doesn't parse).

## Layer 6 — Turn-level orchestration — `ToolLoopController`

The per-response ledger for the server-side loop (shared by GGUF and
safetensors/MLX backends). Differences from Hermes's layer 3 worth noting:

- **Dedup by canonical key**, not by call id: `canonical_tool_call_key(name,
  args)` = `f"{name}:{json.dumps(args, sort_keys=True, separators=(',',':'))}"`.
  A call whose (name, args) pair already **succeeded** this response becomes a
  `"duplicate"` no-op with a static nudge telling the model not to repeat
  itself; a repeat-offender counter (`duplicate_noop_limit`, default 2) forces
  a final no-tools answer pass if nudging alone isn't landing.
  `render_html` (and any declared "one-shot" tool) additionally can't be
  called again at all once it's succeeded once in a response, regardless of
  argument changes.
  - **No fuzzy tool-name repair at all** — this is the sharpest contrast with
  Hermes. An unrecognized/undeclared name is classified `"disabled"` and gets
  a static "that tool is not enabled, answer now" nudge; there is no
  `difflib`/normalize/strip-suffix name-correction step anywhere in this
  codebase. Grammar-constrained decoding is presumably why: on the path where
  Unsloth controls the loop, the model is sampled *inside* a grammar built
  from the declared names, so a hallucinated name is far rarer than in an
  unconstrained agent loop like Hermes's — the tradeoff (reject the rare miss
  vs. build a repair heuristic for it) went the other way here.
- **A forced tool-choice mismatch is its own noop reason** (`"forced_mismatch"`),
  distinct from `"disabled"` — a call for the wrong (but otherwise valid and
  enabled) tool when `tool_choice` pins a specific one.
- **Result-string error classification** (`is_tool_error` /
  `TOOL_ERROR_PREFIXES`) is a plain prefix allowlist (`"Error:"`, `"Execution
  error"`, `"Exit code"`, `"Failed to fetch"`, ...) rather than a strike
  counter with retry — a failed tool result gets `TOOL_ERROR_NUDGE` appended
  and is fed back once; there's no 3-strikes/terminal-exit machinery like
  Hermes's turn validator.

## Key design principles worth carrying over to LiteLLM-side repair

- **Grammar-constrained decoding is the primary defense, repair is the
  fallback.** Unsloth's own server-side loop leans on llama.cpp's GBNF
  grammar to make most malformation structurally impossible; textual
  parsing/healing exists for what leaks through and for the client-tool
  passthrough path where no grammar applies at all. Club-3090/LiteLLM has no
  equivalent (it doesn't control sampling), so it's necessarily closer to
  Hermes's "repair is the only defense" posture — but it's worth knowing
  Unsloth's repair code was written assuming it's a second line of defense,
  not the primary one, and may be less battle-hardened against the failure
  modes grammar mode normally prevents.
- **Healing only ever promotes to a name the client already declared** — the
  same "never invent" principle as Hermes's mixed-batch/name-repair layers,
  enforced here by intersecting with the request's own `tools`/`tool_choice`
  rather than a global valid-name list.
- **Touch the response, never the request, on the passthrough path** — an
  even stronger cache-preservation invariant than Hermes's per-message
  canonicalization: the outbound request to llama-server is untouched byte-
  for-byte specifically so slot/KV-cache reuse holds, and even the one-shot
  nudge retry is built as a byte-identical-prefix continuation for the same
  reason.
- **Verify a coercion round-trips before trusting it** (`_usable_container`:
  re-encode and compare) — don't accept a "fixed" value that would silently
  change shape again the next time it's serialized.
- **Distinguish truncation from malformation by WHERE the parse failed**, not
  by a structural heuristic like "does it end in `}`" — `_looks_like_broken_json`
  inspects the `json.JSONDecodeError`'s message/position to tell "cut off
  mid-token" from "just not valid JSON to begin with," and only refuses to
  heal the former. Same spirit as Hermes's "truncated stream vs.
  malformed-but-complete" distinction, more precise implementation.
- **A failed heal must be recoverable, not silently wrong** — an
  unresolvable bare-string argument becomes a sentinel that reports the real
  failure (arguments never arrived) instead of a guessed key that produces a
  confusing downstream type error blaming the model.
- **Derive healable shape from the schema, not a hand-maintained table** — a
  hand-kept name→key map for "which tools have one healable string arg"
  silently rotted when tools were added; deriving it from each tool's own
  JSON Schema (single required property, type string) can't go stale the
  same way.
- **Never let repair logic itself become the perf bug** — several routines
  are explicitly written to avoid quadratic behavior (bounded lazy-regex
  scans via a required-closing-token check, length-enumerated code-fence
  matching instead of backreferences) with comments citing the exact
  pathological input that used to stall.

## Source files referenced

- `studio/backend/core/tool_healing.py` — shared/lightweight text-based
  tool-call parser (Qwen/Hermes, Qwen3.5 XML, Gemma 4 formats); the
  externally-reusable subset
- `studio/backend/core/inference/tool_call_parser.py` — the rest of the
  vendor formats (Llama, Mistral, DeepSeek, GLM, Kimi K2, TML Inkling);
  imports and extends `core.tool_healing`
- `studio/backend/core/inference/passthrough_healing.py` — response-side
  healing + streaming state machine + nudge-retry for the client-tool
  passthrough path
- `studio/backend/core/inference/tool_loop_controller.py` — argument JSON
  repair (`_repair_json_value`), schema-guided type coercion
  (`coerce_arguments_by_schema`), bare-string healing (`coerce_tool_arguments`),
  and the `ToolLoopController` per-turn ledger (dedup, noop classification,
  final-answer forcing)
- `unsloth_cli/commands/start.py`, `unsloth_cli/commands/studio.py` — CLI
  flags (`--enable-tool-call-healing`, `--enable-tool-call-nudging`) and env
  var plumbing (`UNSLOTH_DISABLE_TOOL_CALL_HEALING`, `UNSLOTH_TOOL_CALL_NUDGE`)
  into the child server process

Installed at (this machine): `~/.unsloth/studio/unsloth_studio/Lib/site-packages/`
— note the on-disk package is `studio`, distinct from the `unsloth`/
`unsloth_cli`/`unsloth_zoo` packages that make up the CLI and training/
inference core; the CLI launches the backend as a subprocess and only passes
config through, it does not itself contain the healing logic.
