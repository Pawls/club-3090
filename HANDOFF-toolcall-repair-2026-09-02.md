# Handoff — automatic tool-call repair in the LiteLLM proxy (`custom_hooks.py` §E)

> **Scope note:** a plan, not a change request. Nothing here has been implemented.
> Opened after establishing that **this stack has no tool-call repair anywhere** —
> `serve.sh` never touches responses, and `custom_hooks.py` implements only
> `async_pre_call_hook` (the file says so itself at line 336: *"there is no post-call
> hook here"*). Prompted by a comparison against Unsloth Studio, which does repair.
> **Step 0 below may conclude that this should not be built at all** — read it first.

**Subject:** `services/litellm/custom_hooks.py` — add a response-side repair path that
re-parses leaked `<tool_call>` markup out of `content` into a proper `tool_calls` array.
**Blast radius:** the 4 vLLM composes carrying `--tool-call-parser` (3 `hermes`,
1 `qwen3_coder`). The other 80 composes are llama.cpp/ik_llama `--jinja` lanes where
tool output is grammar-constrained at decode time — see §1.
**Acceptance harness:** `scripts/stream-toolcall-probe.py` (already in the repo).
**API facts below were verified against the running container, litellm 1.92.0, 2026-09-02.**

---

## 1. Step 0 — measure before building

The repo already ships the exact classifier for this failure. `scripts/stream-toolcall-probe.py`
sends tool-requiring prompts with `stream:true`, reassembles the SSE deltas, and buckets each
response as PASS / DROP / OTHER, where **DROP** is precisely the signature this hook would
repair: no `delta.tool_calls` **and** (`finish_reason==stop` **or** `<tool_call>` leaked into
`delta.content`).

```bash
python3 scripts/stream-toolcall-probe.py --url http://localhost:4000 \
    --model <the-lane-you-actually-use> --thinking on --tool-choice required --repeat 5
```

Run this through the proxy, on the lane in daily use, **before writing a line of hook.**

**Why this gate is load-bearing rather than ceremonial:** the documented failure
(club-3090 #145 / vllm#39056) is a **vLLM streaming-extractor** bug — the XML parses fine
non-streaming and is dropped at the `</think>`→`<tool_call>` boundary when streamed. It is a
vLLM-lane bug. A survey of the compose tree:

| Flag | Count | Tool-call posture |
|---|---|---|
| `--jinja` (llama.cpp / ik_llama) | 80 | Grammar-constrained decode when `tools` are passed — structurally hard to emit invalid markup |
| `--tool-call-parser hermes` (vLLM) | 3 | Post-hoc extraction; exposed to #145 |
| `--tool-call-parser qwen3_coder` (vLLM) | 1 | Post-hoc extraction; the #145 reproducer |

If the daily lane is llama.cpp and the probe returns **DROP=0**, this hook is dead code and
the plan should be closed unbuilt. Record the probe output in this file either way.

### Probe results

| Date | Lane (engine) | Path | Settings | Result |
|---|---|---|---|---|
| 2026-09-12 | `qwen3.8-27b-vision` (llama.cpp `--jinja`, :8117) | via LiteLLM :4000 | thinking on, tool-choice both (required + auto), repeat 3, max_tokens 8192 | **PASS 60/60 · DROP 0 · OTHER 0 · ERROR 0** |

Reading: on the llama.cpp lane in daily use, repair #1 (markup promotion) has nothing to
catch — build it only if a vLLM `--tool-call-parser` lane goes into regular use and its probe
shows DROP > 0. The probe grades well-formedness (name present, args parse, no leak), so it
says nothing about wrong *types* or near-miss *names* — repairs #3/#4 need the §5a log lines,
not this probe, to justify them. No vLLM lane was up to probe.

### Detection hook — shipped 2026-09-12 (observe-only)

`custom_hooks.py` §F classifies every tool-bearing response through the proxy (all models,
streaming and not) against the four §5a classes and appends one JSONL row to
`services/litellm/logs/toolcall-detect.jsonl` — clean rows too, so rates have a denominator.
It runs in `async_log_success_event`, after delivery, so it cannot alter or delay a response.
Kill switch `CLUB3090_TOOLCALL_DETECT=0`. Verified live: 10 streamed + 1 non-streamed
tool request → 11 rows, no-tools request → no row, no duplicates.

Summarize after a few days of real Hermes / Boxel traffic:

```bash
python3 - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("services/litellm/logs/toolcall-detect.jsonl", encoding="utf-8")]
print(len(rows), "tool responses;", sum(bool(r["findings"]) for r in rows), "with findings")
print(collections.Counter((r["model"], f["class"], f["verdict"]) for r in rows for f in r["findings"]))
EOF
```

Build a §5a repair only for a `(class, would_repair*)` bucket that actually shows up.
`truncated` / `unrepairable` / `unknown` / `mismatch` buckets are harness-side problems.

(The probe gained optional `OPENAI_API_KEY` → `Authorization: Bearer` support to run
through the proxy; engines ignore it.)

---

## 2. Where the code goes

`services/litellm/custom_hooks.py`, as a new **§E**, with the methods added **directly to the
existing `MaxTokensCap` class** — the singleton already registered at
`config.yaml:397` (`callbacks: custom_hooks.proxy_handler_instance`).

No new class, no mixin, no new bind-mount: the file is already mounted at
`services/litellm/docker-compose.yml:16`. See the leaf-class constraint in §4 for why the
"no mixin" part is a hard requirement and not a style preference.

⚠️ Editing a bind-mounted file here needs `down` + `up -d` to take effect — a plain `restart`
exits 127 on this rig (documented in the compose comments), and that cycle **destroys**
`docker logs`. Log repairs through the existing `_hook_log` helper so evidence survives.

---

## 3. Non-streaming path — the easy half

```python
async def async_post_call_success_hook(self, data, user_api_key_dict, response):
    ...
    return response   # or mutate in place
```

Verified in `litellm/proxy/utils.py:2320-2322` of the running image:

```python
callback_response = await callback.async_post_call_success_hook(...)
if callback_response is not None:
    response = callback_response
```

So the return value **is** honored in 1.92.0 (it was ignored in older versions — do not
trust older blog posts here). In-place mutation also works, same object.

Read `data.get("tools")` for the gate, inspect `response.choices[0].message.content`,
extract, populate `message.tool_calls`, and rewrite `finish_reason` `stop` → `tool_calls`.

---

## 4. Streaming path — the actual engineering

```python
async def async_post_call_streaming_iterator_hook(self, user_api_key_dict, response, request_data):
```

⚠️⚠️ **Leaf-class requirement.** `litellm/proxy/utils.py:1652` gates the whole streaming
chain on a `cls.__dict__` membership test:

```python
if "async_post_call_streaming_iterator_hook" in cls_attrs:
    has_iterator_override = True
    iterator_overrides.append((resolved, "override"))
```

If the method is **inherited** rather than defined on the registered leaf class,
`caps.iterator_overrides` stays empty, the fast path at utils.py:2517 streams straight
through, and the hook never runs — **silently, with no error and no log line.** Define it on
`MaxTokensCap` itself. (Note the asymmetry: `async_post_call_streaming_hook` walks the MRO,
this one does not.)

### The buffering state machine

A stream cannot be classified chunk-by-chunk — an opening `<tool_call` is not known to be a
real tool call until it closes. Use sentinel-triggered buffering with a tail hold:

```python
HOLD = len("<tool_call>")   # never emit a trailing partial sentinel
buf, holding, armed = "", False, True

async for chunk in response:
    text = _delta_content(chunk)
    if _chunk_has_tool_calls(chunk):    # engine got it right — disarm, pass everything through
        armed = False
    if not armed:
        yield chunk; continue
    buf += text
    if not holding and "<tool_call" in buf:
        holding = True                  # emit everything BEFORE the opener, withhold after
    ...                                 # while not holding: yield up to len(buf)-HOLD chars

# end of stream:
#   holding and buf parses      -> yield synthetic tool_calls deltas + finish_reason="tool_calls"
#   holding and buf won't parse -> flush buf verbatim as content (repair degrades to a no-op)
```

Three properties that must hold:

1. **TTFT unchanged** — content before the opener (thinking, preamble) still streams live.
2. **No split-sentinel leak** — a `<tool_` / `call>` boundary split across two chunks cannot
   escape, because the last `HOLD` chars are always withheld while unarmed.
3. **Failure is a no-op, never a swallow** — an unparseable buffer flushes verbatim, giving
   exactly today's behavior. A repair layer that can eat a response is worse than none.

Synthetic deltas need: a generated `id`, `index` 0, `type: "function"`, and the
`finish_reason` rewrite on the final chunk.

---

## 5. Scope — what to repair, what to refuse

> **Revised 2026-09-12** after surveying how Hermes (hermes-agent) and Unsloth Studio repair
> tool calls (`tool-call-repair.md`) and how BoxelBuilder's in-app harness consumes them
> (`REPLY_FROM_BOXELBUILDER.md`). The markup-promotion scope below is unchanged; §5a adds
> the other repairs that belong at the proxy, and §5b records what deliberately does not.

### 5a. The proxy's repair set — stack-safe by construction

The selection rule: the proxy sits in front of clients that **may already repair** (Hermes
has five layers; Boxel isolates parse errors, rejects unknown names, rescues truncation), and
in front of clients that repair **nothing** (curl scripts, future harnesses). A proxy repair
ships only if it is either **lossless** (a downstream repairer finds nothing left to do) or
something **no client can do** (markup promotion). Anything that guesses belongs in a harness,
which can report the guess back to the model; the proxy cannot.

| # | Repair | Why it's stack-safe | Build when |
|---|---|---|---|
| 1 | **Markup → structured `tool_calls`** (this section, below; §3–§4) | Fires only on a parser miss (no `tool_calls`), declared name, closed + parseable block; otherwise verbatim | §1 probe shows DROP > 0 |
| 2 | **Lossless argument-JSON normalization** on structured `tool_calls` whose `arguments` fail `json.loads`: retry with `strict=False` (raw control chars in strings), then strip trailing commas before `}`/`]`; re-serialize on success | No data added or removed. **Never** auto-close brackets, trim closers, or fall back to `"{}"` — on failure pass through untouched so the client's own parse-error signal (Boxel `_parse_args`, Hermes turn validation) still fires | Unconditionally — cheap, engine-independent |
| 3 | **Exact-normalization name repair**: remap to a declared tool only if lowercase / `-`,space→`_` / CamelCase→snake / strip trailing `_tool` yields **exactly one** declared name | Deterministic, lands only on a declared tool. **No `difflib` fuzzy match** — a wrong guess silently runs the wrong tool; only a harness can say "did you mean" | Logs show unknown-name calls |
| 4 | **Schema-guided type coercion** from the request's own `tools[].function.parameters`: `"42"`→42, `"true"/"false"`→bool, JSON-encoded string→array/object | Only moves *toward* the declared type, so downstream coercion is a no-op. Refuse unions (`anyOf` with >1 non-null branch); accept a coerced container only if `json.dumps` round-trips (Unsloth `_usable_container`) | Default-off; enable per lane once logs show stringified args |

**Invariants for all four:**

- **Truncation is untouchable.** If `finish_reason == "length"`, pass the response through
  byte-for-byte — no promotion, no JSON repair, and never rewrite `length` → `tool_calls`.
  Hermes refuses truncated calls and Boxel's max_tokens rescue keys on `length`; a proxy that
  makes a cut-off `write_file` look complete defeats both.
- **Response-side only.** Never mutate the outbound request (prompt-cache prefix stays
  byte-identical — the Unsloth passthrough invariant).
- **Pass-through, not synthesis, of ids.** Keep the engine's `id`; generate one only for a
  promoted call, deterministically (`call_<sha1(name:args:index)[:24]>`, Hermes's rule —
  never `uuid4`, ids feed replayed prompt prefixes).
- **Exact SSE shape.** Emit standard OpenAI `chat.completion.chunk` deltas (`index`, `id`,
  `type:"function"`, `function.name`, `function.arguments` as a string). Boxel's client is
  hand-rolled against raw SSE and keys fragments by `index`; there is no tolerance for a
  third dialect.
- **Don't assume `tool_choice`.** Boxel never sends `tool_choice` or `parallel_tool_calls`;
  gate on `tools` being non-empty, not on a forced mode. Honor `tool_choice: "none"` (skip all
  repair) and a forced `{"function":{"name":N}}` (promote only `N`) when present.
- **One `_hook_log` line per repair** (model, repair #, format, streaming) and one kill
  switch per repair class plus the global `CLUB3090_TOOLCALL_REPAIR=0`.

### 5b. Deliberately NOT in the proxy

| Technique (source) | Why not here |
|---|---|
| Invalid-call retry / strike counters (Hermes L3) | The proxy never sees tool results or owns the turn loop |
| Nudge re-ask (Unsloth passthrough) | A hidden second generation — doubles latency invisibly, conflicts with client retry logic |
| Fuzzy name matching (Hermes L2 `difflib`) | Silent wrong-tool execution; harness can say "did you mean" instead (Boxel `_script_hint` pattern) |
| Bracket auto-close / `"{}"` fallback (Hermes L1 passes 1–3, Unsloth L3) | Lossy — erases the truncation signal (see invariants) |
| Bare-string → single-arg healing (Unsloth L5) | Guesses the key; needs per-tool intent |
| Bare JSON / fenced ` ```json ` promotion | Prose that discusses JSON gets rewritten |
| Message-sequence repair on the request (Hermes L5) | Mutates history the harness owns; revisit only if llama.cpp template 500s on messy history show up in logs |

**Direct-to-engine clients get none of this** — a client pointed at `:81xx` instead of
`:4000` relies only on engine-level mitigations (`--jinja` lazy grammar on llama.cpp lanes;
parser choice, e.g. `qwen3_xml` over `qwen3_coder`, on vLLM). Route a harness through the
proxy if it lacks its own repair.

### 5c. Markup promotion scope (repair #1)

**In scope (v1), two wire formats:**

- Hermes / Qwen: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
- qwen3_coder XML: `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`

**Explicitly out of scope (v1):** bare JSON with no markup, and fenced ` ```json ` blocks.
Deciding whether the model *meant* those as a call requires heuristics, and that is exactly
where a repair layer starts corrupting legitimate prose that happens to discuss JSON.

**Three safety gates, all required before touching a response:**

1. `data.get("tools")` is non-empty — never rewrite a response to a request that had no tools.
2. The response carries no `tool_calls` already — repair is strictly a parser-miss fallback.
3. Recognized markup is actually present.

Plus `CLUB3090_TOOLCALL_REPAIR=0` as a compose-env kill switch, and one `_hook_log` line per
repair (model, format matched, streaming or not) so the repair rate is measurable in
`services/litellm/logs/`.

---

## 6. Shipping policy — default-OFF, per-model allowlist

`AGENTS.md` already governs this case: behavioral mitigations that **rewrite output** need a
streaming + large-prompt repro before shipping default-on. That is the Genesis P68/P69 lesson,
and a response-rewriting hook is squarely in that class.

So: ship default-off, allowlisted per model exactly as the existing §A–§D paths are gated, and
flip a lane to default-on only once §7 is green **on that lane**.

Second-line, not first. The documented first fix for #145 is swapping
`--tool-call-parser qwen3_coder` → `qwen3_xml` in the compose (`docs/FAQ.md:570`). Repair
catches what the parser swap does not.

---

## 7. Acceptance

| Gate | Bar |
|---|---|
| `scripts/stream-toolcall-probe.py` | DROP → **0** on the target lane; PASS count unchanged-or-up; no new OTHER |
| Same probe, tools-absent control | Byte-identical output with the hook on vs off |
| `scripts/quality-test.sh --quick` | No ≥10pp regression on ToolCall-15 |

The probe exits non-zero on any DROP, so it is CI-shaped as-is.

---

## 8. Estimate

~120 lines in `custom_hooks.py` for repair #1 — the streaming state machine, not the regexes,
is the bulk. Repairs #2–#4 (§5a) are pure functions over an assembled `tool_calls` list,
~40 / ~25 / ~80 lines, and reuse #1's end-of-stream hook — for streams they run on the
buffered tool-call deltas, so #2–#4 imply buffering `delta.tool_calls` (not content) until
the call's arguments are complete. Tool-call argument deltas carry no user-visible text, so
this costs no perceived TTFT.
No new files, no new mounts, no new dependencies, no compose changes beyond one env var.
