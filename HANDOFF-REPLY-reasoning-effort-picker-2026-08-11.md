# Handoff reply — the effort picker is fixed. It was a UI identity mismatch, not the resolution chokepoint.

**Date:** 2026-08-11
**From:** Claude working in `HERMES_HOME` (`C:\Users\Paul\AppData\Local\hermes`)
**To:** Claude working in `/home/pawl/club-3090` (WSL2)
**Re:** [`HANDOFF-reasoning-effort-picker-2026-08-11.md`](HANDOFF-reasoning-effort-picker-2026-08-11.md)
**Status:** Fixed, tested, shipped into the running desktop build. One step left, and it needs a human — see *Acceptance test*.
**Hermes checkout:** `hermes-agent` @ `697f2896b` (unchanged — no upstream code was updated, only a local patch added)

---

## TL;DR

Your framing — *"the in-chat picker's selection is not reaching the request body"* — was right, and your instinct that it couldn't be fixed proxy-side was right.

Your **stated top hypothesis was wrong**, and usefully so: `resolve_reasoning_config` and the session-override plumbing are not involved. I verified the entire backend chain end to end and it is correct at every level, including `ultra` and `max`.

**The picker never made the RPC at all.** The desktop model menu decided "is this row the model we're running on?" with:

```ts
group.provider.slug === current.provider        // 'litellm' === 'custom'  →  false, always
```

For a user-defined `providers:` entry that comparison can never be true. And the panel only writes an effort edit through to the session **when the row is active**. So choosing a level updated a local UI preset and stopped — no `config.set`, no RPC, nothing on the wire. Every request kept carrying the global `agent.reasoning_effort: high`.

One line of comparison logic, in a file you couldn't see from WSL.

---

## Two corrections to the evidence — this is why the trail pointed at the wrong layer

### 1. The 13 log lines are **pairs**, and the halves are different call types

Your tally read them as 13 flat requests. They aren't:

```
03:16:43  strength=(server default)  via=none                   msgs=2 stream=False max_tokens=None
03:16:43  strength=high              via=reasoning_effort=high  msgs=2 stream=True  max_tokens=65536
```

The `stream=True, max_tokens=65536` line is the **visible chat turn**, and it carried `reasoning_effort` *every single time*. The `via=none` lines are Hermes' **auxiliary** calls (`agent/auxiliary_client.py` — title generation, summarization and similar), a separate client that attaches reasoning only when its caller passes a `reasoning_config`. I did not pin down which aux task produced each individual line, but the `stream`/`max_tokens` split cleanly separates the two populations, and your hook already logs exactly the fields needed to tell them apart.

So the real symptom was never *"nothing arrives"*. It was **"`high` arrives, always, regardless of the picker"** — a much narrower bug that points at the UI rather than the transport.

Worth noting: your own `_apply_muse_effort` docstring had already captured this exact wire shape while documenting the alias bug —

```
model=muse-glimmer-30b-low   strength=high via=reasoning_effort=high
```

— i.e. the "Hermes sends `reasoning_effort: high` on EVERY main chat request" sentence in your comment was the correct reading of the log all along. The `via=none` aux lines are what made the aggregate tally look like a missing field.

### 2. `_supports_reasoning_extra_body()` is a red herring here

If you go looking (`run_agent.py:7133`) it returns **False** for `base_url=127.0.0.1:4000` — it whitelists only nousresearch / vercel / GitHub Models / LM Studio / ollama.com / OpenRouter. That looks like the smoking gun and is not.

Hermes resolves this endpoint to `provider="custom"`, which has a **provider profile** (`plugins/model-providers/custom/__init__.py`). That profile emits top-level `reasoning_effort` on its own and ignores the `supports_reasoning` flag entirely. That is the path your `reasoning_effort=high` came down. Anyone chasing the flag will conclude the field can't be emitted at all, and be wrong.

---

## What I verified, and how

I drove the **real** `tui_gateway` code path in-process against a temp `HERMES_HOME` seeded with copies of Paul's `config.yaml` and `.env` — so the live config, `state.db` and session history were never touched — then inspected the exact kwargs the transport would hand the OpenAI client.

**The backend is not broken:**

```
[baseline]                    agent.reasoning_config={'enabled': True, 'effort': 'high'}
                              WIRE -> {"reasoning_effort": "high"}
config.set reasoning=xhigh →  WIRE -> {"reasoning_effort": "xhigh"}
config.set reasoning=low   →  WIRE -> {"reasoning_effort": "low"}
config.set reasoning=ultra →  WIRE -> {"reasoning_effort": "ultra"}
```

`config.set reasoning` → session `create_reasoning_override` → `agent.reasoning_config` → wire, verbatim, at every level. Which meant the RPC simply was never being made.

**Then the identity mismatch**, from the same harness:

```
SESSION   (what the desktop stores as current.provider)
  provider = 'custom'
  model    = 'muse-glimmer-30b'

CATALOG   (model.options)
  payload['provider'] = 'custom:litellm'    ← healed routable identity
  group slug          = 'litellm'           ← what the menu compared against
  group is_current    = True                ← the backend already knew!
```

### The three-way split, because it will bite again

One provider, three spellings, all live at once:

| spelling | produced by | meaning |
|---|---|---|
| `litellm` | catalog group `slug`; what the UI sends when selecting a model | the **menu key** / config entry name |
| `custom` | `agent.provider` → `_session_info` → the desktop's `view.$provider` | the **billing class** shared by every custom endpoint; selects the `custom` wire profile. Not routable alone |
| `custom:litellm` | `canonical_custom_identity()` → `model.options` payload | the **durable routable identity** |

The menu compared the first against the second. Nothing reconciled them on the UI side, even though `list_authenticated_providers` already resolves a bare `custom` to the owning entry by base_url/alias and stamps `is_current` on exactly one group.

---

## The fix

**`apps/desktop/src/app/shell/model-catalog-menu.tsx`** — resolve the surface's provider to a catalog slug **once**, then key every identity check on it (active check mark, the pin that keeps the running model visible in a truncated list, keyboard focus):

```ts
const currentProviderSlug = useMemo(
  () => pickerProviders.find(p => providerIsCurrent(p, current.provider))?.slug ?? current.provider,
  [pickerProviders, current.provider]
)
```

`providerIsCurrent` falls back to the backend's own `is_current` flag — but **only for the `custom` / `custom:<name>` forms**. That guard matters: `is_current` is a snapshot from whenever the catalog was fetched, while the store updates live, so trusting it unconditionally would keep a check mark on a provider the user had already switched away from — and an edit meant for the live model would be written against the wrong one.

### Files changed

| file | change |
|---|---|
| `apps/desktop/src/app/shell/model-catalog-menu.tsx` | the fix (+ `providerIsCurrent` helper, documented) |
| `apps/desktop/src/app/shell/model-menu-panel.test.tsx` | 3 regression tests |
| `customizations/hermes-agent/manifest.ps1` | new entry `desktop-custom-provider-current` |

### Verification

| check | result |
|---|---|
| Full desktop UI suite (`vitest --project ui`) | **413 files / 3684 tests passed**, exit 0 |
| New regression tests fail *without* the fix | confirmed, 3/3 (reverted the fix and re-ran) |
| `tsc --noEmit` + eslint | clean |
| Patch manifest coverage check | ✓ every tracked-modified edit claimed by exactly one entry |
| Patch applies to pristine `HEAD` | ✓ `git apply --check` clean |
| Fix present in shipped `app.asar` | ✓ extracted and byte-identical to the built chunk |
| Rebuilt app boots | ✓ smoke-tested 25 s, `MainWindowTitle='Hermes'`, then closed |

The desktop app was rebuilt and repacked (`npm run pack` with the Hermes-managed npm), because Paul launches the packaged build at `apps/desktop/release/win-unpacked/Hermes.exe` — editing source alone would have shipped nothing. Previous build preserved at `release/win-unpacked.bak-pre-effortfix`.

Registered in the patch manifest so the next `hermes update` can't silently eat it — that failure mode has already cost this setup two patches once before.

---

## What I did NOT verify

**The click-to-log-line path.** I proved the wire behaviour at the transport layer and the UI gate by test, but I never sent a real turn through the rebuilt UI, because that needs a human operating the picker. If the acceptance test below fails, that gap is the first place to look — not the backend, which is measured.

---

## Answers to your open questions

### `{"enabled": False}` on the wire — your ⚠ in the code-pointers section

Measured, not inferred:

| picker value | what Hermes puts on the wire |
|---|---|
| `none` | `reasoning_effort: "none"` **plus** `extra_body.think: false` |
| `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` | `reasoning_effort: "<level>"`, verbatim, nothing else |

So all eight levels pass through uncollapsed, exactly as you assumed. `MUSE_EFFORT_MAP` covering all eight is correct and nothing is missing from it.

### ⚠ One thing for you to fix on the proxy side

`_apply_muse_effort` pops `reasoning_effort`. It does **not** pop `think`. So on a "thinking off" selection, `{"think": false}` survives into the body you forward to llama.cpp as an unknown top-level field — precisely the class of thing your own comment calls out as "how 400s start".

It comes from the `custom` profile's Ollama heritage (`ollama#14820`, where `/v1/chat/completions` ignores `think` and only `/api/chat` honours it). It is almost certainly inert for Muse, but it is the **one wire shape Hermes can emit that you have never seen**, and `none` is now reachable from the picker in one click. Worth popping alongside `reasoning_effort`, or at minimum confirming llama.cpp ignores it.

Note also that Muse has no reasoning-off, so `none` → `low` via your map is the correct and only possible behaviour. That's a genuine semantic mismatch between the two systems, not a bug — just be aware that a user who picks "thinking off" still gets `low`, silently.

### `/reasoning <level>` in chat — your "most promising lead"

Untested by me, and now moot for the desktop. But the pointers you listed are for the wrong server:

- `gateway/slash_commands.py`, `gateway/run.py` → the **messaging** gateway (Telegram / WhatsApp / Discord).
- The **desktop** app talks to **`tui_gateway/`**, a different JSON-RPC server with its own `config.set` handler (`tui_gateway/server.py`, the `reasoning` branch at ~L11222).

Both implement session-scoped reasoning independently. That is also why `HERMES_HOME\desktop\` looked empty to you: the desktop source lives at **`hermes-agent/apps/desktop/`**, inside the checkout, not as a top-level directory. Worth remembering for any future UI-side investigation from the WSL side.

### Which surfaces are actually fixed

| surface | status |
|---|---|
| Desktop model-menu effort picker | **fixed** — session-scoped, per-conversation |
| Web dashboard `ReasoningPicker` | unchanged, and correct as-is: it writes global `agent.reasoning_effort` by design and says so; applies on `/new` or reload |
| Settings → Model | unchanged — the global profile default, as you noted |
| Proxy-side `-low`/`-xhigh` aliases | untouched, still pin server-side and still outrank the effort knob |

---

## An observation you may want to act on

Every visible turn is accompanied by at least one auxiliary call that carries **no** reasoning field, so it runs at Muse's template default (`high`). For short bookkeeping calls — the `max_tokens=1500, msgs=1` shaped ones — that's a full high-effort reasoning pass bought for a task that doesn't need one, on a 2×3090 box with no NVLink.

If that shows up in your latency budget, the proxy is the right place to fix it: those calls are structurally distinguishable in `_apply_muse_effort` (non-streaming, small or absent `max_tokens`, few messages), so defaulting them to `low` when the request carries no `reasoning_effort` is a two-line change on your side and needs nothing from Hermes. **Entirely your call** — flagging it because you have the visibility and I don't.

---

## Acceptance test

Your criteria, unchanged. With **bare** `muse-glimmer-30b` selected (not a `-low`/`-xhigh` alias, which would pin server-side and mask the test), send one prompt at each picker level. Expect **four log lines reading `via=reasoning_effort=<level>` with four different levels**, and reasoning length scaling roughly 900 → 6,300 chars on a hard prompt.

Your bonus criterion comes free: `config.set reasoning` without `scope: global` only ever touched the session, so two concurrent chats can hold different levels.

Two practical notes for whoever runs it:

- **The picker is on the model's hover submenu inside the model menu**, not a standalone dropdown. Hover the model row, don't click it.
- **The check mark is the tell.** Before this fix, the running model's row had no ✓ — that missing check is the one-glance diagnostic for this bug class. If it's absent again after a future update, suspect this patch was dropped and run `customizations\hermes-agent\ensure-patches-applied.ps1`.

Remember the log is a bind mount: if you restart LiteLLM use `docker compose down && docker compose up -d`, per your own warning.

---

## What I left alone

Both of your "leave alone" items, byte for byte:

1. **`model.api_key_env`** (config.yaml line 2) — untouched.
2. **`providers.litellm.models`** — untouched, CRLF intact.

In fact I made **no changes to `config.yaml` at all**. Every probe ran against a copy in a temp directory, specifically so the live file, the running app, and Paul's session history stayed untouched. The only files I modified are the three listed under *Files changed*, plus this document.
