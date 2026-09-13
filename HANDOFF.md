# Session Handoff — Dual RTX 3090 / WSL2 / Qwen3.6-27B

> **Scope note:** this is a snapshot of one user's rig setup and troubleshooting session (2026-07-03), not evergreen project documentation. It's rig-specific (exact wattages, WSL2 config, personal client setup) — safe to delete once superseded by the next session, or once its durable findings graduate into the real docs. The one repo-level bug found here is also tracked in `AGENTS.md` (→ `CLAUDE.md`) under "Compose layout."
>
> **2026-09 retirement note:** `qwen3.6-27b`, the model this whole handoff is about, was removed
> from the catalog (superseded by `qwen3.8-27b`). Everything below is stale — kept only per this
> file's own "safe to delete" scope note above.

## Update — 2026-07-04 (firmware/ReBAR update + new bench baseline)

**Hardware change:** both cards' VBIOS updated. **Resizable BAR is now ON and symmetric**
(BAR1 = 32 GB on both, verified via `nvidia-smi -q -d MEMORY`). Previously asymmetric — the
MSI Suprim X (GPU0) had ReBAR OFF (256 MB BAR1) while the FE (GPU1) had it on. GPU1 is still
PCIe **Gen4 x4** (physical slot; ReBAR doesn't change lane count). GPU0 = x16.

**New canonical `bench.sh` baseline (decode TPS, 3-warm/5-measured):**

| Config | Narrative | Code | Context | Notes |
|---|---|---|---|---|
| **Dual TP=2** (fp8-mtp) | **53.98** (48–58) | **67.77** (58–76) | 262K | up from old 46.19 / 62.40 → **+17% / +9%** |
| **Single-card fp8+MTP** | **57.82** (54–62) | **79.96** (70–85) | ~28K | **faster than dual: +7% narr / +18% code** |

**Key findings:**
- The **firmware/VBIOS update raised TPS** (dual +9–17%), attributed mostly to a memory-subsystem
  improvement (decode is memory-bandwidth-bound) more than ReBAR's addressing window — but they
  changed together, so not cleanly separable.
- **Single-card is now FASTER than dual** (was a tie on old firmware). The firmware freed memory
  bandwidth; the single card captures it fully, while **dual stays capped by the x4 all-reduce**.
  So the x4 interconnect tax is now *visible* (~7% narr / ~18% code) — it was masked when both
  configs were equally memory-limited. The user's original x4 hypothesis was right; it just only
  shows up once memory stops being the bottleneck.
- **Stability likely fixed:** dual booted TP=2 clean with **no `shm_broadcast` hang** (the old
  failure). Symmetric BAR is the plausible cause — asymmetric BAR corrupts the multi-GPU IPC path.
- **Power does NOT bind decode:** during the dual bench both cards drew only ~180 W (GPU0 192 /
  GPU1 176) vs the ~288 W cap, at 65–68% util each and 52–68 °C. So **raising the power cap won't
  speed up token generation** — decode is memory-bound. (Single-card pulls ~281 W near the cap, so
  it's mildly power-limited there; and prefill/TTFT on big prompts could still touch the cap.)
  Thermals are healthy; the sandwiched x4 GPU1 ran coolest (52 °C).

**Recommendation (updated):** the **hybrid now has real merit** — single-card (port 8021) for
sub-28K chat/coding (measurably faster), dual TP=2 (port 8010) for big-context refactors (262K).
Measure the token sizes Hermes actually sends to pick the daily driver. **Don't overclock memory**
(junction risk, no need — firmware gave the gains). **Skip power-cap tuning for decode speed.**

**Config housekeeping this session:**
- Authored `single/autoround-int4/fp8-mtp.yml` (TP=1 + fp8 + MTP, util 0.94, ~28K ceiling) + its
  co-located `.env`. Experimental (untracked→now on branch `feat/asymmetric-pcie-preflight`).
- Added a PCIe-lane-width preflight warning (`preflight_pcie_lane_width`) — fires at `switch.sh`/
  `launch.sh` on a TP≥2 target with a narrow/asymmetric link. Committed on the same branch.
- **Cleaned the root `.env`:** removed `GPU_MEMORY_UTILIZATION` + `PYTORCH_CUDA_ALLOC_CONF` from it.
  `switch.sh:99` exports every root-`.env` var into the shell, and Compose gives shell vars
  precedence over a compose-dir `.env` — so those in root were *overriding* every compose's per-dir
  value (would force single-card to 0.92 instead of 0.94). They now live only in each compose's dir
  `.env` (dual→0.92, single→0.94). Root keeps launcher vars (MODEL_DIR, HF_TOKEN, default pin).
- Supersedes old open-items #1/#2 below: power-cap sweep is low-value (decode isn't power-bound);
  the `shm_broadcast` hang (#3) is likely resolved by symmetric ReBAR.

## Update — 2026-07-04 (correction: single-card hybrid conflicts with Hermes's min context)

**The single-card hybrid recommendation above (port 8021 for sub-28K chat) does not actually
work with Hermes.** Hermes enforces a **minimum 65536 context**, but
`single/autoround-int4/fp8-mtp.yml` caps `--max-model-len` at **28672** (line 130) — well under
Hermes's floor. Don't route Hermes at the single-card endpoint as-is; either bump that compose's
`MAX_MODEL_LEN` to ≥65536 (re-benchmark after — TPS/VRAM numbers above were measured at 28K) or
keep Hermes on dual (port 8010, 262K) and reserve single-card for a client without a context floor.

**Also confirmed while investigating:** the model currently served is **AutoRound INT4 safetensors
via vLLM** (`--quantization auto_round`), **not GGUF** — GGUF variants exist in the registry
(`qwen3.6-27b.yml` weights map: `gguf`, `unsloth-q4km`, `ubergarm-iq4ks`, etc.) but route to
different engines (`llama-cpp` / `ik-llama` / `beellama`), none of which are what's running.

**Sampling params** (temperature/top_p/top_k/min_p) are set server-side in
`fp8-mtp.yml` (dual) via `--override-generation-config`:
`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, repetition_penalty=1.0` — overridable per-compose
via `.env` (`TEMP`/`TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `REPEAT_PENALTY`). These are only the
*server default* — a client (Hermes) that sends its own sampling params in the request body should
still win per normal OpenAI-API precedence; hasn't been verified against Hermes's actual request
behavior.

**"Refusal-free" Qwen3.6-27B variant — blocked on a real source, not yet added.** Searched HF for an
abliterated/uncensored release matching this exact checkpoint; none exists, because `Qwen3.6-27B` is
this repo's own catalog naming — not a real upstream Qwen release, so no third-party abliteration
(huihui-ai, mlabonne, etc. all publish against real Qwen versions — closest analog:
`huihui-ai/Huihui-Qwen3.5-27B-abliterated`, BF16 safetensors) targets it directly. Two real paths
if the user finds/names a specific HF repo: (a) serve it as a new `bf16-mtp.yml`-style compose
(loses AutoRound INT4 footprint + likely loses the embedded MTP head unless the release happens to
carry one), or (b) self-quantize with AutoRound to match the existing `autoround-int4` pipeline
exactly — mirrors how `carnice-bf16mtp` / `qwopus-bf16mtp` are already handled in the registry
(both marked experimental, "no directly downloadable packed artifact" — manual requant already the
precedent here). Needs the user to pick a source before any compose gets written.

## Environment

- **Host**: Windows + WSL2 (Ubuntu), repo cloned at `~/club-3090` (correct — on WSL2's native ext4, not `/mnt/c`).
- **Hardware**: 2× RTX 3090, PCIe-only (no NVLink). Actual power limits per-card (via `nvidia-smi -q -d POWER`): **Default 420W, Max 450W** — notably higher stock TDP than club-3090's own reference-rig numbers (370-390W in `docs/HARDWARE.md`). Don't assume the repo's absolute wattage/TPS figures transfer 1:1 to this rig.
- **PSU**: 1000W. Pre-cap, dual cards at up to 450W each = 900W combined left only ~100W margin — a real risk, now mitigated by the power cap below.
- **Power**: ~290W/card via **MSI Afterburner's Power Limit slider (~69%, computed as 290/420)**, stacked on top of an existing **undervolt** (Curve Editor — a separate, complementary setting, not redundant with the power cap: undervolt improves efficiency within a power budget, the cap sets the ceiling itself). `nvidia-smi -pl` write operations don't work from inside WSL2 (read-only telemetry only) — power changes must be made from the Windows side.
- **`.wslconfig`**: `memory=56GB, processors=16, swap=4GB, networkingMode=mirrored, autoMemoryReclaim=gradual` on a 64GB host. Mirrored networking means `localhost` from native Windows reaches WSL2 services directly — relevant for the Hermes → club-3090 connection below.
- **`nvidia-smi` in WSL2**: binary lives at `/usr/lib/wsl/lib/nvidia-smi`, not on `sudo`'s default `secure_path`, so `sudo nvidia-smi ...` fails with "not found in PATH" until fixed. Fix applied: `sudo ln -s /usr/lib/wsl/lib/nvidia-smi /usr/local/bin/nvidia-smi`.

## Model serving

- **Model**: `qwen3.6-27b`, AutoRound INT4 (`Lorbus/Qwen3.6-27B-int4-AutoRound`), `MODEL_DIR=/home/pawl/models`.
- **Compose**: `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml` — the repo's documented dual-card "fast tier" default (⭐ in `docs/DUAL_CARD.md`). Pinned as default via `bash scripts/switch.sh --set-default vllm/dual` → `CLUB3090_DEFAULT_QWEN3_6_27B` in root `.env`.
- **Container**: `vllm-qwen36-27b-dual`, port **8010**. TP=2, 262K context, MTP n=3 (embedded in the checkpoint — no separate drafter download), fp8_e5m2 KV cache, FlashAttention-2, `--gpu-memory-utilization 0.92`, `--max-num-seqs 2`.
- **Launch / relaunch**: `bash scripts/launch.sh --variant qwen3.6-27b/default` (or bare `bash scripts/launch.sh`, since this is the only installed model). Must run from an actual WSL2 shell — not PowerShell, not a UNC-path-rooted shell.

## ⚠️ Repo bug found and fixed this session

**`.env` overrides placed at `<engine>/compose/.env` are silently never read** by any nested `<topology>/<quant>/` compose, because Docker Compose only auto-loads `.env` from the directory it's invoked from, and `switch.sh`/`launch.sh` `cd` two directories deeper before running `docker compose up`. No error, no warning — the compose just silently falls back to its own hardcoded defaults.

**Confirmed-affected**: `scripts/setup.sh`'s WSL2 auto-detection writes its `PYTORCH_CUDA_ALLOC_CONF` boot-crash-workaround override to `models/qwen3.6-27b/vllm/compose/.env` — one level too high. Verified via `docker compose config` showing the hardcoded fallback instead of the file's value. Likely affects every nested compose across every model/engine, not just this one.

**Fix applied**: copied the `.env` into the correct directory:
```
models/qwen3.6-27b/vllm/compose/dual/autoround-int4/.env
```
containing:
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:512
GPU_MEMORY_UTILIZATION=0.92
```
(`GPU_MEMORY_UTILIZATION` was also corrected from a mistaken `0.94` — a single-card WSL2 value copied from `docs/WSL_SETUP.md` — to the correct dual-card default of `0.92`.) Confirmed applied via `docker inspect`.

**Not yet fixed at the source** — see `AGENTS.md` (→ `CLAUDE.md`) "Compose layout" section for the durable version of this note, aimed at whoever next touches `setup.sh`'s WSL2 block.

## Verified current state (end of session)

- `verify-full.sh`: 8/8 pass (Genesis check N/A — this model doesn't use Genesis).
- `bench.sh` (canonical 3-warmup / 5-measured protocol): **narrative mean 46.19 TPS** (39.5-51.4 range), **code mean 62.40 TPS** (54.4-70.7 range).
- Documented peak for this exact compose (uncapped, reference rig): 69/89 TPS — the gap here is explained by the ~290W cap + WSL2 overhead, not a misconfiguration.
- MTP spec-decode healthy: acceptance length ~3.0-3.4, draft acceptance ~80-97%.
- Live usage confirmed matching bench data ("30 to 50 tps" observed directly).

## Client setup

- **Hermes Agent TUI** (Nous Research), Windows-native install. Config at **`C:\Users\Paul\AppData\Local\hermes\config.yaml`** — not `~/.hermes/config.yaml` (that path doesn't exist on this install; disregard generic Hermes docs referencing it).
  ```yaml
  model:
    default: qwen3.6-27b
    provider: custom
    base_url: http://localhost:8010/v1
    api_key_env: LM_STUDIO_API_KEY   # harmless leftover placeholder — this vLLM compose has no auth
  ```
- Must be launched from a native Windows path. Launching/`cd`-ing into a `\\wsl.localhost\...` UNC path causes Hermes's Node "gateway" subprocess to crash (classic cmd.exe UNC-cwd limitation), surfacing as a "gateway exited — recovering your session" error mid-response.

## Open items / natural next steps

1. **`scripts/power-cap-sweep.sh --cooling air`** hasn't been completed. The 290W cap was borrowed from the repo's reference rig (370-390W stock); this card's actual stock is 420-450W, so its real efficiency knee is unverified. `nvidia-smi` PATH is already fixed; just needs `sudo bash scripts/power-cap-sweep.sh --cooling air` run from a real WSL2 shell (~6-8 min).
2. **`sudo nvidia-smi -pm 1`** (persistence mode) was recommended but never confirmed as actually run.
3. If the earlier `shm_broadcast.py` engine hang (diagnosed as likely VRAM/allocator pressure under TP=2, no crash/traceback in logs) recurs, the next lever is dropping `GPU_MEMORY_UTILIZATION` further (0.92 → 0.88-0.90) in the now-correctly-located `.env`.
4. If more models get installed later, check their compose directories for the same `.env`-location bug before assuming any WSL2/rig-specific overrides are actually active — `docker compose config` (run from the compose's own directory) is the fast way to verify.
5. User's stated goal from here: continue "optimizing things" — open-ended, likely concurrency tuning, KV-cache tradeoffs, possibly `dual-turbo` for multi-stream throughput, or quality-vs-speed tuning via `quality-test.sh`.

---

# Update — 2026-08-09 (code change: LiteLLM preserve-state launcher gap + SPEC_N parser)

> **Different in kind from the sections above.** Those are rig-tuning snapshots. This one is a
> **code change** sitting UNCOMMITTED in the working tree. Read this before touching
> `scripts/preserve-state.sh`, `serve.sh`'s tail, `CockpitData.serve()`, or `app.py::_with_force`.
>
> The load-bearing invariants are also written as comments at each site — those are the durable
> copy. This section is the narrative: *why* the change exists and what nearly went wrong.

## 1. The bug this started from

`services/litellm/custom_hooks.py` §C re-inlines prior-turn `<think>` blocks into requests. That
hook runs **in a separate container** and cannot read the GPU compose's environment, so the
launcher must hand it the resolved mode out-of-band. The transport is
`services/litellm/preserve_state.json` (bind-mounted into the litellm container).

**Only `serve.sh` ever wrote that file.** The serve-cockpit boots models via
`bash scripts/switch.sh <slug>`, which never touched it — so a cockpit-launched model inherited
whatever mode the *last* `serve.sh` boot left behind.

This fails **silently**: nothing errors, the hook just replays `<think>` into a model whose compose
says not to. Caught live — the file held `{"mode":"full"}` while `vllm/qwen-35b-a3b-dual` resolves
to `off` (its co-located `.env` sets `PRESERVE_THINKING=false`). That carryover is exactly what
seeded thought-loops on the always-reasoning a3b lanes (see `MODEL_REFERENCE.md` fn 10).

## 2. What was built

**`scripts/preserve-state.sh` (NEW)** — the single resolver. Takes `--slug` (registry lookup),
`--compose <path>`, or an explicit `--mode off|full|window [--window N]`. Honours `NO_LITELLM=1`
and no-op's cleanly when `services/litellm/` is absent.

- **`serve.sh`** — its inline 18-line resolution block now delegates here, passing an explicit
  `--mode` when `--preserve` / `--no-preserve` / `--preserve-window` was given, else `--compose`.
- **The cockpit** — `CockpitData.serve()` now emits
  `["bash", "-c", <script>, "cockpit-serve", <slug>]`, where the script runs `switch.sh` then
  `preserve-state.sh`.

**Design decision: `switch.sh` was deliberately NOT modified.** `AGENTS.md` says launchers
auto-derive from the registry and shouldn't be hand-edited, and it's upstream-tracked — changes
there become merge conflicts on every upstream pull. If a future launcher boots GPU models, it must
call `preserve-state.sh` too, or it re-opens this exact gap.

## 3. Invariants you must not break

| Invariant | Why | Enforced by |
|---|---|---|
| **Slug is the LAST element of `plan.cmd`** | `app.py` recovers it via `plan.cmd[-1]` for the pending-serve watch (~L9183) and the problem reporter (~L9394) | `assert_serve_cmd()` in `tests/conftest.py` |
| **Slug is POSITIONAL, never interpolated into the script text** | otherwise a slug can inject shell | same helper (asserts `slug not in script`) |
| **`--force` lives INSIDE the script text** | keeps it from displacing the slug from the end of `cmd` | same helper (`force=` arg) |
| **`preserve_state.json` is truncated IN PLACE (same inode)** | the litellm container bind-mounts that inode; a temp-file + `os.replace()` swaps it and the container keeps reading the old one. **This is a deliberate exception to AGENTS.md's write-atomically rule.** | comment in `preserve-state.sh` |
| **One global state file** | only correct because of the GPU-mutex (one live model). If that stops holding, this must become per-model. | comment in `preserve-state.sh` |

## 4. The trap I fell into — read this before changing the cmd shape

Moving `scripts/switch.sh` from an argv element into script text **silently broke forcing.**
`app.py::_with_force()` splices argv and guards on `if "scripts/switch.sh" in cmd` — a list-
membership test that quietly evaluated False, so `--force` was never injected. Pressing `f` on an
unsafe gate would have booted *unforced*.

That is the **same bug class the whole change set is about**: something changed shape, and a
consumer pattern-matching on it silently stopped working. I reintroduced it one layer up.

- The fast tests did **not** catch it. Only the full `test_app_headless.py` did.
- 5 of the 7 failures printed `switch.sh "$slug"` with no `--force` — the tests were saying exactly
  what was wrong, and I first misread them as assertion churn. **Read the assertion payload.**
- Fix: `_with_force()` now swaps in `CockpitData._serve_script(force=True)` — the same builder
  `serve()` uses — instead of splicing argv. A legacy branch still handles the old shape.

**If you change the serve cmd shape again**, grep for consumers first:
`grep -rn 'scripts/switch.sh' tools/serve-cockpit/club3090_cockpit/*.py` and
`grep -rn 'cmd\[-1\]\|"--force" in' tools/serve-cockpit/`.

## 5. Second bug, found in the same suite run (SPEC_N)

Two `TestServeOverrides` failures were **pre-existing** (baselined against a clean stash), and
genuinely broken — not flaky. Commit `72272e60` parameterized a compose:

```diff
- '{"method":"mtp","num_speculative_tokens":3}'
+ '{"method":"mtp","num_speculative_tokens":${MTP_SPEC_TOKENS:-3}}'
```

`serve_override_defaults()` matched `"num_speculative_tokens"\s*:\s*(\d+)` — literal digits only —
so it returned `SPEC_N=""` and degraded the drafter label from `MTP n=3` to `MTP`.

**Blast radius was wider than the tests showed:** the tests only cover `vllm/dual`, but **13
composes** use the interpolated form under **five** variable names (`SPEC_N`, `SPEC_N_MAX`,
`NUM_SPEC_TOKENS`, `MTP_SPEC_TOKENS`, …). Fixed by matching an optional `${VAR:-` prefix and
reading the default.

Two things verified rather than assumed:
- The 11 **Gemma** composes still parse empty — **correct**. Their `num_speculative_tokens` hits are
  inside comment blocks; MTP is deliberately disabled there (v0.24.0 breaks tool-calling). No active
  drafter → no label.
- That exposed that the parser reads commented lines, so every slug that *does* match was checked to
  be matching a live, uncommented config line.

**Left deliberately:** the `"method"` regex stays literal-only — it's a quoted literal in all 28
shipped composes, so generalizing would be speculative. Noted in-code so it reads as a decision.

## 6. Verification performed (reproduce before trusting a refactor)

- **77/77 compose parity.** The ported resolver was diffed against `serve.sh`'s original logic
  across every compose — zero mismatches. This is THE test that matters if you touch resolution:
  extract serve.sh's `_compose_default`/`_env_override`/`_effective` into a harness and compare.
- **Full cockpit suite: `876 passed, 0 failed`** (~8.7 min).
  ```bash
  cd tools/serve-cockpit && uv run --with pytest --with pytest-asyncio \
    --with-editable ../tui-core --with-editable . pytest tests/ -q > /tmp/suite.log 2>&1
  ```
  ⚠️ `pytest` is **not** installed in the repo `.venv`; the above uses an ephemeral `uv` env so the
  venv isn't mutated. It writes `tools/serve-cockpit/uv.lock` — **delete it afterwards.**
  ⚠️ Redirect to a log, never pipe to `tail` (AGENTS.md: a pipe buffers until exit).
- **Behavioral tests with stubs:** switch.sh failure propagates its exit code and skips the sync;
  preserve-state failure is non-fatal (model is already up); slug `evil; touch /tmp/PWNED` does not
  inject; forcing is idempotent; non-serve plans untouched.
- `scripts/tests/test-locale-utf8.sh` passes with the new script (it carries the required
  `export PYTHONUTF8="${PYTHONUTF8:-1}"` above its first `python3` call).

**Known flaky:** `test_one_tab_from_tab_bar_reaches_scene_table` failed once in a combined run,
then passed 3/3 alone and 571/571 in its own file. It's a Textual focus assertion driven by
`pilot.pause()`. Nothing in this change touches focus or tab order.

## 7. State of the tree / open items

**UNCOMMITTED** — 7 modified files + `scripts/preserve-state.sh` (new, +x) + `DAILY_DRIVERS.md`
(new, a model tier-list unrelated to this code change). Net ≈ +134/−49.

1. **`services/litellm/preserve_state.json` was left at `{"mode":"off","window":0}`** (was `full`).
   Changed as a side effect of testing; left at `off` deliberately — `full` was the stale buggy
   value, nothing was running, and the next boot rewrites it. Flip it back if you disagree.
2. **Nothing is committed.** Suggested split: (a) `preserve-state.sh` + `serve.sh` + cockpit wiring
   + tests, (b) the SPEC_N parser fix, (c) `DAILY_DRIVERS.md`.
3. The SPEC_N fix is arguably **upstreamable** — the parser bug affects any rig using a
   parameterized compose, not just this one.
4. `DAILY_DRIVERS.md` §"9.0" documents that both launchers are now equivalent for correctness; if
   this change is reverted, that line becomes wrong.
