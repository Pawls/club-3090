# MODEL_REFERENCE — which model for which task (Paul's rig)

Unified successor to the scattered tables in `REMINDER.md`. `REMINDER.md` stays the *run-notes*
(gotchas, Hermes config, WSL2 lore); **this file is the picker**: what each model can do as
configured today, what it could do with tweaks, and the current `.env` override state.

- **Fact-checked 2026-07-10** against the actual compose files and the `.env` sitting in each
  compose's own directory (the only `.env` docker compose loads). §5 is the dated env snapshot.
- **Launch everything with `./serve.sh <name>`** (GPU-mutex aware: evicts the current model
  first). `./serve.sh --list` groups canonical vs ours and flags missing weights;
  `./serve.sh --pull <name>` downloads them.
- **GPU-mutex:** exactly ONE GPU model at a time. LiteLLM (`:4000`) + Hermes stay up always.

## 0. Reading the numbers (rig context)

This rig: 2× RTX 3090, WSL2, **GPU0 = PCIe x16 (also drives the Windows desktop, ~1.7 GB),
GPU1 = PCIe 4.0 x4**. The x4 link chokes vLLM TP=2 all-reduce → local dual-vLLM decode is
~half the repo's reference numbers ([[rig-pcie-x4-bottleneck]]).

TPS labels in the tables:
- **(L)** = measured on THIS rig (REMINDER/compose notes, 2026-07).
- **(R)** = repo reference rig (`@noonghunna`, 2×3090 with proper lanes, mostly 370 W caps).
  For **vLLM TP=2** expect roughly *half* of (R) here. For **llama.cpp/ik/beellama layer-split**
  (deckard, hauhau, carnice) the cards exchange one small activation per token, not per-layer
  all-reduces, so (R) should transfer *approximately* — but none of those three have been
  independently re-benched on this rig. Single-card numbers transfer best.

## 1. Capability table — as configured TODAY (compose + current `.env`)

Single card (GPU1 → desktop stays snappy on GPU0):

| serve.sh name | Port | Engine | Ctx | TPS narr/code | Conc. | MTP | Vision | Thinking | Alignment | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **apex** ⭐ daily text | 8056 | ik-llama | **262K** | ~84–90 (L) | 1 | off¹ | — | ON, strip¹⁰ | stock | ✅ |
| **apex-vision-ik** ⭐ screenshots | 8057 | ik-llama | 131K | ≈apex (est.) | 1 | off¹ | ✅ image | ON, strip¹⁰ · windowable¹¹ | stock | 🧪 |
| apex-vision-mainline (backup) | 8058 | llama.cpp | 200K | unmeasured² | 1 | none | ✅ image | OFF | stock | 🧪 |
| **byteshape-vision** (stock A/B) | 8059 | ik-llama | 131K | ≈apex (est.) | 1 | off (dropped) | ✅ image | OFF, windowable¹¹ | **stock** (not apex) | 🧪 |
| apex-fit (alt lane) | 8057 | ik-llama | 262K³ | 103/149 (R, w/ MTP)³ | 1 | off¹ | — | ON, strip¹⁰ | stock | ✅ |
| **27b-vision** ⭐ BoxelBuilder | 8020 | ik-llama | **160K** | ~51 (est.) | 1 | n=2 | ✅ image | OFF | stock | ✅ |
| 27b-single | 8021 | vLLM | **28K** | 57.8/80.0 (L) | 1 (hard) | n=3 (hard) | ✅ image | OFF + preserve | stock | 🧪 |
| 27b-minimal (fallback) | 8021 | vLLM | 65K | ~32/33 (R) | 1 | none | — | OFF | stock | ✅ |

Dual card (both 3090s):

| serve.sh name | Port | Engine | Ctx | TPS narr/code | Conc. | MTP | Vision | Thinking | Alignment | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **deckard** ⭐ hard reasoning | 8199 | llama.cpp | 131K | 36/46 (R)⁴ | 1 | n=2 (+59/104%) | — | ON + preserve | **uncensored** | ✅ |
| deckard-vision | 8200 | llama.cpp | 131K | ≈deckard −ε | 1 | n=2 kept | ✅ image | ON + preserve | **uncensored** | 🧪 |
| hauhau | 8073 | llama.cpp | **262K** | 113/~150 (R)⁴ | 1 | n=3 (code-max)⁵ | — | ON, strip¹⁰ | **uncensored** | 🧪 |
| hauhau-vision | 8073 | llama.cpp | 262K | ≈hauhau (est.) | 1 | n=3 kept | ✅ image | ON, strip¹⁰ | **uncensored** | 🧪 |
| carnice (agent brain) | 8070 | beellama | **262K** | 40.7/44.0 (R)⁴ | 1 | n=1 (n=2 = +13%) | — | ON⁶ + preserve | stock (agentic SFT) | 🧪 |
| **27b** ⭐ big-ctx images | 8010 | vLLM TP=2 | **262K** | 54.0/67.8 (L) | 2 | n=3 | ✅ image | ON + preserve | stock | ✅ |
| **35b-a3b** ⭐ subagents | 8051 | vLLM TP=2 | **262K** | ~85 (L); **agg ~265 @N=4** | **4** | none⁷ | ✅ 2 images | ON, strip¹⁰ | stock | ✅ |
| agents-a1 | 8072 | vLLM TP=2 | **262K** | ~85 (L est.); 154 (R) | 1 | none (no head) | ✅ 2 images | OFF⁸ (hook: think ON, strip¹⁰) | stock | ⚠️ |
| omni (media in) | 8042 | vLLM-Omni | **48K**⁹ | ~164 text (R) | 8/stage | none | ✅ image/audio/video | not a thinking model | stock | 🧪 |
| apex-yarn-1m (park) | 8057 | ik-llama | 1M (YaRN 4×) | unmeasured | 1 | n=5 | — | OFF (no .env here) | stock | 🧪 deprecate? |

¹ MTP A/B'd on this rig 2026-07-08: **loses at every depth on the APEX MoE** (off ~84 · n=2 ~55 ·
n=4 ~65 TPS) — a3b's ~3B active params make base decode cheaper than the draft pass. Kept off via
`MTP_DRAFT_N_MAX=0` in the shared apex `.env`.
² Mainline llama.cpp CUDA decode may actually be *faster* than ik on this MoE (LM Studio's mainline
build hit 135–145 TPS single-GPU) — never A/B'd in our stack; separate investigation if it matters.
³ apex-fit defaults 196K/MTP n=4, but it shares the apex `.env` → effectively 262K/MTP-off here,
i.e. ≈ `apex` plus asymmetric q8_0/q5_0 KV + `--no-mmap` + `--cache-ram`. The 103/149 (R) was
measured *with* MTP n=4; expect ≈apex speeds as configured. Port 8057 collides with apex-vision-ik
(GPU-mutex makes it moot, but don't register both).
⁴ Reference-rig number; layer-split ⇒ should approximately transfer, but **not re-measured here**.
⁵ Hauhau MTP n=3 maximizes code (+10% @262K) and costs −9% prose; `MTP_DRAFT_N_MAX=1` is the
prose-optimal setting, 2 balanced.
⁶ Carnice thinking is ON via the *compose default*. Its `.env` had set `ENABLE_THINKING=true`,
which the beellama compose never reads (it reads `REASONING`) — fixed to `REASONING=on`
2026-07-10, same effective behavior. `--reasoning-budget 4096` caps think-block length.
⁷ MTP measured **−45–51%** on the 35B MoE under vLLM TP=2 (draft pass adds inter-GPU syncs) —
deliberately not in the compose. Not a missing feature.
⁸ agents-a1's compose hardcodes `enable_thinking: false` (generated file, "never hand-repaired").
Thinking-ON — its stronger mode, 110/150 — only happens via the LiteLLM `custom_hooks.py`
`AGENTS_A1_THINKING` hook, i.e. **through `:4000`**, or a per-request `chat_template_kwargs`.
Hitting `:8072` directly = thinking OFF.
⁹ Omni's stages run 49152 in the deploy-config because GPU0 shares ~1.7 GB with the desktop;
native max is 65536 (see §3). Also: requests MUST carry `"modalities":["text"]` and a sane
`max_tokens` — the LiteLLM hook injects both; **use it via `:4000` or Open WebUI, not raw**.
¹⁰ Preserve default flipped to **strip** on the a3b MoE lanes (2026-07-20): replaying every
prior turn's `<think>` seeded thought-loops + cost quality on always-reasoning a3b (apex family,
hauhau, 35b-a3b, agents-a1). Thinking itself stays ON — only the prior-turn carryover is dropped.
`./serve.sh <name> --preserve` re-enables per-boot. For the vLLM `35b-a3b`/`agents-a1` routes the
LiteLLM re-inline hook (`custom_hooks.py`) was also updated so it can't re-inject the reasoning.
27b keeps preserve (real replay path, not a loop source).
¹¹ byteshape-vision (2026-07-20): the **stock** Qwen3.6-35B-A3B single-card vision lane (byteshape
IQ4_XS = stock base + MTP head, NOT the apex fine-tune), added as (a) the apex-vs-stock loop A/B —
run it with `--preserve` and see if stock loops the way apex does, isolating fine-tune vs base — and
(b) the `--preserve-window` test bed. It mounts the custom template (same standard Qwen3.6 template +
the `preserve_window` branch), so the A/B controls for template. 🧪 pending the full gate.
Same day, **apex-vision-ik** was converted from native `--jinja` to this same custom template
(`--jinja` + `--chat-template-file`) so `--preserve-window` works on the screenshot driver too —
the template's `render_content` handles image/video markers (Qwen3-VL), so vision is retained;
re-verify an image round-trip at first boot since the template path changed.

### Refusals / alignment (asked for "refusal rates" — honest answer)

**No refusal-rate benchmark has ever been run in this repo** — there is no measured number to
put in a column. What exists is categorical:

- **deckard** (DavidAU Opus-Deckard merge) and **hauhau** ("HauhauCS-Aggressive") are community
  **uncensored/abliterated** fine-tunes — "complies with content a base model refuses." Repo's own
  caveat: uncensoring buys *compliance, not capability*; don't expose them on shared endpoints.
- Everything else runs stock Qwen alignment (occasional refusals on edgy-but-legit asks).
- If you want a real number: `benchlocal-cli` has no refusal pack; it would be a custom prompt set.

## 2. Coding / agentic quality — "can it write a program end to end?"

8-pack = `quality-test.sh` behavioral suite, /150 (higher = better). The packs that predict
*end-to-end coding with self-correction* are **cli-40** (drive a real CLI sandbox), **bugfind**,
**aider-polyglot** (edit-compile-test loops), and ToolCall. All rows below are reference-rig runs.

| Model | 8-pack | ToolCall | cli-40 | bugfind | aider-30 | Read |
|---|---|---|---|---|---|---|
| agents-a1 (thinking ON) | **110** | 15/15 (off) | **23/40** ← best CLI | — | — | best pure agent; hermes-pack *regresses* with thinking ON (12→9) |
| 27b (dual) | **109** | — | — | — | — | best all-rounder score; vision + 262K + MTP speed |
| 35b-a3b (dual) | det 79/90 (88%) | — | 14/40 | — | 13/30 | deterministic packs strong; sandboxed agentic middling; the *speed* pick |
| deckard | 105 | 15/15 | 15/40 | 13/15 | — | strongest thinker per-token, slowest per-token — deliberation trade |
| hauhau | 103 off / 105 on | 14/14 | 17→19/40 | 13/13 | — | uncensored MoE, quality ≈ carnice/deckard tier |
| carnice | 103 off / 105 on | 13/14 | 17/40 | 12/13 | — | agentic-SFT flavor; MTP accept ~81% |
| apex-fit (≈ apex weights) | det 76/90 (84%) | 14/15 | 12/40 | 10/15 | 12/30 | daily driver's quality tier — fine for everyday code, not the deep-agent pick |

Rules of thumb (unchanged from REMINDER, now with receipts):
- **Fast iterative coding, one task at a time** → `apex` (262K, ~90 TPS, thinking ON).
- **Parallel subagents / agent fan-out** → `35b-a3b` (only real concurrency: N=4, agg ~265 TPS).
- **Hardest debugging / architecture with images of the failing UI** → `27b` (109/150 + vision + 262K)
  or `deckard` when you want maximum deliberation and don't care about speed.
- **Agent-harness workloads (Hermes/CLI agents)** → `agents-a1` *via LiteLLM* — best cli-40 — or `carnice`.
- **Anything a stock model refuses** → `deckard` / `hauhau`.

## 3. Hypothetical max context — what tweaks buy (goal: 262K, without lobotomizing code)

Already AT 262K as configured — nothing to do: **apex, apex-fit, hauhau, hauhau-vision, carnice,
27b (dual), 35b-a3b, agents-a1.** Three of those keep vision at 262K: **27b, 35b-a3b, hauhau-vision.**
So "262K + vision + clever code" is not hypothetical — it's `27b` / `35b-a3b` today; the tweaks
below only matter for the models NOT already there.

| Model | Today | Realistic max | Lever(s) | Cost / risk |
|---|---|---|---|---|
| 27b-single | 28K | ~31–32K | util already 0.94 (max safe); nothing else left — MTP n=3 is hardcoded | dead end on one card |
| ↳ via 27b-minimal | 65K | 65K | drop MTP **and** vision (that's what minimal.yml is) | ~32 TPS (−60%), no images; still not 262K |
| ↳ real fix | — | **262K** | run the dual compose (`27b`) | costs both cards; that's the whole trade |
| apex-vision-ik | 131K | **~160K** probe | raise `VISION_CTX_SIZE` stepwise (KV is already q4_0); the ik `27b-vision` sibling ships at ~160K, expect similar | boot-OOM risk; test, don't assume. 262K+vision on ONE card: not realistic |
| 27b-vision | 160K | 160K (shipped default) | already at the verified [8/8] vision ceiling (KV q4_0 + fill overhead, not the image buffer); 180K OOMs at fill. Push higher only by dropping MTP + asym KV, then re-verify at fill | boot-OOM risk above 160K; the 262K text sibling is dual-card only |
| apex-vision-mainline | 200K | **262K** maybe | raise `CTX_SIZE` toward 262144 (compose header: "if boot VRAM allows"); V-cache already q5_0 | unverified; mainline engine, thinking currently OFF here |
| deckard | 131K | ~192K unverified | (a) `KV_TYPE=q4_0` (~halves KV; untested on this model) (b) `MTP_DRAFT_N_MAX=0` frees ~1.2 GB draft ctx (c) Q5_K_M weights (−4.6 GB) | (a) quality risk on a *deliberation* model — bad trade; (b) loses +59–104% speed; (c) quant-quality hit. **Recommendation: leave at 131K**; 40B dense KV is the expensive kind. 192K OOM'd at q8. |
| deckard-vision | 131K | same as deckard −2.1 GB headroom (mmproj+draft) | same levers | same verdict |
| omni | 48K | **65,536 (hard ceiling)** | restore `max_model_len: 65536` in `qwen3_omni_3090.yaml` stages 0+1 — fits only if GPU0 isn't carrying the Windows desktop (~1.7 GB); it missed by ~20 MiB last time | can't exceed 64K ever (model native max). Speech output needs `KV_CACHE_DTYPE=auto` → even less ctx |
| apex-yarn-1m | 1M | 1M (it boots… allegedly) | already YaRN 4× | static YaRN degrades accuracy at **all** positions incl. short prompts — exactly the "worthless for code" failure you don't want. Park/deprecate; weights (I-Quality, ~23.5 GB) not even downloaded |

Levers that DON'T exist, to save you looking:
- vLLM composes here have **no vision-off switch** that buys context. The 35B's vision flag is
  `--limit-mm-per-prompt image=2` (→0 disables images but frees ~nothing — the DeltaNet-hybrid KV
  is already cheap and 262K already fits). The only compose where dropping vision bought context
  is 27b `minimal.yml`, and it dropped MTP at the same time.
- Deckard "disable vision": the **text compose (`deckard`) IS the no-vision option** — that's the
  answer to "the tables don't show Deckard without vision." Same weights, same 131K, `-vision` just
  adds the mmproj (+~2.1 GB VRAM incl. draft ctx). Both are in `./serve.sh --list`.

## 4. Vision decision table (since it's the recurring question)

| Need | Pick | Why |
|---|---|---|
| Quick screenshot on the daily driver | **apex-vision-ik** :8057 | keeps ~90 TPS single-card; 131K is plenty for chat+image |
| Image analysis inside a BIG context (Hermes ~40K overhead + code) | **27b** :8010 | 262K + vision + MTP; best 8-pack |
| Vision + parallel subagents | **35b-a3b** :8051 | 262K + vision + N=4 |
| Uncensored + images | **deckard-vision** / **hauhau-vision** | only uncensored vision paths |
| Video / audio understanding | **omni** :8042 via Open WebUI/LiteLLM | only multimodal-in model; upload extracted FRAMES, never the video file (filename-hallucination trap — REMINDER §11) |

## 5. Current `.env` overrides — snapshot **2026-07-10** (volatile; re-check date before trusting)

All `.env`s are tracked on `pawl-custom` (the blanket gitignore was removed 2026-07-10).
Docker compose only loads the `.env` in the compose's OWN directory — the values below are the
ones actually in effect.

**Override thinking / preserve at launch — no `.env` edit** (added 2026-07-12). The `.env` values
below are just the *defaults*; `serve.sh` can override the two request-shaping knobs per launch,
because an exported shell var beats the compose-dir `.env` in docker-compose interpolation:

```
./serve.sh 27b --no-think              # this boot only; .env unchanged
./serve.sh apex --think --no-preserve
```

| Flag | Effect | Wires (auto-detected per engine) |
|---|---|---|
| `--think` / `--no-think` | reasoning on/off | vLLM `ENABLE_THINKING` true/false · llama.cpp/ik/beellama `REASONING` on/off |
| `--preserve` / `--no-preserve` | keep vs strip prior-turn `<think>` from ctx | `PRESERVE_THINKING` true/false (all engines) |
| `--preserve-window <N>` | **bounded** carryover: keep `<think>` from only the last N query blocks (0=off · 1=current only · 2=last two · ≥total=all). The anti-thought-loop middle ground between strip and preserve-all. | `PRESERVE_THINKING_WINDOW` → custom `apex-qwen-chat-template.jinja` `preserve_window` kwarg. **Custom-template lanes only** (`apex`, `apex-fit`, `apex-vision-ik`, `byteshape-vision`, + apex `mtp`/`long`); errors on native-template / vLLM lanes. Mutually exclusive with `--no-preserve`. |

- **Neither costs VRAM.** KV pool, concurrency (`max_num_seqs`), MTP depth, ctx, and quant are all
  fixed at engine boot — these flags only shape the request. So they're flags, not separate composes;
  disabling thinking does **not** free room to raise any of those (it only frees *runtime* tokens/KV).
- `./serve.sh --list` shows each model's current state as `{think:X preserve:Y}`. `·fixed` =
  hardcoded in the compose (**agents-a1**, **27b-minimal** — not env-togglable; agents-a1's
  thinking-ON is the LiteLLM hook, n.8); `n/a` = **omni** (not a thinking model) or **apex-yarn-1m**
  (compose doesn't thread `preserve_thinking`). A flag that can't be honored errors, never no-ops.
- Ideal for A/B through Hermes (GPU-mutex serializes anyway): `./serve.sh 27b --think` → run pack →
  `./serve.sh 27b --no-think` → run pack; same served name through `:4000`.

| Compose dir (`models/…`) | Effective overrides |
|---|---|
| `qwen3.6-35b-a3b/ik-llama/…/single/mudler-apex-compact/` (apex, apex-fit, apex-vision-ik share it) | `CUDA_VISIBLE_DEVICES=1` · `MTP_DRAFT_N_MAX=0` · `NP=1` · `CTX_SIZE=262144` · `UBATCH_SIZE=1024` · `REASONING=on` · `PRESERVE_THINKING=false`¹⁰ |
| `qwen3.6-35b-a3b/llama-cpp/…/single/mudler-apex-compact/` (apex-vision-mainline) | `CUDA_VISIBLE_DEVICES=1` · `CTX_SIZE=200000` · `KV_TYPE_K=q8_0` · `KV_TYPE_V=q5_0` · `REASONING=off` · `PRESERVE_THINKING=false` · `SERVED_NAME=apex-35b-vision` · `PORT=8058` |
| `qwen3.6-40b-deckard/llama-cpp/…/dual/piehsoft-q6k/` | `CUDA_VISIBLE_DEVICES=0,1` · `TENSOR_SPLIT=1.1,1.0` (mtp.yml only — vision.yml uses its own even-split var) · `CTX_SIZE=131072` · `KV_TYPE=q8_0` · `NP=1` · `MTP_DRAFT_N_MAX=2` · `REASONING=on` · `PRESERVE_THINKING=true` · `PORT=8199` |
| `qwen3.6-35b-a3b/llama-cpp/…/dual/morikomorizz-q6kp/` | `CUDA_VISIBLE_DEVICES=0,1` · `TENSOR_SPLIT=0.5,0.5` (even — GPU0 carries the desktop) · `CTX_SIZE=262144` · `KV_TYPE=q8_0` · `NP=1` · `MTP_DRAFT_N_MAX=3` · `REASONING=on` · `PRESERVE_THINKING=false`¹⁰ · `SERVED_NAME=hauhau-35b` · `PORT=8073` |
| `qwen3.6-27b/beellama/…/dual/carnice-v2-q8/` | `REASONING=on` (was `ENABLE_THINKING=true`, a no-op — fixed 2026-07-10, behavior unchanged) · `PRESERVE_THINKING=true` · `NO_MMAP=true` |
| `qwen3.6-27b/vllm/…/single/autoround-int4/` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:512` · `GPU_MEMORY_UTILIZATION=0.94` · `ENABLE_THINKING=false` · `PRESERVE_THINKING=true` |
| `qwen3.6-27b/vllm/…/dual/autoround-int4/` | `GPU_MEMORY_UTILIZATION=0.83` (desktop headroom) · `ENABLE_THINKING=true` · `PRESERVE_THINKING=true` (alloc-conf + `MTP_SPEC_TOKENS` commented out) |
| `qwen3.6-35b-a3b/vllm/…/dual/autoround-int4/` | `GPU_MEMORY_UTILIZATION=0.83` · `ENABLE_THINKING=true` · `PRESERVE_THINKING=false`¹⁰ (+ LiteLLM re-inline hook drops `qwen3.6-35b-a3b`) · `MAX_NUM_SEQS=4` |
| `qwen3-omni-30b-a3b/vllm-omni/…/dual/autoround-int4/` | `PORT=8042` only (ctx lives in `qwen3_omni_3090.yaml`, not `.env`) |
| `agents-a1/vllm/…/dual/fp8-dynamic/` | `TEMP=0.85` `TOP_P=0.95` `TOP_K=20` `MIN_P=0.0` `REPEAT_PENALTY=1.0` · `PORT=8072` (util + max-len commented → 0.92 / 262144) |
| `qwen3.6-35b-a3b/ik-llama/…/dual/mudler-apex-quality/` (yarn-1m) | **no `.env` exists** → pure compose defaults (incl. `REASONING=off`) |

Traps confirmed while fact-checking:
- **agents-a1 has a second `.env`** at `agents-a1/vllm/compose/.env` carrying the WSL2
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` fix — **silently ignored** when launching
  from the quant dir (the known .env-location bug, CLAUDE.md). If agents-a1 hits the
  `gptq_marlin_repack`/`cudaErrorNotReady` boot crash, copy that line into
  `dual/fp8-dynamic/.env`.
- The agents-a1 `.env` comment "weights NOT downloaded" is **stale** — 36 GB present on disk.
- 27b/35b vLLM duals run `GPU_MEMORY_UTILIZATION=0.83`; the 262K/quality numbers were validated
  at 0.92. Both boot at 0.83, but the KV pool (and 27b's 2-seq @262K headroom) is smaller than
  the compose headers advertise.

## 6. Fact-check deltas vs REMINDER.md (found 2026-07-10)

- **27b-single does NOT sit on GPU1.** REMINDER §1 groups it under "single-card models sit on
  GPU1"; the compose's `CUDA_VISIBLE_DEVICES=0` example is *commented out* and TP=1 grabs
  device 0 → it lands on **GPU0, the desktop card**. Pin it in the compose-dir `.env` if that matters.
- **Carnice `ENABLE_THINKING` was a dead variable** (compose reads `REASONING`); thinking was ON
  only because the compose default is `on`. `.env` fixed to say what it means.
- **serve.sh could never evict carnice** — its container prefix `beellama-` wasn't in the
  eviction regex. Fixed in serve.sh 2026-07-10.
- **Deckard's ~41 TPS is a reference-rig number** reused locally, not a local measurement
  (likely close — layer-split — but unmeasured; same for hauhau's 113 and carnice's 41/44).
- The hauhau HF repo ships its **own mmproj** (`mmproj-…HauhauCS-MTP-f16.gguf`); our vision
  compose uses the shared unsloth `mmproj-BF16` instead. Works (same base arch), but if hauhau
  vision ever misbehaves, the model-matched projector is the first thing to try.
- **apex-fit + apex-vision-ik + apex-yarn-1m all default to port 8057.** GPU-mutex means they
  never collide at runtime; just don't hand three Hermes providers the same port.
