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
  (hauhau) the cards exchange one small activation per token, not per-layer
  all-reduces, so (R) should transfer *approximately* — but it hasn't been
  independently re-benched on this rig. Single-card numbers transfer best.

## 1. Capability table — as configured TODAY (compose + current `.env`)

Single card (GPU1 → desktop stays snappy on GPU0):

| serve.sh name | Port | Engine | Ctx | TPS narr/code | Conc. | MTP | Vision | Thinking | Alignment | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **apex** ⭐ daily text | 8056 | ik-llama | **262K** | ~84–90 (L) | 1 | off¹ | — | ON, strip⁹ | stock | ✅ |
| **apex-vision-ik** ⭐ screenshots | 8057 | ik-llama | 131K | ≈apex (est.) | 1 | off¹ | ✅ image | ON, strip⁹ · windowable¹⁰ | stock | 🧪 |
| apex-vision-mainline (backup) | 8058 | llama.cpp | 200K | unmeasured² | 1 | none | ✅ image | OFF | stock | 🧪 |
| **byteshape-vision** (stock A/B) | 8059 | ik-llama | 131K | ≈apex (est.) | 1 | off (dropped) | ✅ image | OFF, windowable¹⁰ | **stock** (not apex) | 🧪 |
| apex-fit (alt lane) | 8057 | ik-llama | 262K³ | 103/149 (R, w/ MTP)³ | 1 | off¹ | — | ON, strip⁹ | stock | ✅ |

Dual card (both 3090s):

| serve.sh name | Port | Engine | Ctx | TPS narr/code | Conc. | MTP | Vision | Thinking | Alignment | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| hauhau | 8073 | llama.cpp | **262K** | 113/~150 (R)⁴ | 1 | n=3 (code-max)⁵ | — | ON, strip⁹ | **uncensored** | 🧪 |
| hauhau-vision | 8073 | llama.cpp | 262K | ≈hauhau (est.) | 1 | n=3 kept | ✅ image | ON, strip⁹ | **uncensored** | 🧪 |
| **35b-a3b** ⭐ subagents | 8051 | vLLM TP=2 | **262K** | ~85 (L); **agg ~265 @N=4** | **4** | none⁶ | ✅ 2 images | ON, strip⁹ | stock | ✅ |
| agents-a1 | 8072 | vLLM TP=2 | **262K** | ~85 (L est.); 154 (R) | 1 | none (no head) | ✅ 2 images | OFF⁷ (hook: think ON, strip⁹) | stock | ⚠️ |
| omni (media in) | 8042 | vLLM-Omni | **48K**⁸ | ~164 text (R) | 8/stage | none | ✅ image/audio/video | not a thinking model | stock | 🧪 |
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
⁶ MTP measured **−45–51%** on the 35B MoE under vLLM TP=2 (draft pass adds inter-GPU syncs) —
deliberately not in the compose. Not a missing feature.
⁷ agents-a1's compose hardcodes `enable_thinking: false` (generated file, "never hand-repaired").
Thinking-ON — its stronger mode, 110/150 — only happens via the LiteLLM `custom_hooks.py`
`AGENTS_A1_THINKING` hook, i.e. **through `:4000`**, or a per-request `chat_template_kwargs`.
Hitting `:8072` directly = thinking OFF.
⁸ Omni's stages run 49152 in the deploy-config because GPU0 shares ~1.7 GB with the desktop;
native max is 65536 (see §3). Also: requests MUST carry `"modalities":["text"]` and a sane
`max_tokens` — the LiteLLM hook injects both; **use it via `:4000` or Open WebUI, not raw**.
⁹ Preserve default flipped to **strip** on the a3b MoE lanes (2026-07-20): replaying every
prior turn's `<think>` seeded thought-loops + cost quality on always-reasoning a3b (apex family,
hauhau, 35b-a3b, agents-a1). Thinking itself stays ON — only the prior-turn carryover is dropped.
`./serve.sh <name> --preserve` re-enables per-boot. For the vLLM `35b-a3b`/`agents-a1` routes the
LiteLLM re-inline hook (`custom_hooks.py`) was also updated so it can't re-inject the reasoning.
¹⁰ byteshape-vision (2026-07-20): the **stock** Qwen3.6-35B-A3B single-card vision lane (byteshape
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

- **hauhau** ("HauhauCS-Aggressive") is a community **uncensored/abliterated** fine-tune —
  "complies with content a base model refuses." Repo's own caveat: uncensoring buys
  *compliance, not capability*; don't expose it on shared endpoints.
- Everything else runs stock Qwen alignment (occasional refusals on edgy-but-legit asks).
- If you want a real number: `benchlocal-cli` has no refusal pack; it would be a custom prompt set.

## 2. Coding / agentic quality — "can it write a program end to end?"

8-pack = `quality-test.sh` behavioral suite, /150 (higher = better). The packs that predict
*end-to-end coding with self-correction* are **cli-40** (drive a real CLI sandbox), **bugfind**,
**aider-polyglot** (edit-compile-test loops), and ToolCall. All rows below are reference-rig runs.

| Model | 8-pack | ToolCall | cli-40 | bugfind | aider-30 | Read |
|---|---|---|---|---|---|---|
| agents-a1 (thinking ON) | **110** | 15/15 (off) | **23/40** ← best CLI | — | — | best pure agent; hermes-pack *regresses* with thinking ON (12→9) |
| 35b-a3b (dual) | det 79/90 (88%) | — | 14/40 | — | 13/30 | deterministic packs strong; sandboxed agentic middling; the *speed* pick |
| hauhau | 103 off / 105 on | 14/14 | 17→19/40 | 13/13 | — | uncensored MoE, strong reasoning quality |
| apex-fit (≈ apex weights) | det 76/90 (84%) | 14/15 | 12/40 | 10/15 | 12/30 | daily driver's quality tier — fine for everyday code, not the deep-agent pick |

> `qwen3.6-27b` ("27b", 109/150) and its beellama "Carnice" fine-tune (103/105) were retired from
> the catalog 2026-09; `deckard` (105) was retired too. No like-for-like replacement was
> re-benched here — see the rules of thumb below for the closest current substitute per use case.

Rules of thumb (unchanged from REMINDER, now with receipts):
- **Fast iterative coding, one task at a time** → `apex` (262K, ~90 TPS, thinking ON).
- **Parallel subagents / agent fan-out** → `35b-a3b` (only real concurrency: N=4, agg ~265 TPS).
- **Hardest debugging / architecture with images of the failing UI** → `35b-a3b` (262K + vision +
  N=4; the former `27b`/`deckard` picks are retired — nothing here has re-earned that slot with a
  fresh 8-pack run, so don't assume `35b-a3b` matches their old scores, just that it's the closest
  living option).
- **Agent-harness workloads (Hermes/CLI agents)** → `agents-a1` *via LiteLLM* — best cli-40.
- **Anything a stock model refuses** → `hauhau`.

## 3. Hypothetical max context — what tweaks buy (goal: 262K, without lobotomizing code)

Already AT 262K as configured — nothing to do: **apex, apex-fit, hauhau, hauhau-vision,
35b-a3b, agents-a1.** Two of those keep vision at 262K: **35b-a3b, hauhau-vision.**
So "262K + vision + clever code" is not hypothetical — it's `35b-a3b` today; the tweaks
below only matter for the models NOT already there.

| Model | Today | Realistic max | Lever(s) | Cost / risk |
|---|---|---|---|---|
| apex-vision-ik | 131K | **~160K** probe | raise `VISION_CTX_SIZE` stepwise (KV is already q4_0) | boot-OOM risk; test, don't assume. 262K+vision on ONE card: not realistic |
| apex-vision-mainline | 200K | **262K** maybe | raise `CTX_SIZE` toward 262144 (compose header: "if boot VRAM allows"); V-cache already q5_0 | unverified; mainline engine, thinking currently OFF here |
| omni | 48K | **65,536 (hard ceiling)** | restore `max_model_len: 65536` in `qwen3_omni_3090.yaml` stages 0+1 — fits only if GPU0 isn't carrying the Windows desktop (~1.7 GB); it missed by ~20 MiB last time | can't exceed 64K ever (model native max). Speech output needs `KV_CACHE_DTYPE=auto` → even less ctx |
| apex-yarn-1m | 1M | 1M (it boots… allegedly) | already YaRN 4× | static YaRN degrades accuracy at **all** positions incl. short prompts — exactly the "worthless for code" failure you don't want. Park/deprecate; weights (I-Quality, ~23.5 GB) not even downloaded |

Levers that DON'T exist, to save you looking:
- vLLM composes here have **no vision-off switch** that buys context. The 35B's vision flag is
  `--limit-mm-per-prompt image=2` (→0 disables images but frees ~nothing — the DeltaNet-hybrid KV
  is already cheap and 262K already fits).

## 4. Vision decision table (since it's the recurring question)

| Need | Pick | Why |
|---|---|---|
| Quick screenshot on the daily driver | **apex-vision-ik** :8057 | keeps ~90 TPS single-card; 131K is plenty for chat+image |
| Image analysis inside a BIG context (Hermes ~40K overhead + code) | **35b-a3b** :8051 | 262K + vision + N=4; best living 8-pack option |
| Vision + parallel subagents | **35b-a3b** :8051 | 262K + vision + N=4 |
| Uncensored + images | **hauhau-vision** | only uncensored vision path (deckard-vision retired) |
| Video / audio understanding | **omni** :8042 via Open WebUI/LiteLLM | only multimodal-in model; upload extracted FRAMES, never the video file (filename-hallucination trap — REMINDER §11) |

## 5. Current `.env` overrides — snapshot **2026-07-10** (volatile; re-check date before trusting)

All `.env`s are tracked on `pawl-custom` (the blanket gitignore was removed 2026-07-10).
Docker compose only loads the `.env` in the compose's OWN directory — the values below are the
ones actually in effect.

**Override thinking / preserve at launch — no `.env` edit** (added 2026-07-12). The `.env` values
below are just the *defaults*; `serve.sh` can override the two request-shaping knobs per launch,
because an exported shell var beats the compose-dir `.env` in docker-compose interpolation:

```
./serve.sh 35b-a3b --no-think          # this boot only; .env unchanged
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
  hardcoded in the compose (**agents-a1** — not env-togglable; agents-a1's
  thinking-ON is the LiteLLM hook, n.7); `n/a` = **omni** (not a thinking model) or **apex-yarn-1m**
  (compose doesn't thread `preserve_thinking`). A flag that can't be honored errors, never no-ops.
- Ideal for A/B through Hermes (GPU-mutex serializes anyway): `./serve.sh 35b-a3b --think` → run pack →
  `./serve.sh 35b-a3b --no-think` → run pack; same served name through `:4000`.

| Compose dir (`models/…`) | Effective overrides |
|---|---|
| `qwen3.6-35b-a3b/ik-llama/…/single/mudler-apex-compact/` (apex, apex-fit, apex-vision-ik share it) | `CUDA_VISIBLE_DEVICES=1` · `MTP_DRAFT_N_MAX=0` · `NP=1` · `CTX_SIZE=262144` · `UBATCH_SIZE=1024` · `REASONING=off` (2026-08-07: was `on`; back to the upstream compose default) · `PRESERVE_THINKING=false`⁹ |
| `qwen3.6-35b-a3b/llama-cpp/…/single/mudler-apex-compact/` (apex-vision-mainline) | `CUDA_VISIBLE_DEVICES=1` · `CTX_SIZE=200000` · `KV_TYPE_K=q8_0` · `KV_TYPE_V=q5_0` · `REASONING=off` · `PRESERVE_THINKING=false` · `SERVED_NAME=apex-35b-vision` · `PORT=8058` |
| `qwen3.6-35b-a3b/llama-cpp/…/dual/morikomorizz-q6kp/` | `CUDA_VISIBLE_DEVICES=0,1` · `TENSOR_SPLIT=0.5,0.5` (even — GPU0 carries the desktop) · `CTX_SIZE=262144` · `KV_TYPE=q8_0` · `NP=1` · `MTP_DRAFT_N_MAX=3` · `REASONING=on` · `PRESERVE_THINKING=false`⁹ · `SERVED_NAME=hauhau-35b` · `PORT=8073` |
| `qwen3.6-35b-a3b/vllm/…/dual/autoround-int4/` | `GPU_MEMORY_UTILIZATION=0.86` · `ENABLE_THINKING=false` (2026-08-07: was `true`; flipped on a measured tie — GSM-Symbolic-30 30/30 both arms, ~3.8× latency for thinking) · `PRESERVE_THINKING=false`⁹ (+ LiteLLM re-inline hook drops `qwen3.6-35b-a3b`) · `MAX_NUM_SEQS=4` |
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
- The 35b vLLM dual runs `GPU_MEMORY_UTILIZATION=0.83`; the 262K/quality numbers were validated
  at 0.92. It boots at 0.83, but the KV pool is smaller than the compose header advertises.

## 6. Fact-check deltas vs REMINDER.md (found 2026-07-10)

- **Hauhau's ~113 TPS is a reference-rig number** reused locally, not a local measurement
  (likely close — layer-split — but unmeasured).
- The hauhau HF repo ships its **own mmproj** (`mmproj-…HauhauCS-MTP-f16.gguf`); our vision
  compose uses the shared unsloth `mmproj-BF16` instead. Works (same base arch), but if hauhau
  vision ever misbehaves, the model-matched projector is the first thing to try.
- **apex-fit + apex-vision-ik + apex-yarn-1m all default to port 8057.** GPU-mutex means they
  never collide at runtime; just don't hand three Hermes providers the same port.
