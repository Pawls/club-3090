# DAILY_DRIVERS — tier lists for Paul's rig (2026-08-09)

Companion to [MODEL_REFERENCE.md](MODEL_REFERENCE.md) (the capability picker, fact-checked
2026-07-10). This file is the **ranking**, and it re-ranks for two things MODEL_REFERENCE predates:

1. The **2026-08-01 status flip** that demoted `vllm/dual` and five siblings ✅ → ⚠️.
2. **Scope change:** ComfyUI / asset generation has moved to **LM Studio**. club-3090 is now used
   *only* when the goal is **highest precision + largest context + best performance at those
   values, with vision**. Both 3090s are available; nothing needs to stay resident on GPU0.

**Vision is a hard requirement**, and it prunes the field — several otherwise-top picks ship
`Vision: no` in their default compose and need a *sibling* compose to get it. §3a is the vision map.

Derived from the registry (`scripts/lib/profiles/compose_registry.py`), compose profile headers,
`BENCHMARKS.md`, and `docs/UPSTREAM.md` — not from the cockpit table alone.

**Legend:** (L) measured on this rig · (R) reference rig · ✅ production · ⚠️ caveats · 🧪 experimental.
In the cockpit, orange `✔` = `caveats`, green `✅` = `production` — that's the "purple checkmark".

---

## 0. The rig constraint that still decides everything

**GPU1 negotiates PCIe 4.0 x4** (GPU0 is x16), confirmed 2026-07-08 under sustained load. This
does **not** penalize all dual-card work equally — it penalizes exactly one thing:

| Path | Cross-card traffic | Penalty here |
|---|---|---|
| **vLLM TP=2** | all-reduce **every layer** | **~½ of reference** — 35B-A3B 182 (R) → **~85 (L)**; 27B 71/94 (R) → **54.0/67.8 (L)** |
| **llama.cpp / ik-llama layer-split** (`-ts`) | one small activation **per token** | should approximately transfer (not re-measured here) |
| **Single card** | none | transfers cleanly |

Under load the cards draw ~140 W of 350 at 55–77% util — a comm-bound stall, not a compute limit.

So with both cards free, the ordering that matters for you is:

> **layer-split dual (llama.cpp / ik-llama) ≥ single card > vLLM TP=2**

vLLM TP=2 only wins back its ground under **concurrency**, where batching amortizes the all-reduce
(you measured aggregate ~265 TPS @ N=4 on 35B-A3B against ~85 single-stream).

---

## 1. Why `vllm/dual` was demoted — and it is NOT "no benchmarks"

**Cause: [vllm#50021](https://github.com/vllm-project/vllm/pull/50021) — MTP × hybrid-GDN wild write. Open upstream, live in our pinned image.**

The GDN speculative-decode path builds a state-block index from an **unbounded** accepted-token
count. A count that is zero, stale, or too large indexes outside its tensor, and the garbage read
back gets dereferenced as a state-block address:

```
CUDA error: unspecified launch failure  +  Xid 13 (SM address exception) / Xid 31 (MMU fault)
→ dead worker
```

What matters:

- Fires **after minutes of agent-shaped traffic**, not at boot. A clean `verify-full` proves nothing.
- **Not quant-, topology-, or depth-specific.** Hit on Ampere and Ada (#758), and on 2× 5090 across
  NVFP4 *and* W4A8 (#838).
- **Lowering `n` cuts frequency, not exposure.** #758 called n=3 stable; #838 crashed *at* n=3.
- Our vendored [#48375](https://github.com/vllm-project/vllm/pull/48375) fixes a *different* fault
  (#43559, corruption) and explicitly does not cover this one.
- **Mitigation: run the drafter off** — delete `--speculative-config` from the compose. Clears when
  #50021 merges and a pin bump inherits it; our `v0.26.0` pin predates the fix.

Maintainer call 2026-08-01: rather than un-ship the composes, six exposed slugs (7 registry entries,
incl. the `vllm/dual` default and its alias) flipped to ⚠️ with a formal `Caveats:` line.
**Nine composes carried it**, seven of them on `qwen3.6-27b` (`vllm/dual` = `vllm/qwen-27b-dual-fast`,
`vllm/qwen-27b-dual-max`, `vllm/qwen-27b-dual-nvfp4`, `vllm/qwen-27b-single-nvfp4`,
`vllm/qwen-27b-multi-fast`, `vllm/qwen-27b-multi-max`, `vllm/qwen-27b-dual-lmcache`) — all retired
2026-09 along with the model. The two that remain live: `vllm/tess-dual-w4a16`,
`vllm/thinkingcap-dual-w4a8`.

**The common factor is `vLLM + built-in MTP drafter on a Qwen3.5/3.6 hybrid-GDN trunk`** — not the
weights, not the KV format, not TP. Hence the escape hatches:

- **Drafter-less vLLM slugs are unaffected** — `vllm/qwen-35b-a3b-dual` (no drafter by design),
  the a3b NVFP4 family. The a3b-nvfp4 BENCHMARKS row states it outright:
  *"MTP confirmed off in the boot log — vllm#50021 does not apply to this slug."*
  (`vllm/minimal`, the other example here, was the `qwen3.6-27b` fallback slug — retired 2026-09.)
- **llama.cpp / ik-llama MTP is unaffected.** Different engine, different kernel; `hauhau` runs
  `--spec-type draft-mtp` safely. (`ik-llama/iq4ks-mtp`, the other example here, was a
  `qwen3.6-27b` slug — retired from the catalog 2026-09.)

### "Is the only caveat that it has no structured benchmarks?"

No — unrelated things, rendered in different sections of the explain modal.

- **`Caveat`** = `status_note` / the compose `Caveats:` line = the #50021 exposure. `vllm/dual`'s modal
  showed nothing because of a registry gap: it carries `status="caveats"` **with no `status_note=`**
  (its alias `vllm/qwen-27b-dual-fast` *does* have one). The real caveat lives in the compose header,
  which the modal doesn't read. Worth a one-line PR upstream.
- **"no structured benchmarks for this slug"** = `detail["benchmarks"]` came back empty. The modal
  wants **local measurement records** (`results/rebench/…`, `results/bench/…`) — both empty on this
  box; you've never run `bench.sh --save-json` / `rebench-full.sh` here. The `71/94†` in the TPS
  column comes from the BENCHMARKS.md scrape, and `†` = "measured on an older engine pin — re-bench
  owed". The **Cross-rig** block below it is other people's rigs, correctly labelled.

---

## 2. The uncomfortable part: on this stack, weight precision above INT4 buys ≈ nothing measurable

This is the finding that should shape your picks, because it's the exact axis you're optimizing.
Three same-harness, same-day A/Bs in this repo:

| A/B | Weights | 8-pack /150 | Delta |
|---|---|---:|---|
| 3-way quant A/B, 27B dual (same harness, same day) | AutoRound **INT4** (`fast`) | **109** | — |
| | AWQ-BF16-**INT4** (`balanced`) | 105 | −4 |
| | Official **FP8** (`max`) | **110** | **+1** |
| 8-bit corner A/B, 27B dual, v0.24.0 | **FP8** W8A16 + int8-PTH KV | **107** | — |
| | **INT8 W8A8** (Avesed) + int8-PTH KV | **107** | **0** |
| KV-format A/B, 27B dual-max | int8-PTH KV | 107 | — |
| | fp8/e4m3 KV | 109 | +2 |

The `fp8-mtp.yml` header states the conclusion in its own words: *"fast 109 · balanced 105 · max 110
— a TIE (the 8-pack doesn't separate the quants)."* All of those deltas sit inside the repo's own
±5–7 noise band.

*(These A/Bs were run on `qwen3.6-27b`, retired from the catalog 2026-09 — the finding itself
("weight precision above INT4 buys ≈ nothing measurable here") is kept as historical record; none
of the specific slugs below are launchable anymore.)*

**What it costs to buy that +1.** Going INT4 → official FP8 on the 27B means a **29 GB** download and
a KV pool that shrinks from **622K tokens / 2.37× concurrency to 295K / 1.13×** — the smallest pool of
any tier. On Ampere sm_86 there is no native FP8 compute, so FP8 weights run through **Marlin W8A16**:
a memory win, not a compute win (vLLM itself warns it "may degrade compute-heavy workloads").

**What *does* separate models, measurably:**

| Axis | Spread |
|---|---|
| Weight quant (INT4 → FP8 → INT8) | **0–1 points** |
| KV format among ≥8-bit options | 0–2 points |
| **Model / fine-tune choice** | **ThinkingCap 113/120 vs Tess 108/115** (like-for-like v0.9.7) — and on agentic packs the spread is far wider: cli-40 runs 12/40 (apex) → 14/40 (35B-A3B) → 17/40 (byteshape) → 23/40 (agents-a1) → **28–29/40** (ThinkingCap) |
| **KV bit-width below 8** | tail precision 94.3% (q8/q6) → **88.9% (q4_0)** → 81.6% (turbo3) — see §7 |

So the precision that actually pays on this rig is **KV ≥ 8-bit**, and after that it's **which model
you run**, not how many bits you spent on its weights. A Q6_K of a mid-tier fine-tune loses to a
4-bit quant of a stronger one — `hauhau` at Q6_K_P scores 103/105 while `byteshape` at IQ4_XS scores
110, and ThinkingCap at **W4A8** tops the whole catalog.

⚠️ **Honest caveat on that ThinkingCap number:** its headline 118/123 was measured on the v0.9.8
harness while Tess (108/115) and Qwen-fast (109) were pre-v0.9.8 — **not like-for-like**. The
like-for-like pairing is the v0.9.7 one, ThinkingCap **113/120** vs Tess **108/115**, where it still
led both legs. Don't rank across harness generations.

---

## 3a. Vision — what actually has it, and one important correction

**The correction first: vision and MTP are NOT mutually exclusive.** That was believed in this repo
and is **wrong** for llama.cpp b9570 — confirmed on Deckard 2026-07-10: `--mmproj` and
`--spec-type draft-mtp` both load (`[mtmd] mmproj 873 MiB` + `speculative decoding context
initialized`), an image round-trips (`finish_reason=stop`), **and MTP still drafts at ~0.71
acceptance**. MTP is lossless and image tokens live in prefill, so the two are orthogonal — you keep
the speedup *and* the images. If you internalized the old belief, drop it.

**Second thing to know:** the vision variants of your two best precision picks are **separate sibling
composes that are NOT in the registry** — `switch.sh` cannot see them, which is exactly why `serve.sh`
exists. They're drop-in swaps that bind the same port as their text sibling (mutually exclusive:
`down` the text one, `up` the vision one).

| Lane | Compose | Registry? | Ctx | KV | mmproj | On disk |
|---|---|---|---|---|---|---|
| `hauhau-vision` | `qwen3.6-35b-a3b/llama-cpp/dual/morikomorizz-q6kp/**vision.yml**` | ❌ serve.sh only | **262K** | **q8_0/q8_0** | `mmproj-BF16.gguf` (861 MiB) | ✅ verified present, path matches |
| `vllm/qwen-35b-a3b-dual` | `…/vllm/dual/autoround-int4/fp8.yml` | ✅ | **262K** | fp8_e4m3 | in-checkpoint tower, `image=2` | ✅ |

*(Two rows used to sit here: `ik-llama/iq4ks-mtp-vision` and `ik-llama/prism-pro-dq-dual-vision` —
both `qwen3.6-27b`-family slugs, retired from the catalog 2026-09 along with the rest of that
model. See §3b below for why `prism-pro-dq` wasn't worth pulling anyway.)*

---

## 3b. `prism-pro-dq` — checked, and it's a no (don't spend the download)

*(`prism-pro-dq` was a `qwen3.6-27b` fine-tune slug — `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ` — retired
from the catalog along with the base model 2026-09. Moot as a "should I pull this" question now,
kept below as a worked example of registry-vs-compose verification.)*

I flagged this last round as "262K + q8_0 KV + vision, in the registry, unknown-not-bad." **I was
reading registry metadata that the compose contradicts.** Checked against the actual files:

| | Registry says | Compose actually does |
|---|---|---|
| Weights | `ex0bit-prism-pro-dq` (unlabelled) | **Q3_K_M** — `quant_label: q3km`, from `general.file_type=12` read out of the GGUF header, verified 2026-07-06 |
| KV | `q8_0` | **`-ctk/-ctv ${KV_TYPE:-q4_0}`** |
| Max ctx | `262144` | **`--ctx-size ${CTX_SIZE:-196608}`** |

So the one slug that looked like it matched your spec on paper is, in reality, **the lowest-precision
option in this entire document on both axes at once** — ~3.5 bpw weights (below every other candidate:
Q6_K_P, Q8_0, IQ4_XS, INT4, Q4_K_M) *and* the 88.9%-tail q4_0 KV — at 196K, not 262K. Its own header
says `Status: 🧪 EVAL ONLY`, and it has no bench or quality numbers anywhere in `BENCHMARKS.md`.

For a "highest precision" objective that's the exact wrong direction, so I'm not pulling it. If you
want it anyway for curiosity it's a one-line `hf download Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ` — the
mmproj it needs (`qwen3.6-27b-gguf/mmproj-F16.gguf`) is already on disk. Worth a separate PR to fix
the registry↔compose mismatch regardless, since anyone reading `switch.sh --list` gets told q8_0/262K.

---

## 3. Tier list A — maximum precision × maximum context × vision, both cards

Ranked for *this* rig. "Precision" = weight bpw **and** KV bits, with the §2 caveat that above
INT4 / 8-bit-KV the measured quality return is flat.

**Constraint added 2026-08-09: uncensored / low-refusal models are a specialty lane, not daily
drivers.** That removes `hauhau` from the S tier below and puts it in §3c. (`deckard` and `carnice`
were also in that specialty lane; both were retired from the catalog 2026-09.)
It leaves the stock field thin — deliberately so.

### S — the stock daily driver

| Pick | Status | Weights | KV | Ctx | Vision | TPS | On disk |
|---|---|---|---|---|---|---|---|
| **`vllm/qwen-35b-a3b-dual`** | ✅ **Production** | AutoRound INT4 | fp8_e4m3 | **262K** | ✅ **validated live @262K** | ~85 (L) single · **agg ~265 @ N=4 (L)** | ✅ 21 GB |

With uncensored models excluded, this isn't merely the best option — **it is the only stock,
✅-Production, #50021-immune config in the catalog that does 262K with vision on this rig.** The
vision claim is measured, not inferred: the 2026-05-30 promotion gate included a live vision smoke at
262K, alongside NIAH-clean to 240K, soak PASS (0 growth / 0 err / 100% retention), and a 2.05M-token
KV pool. Drafter-less by design, so the §1 crash class cannot reach it.

Its two real costs: **agentic quality** (cli-40 14/40, hermesagent 11/20, aider 13/30) and **~85 TPS
(L)** single-stream from the x4 tax. Per §2 its INT4 weights are *not* one of the costs worth worrying
about.

### A — stock alternates (single card, and they're competitive here)

| Pick | Status | Weights | KV | Ctx | Vision | Why |
|---|---|---|---|---|---|---|
| **`apex-vision-ik`** (`mudler-apex-compact/vision.yml`) | 🧪 | APEX Q4_K_M | **q8_0 K / q5_0 V** | **262K** ✔ | ✅ | **Verified on this rig 2026-07-18** at full 262K: 20.8 GB used / ~3.7 GB free, image described correctly. Best KV precision of any vision lane you can boot. A mudler *fine-tune* — but a capability fine-tune, **not** abliterated, so it's inside your constraint. Compose default is `VISION_CTX_SIZE=131072`; your co-located `.env` raises it to 262144. |
| `unsloth-ud-iq4xs/vision.yml` (35B-A3B single) | 🧪 | unsloth UD IQ4_XS (**stock** base) | q4_0 (K/V overridable) | 131072 | ✅ | On disk, stock, not in the registry. Sits exactly at your floor. Untested here. |

*(`ik-llama/iq4ks-mtp-vision` — the purest stock+production dense-27B vision lane — used to be the
other row here. It was a `qwen3.6-27b` slug, retired from the catalog 2026-09; no replacement
dense-model vision lane exists today.)*

> **Call it:** `vllm/qwen-35b-a3b-dual` is the stock daily driver — it's the only one that clears every
> constraint at once. `apex-vision-ik` is the one to reach for when you want the KV precision and
> single-stream speed and can accept a fine-tune; you've already proven it boots at 262K.

### 3c. The uncensored lane — kept, and here's what it actually costs

Your reasoning is **directionally right but wrong on the specific mechanism**, and the distinction
matters for how you use them:

- **Overall quality: yes, mid-band.** hauhau 103 off / 105 on — against stock byteshape's
  110 and ThinkingCap's 113. The repo's own framing: *"uncensoring buys compliance, not capability."*
- **Tool calling: not degraded.** This is the part to drop. hauhau scores **toolcall 14/15** — at or
  near the top of the whole catalog. There is no measured tool-call penalty.
- **Where the low-refusal cost is actually measurable: `cli-40` safety scenarios.** The one external
  attended judge run in the repo (@kevinb361 on Tess) returned a general-corpus win (13W/13T/4L) but a
  **safety-corpus loss (5W/2T/8L)**, attributed to "Hermes-lineage low-refusal; chronic cli-40
  safety-scenario failures across every quant/engine" — explicitly **model-level, not recipe-fixable**.
- **Not technically "abliterated."** hauhau is an uncensored *fine-tune*. Abliteration (weight
  orthogonalization) is a specific technique, and the repo doesn't attribute it to this model. Also
  worth knowing: **no refusal-rate benchmark has ever been run in this repo** — there is no measured
  refusal number for anything, in either direction.

So: keeping it as a specialty lane is a sound call, but make it on the **overall 8-pack and the
safety-scenario data**, not on a tool-calling fear that the numbers don't support.

| Specialty pick | Weights | KV | Ctx | Vision | Note |
|---|---|---|---|---|---|
| `hauhau-vision` | **Q6_K_P** | **q8_0/q8_0** | **262K** | ✅ (unverified boot) | Still the highest precision × context × vision config you own. 262K + MTP + 861 MiB mmproj — first boot must be checked: mmproj loads, `speculative decoding context initialized`, per-card free VRAM. If tight: `CTX_SIZE=229376` or rebalance `VISION_TENSOR_SPLIT`. Unpinned community digest. MTP n=3 is code-max, **−4% prose** (`MTP_DRAFT_N_MAX=1` prose-safe). `REASONING=on`. |

*(`deckard-vision` used to be the other specialty pick here — Q6_K 40B dense, q8_0/q8_0 KV, 131K,
confirmed vision round-trip. `deckard` was retired from the catalog 2026-09; no replacement dense
uncensored vision lane exists today.)*

### B — high precision, but the trade doesn't clear

| Pick | Status | Vision | Why it's not S |
|---|---|---|---|
| `ik-llama/prism-pro-dq-dual-vision` | 🧪 | ✅ | **RULED OUT — do not pull.** See §3b: it's a **Q3_K_M**, and its registry metadata misdescribes it. |
| `ik-llama/ornith35b-dual` | 🧪 | ❌ | **Q8_0 weights** (~35 GB) + q8_0 KV at 262K — the highest weight precision at full context. Coding-leaning (aider 15/30 vs the base's 12–13, bugfind 15/15), 8-pack 105 off==on, 106.5/103.6 (R). No vision, 35 GB pull. Out on the vision requirement alone. |
| `ik-llama/apex-mtp-quality-dual` | ❌ | ❌ | q8_0 KV, 196K, dual. Not on disk, no vision variant. |

### C — skip on this rig
- **NVFP4 family** — needs sm_90+ to execute natively; on Ampere it dequants through Marlin W4A16 and
  AutoRound is faster. Every headline number is a 5090 run.
- `vllm/agents-a1-dual` — best cli-40 in the vLLM set (23/40) but **crash-loops on this box** in the
  FP8-MoE→Marlin repack (`CUDA driver error: device not ready`); prime suspect driver 610.62.
- **`beellama/*`** — engine retired 2026-07-27 (Anbeeld #98 won't-fix); all remaining slugs deprecated,
  launch needs `--force`. (`carnice`, the beellama-hosted `qwen3.6-27b` fine-tune, was removed from
  the catalog entirely 2026-09 along with its weights.)

---

## 4. Tier list B — best performance *at* those precision/context values

Same objective, ranked by throughput instead of bits. This is where the x4 link does the sorting.

Stock lanes only (the uncensored §3c picks would sit at ranks 1 and 5).

| Rank | Pick | Ctx | KV | Vision | TPS | Note |
|---|---|---|---|---|---|---|
| 1 | `apex-vision-ik` (single card) | **262K** | **q8_0 K / q5_0 V** | ✅ | ~84–90 (L) | **Verified on this rig 2026-07-18** at full 262K: 20.8 GB used / ~3.7 GB free, image correct. Single-card ⇒ transfers cleanly. Dropping `--spec-type` reclaimed 2.3 GB. Fine-tune, not abliterated. |
| 2 | `vllm/qwen-35b-a3b-dual` | 262K | fp8_e4m3 | ✅ | ~85 (L) single · **~265 agg @ N=4 (L)** | The validated stock answer, and the only concurrency answer. |
| — | ~~`ik-llama/byteshape-iq4xs-mtp`~~ | 262K | q4_0 + Hadamard | ❌ **text-only** | 115.6 / 137.1 (R) | **Correction:** I listed this as a vision lane last round — wrong. Its header reads *"Vision: NO (mmproj-bf16.gguf exists upstream — vision not wired here yet)."* I'd matched a grep on the header prose, not on an actual `--mmproj` mount. Still the best-measured single-card 35B-A3B (8-pack **110/150**, ToolCall 15/15) — just not with images. Your local `byteshape-vision` lane at 131K is your own compose, not a repo one. |

*(Rank 3 used to be `ik-llama/iq4ks-mtp-vision`, a stock dense-27B vision lane at 163840 ctx — it
was a `qwen3.6-27b` slug, retired from the catalog 2026-09.)*

**The uncomfortable comparison:** #1 on **one card** matches #2 on **two** for single-stream decode at
*higher* KV precision, same 262K, same vision. That's the x4 tax in one line — "use both cards" is not
automatically right even when both are free.

---

## 5. `vllm/qwen-35b-a3b-dual` — verdict on your preliminary plan

**Its caveats aren't deal-breaking because it has none.** You may have read the wrong row: it's
`status="production"` — the **green ✅**, not the orange `✔`.

```
model=qwen3.6-35b-a3b  engine=vllm-stable  tp=2  max_ctx=262144
kv=fp8_e4m3  drafter=None  status=production
compose: models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml
```

Gate passed 2026-05-30: 262K max-ctx probe (7.81× concurrency, 2.05M-token KV pool) · `verify-stress`
NIAH-clean to 240K, no Cliff 2 · `soak-continuous` PASS (0 growth / 0 err / 100% retention) · quality
`--full` · live vision smoke at 262K. **Drafter-less by design** (built-in MTP measured −45% at n=2,
−51% at n=3 on this MoE under TP=2), so **#50021 cannot touch it**. Weights already on disk (21 GB).

Three things to weigh against your new objective:

1. **It is not the precision pick.** INT4 weights, fp8 KV. Fine by §2 — but if you're explicitly
   optimizing bits, `hauhau-vision` (Q6_K_P + q8_0/q8_0 KV, same 262K, also with vision) dominates it
   on that axis *and* on speed here. What a3b-dual has that hauhau-vision doesn't is a **validated**
   262K vision boot.
2. **Agentic quality is its weak axis.** `--full` 2026-05-30: deterministic **79/90 (88%)** — strong.
   But **cli-40 14/40**, hermesagent 11/20, aider-polyglot 13/30 — the packs that predict drive-a-real-CLI
   and edit-compile-test loops.
3. **Where it is unbeatable: fan-out.** Registry ships `max_num_seqs=1`; your `.env` runs 4, and you
   measured **agg ~265 TPS @ N=4**. Batching amortizes the all-reduce — the one regime where the x4
   link stops mattering. Nothing else in your catalog is close.

**Verdict (revised for vision):** keep it as a **first-class pick**, not just the concurrency profile.
With vision required at 262K on dual, the field narrows to it and `hauhau-vision` — and it is the only
one of the two whose vision-at-262K is validated rather than inferred. Run `hauhau-vision` when its
boot check passes and you want the precision; keep this as the fallback and the fan-out lane.

---

## 6. What is `tess-4-27b`, and is q4_0 "really bad precision"?

**Tess-4-27B** is migtissera's dense **Qwen3.5-based 27B instruct/agentic fine-tune**
(`family: qwen35-dense`). Architecturally a **hybrid**: 64 layers = 48 linear-attention (SSM-style) +
16 full-attention, 3:1 interleave. KV exists only on those 16 layers — cheap at long context, and why
its registry entry ships `kvcalc_key="SKIP"` (naive 64-layer math overestimates ~4×). Same family as
Deckard. Base is VL-capable but the GGUF entry serves **text-only**. Its chat template is stock-broken
(developer-role crash), so composes pin `froggeric`.

Three slugs — and **the `q4` in the name is not the KV**:

| Slug | Weights | **KV** | Ctx | Status |
|---|---|---|---|---|
| `llamacpp/tess-dual-mtp` | Q4_K_M GGUF + external MTP draft GGUF | **q4_0** | 262K | ✅ |
| `vllm/tess-dual-w4a16` | AutoRound **W4A16** (INT4 g128, `mtp.fc` kept BF16) | fp8_e4m3 | 262K | ⚠️ (#50021) |
| `vllm/tess-dual-nvfp4` | NVFP4 W4A4 | fp8_e4m3 | 131K | 🧪 |

So the cockpit's `q4_0` on the Tess row is the **KV cache type of the llama.cpp variant**; `Q4_K_M` is
the *weight* quant. Two different axes that both happen to say "4-bit". Neither means "the model is q4_0".

The W4A16 row is the interesting one: the built-in MTP head accepts **0%** in the NVFP4 export but
**80% (accept-len 5.0)** in the W4A16 export, because LeaderboardModel1 kept `mtp.fc` and
`linear_attn.in_proj` at BF16 via `extra_config`. The export was the killer, not the head. It benches
73.5/108.2 (R), 8-pack 108 off / 115 on. **Not on disk**, 18 GB, and it lands in the #50021 set —
given §2 and your objective, it isn't worth the pull.

---

## 7. The q4_0 KV question — your instinct is right, with one rescue

You're right that KV quant is where quality quietly dies, and the metric is **tail precision**
(99.9th-percentile KL divergence), not perplexity — the worst 0.1% of positions are exactly where
quantization breaks JSON keys, closing braces, and tool-call grammar. Anbeeld measured this on
**Qwen3.6-27B on a single 3090** — same model, same GPU class:

| K / V | % of bf16 KV | tail precision | use |
|---|---:|---:|---|
| `q8_0` / `q6_0` | 47% | 94.3% | best you'd actually run |
| `q5_0` / `q5_0` | 34% | 93.2% | quality default (coding / agents / JSON) |
| `q5_0` / `q4_1` | 33% | 92.7% | VRAM-constrained quality |
| **`q4_0` / `q4_0`** | **28%** | **88.9%** | catalog default — favors max context |
| `turbo3_tcq` | 20% | 81.6% | extreme context only — visible structured-output loss |
| `turbo2` | 14% | 54.4% | last resort (no code / JSON / math) |

Four resolutions:

1. **q4_0 is the default because the catalog optimizes max context on 24 GB**, not because it's
   neutral. `docs/FAQ.md` says the loss is small on average but **meaningful on the tail for
   structured output** — and the gap **grows with context length**, which matters precisely because
   you run 131–262K.
2. **`KV_TYPE` is an env override on every llama.cpp/ik compose** (shell env beats `.env`):
   ```bash
   KV_TYPE=q8_0 bash scripts/switch.sh ik-llama/byteshape-iq4xs-mtp   # caps ctx; check the fill ladder
   ```
3. **K is the sensitive cache, V is not.** Asymmetric beats symmetric at equal size (`q5_0`/`q4_1`
   beats symmetric `q4_1`). The strong version — **q8_0 K / q5_0 V** — is what `apex-fit-q8q5` ships,
   and it's the pattern to copy.
4. **ik-llama's `-khad` / `-vhad` (Hadamard transform on the K/V caches) recovers accuracy lost to KV
   quantization at zero VRAM cost.** Every ik compose sets both. So `ik-llama/byteshape-iq4xs-mtp` at
   q4_0 is **not** the same as mainline `llamacpp/mtp` at q4_0 — ik rows are q4_0 **+ Hadamard**,
   mainline rows are bare q4_0. The cockpit's `kv` column cannot show you that difference.

**Reconciling this with §2:** those A/Bs all compared **≥8-bit** KV formats against each other and
tied. The Anbeeld table is the one that tests **4-bit** KV. No contradiction — the cliff is *below*
8 bits, which is exactly where the 8-pack comparisons never went.

Also: **`verify-stress` 7/7 including the 91K needle does not certify KV-quant tail safety.** Synthetic
needle retrieval is blind to this drift. "It passed stress" is not evidence your q4_0 KV is fine for
JSON-heavy traffic.

**Rule for your objective:** run **K at q8_0** wherever the context budget allows. `hauhau` (q8_0/q8_0
@262K) and `apex-fit-q8q5` (q8_0/q5_0 @196K) already satisfy it. (`deckard`, q8_0/q8_0 @131K, used to
be a third example here — retired from the catalog 2026-09.)

---

## 8. `vllm/minimal` and `vllm/dual` — are those user-defined slugs?

*(Both were `qwen3.6-27b` compose slugs; the model — and these two registry entries — were retired
from the catalog 2026-09. Kept below as historical context for the naming convention it explains.)*

**No — maintainer-owned registry tags**, literal keys in
`scripts/lib/profiles/compose_registry.py::COMPOSE_REGISTRY`, shipped with the repo. The naming is
historical, not systematic: they predate the current `<engine>/<model>-<topology>-<feature>`
convention (`vllm/qwen-27b-dual-max`, `vllm/qwen-35b-a3b-dual-nvfp4`), and are grandfathered because
renaming a slug re-paths `compose_path` and breaks everyone's scripts. `vllm/qwen-27b-dual-fast` is an
explicit **alias** of `vllm/dual` — same compose, same port 8010 — created just to give the fast tier
a convention-conforming name.

Slugs also decouple from the filesystem: the path encodes
`<model>/<engine>/compose/<topology>/<quant>/<serving>.yml`, while `DEFAULTS` maps
`(model, engine, topology) → slug`. `("qwen3.6-27b","vllm","single") → "vllm/minimal"` and
`("qwen3.6-27b","vllm","dual") → "vllm/dual"` — which is why those two have the plainest names: they
*are* the defaults for their cell.

**Why would you ever pick the 33K one?** For your current purpose, you wouldn't.
`vllm/minimal` is `tp=1, max_ctx=32768, max_num_seqs=1, drafter=None`, and its header says the quiet part:

> *Best for: Debugging baseline / 20 GB Ampere fallback when TQ3 paths don't fit / first-time setup verification*

It exists because community feedback showed users hitting the spec-decode tool-call cascade by booting
the wrong compose — **minimal removes that whole failure class by never enabling spec-decode**. It's
the "is vLLM itself working on this box" control at ~32/33 TPS, and incidentally one of the few vLLM
slugs immune to #50021 for the same reason.

Two footnotes: the ctx number disagrees across sources — the compose *header* claims 65K, the flag is
`MAX_MODEL_LEN:-32768`, and the registry says 32768, which is what the cockpit renders. **The header is
stale.** (This applied to the retired `serve.sh 27b-minimal` local override — moot now that the
compose is gone, but the header-vs-flag discrepancy is a good example of the class of bug to watch
for elsewhere.)

---

## 9. What I'd actually do

0. **Launcher: either one is now safe.** `serve.sh` used to be the only launcher that wrote
   `services/litellm/preserve_state.json` — the file the LiteLLM `custom_hooks.py` §C hook reads to
   decide whether to replay prior-turn `<think>`. Cockpit/`switch.sh` boots silently inherited the
   last `serve.sh` mode (caught live: the file held `mode=full` while a3b-dual resolves to `off`,
   which is the carryover that seeded thought-loops on the a3b lanes). Fixed 2026-08-09:
   `scripts/preserve-state.sh` is the shared resolver, `serve.sh` delegates to it, and the cockpit
   runs it right after `switch.sh`. Pick by ergonomics now, not correctness — `serve.sh` still owns
   the per-boot flags (`--think/--preserve/--ctx/--gpu`) and auto-starts LiteLLM if it's down.
1. **Run `vllm/qwen-35b-a3b-dual` as the stock daily driver**, and `apex-vision-ik` when you want the
   q8_0/q5_0 KV and single-card speed at the same 262K. Those are the two that clear every constraint
   (stock-or-non-abliterated, vision, ≥131K, validated).
2. **Keep `hauhau-vision` as the specialty lane** (§3c) — booted deliberately, not
   as a default. If you do boot it, its 262K + MTP + 861 MiB mmproj combination has **never
   been verified**: confirm mmproj loads, `speculative decoding context initialized` appears, and
   per-card free VRAM is healthy; if tight, `CTX_SIZE=229376` or rebalance `VISION_TENSOR_SPLIT`.
   (`deckard-vision` used to be the other specialty pick here; retired from the catalog 2026-09.)
3. **Don't pull `prism-pro-dq`** — §3b. Q3_K_M weights + q4_0 KV @196K is the wrong direction on every
   axis you care about, and its registry row misdescribes all three.
4. ~~Demote `vllm/dual` out of daily use~~ — moot: `vllm/dual` was the `qwen3.6-27b` default slug,
   retired from the catalog 2026-09 along with the rest of that model's composes. The underlying
   #50021 bug still matters for `vllm/tess-dual-w4a16` / `vllm/thinkingcap-dual-w4a8` — watch
   `docs/UPSTREAM.md` #50021 row for the merge.
5. **Run `bash scripts/rebench-full.sh` once per slug you actually use.** You have zero local
   measurement records, which is why the explain modal is empty and why every number you're reading is
   someone else's rig — inflated ~2× on any TP=2 path here. This is the single highest-value thing on
   the list: right now you are choosing between models on other people's hardware.
6. **The stock field is thin — that's the real finding.** With uncensored excluded, exactly one
   catalog slug clears stock + ✅ + 262K + vision + crash-free. If you want genuine choice at that spec,
   the gap is worth filling: wiring vision onto `byteshape-iq4xs` (mmproj exists upstream, header says
   "not wired here yet") would give you a **stock** 262K vision lane with the best-measured
   single-card 8-pack in the catalog (110/150, ToolCall 15/15). That's a small, upstreamable PR.
7. **Don't buy precision you can't measure.** Per §2, spending on the top weight-precision tier
   historically returned only ~+1 point on the 8-pack (the `qwen3.6-27b` FP8 vs INT4 A/B) while
   halving the KV pool. If you want a *real* quality jump, the lever is the model:
   `vllm/thinkingcap-dual-w4a8` is top-of-catalog (113/120 like-for-like; cli-40 28–29/40) at 4-bit
   weights — but note for your requirement that its **vision tower is resident but untested**
   (text-only validated), and single-card W4A8 tops ~76K *because* of the vision tower. Plus MTP n≥4
   crashes (#758), a thin 69 MB margin at the 262K ceiling, and #50021. Not a fit while vision is
   non-negotiable.
8. **Housekeeping:** `carnice`, `deckard`, and the `qwen3.6-27b` catalog entry were retired 2026-09
   and their local weights removed. Your live `HF_TOKEN` still sits in `.env`; it's `.gitignore`d
   (`/.env`, line 53) so it won't be committed, but it surfaces in any full-file paste or log
   capture — worth rotating if one ever left the box.

---

*Sources: `scripts/lib/profiles/compose_registry.py` · compose profile headers · `BENCHMARKS.md`
rows 196–200, 246–248, 403–404, 422, 528–542, 558 · `docs/UPSTREAM.md` #50021 row · `docs/FAQ.md`
"Which KV-cache quant should I use?" · `docs/QUANTIZATION.md` ·
`tools/serve-cockpit/club3090_cockpit/app.py` · local memory: rig PCIe x4, agents-a1 crash.*
