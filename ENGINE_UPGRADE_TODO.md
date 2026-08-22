# ⚙️ ENGINE UPGRADES — deferred / optional

**Status:** deferred by decision, 2026-08-22. Nothing here is required; the estate runs
fine as-is. Two independent tracks, listed in the order I'd actually do them.

Research is complete and verified — **do not redo it**, just re-check the freshness lines
before acting (upstream moves).

- **Track A** — bump the llama.cpp pin `b10236` → `b10548`. Modest, maintenance-shaped.
- **Track B** — evaluate the syv-ai single-card Qwen3.8 stack. **Higher upside for this
  rig**, because it routes around the PCIe bottleneck instead of paying it.

---

## ⚠️ Read this first: the three-channel trap

This cost two wrong answers before it was pinned down. llama.cpp publishes the same build
through **three independent pipelines with different cadences**:

| Channel | What it is | Where |
|---|---|---|
| **git tag** | every build gets one | `refs/tags/bNNNNN` |
| **GitHub Release** | binary archives, a *subset*, some flagged `prerelease` | `/releases/tag/bNNNNN` |
| **ghcr container image** | **what our composes pull** — a *different* subset | `/pkgs/container/llama.cpp` |

Measured 2026-08-22:

| Build | git tag | GitHub Release | ghcr container |
|---|:--:|:--:|:--:|
| b10454 | ✅ | ❌ none | ✅ |
| **b10548** | ✅ | ❌ none | ✅ **← the target** |
| b10549 | ✅ | ✅ **full release** | ❌ 404 |
| b10566 / `v0.2.0` | ✅ | ✅ prerelease | ❌ 404 |
| b10573 | ✅ | ✅ prerelease | ✅ |

**Two traps this encodes:**

1. **`prerelease=true` is release-page metadata only.** It says nothing about whether a
   container image exists. b10549 is a *full* release with no image; b10573 is a
   *prerelease* with one.
2. **No llama.cpp release ships a Linux CUDA binary — ever.** CUDA archives are
   Windows-only (`win-cuda-12.4/13.3/13.4`). Every Ubuntu asset is CPU / Vulkan / SYCL /
   OpenVINO / s390x / arm64. **Linux CUDA exists only as the container image.** That is
   why our pin is a container ref and there is no binary fallback.

> **Rule: pick the pin from the *Packages* page, then look up which release it maps to —
> never the reverse.** Starting from the Releases page keeps landing on tags with no image,
> because roughly 1 build in 25 gets a container and a *different* subset gets a release.

Verify any candidate in one line:

```bash
docker manifest inspect ghcr.io/ggml-org/llama.cpp:server-cuda-b10548   # prints a manifest
docker manifest inspect ghcr.io/ggml-org/llama.cpp:server-cuda-b10549   # manifest unknown
```

---

## Track A — llama.cpp pin `server-cuda-b10236` → `server-cuda-b10548`

**Current pin:** `scripts/lib/profiles/engines/llama-cpp-mainline.yml` (engine id
`llama-cpp-local`), `install.spec: ghcr.io/ggml-org/llama.cpp:server-cuda-b10236`
— pinned 2026-08-06, commit dated 2026-08-03.

**Blast radius:** **31 registry composes** ride `llama-cpp-local`, plus the two unregistered
serve.sh Qwen3.8 lanes (`38b-dual` and the q6kxl sibling).

**Why b10548** — 312 commits ahead of the pin, commit dated 2026-08-21T04:36Z, container
published. b10573 is newer (337 commits, includes the v0.2.0 bump + 7) but was ~12 h old at
the time of writing, and this pin's own note says it was deliberately *not* set to the
newest build at bump time: *"shipping an unmeasured image is the failure this pin exists to
prevent."* b10549 is the newest **full release** and is tempting for that reason — but it
has no container, and its only delta over b10548 is `TP: enable tensor split for
LFM2/LFM2MOE (#26993)`, and **there is no LFM2 anywhere in this estate** (no model profile,
no compose, no route — checked).

### What we gain

| Commit | Effect here |
|---|---|
| **`chat : pass reasoning_effort to template`** (2026-08-14) | **The one that justifies the bump.** `services/litellm/custom_hooks.py` §E exists *only* because a top-level `reasoning_effort` never reached the llama.cpp template — we translate it into `chat_template_kwargs` and clamp it by hand. This makes it native and retires bespoke code that already cost one debugging cycle (see `HANDOFF-REPLY-reasoning-effort-picker-2026-08-11.md`). |
| `chat : fix muse-glimmer detection of tool calls after EOM (#26879)` | Five muse-glimmer routes in `services/litellm/config.yaml`. Fixed by name. |
| `chat : tighten bare function parsing for Qwen models (#26793)` | Tool-call correctness on the main family. |
| `mtmd: add --mmproj-device argument (#23255)` | Pin the projector to a chosen card. With GPU0 carrying the Windows desktop (see `[[gpu0-desktop-tenant]]` note below), being able to place the 885 MB F16 projector on GPU1 is a lever we don't have today. |
| `server: spec-decode counters on /metrics (#26389)` | MTP acceptance becomes a metric instead of a log scrape (today: 0.686, read from logs). |
| `spec : auto-detect mtp draft model type (#27005)`, `common : auto-detect spec type from draft GGUF metadata (#26814)` | Less fragility around `--spec-type draft-mtp`. |
| `llama: Restore quantization of mmprojs (#26818)` | Possibly a smaller projector than our F16. |
| `server : slot save/restore with media inputs (#26640)`, `server: save processed mtmd chunks (#27278)`, `mtmd: sha256 input hashing (#27274)` | Avoids re-processing images across requests — matters on a vision lane with a 5400 s timeout. |
| CUDA: `fuse rms_norm+mul+rope (#26767)`, `block_reduce data-race fix (#26385)`, `CUDA-graph stream-sync fix (#26802)` | Small and unquantified. **Do not budget a TPS number for these.** |

### What it does NOT fix — checked, not assumed

- **The `--spec-type draft-*` + `--mmproj` GGML_ABORT.** Upstream
  [#24232](https://github.com/ggml-org/llama.cpp/issues/24232) is closed **not_planned /
  stale**, and nothing in the 312 commits touches the `ggml_cuda_pool_vmm::alloc` →
  unconditional `GGML_ABORT` path. **`-ub 512` stays mandatory on 38b-dual.** Full row in
  `docs/UPSTREAM.md`.
- **[#26475](https://github.com/ggml-org/llama.cpp/issues/26475)** (drafter alone on its own
  device) — still open; moving a drafter off-GPU is still unavailable.
- **Neither real bottleneck on this rig.** The GPU1 PCIe 4.0 **x4** link and GPU0's desktop
  tenancy are untouched by any llama.cpp version.
- `fit: Fix memory allocation for MTP layers (#26605)` **looked** relevant but is about
  **auto** VRAM fitting. Our lanes place layers explicitly (`-ngl 99 -ts 0.45,0.55`), so it
  almost certainly does not apply.

### New risks the bump introduces

- ⚠️ **`server: notice for upcoming default port change 8080 → 9931 (#26508)`.** Our composes
  hardcode `--port 8080` and map `…:8080` container-side. **b10548 is only the *notice*** —
  we are safe on it — but when that default actually changes it breaks *every* llama.cpp
  compose in the estate. Watch for it on the next bump.
- `common: migrate the deprecated --mmap/--no-mmap to --load-mode (#26934)` — **8 composes**
  use `--no-mmap` (deepseek-v4-flash ×5, inkling-small ×3).
- `sampler : remove "full-context windows" from history-based samplers (#26524)` — several
  lanes use `--dry-*` / `--repeat-*`.
- Spec **auto-detection** (#26814 / #27005) could change how our explicit
  `--spec-type draft-mtp` resolves. 53 occurrences across the estate.
- 27 server commits including `server: re-design yield_to_queue thread model (#27133)` —
  meaningful internals churn; smoke-test the server surface, not just a boot.

### Procedure

```bash
# 1. mechanical half — spec + compose defaults + a bucketed report of what needs hands
bash scripts/engine-pin-bump.sh llama-cpp-local server-cuda-b10548 --check

# 2. judgment half (this is the part that matters)
#    - live boot 38b-dual, confirm -ub 512 still holds the image-on-cold-boot case
#    - verify-full on the daily driver + one deepseek lane (--no-mmap surface)
#    - re-check --spec-type draft-mtp still resolves after the auto-detect change
#    - grep the boot log for the 8080->9931 deprecation notice
```

Per `AGENTS.md`: bump via PR with a `verify-full.sh` + `bench.sh` re-run, never silently.
A quality A/B is warranted if think-ON behaviour shifts — the b9246→b9967 precedent was
think-OFF-neutral but **+4 think-ON with 3 scenario-level flips**.

### Verdict

**Modest, maintenance-shaped.** The `reasoning_effort` item alone probably justifies it.
Not a performance bump — nothing here makes a dense Qwen decode faster on Ampere. Do it when
the rig is idle for a validation pass; it is not a priority.

---

## Track B — evaluate the syv-ai single-card stack (higher upside)

<https://github.com/syv-ai/qwen38-27b-rtx3090>

**We already run half of it.** The DFlash2 tiers merged from upstream on 2026-08-22 *are*
syv-ai's work — `models/qwen3.8-27b/vllm/patches/vllm-dflash2-backport` is their v0.27.1
backport and the drafter is `syvai/Qwen3.8-27B-DFlash2-W4A16`.

**What we did NOT take is the part that matters here: their stack is single-card.**
Single-GPU has no all-reduce, so **GPU1's x4 link stops mattering** — PCIe width then costs
only a one-time weight load. GPU1 is a clean 24 GB card with no desktop tenant. That
sidesteps *both* rig bottlenecks at once and leaves GPU0 free for ComfyUI / Boxel.

Their single-user claim: **122 tok/s sampling / 131 greedy** on one 3090 at a 250 W cap —
against a dual-card `ultrafast` tier measuring 231 on a *reference* rig we cannot reproduce.

**Documented trades (they are unusually honest about these):**

- **Accuracy: IFBench 78.3 vs 79.5 unquantized, GSM8K 96.5%, ppl 8.09** — ~1.2 points for
  W4A16 AutoRound. ⭐ This is the **only published accuracy anchor for any Qwen3.8 quant**;
  club-3090 has *zero* quality data for any 3.8 config, ours included.
- **262K requires KVarN 4/2-bit KV, which they call lossy** — the 262K mode and the accuracy
  numbers above are *not the same config*. Standard is 150K; `CTX=fast` is 64K.
- ⚠️ **Correctness landmine:** `FULL_AND_PIECEWISE` capture corrupts output — special-token
  ids leaking at ~1 char in 1,176. Must run `CUDAGRAPH_MODE=PIECEWISE`.
- **WSL2 (us):** KVarN needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. They report WSL2 95.2% GSM8K
  vs 95.0–97.0 bare metal, with a **5–8% run-to-run spread** — their own numbers have a wide
  band.
- FlashInfer + 4 drafts = illegal memory access. Reproduction mode and 240K are mutually
  exclusive. `DFLASH_TOKENS=15` halves request slots and costs 8K context.
- **Maintenance:** nine vLLM patches plus seven more in the KVarN runner patch, all pinned to
  vLLM 0.27.1. Different checkpoint (`dbirks/Qwen3.8-27B-W4A16-AutoRound`, not Frozenlock).
  That is a lot of vendoring to carry, and per `AGENTS.md` vendored patches force a pin and
  drift when upstream moves.

**This is a parallel stack, not a club-3090 compose** — it would need onboarding via
`docs/ADDING_MODELS.md` if it graduates.

---

## Also deferred: the dual-card DFlash2 tiers (evaluated, not pursued)

From the same 2026-08-22 merge. Recorded so the analysis is not redone:

| Slug | Drafter · KV | Ctx | ref narr / **code** |
|---|---|--:|--:|
| `vllm/qwen38-27b-dual-superfast` | DFlash2 · fp8 · **W4A8** | 262K | 78 / **141** |
| `vllm/qwen38-27b-dual-ultrafast` | DFlash2 · bf16 · W4A16 | ~200K | 132 / **231** |

- **Accuracy read (reasoned, not measured):** our UD-Q5_K_XL is ~5.6 bpw dynamic vs INT4
  AutoRound's ~4.3. `ultrafast` partly buys that back — **bf16 KV beats our q8_0** and
  W4A16 keeps activations at 16-bit. `superfast` does the opposite: fp8_e4m3 KV is a slight
  step *down* from q8_0, and it stacks **int8 activations** on 4-bit weights. **So
  `ultrafast` is plausibly close to our lane; `superfast` is the weakest of the three.**
- Spec decoding is **lossless** (verified against the target), so the drafter costs no
  accuracy on any of these.
- **Blockers on this rig:** ⓐ TP=2 pays the x4 link — though DFlash2 at accept-len ~3.6
  amortises the per-token all-reduce ~3.6×, so these may degrade *less* than plain TP=2 does
  (unmeasured); ⓑ `--gpu-memory-utilization 0.90` ≈ 21.6 GiB/card with no `-ts` equivalent to
  give GPU0 less, against a desktop tenant measured growing to **4705 MiB mid-session** →
  expect to need ~0.80, which eats the KV pool; ⓒ `max_num_seqs=1`; ⓓ vision "tower present,
  **untested on this tier**" — our lane is the only 3.8 config with *verified* image serving.
- **Downloads needed:** `qwen3.8-27b:autoround-int4` (19.0 GB) + `qwen3.8-27b:dflash2`
  (1.2 GB). Not on disk today.

```bash
WEIGHT_KEY=qwen3.8-27b:autoround-int4 WITH_DFLASH_DRAFT=1 bash scripts/setup.sh qwen3.8-27b
bash scripts/switch.sh --force vllm/qwen38-27b-dual-ultrafast
```

---

## Freshness — re-check before acting

Everything above was verified **2026-08-22** against upstream at `b10573`. Re-run these
three before committing to anything:

```bash
# is there a newer CONTAINER image than b10548?  (Packages, not Releases)
docker manifest inspect ghcr.io/ggml-org/llama.cpp:server-cuda-bNNNNN

# has the spec+mmproj abort been reopened/fixed?
gh api repos/ggml-org/llama.cpp/issues/24232 --jq '.state'

# has the drafter-on-own-device blocker moved?
gh api repos/ggml-org/llama.cpp/issues/26475 --jq '.state'
```

Related: `docs/UPSTREAM.md` (both llama.cpp rows), `DAILY_DRIVERS.md`,
`models/qwen3.8-27b/CHANGELOG.md`.
