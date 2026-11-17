# Handoff — `--slot-save-path` spike (Qwen3.8-27B vision lane)

> **Scope note:** a decision spike, not a change request. Nothing here has been implemented.
> The point is to answer four questions in ~20 minutes and decide whether to build the
> escape hatch described at the end. Safe to delete once the decision is recorded.

**Target lane:** `llamacpp` dual Q5_K_XL vision — `models/qwen3.8-27b/llama-cpp/compose/dual/unsloth-ud-q5kxl/mtp-vision.yml`, `./serve.sh 38b-dual`, port **8117**, `n_ctx` 262144, `-np 1`, q8_0 K+V, `--spec-type draft-mtp --spec-draft-n-max 2`, `--mmproj mmproj-F16.gguf`.

---

## 1. The contention (why this exists)

The maintainer runs **GPU-heavy OpenGL / game-engine benchmarks on the same two cards that serve the model**. A resident model leaves ~4-6 GiB free per card, which skews those benchmark results. The preferred workflow is therefore to **unload deliberately before a benchmark run** rather than let an idle timer evict unpredictably.

The obstacle is what unloading costs. `./serve.sh --down` tears down the container, which discards the entire KV pool and every cached prefix. The next deep prompt re-prefills from token zero. At the lane's **measured** prefill curve (1200 t/s @94K degrading to 755 t/s @240K, verify-stress 2026-08-18) a full-depth context costs **roughly 4-6 minutes** to rebuild.

That cost is the whole problem. It is high enough to discourage unloading, which pushes toward leaving the model resident, which is what skews the benchmarks. **If reload were cheap, the tension disappears** — unload freely before benchmarks, restore in seconds afterwards.

llama.cpp has a mechanism that could do this: `--slot-save-path` enables `POST /slots/{id}?action=save|restore`, writing a slot's KV state to disk and reading it back. **It is manual-API only** — the server never calls it itself ([`server-context.cpp:4746`](/mnt/c/Users/Paul/.unsloth/llama.cpp/tools/server/server-context.cpp#L4746) gates the endpoint on the flag, and nothing else invokes it). The compose does not pass the flag today.

The economics look strongly favourable on paper: a 262K q8_0 slot is **~8.5 GiB**, which is seconds of NVMe I/O against ~5 minutes of re-prefill — call it 20-30×. The spike exists because two things could make that number irrelevant.

---

## 2. What is already known statically (do not re-derive)

Read from the b10687 source at `C:\Users\Paul\.unsloth\llama.cpp`. **Caveat: the pinned engine for this lane is `ghcr.io/ggml-org/llama.cpp:server-cuda-b10236`, ~450 builds older. Both `--slot-save-path` and `-ctxcp` exist in b10236 (verified via `--help` in the image), but the implementation below was read from b10687 — treat behaviour as "expected", confirm against b10236.**

| Question | Answer | Evidence |
|---|---|---|
| Is the **MTP draft context** saved? | **NO.** Only the target context. | `llama_state_seq_save_file(ctx_tgt, ...)` — `ctx_dft` is never passed ([`server-context.cpp:2549`](/mnt/c/Users/Paul/.unsloth/llama.cpp/tools/server/server-context.cpp#L2549)) |
| Are **context checkpoints** saved? | **NO.** Only `prompt.tokens` + target seq state. | `slot->prompt.tokens.serialize()` is the entire token payload |
| Are **vision / media chunks** saved? | **YES, by design.** Versioned state format with a media-key table. | `server_tokens::serialize()` writes `media_keys` then `write_media_chunk()` per item ([`server-common.cpp:553`](/mnt/c/Users/Paul/.unsloth/llama.cpp/tools/server/server-common.cpp#L553)); `deserialize(packed, has_mtmd)` reads them back |
| Can restore fail? | Yes — explicit error path. | `"No available space in KV cache or invalid slot save file"` |
| Startup gotcha | `--slot-save-path` calls `fs_is_directory()` and **throws if the dir does not exist** ([`arg.cpp:3627`](/mnt/c/Users/Paul/.unsloth/llama.cpp/common/arg.cpp#L3627)) — the volume mount must pre-exist or the container will not boot. |

**So the vision risk is largely retired before the spike starts, and the MTP risk is the live one.** The sharp question is: *what happens on the first request after restoring a slot whose target context is populated but whose draft context is empty?* This matters more than it would on a plain model because Qwen3.8 is the hybrid — llama.cpp classifies it `COMMON_CONTEXT_SEQ_RM_TYPE_RS` (partial rollback **bounded** by `n_rs_seq`, because of the 48 DeltaNet recurrent layers), and the spec-decode path leans on checkpoints precisely when a draft exceeds that bound. A restored slot has neither draft state nor checkpoints.

---

## 3. Acceptance criteria — what makes this worth incorporating

Gates 1-3 are **binary**. Gate 4 is the economics. Fail any of 1-3 and the answer is no (or a documented workaround); miss gate 4 and the complexity stops paying for itself.

| # | Gate | Pass condition | Fail means |
|---|---|---|---|
| **1** | **Restore works at all** | After save → `--down` → up → restore, a follow-up sharing the prefix returns **coherent** output, and `n_prompt_tokens_cache` in the response (or the timing) shows the prefix was **reused, not reprocessed** | Reject. Nothing else matters. |
| **2** | **MTP survives** | Draft acceptance on the post-restore turn is in the same ballpark as the lane's measured baseline — **0.686 acceptance / mean len 2.37**. No `GGML_ABORT`, no crash. | If MTP collapses (~0 acceptance) or aborts: reject, **unless** disabling the drafter for just the restore turn is cheap and recovers on the next. Document either way. |
| **3** | **Vision survives** | A restored prefix containing an image still answers correctly about that image (reuse the blue-circle / red-7 probe, or the 848×1241 PNG) | Reject for this lane — vision is the reason the lane exists. |
| **4** | **Economics** | **Restore < 60 s** for a ~240K prefix (against ~5 min re-prefill), **and save < 60 s** so `--down` does not become the new annoyance | 60-120 s: marginal, maintainer's call. >120 s: reject — the win no longer justifies ~100 lines and a new failure mode. |
| **5** | **Disk** | ~8.5 GiB per snapshot lands somewhere with room; decide **where** (WSL ext4 vs `F:`) — a 9p mount across the WSL/Windows boundary will be slower than ext4 and may sink gate 4 | Pick the fast path or reject. |

**Secondary observation worth recording either way:** checkpoints are not saved, so after a restore the slot has no checkpoint to rewind onto until new ones accumulate (defaults: 32 checkpoints, min-step 8192 — which tile 262,144 exactly). This is not a gate; note the first-rewind-after-restore cost if it shows up.

---

## 4. Procedure (~20 min, no repo changes)

Hand-edit the compose; **do not commit anything during the spike.**

1. `mkdir -p models/qwen3.8-27b/llama-cpp/cache/slots` — must exist before boot (see gotcha above).
2. Add to the compose `command:` — `--slot-save-path /slots` — and a mount `../../../cache/slots:/slots`. Verify `--slots` is not additionally required for route registration; the handler checks only `slot_save_path`, but confirm the route responds.
3. `./serve.sh 38b-dual`, then send a **long** prompt including an image, deep enough to be interesting (≥100K; ideally near the 240,635-token measured ceiling).
4. Save and time it:
   `curl -s -w '\n%{time_total}s\n' -X POST localhost:8117/slots/0?action=save -H 'Content-Type: application/json' -d '{"filename":"spike.bin"}'`
   Record `n_tokens`, `n_bytes`, `t_ms` from the response, and `ls -l` the file.
5. `./serve.sh --down`, then `./serve.sh 38b-dual`.
6. Restore and time it: same curl with `action=restore`.
7. Send a follow-up **sharing the prefix**, plus a question about the image. Record: coherence, `n_prompt_tokens_cache`, prompt-eval time, and **draft acceptance / mean len** from the server log.
8. Boot with `-lv 3` if the trace lines are needed (`created context checkpoint N of 32 … size = X MiB` is `SLT_TRC`).

**Record every number against the gates above and write the verdict into this file**, then delete the hand edits.

---

## 5. If it passes — what shipping looks like

~100 lines, no refactor:

- **Compose:** `--slot-save-path /slots` + the mount + `.gitignore` for the slot dir. Header table and profile-schema `Caveats:` updated per repo convention.
- **`scripts/slot-snapshot.sh save|restore`** (~60 lines): curl the endpoint, resolve the port from the registry, `PYTHONUTF8` line if it shells out to python3.
- **`serve.sh` hook** (~35 lines): snapshot before `--down`, restore after boot-ready, **opt-in behind an env var** so the default path is unchanged.
- **Tests:** this is a scoped change, not catalog-shape — run the guards that touch it, not the full sweep. A `verify-full` re-run on the lane before any push.
- ⚠️ Two guards, `test-litellm-ports-resolve` and `test-litellm-generate`, are **already red on `pawl-cleanup`** (confirmed pre-existing by stashing, 2026-08-30). Baseline against that; do not treat them as regressions from this work.
