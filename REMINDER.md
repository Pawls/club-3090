# REMINDER — Paul's local run notes (club-3090 + Hermes)

Personal cheat-sheet for running the stack **without the cockpit**. Untracked by intent.

> **2026-09 retirement note:** `qwen3.6-27b` ("27b"/"27b-single"), `qwen3.6-40b-deckard`
> ("deckard"/"deckard-vision"), and the beellama "Carnice" fine-tune ("carnice"/"carnice-v2") were
> removed from the catalog — their `serve.sh` entries, composes, and registry rows are gone.
> References to them below are stale run-notes, kept as-is per this file's scratch-notes
> convention; don't expect `./serve.sh 27b` / `deckard` / `carnice` to work anymore.
> `qwen3.6-35b-a3b` ("35b-a3b") is unaffected and still current.

---

## 0. The one rule that bites: recreate, don't restart

A compose **file edit** (new flag, new `.env` value) only takes effect when the container is
**re-created**, not restarted. `docker compose up -d` re-creates on config change; `docker restart`
/ stop→start reuses the OLD args.

```bash
# apply a compose/.env change:
docker compose -f <file>.yml up -d          # re-creates if changed (add --force-recreate to force)
# verify a flag actually landed:
docker inspect <container> --format '{{json .Args}}' | grep -o -- <flag>
```

---

## 0b. KNOWN RIG BOTTLENECK — GPU1 is on PCIe 4.0 **x4** (asymmetric)

Confirmed 2026-07-08 under load (stays x4, not idle ASPM): GPU0 = x16, **GPU1 = x4** (slot max x16).
TP=2 all-reduce is choked by the x4 link → 35B-A3B decode ~**85 TPS** vs the reference ~178 (same
software/quant). Under load the cards pull only ~140 W / 55–77% util = **comm-bound stall, not compute**.
- This is why single-GPU LM Studio (135–145, +MTP) *beat* vLLM-dual: no TP all-reduce → sidesteps x4.
  Quant (Q4_K_M vs INT4) was NOT the cause — near-identical bit-width.
- **Fix (high leverage):** BIOS bifurcation → x8/x8, move GPU1 to a CPU-lane slot, or check riser.
  x8 would ~2× TP throughput. Until then: prefer single-GPU serving for fit-on-one models.
- Concurrency (`max_num_seqs`) still helps — batching amortizes the all-reduce, partially hiding x4.

## 1. Which model for what (GPU-mutex: only ONE runs at a time)

2× 3090 = 48 GB, no NVLink. **Single-card** models sit on GPU1 and leave GPU0 free for the Windows
desktop; **dual-card** models use both. Only one GPU model runs at a time — bring the current one DOWN
first (`docker compose … down`, `./serve.sh <name>` does the swap for you, or `gpu-mode`).

**Single card (GPU1, frees the desktop GPU) — fastest, snappiest:**

| Model · port | Use for |
|---|---|
| **apex-35b-compact** · 8056 | ⭐ daily **TEXT** driver — 35B MoE, ~90 TPS, thinking-ON+preserve, 262K ctx |
| **apex-35b-vision-ik** · 8057 | ⭐ daily driver **+ VISION** (share UI screenshots) — same speed/thinking, 131K ctx |
| qwen3.6-27b (single) · 8021 | dense 27B + vision + MTP, ~28K ctx — fast solo, no x4 tax |

**Both cards (dual) — bigger models / more context / concurrency:**

| Model · port | Split | Use for |
|---|---|---|
| **deckard-40b** · 8199  (+`-vision` · 8200) | 1,1 | ⭐ **deliberation / hard reasoning** — uncensored dense 40B, thinking-ON, MTP (+59/104%). Vision variant keeps MTP + adds images |
| **hauhau-35b** · 8073  (+vision swap) | even | uncensored **35B-A3B** MoE, thinking-ON, MTP n=3, 262K. Vision swap adds images |
| **carnice-v2** · 8070 | 0.55/0.45 | agentic-SFT "agent brain", Q8 quality, thinking-ON+preserve |
| **qwen3.6-27b** (dual) · 8010 | TP=2 | dense 27B + vision + **262K**, tools/MTP — big-ctx **image analysis** |
| **qwen3.6-35b-a3b** · 8051 | TP=2 | MoE + vision + 262K + **concurrency** (parallel subagents, MAX_NUM_SEQS=4) |
| **agents-a1** · 8072 | TP=2 | agentic thinking model (thinking forced ON via LiteLLM hook) |
| **omni-30b** · 8042 | thinker→GPU0 / talker→GPU1 | image / audio / **VIDEO** understanding — via a light UI, **not** Hermes (§11) |
| apex-yarn-1m · 8057 | TP=2 | 1M-context experiments (🧪 candidate to deprecate; **8057 clashes with apex-vision-ik**) |
| video generation | DiT | `gpu-mode ai-studio` (ComfyUI/LTX) |

Rule of thumb: **apex for speed, Deckard for deliberation, add `-vision`/`-ik` when you need images.**
Thinking is a per-model `.env` flip (§12), not a reason to switch models. Uncensored → Deckard or hauhau.

---

## 2. Models — port · container · compose · served-name

| Model | Port | Container | Compose (`-f`) | Served model id | HF source |
|---|---|---|---|---|---|
| **Carnice V2 27B Q8** | 8070 | `beellama-carnice-v2-dual` | `models/qwen3.6-27b/beellama/compose/dual/carnice-v2-q8/mtp-q8kv.yml` | `Carnice-V2-27B-Q8_0-mtp.gguf` | [`stuchapin/Carnice-V2-27B-MTP-GGUF`](https://huggingface.co/stuchapin/Carnice-V2-27B-MTP-GGUF) (`Carnice-V2-27B-Q8_0-mtp.gguf`) |
| **Qwen3.6-27B** (vLLM) | 8010 | `vllm-qwen36-27b-dual` | `models/qwen3.6-27b/vllm/compose/dual/autoround-int4/fp8-mtp.yml` | `qwen3.6-27b` | [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) |
| **Qwen3.6-35B-A3B** (vLLM) | 8051 | `vllm-qwen36-35b-a3b-dual` | `models/qwen3.6-35b-a3b/vllm/compose/dual/autoround-int4/fp8.yml` | `qwen3.6-35b-a3b-autoround` | [`Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound`](https://huggingface.co/Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound) |
| **Qwen3-Omni-30B** (video) | 8042 | `vllm-omni-qwen3-omni-30b` | `models/qwen3-omni-30b-a3b/vllm-omni/compose/dual/autoround-int4/omni.yml` | auto (path id) | [`Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) |
| **APEX-Quality 1M** (ik, YaRN) | 8057 | `ik-llama-qwen36-35b-a3b-apex-quality-yarn-1m` | `models/qwen3.6-35b-a3b/ik-llama/compose/dual/mudler-apex-quality/yarn-1m.yml` | auto | [`mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF`](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF) (`...I-Quality.gguf`) |
| **apex-35b-compact** (ik, single, ⭐daily text) | 8056 | `ik-llama-…apex-compact-long` | `models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact/long.yml` (mtp.yml is the :8054 eval lane) | auto (path id) | mudler (`…I-Compact.gguf`) |
| **apex-35b-vision-ik** (ik, single, ⭐daily+vision) | 8057 | `ik-llama-apex-35ba3b-vision` | `…/ik-llama/compose/single/mudler-apex-compact/vision.yml` | `apex-35b-vision-ik` | mudler I-Compact + [`unsloth/…35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) `mmproj-BF16` |
| **Deckard-40B** (llama.cpp, dual) | 8199 (+vision 8200) | `llama-cpp-deckard-40b[-vision]` | `models/qwen3.6-40b-deckard/llama-cpp/compose/dual/piehsoft-q6k/{mtp,vision}.yml` | `deckard-40b[-vision]` | [`PiehSoft/…Deckard-MTP-Q6_K`](https://huggingface.co/PiehSoft/Qwen3.6-40B-Deckard-MTP-Q6_K) (+ mradermacher mmproj) |
| **hauhau-35B** (llama.cpp, dual) | 8073 | `llama-cpp-hauhaucs-35ba3b[-vision]` | `models/qwen3.6-35b-a3b/llama-cpp/compose/dual/morikomorizz-q6kp/{mtp,vision}.yml` | `hauhau-35b` | [`morikomorizz/…HauhauCS-MTP`](https://huggingface.co/morikomorizz/Qwen3.6-35B-A3B-Uncensored-HauhauCS-MTP) (+ unsloth mmproj) |

**WSL2 GPU-pinning gotcha:** Docker `device_ids: ["1"]` does NOT isolate GPUs under WSL2 (both stay
visible to CUDA → `--fit` splits the model across both over the x4 link). Pin via **`CUDA_VISIBLE_DEVICES`
inside the container** (an `environment:` block), not device_ids. The ik single compose now has this.

**Single-card 35B on GPU1 (`ik-llama/apex-mtp-compact-long`, `:8056`):** `CUDA_VISIBLE_DEVICES=1`, `NP=1`,
`CTX 196608`, MTP off = **~84–90 TPS single-stream** (beats dual-vLLM's ~85, frees GPU0). Weights 20 GB on
GPU1. llama.cpp concurrency is poor — keep dual-vLLM for parallel subagents.
- **MTP A/B DONE 2026-07-08 → keep it OFF.** off ~84 · n=2 ~55 (accept 60%) · n=4 ~65 (accept 40%).
  MTP LOSES at every depth on this MoE: a3b = ~3B active params → base decode already cheap (~12ms/tok),
  so the MTP head's extra forward pass costs more than the ~1.6 tok/step it recovers. Same net-negative
  as vLLM. LM Studio's 135–145 is NOT from MTP (MTP hurts here) — it's engine/kernel/KV differences
  (mainline llama.cpp CUDA decode vs ik + q8_0/Hadamard KV); separate investigation if ever worth it.
- **NP↔CTX coupling** (llama.cpp, NOT vLLM): `--ctx-size` is carved into NP EQUAL FIXED slots;
  per-request cap = CTX/NP, hard (idle slots not lendable). NP=1 hands the full window to one request.
  The shared-pool "sum ≤ total" model is vLLM PagedAttention only. Details in the compose-dir `.env`.

**Concurrency (35B MoE, vLLM dual):** its `.env` sets `MAX_NUM_SEQS=4` (A/B 2026-07-08). Single-stream
unchanged (~85 TPS); N=4 concurrent = agg **265 TPS ~3×**, 4-way subagent fan-out 11.9s→3.9s, no VRAM
growth. Full-262K parallelism is ~2 (KV-bound); short/moderate = full 4. Registry default stays 1
(conservative); the `.env` override is machine-local. To change: edit `MAX_NUM_SEQS` + recreate
(torch.compile makes that boot take ~3–4 min — normal, not a hang).

**Launch pattern** (cd into the compose dir so its `.env` auto-loads):
```bash
cd models/<...>/compose/dual/<quant>/
docker compose -f <file>.yml up -d
docker compose -f <file>.yml logs -f          # watch boot
docker compose -f <file>.yml down             # stop (frees the cards)
```

---

## 3. Weight downloads (weights root = `/home/pawl/models`)

```bash
# Qwen3-Omni int4 (~25 GB) + its image
hf download Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --local-dir /home/pawl/models/qwen3-omni-30b-a3b-instruct-int4-autoround
docker pull vllm/vllm-omni:v0.20.0            # ONLY self-consistent tag

# APEX-Quality 1M GGUF (shared with apex-mtp-quality-dual; ~20 GB) — via cockpit Download,
# or CLI:
bash scripts/pull.sh mudler/Qwen3.6-35B-A3B-APEX-MTP-GGUF \
  --file Qwen3.6-35B-A3B-APEX-MTP-I-Quality.gguf   # confirm flags: scripts/pull.sh --help
```

---

## 4. WSL2 memory — `C:\Users\Paul\.wslconfig`

```ini
[wsl2]
memory=56GB          # high ceiling for WSL2 AI training; reclaim gives it back when idle
swap=16GB
[experimental]
autoMemoryReclaim=gradual   # SAFE mode; returns cold cache to Windows between workloads
```
Apply: close Docker Desktop → PowerShell `wsl --shutdown` → reopen.

- **`NO_MMAP=true`** is set in Carnice's `.env` — loads weights straight to VRAM, no ~29 GB GGUF
  page-cache resident in WSL2. Opt-in only; remove/blank the line to revert to mmap.
- vLLM WSL2 boot crash (`gptq_marlin_repack` / `cudaErrorNotReady`)? Add to the compose-dir `.env`:
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. (Test without it first — a driver update may
  have fixed it.) `.env` MUST sit in the SAME dir as the compose you launch, not a parent.

---

## 5. Video recipes

**Analysis (Omni) — SWAP VALIDATED 2026-07-09:** weights (25GB) + image (`vllm/vllm-omni:v0.20.0`,
32GB) staged. Swap = `cd models/qwen3.6-35b-a3b/ik-llama/compose/single/mudler-apex-compact && docker
compose -f long.yml down` (free GPU1) → `cd models/qwen3-omni-30b-a3b/vllm-omni/compose/dual/autoround-int4
&& docker compose -f omni.yml up -d` → ready ~40s. Both cards (thinker→GPU0 24GB, talker→GPU1 16.5GB).
- **TEXT + IMAGE validated** through the full Hermes→litellm(`omni-30b`)→:8042 chain: text coherent;
  vision correctly ID'd colors+layout of a test image (red-top/green-bottom). In Hermes: pick **omni-30b**,
  attach media, ask.
- **Tool-calling** — Hermes sends `tool_choice:auto` (it's an agent) → omni 400'd without tool flags.
  FIXED 2026-07-09: added `--enable-auto-tool-choice --tool-call-parser hermes` to the compose (env
  `TOOL_PARSER`, default hermes). Verified 200 direct + via litellm. Required for ANY Hermes agent use.
- **VIDEO still unvalidated** — image path works; video adds frame-decode on top. Test ONE clip before a
  folder sweep ("which of these has trees").
- **`modalities:["text"]` — MUST be forced (crashed the engine 2026-07-09):** plain TEXT w/o it is fine,
  but a MEDIA (image/video) or tool request w/o it routes through the Talker→Code2Wav (audio) stage, which
  has a bf16 dtype bug (`SnakeBeta Expected fp32 got bf16` / `chunked_decode_streaming missing
  left_context_size`) that KILLS the EngineCore. Hermes never sends modalities → every media request would
  crash it. FIX: `custom_hooks.py` now also injects `data["modalities"]=["text"]` for any "omni" model
  (same hook as the max_tokens cap). Text-only output = what we want anyway. Verified: full Hermes-shape
  request (no modalities + tools + max_tokens 65536) + image → 200, engine stays ALIVE.
- **Max ctx 65536** (fp8 KV) — a long video = many frames = lots of tokens; may hit the ceiling.
- **max_tokens trap (crashed the engine 2026-07-09) — FIXED via LiteLLM pre-call hook:** Hermes has NO
  max_tokens knob → uses the provider's per-model `context_length` AS max_tokens. Omni's real ctx 65536
  ≈ Hermes' HARD 64K minimum (it refuses any model with context_length <64K — so you CANNOT just lower
  it). So Hermes asks for the full 64K as OUTPUT → 0 input room → 400 on ANY prompt, and the bad request
  crashes the omni EngineCore (recover with `docker compose down && up`; a plain restart loops on
  "StageEngineCoreProc died during READY"). LiteLLM `litellm_params.max_tokens` is only a default (client
  wins), so the cap MUST be forced. FIX: `services/litellm/custom_hooks.py` — a `callbacks:` pre-call hook
  that clamps `max_tokens`→8192 for any model whose id contains "omni" (only omni; text models keep full
  output). Mounted via the compose, registered in `litellm_settings: callbacks`. Hermes context_length
  stays 65536 (true value, satisfies the floor). Verified: max_tokens=65536 via proxy → 200 (clamped).
  Keep clips short (video still eats the ~57K input budget).
- **To go back:** omni down → ik `long.yml` up (restores apex-35b-compact on :8056, your Hermes default).

**Generation (stretch, mutex swap):** `gpu-mode ai-studio` → ComfyUI `:8188` · director `:8090` ·
gallery `:8189`, LTX-2.3 lane, both cards. Swap back afterward. Hermes `video_generate` → ComfyUI
bridge is NOT built yet.

---

## 6. Hermes model switching (`C:\Users\Paul\AppData\Local\hermes\config.yaml`)

Hermes here is the **NATIVE Windows app** (not containerized). So:
- **`localhost:<port>` is CORRECT.** The Docker-networking docs (container-name / `host.docker.internal`)
  are ONLY for when Hermes runs *inside* Docker — do NOT switch these URLs; it would break it. Native
  Windows reaches WSL2-published Docker ports at `localhost`.
- **`api_key` must be non-empty** even for keyless vLLM (it needs the header, doesn't validate it).
  Every `custom_providers` entry has `api_key: none`. Missing key = silent connection fail.
- **`model:` must match the endpoint's `/v1/models` id** (= vLLM `--served-model-name`, or the model
  path if none). Verify with `curl localhost:<port>/v1/models`. A mismatch = 404.
  - omni has no `--served-model-name`, so its id is the full path `/models/qwen3-omni-...` (set as-is).
    Cleaner long-term: add `--served-model-name qwen3-omni-30b` to the omni compose.
- **No trailing slash** on `base_url` (all end `/v1`). ✓
- The composer **hot-swap picker reflects your `model.default`**, NOT live endpoints. To use another
  model: Settings → Model → pick provider + model → Apply (new sessions), or start it + Refresh Models.
- `provider: custom:<name>` + blank top-level `base_url` is CORRECT (sourced from the provider entry).
- `.env` files do NOT affect Hermes — Docker-serving only.
- Provider → port map: carnice `:8070` · 27b `:8010` · 35b `:8051` · omni `:8042` · ik-single-35b `:8056`.
- **Served-id gotcha (bit us 2026-07-08):** the Hermes `model:` must be the endpoint's EXACT `/v1/models`
  id, which is NOT the model path. vLLM uses `--served-model-name` (strips `-int4`: path
  `...-autoround-int4` → served `qwen3.6-27b-autoround` / `qwen3.6-35b-a3b-autoround`). ik-llama has NO
  served-name → id IS the full mount path (`/models/qwen3.6-35b-a3b-gguf/.../...-Compact.gguf`, not the
  basename). Always `curl localhost:<port>/v1/models` and copy the `id` verbatim.
- **supports_vision:** 27b ✓, 35b-a3b ✓ (both have vision towers), carnice ✗ (text SFT), ik-single-35b ✗
  (text GGUF, no mmproj), omni ✓. Set false on the text-only ones or Hermes offers a phantom image button.

---

## 7. Quick health checks

```bash
docker ps --format '{{.Names}}\t{{.Ports}}\t{{.Status}}'
curl -s localhost:<port>/v1/models | python3 -m json.tool     # what a live endpoint serves
wsl -e free -h                                                 # WSL2 host RAM (used vs buff/cache)
docker stats --no-stream <container>                          # a container's RAM
nvidia-smi dmon -s u                                          # real per-card GPU util (not Task Mgr)
```

---

## 8. AMD DDR4 upgrade options (fix the x4 → x8/x8, keep DDR4)

Goal is **PCIe x8/x8**, NOT CPU speed (inference is GPU-bound; RAM speed barely matters). Zen 3
single-thread is within ~5–10% of the 12600K — the "AMD = weak ST" thing is Bulldozer-era, untrue now.
**Platform must be X570** (B550's 2nd slot is chipset x4 — same problem). Confirm "PCIe bifurcation
x8/x8" in the board manual. x8/x8 PCIe 4.0 ≈ 15.8 GB/s/card = ~4× GPU1's current x4, ~2× the TP ceiling.

| CPU (AM4, keep DDR4) | C/T | ST vs 12600K | MT vs 12600K | ~Price used→new | Verdict |
|---|---|---|---|---|---|
| Ryzen 5 5600 | 6/12 | ~5–10% slower | slightly less | ~$90–130 | cheapest sane; fine for inference |
| **Ryzen 7 5700X** ⭐ | 8/16 | ~5–10% slower | ~matches | ~$130–170 | closest overall match, best value |
| Ryzen 7 5800X3D | 8/16 | ~matches | ~matches | ~$250–330 | gaming king; overkill for AI |
| Ryzen 9 5900X | 12/24 | ~5–10% slower | more (12c) | ~$180–260 | MT headroom |
| Ryzen 9 5950X | 16/32 | ~5–10% slower | much more | ~$300–420 | overkill |
| **X570 motherboard** | — | — | — | ~$120–220 used | the actual fix (x8/x8) |

**Total DDR4-keeping upgrade:** 5700X + used X570 ≈ **~$300**, no RAM change, ~2× dual-card TP.
Prices volatile (early-2026 vantage) — verify current. Optional: only helps DUAL-card paths; single-GPU
serving already dodges the x4. **RAM note:** 3600 CL16 is AM4's 1:1 sweet spot with 2 DIMMs; a 4×
dual-rank (64 GB) loadout limits BOTH platforms to ~3200–3400 — 4×DR is hard everywhere.

---

## 9. LiteLLM unified provider (`services/litellm`, `:4000`)

One proxy fronting ALL backends → ONE Hermes provider whose dropdown lists every model
(the "one provider, many models" goal). Master key `sk-litellm-master-key` (Hermes `api_key`
must be this, NOT `none`). Bring up: `start-litellm.bat`, or `cd services/litellm &&
docker compose up -d`. Routes: 27b · 35b-a3b · gemma-4-31b/12b · agents-a1 · deckard-40b · **carnice-v2
· apex-35b-compact · omni-30b** (last 3 added 2026-07). GPU-mutex still applies — the menu lists all,
only the running backend answers; the rest 404 until launched. ik/omni have no served-name so their
litellm `model:` is `openai/` + the FULL mount path (note the `//`).

**Does LiteLLM reshape requests? Tested 2026-07-09 (direct :8056 vs proxied :4000, seeded diff):**
- **Sampling params preserved** — `top_k`/`min_p`/`repetition_penalty`/`seed` forwarded intact
  (byte-identical seeded generation), even with `drop_params: true`. Your OpenAI→OpenAI path (`openai/`
  prefix) is passthrough, NOT the cross-protocol (Anthropic/Gemini) reshaping people warn about online.
- **`drop_params: true` is REQUIRED** (not false). Hermes sends `reasoning_effort` (from
  `agent.reasoning_effort`) on every request; llama.cpp/vLLM chat endpoints reject it → LiteLLM 400s the
  whole request unless it can DROP it. drop_params:true drops ONLY params in LiteLLM's known-OpenAI map
  (reasoning_effort); non-OpenAI sampling extensions are unknown-passthrough → survive. (drop_params:false
  was my mistake — it turned that benign drop into a hard 400. The real fidelity guard is the diff test,
  not the flag.)
- **Streaming (Hermes default) = passthrough** — correct content deltas, `[DONE]`, `<think>` stays inline
  same as direct. This is the mode Hermes uses → lowest risk.
- **Non-streaming ONLY:** LiteLLM lifts `<think>…</think>` out of `content` into `reasoning_content`
  (OpenAI-standard, non-lossy; Hermes reads reasoning_content natively). Benign.
- **UNTESTED (backends were down, GPU-mutex w/ ik):** vision image_url payloads (spec-standard, low risk);
  a full reasoning-ON chain-of-thought extraction (mechanism confirmed on empty tags, non-lossy).
- Verdict: **safe** — no request-breaking reshape; the one transform (reasoning extraction) is
  non-streaming-only, non-lossy, and Hermes streams anyway.

---

## 10. CONFIG CATALOG — which to run when (master reference)

**GPU-mutex:** exactly ONE GPU model at a time (single- or dual-card). LiteLLM (`:4000`, CPU-only)
and Hermes stay up always. **Two launch methods:**
- **Cockpit (c3):** activate the repo-root venv (`source .venv/bin/activate`) → run `c3` → pick the slug in the TUI (start/stop). See [[cockpit-launch-setup]].
- **CLI (registry slugs):** `bash scripts/switch.sh <slug>` from repo root — GPU-mutex-aware (brings others down, this one up).
- **⭐ `./serve.sh` (project root — for models NOT in the registry/cockpit):** evicts whatever GPU model is
  running, then boots the target. `./serve.sh <name>` where name is a shortcut (`deckard`, `deckard-vision`,
  `hauhau`, `hauhau-vision`, `apex`, `apex-vision-ik`, `apex-vision`, `carnice`, `27b-single`) or a compose
  path. `./serve.sh --list` shows shortcuts, `--status` shows what's on the cards, `--down` just evicts.
  This is how you launch the vision/deckard/hauhau variants (none are registered).
- **Manual:** `cd <compose dir> && docker compose -f <file> up -d`.

| # | Config · port | Launch (slug or manual) | Custom vs stock repo | Use case | Status |
|---|---|---|---|---|---|
| 1 | **apex-35b-compact** · 8056 | `ik-llama/apex-mtp-compact-long` (or `./serve.sh apex`) | repo compose **+ our `CUDA_VISIBLE_DEVICES` env block** ([[rig-pcie-x4-bottleneck]]); `.env`: GPU1, NP=1, CTX **262144**, MTP off, **thinking-ON+preserve** | ⭐ **Daily TEXT driver** — 35B MoE, ~90 TPS, 262K ctx, single card (frees GPU0), reasoning blocks | ✅ keep — Hermes default |
| 2 | **qwen3.6-27b** · 8010 | `vllm/dual` | stock repo compose | Dense 27B **+ VISION** + 262K, dual, tools/MTP | ✅ keep — best for **IMAGE analysis** (262K ctx absorbs Hermes' ~40K agent overhead) |
| 3 | **qwen3.6-35b-a3b** · 8051 | `vllm/qwen-35b-a3b-dual` | stock + `.env` MAX_NUM_SEQS=4 | MoE + vision + 262K dual, real concurrency (subagents) | ⚠️ **overlaps #1** — keep only if you need MoE vision/concurrency; else redundant |
| 4 | **carnice-v2** · 8070 | `beellama/carnice-v2-dual-q8-mtp` | repo compose **+ our NO_MMAP toggle**; `.env` NO_MMAP=true | Agentic-SFT reasoning brain, Q8 quality, both cards | ✅ keep — the "agent brain" flavor |
| 5 | **omni-30b** · 8042 | MANUAL: `cd models/qwen3-omni-30b-a3b/vllm-omni/compose/dual/autoround-int4 && docker compose -f omni.yml up -d` | repo compose **+ our tool-choice flags**; `.env` PORT=8042; **NOT registered** | Image/audio/**video** UNDERSTANDING, both cards, 64K ctx | ✅ keep — only multimodal-IN model. Use via a **light UI, NOT Hermes agent** (64K too tight for agent overhead) |
| 6 | **apex-yarn-1m** · 8057 | `ik-llama/apex-yarn-1m-dual` | **NEW compose we authored** + registry entry | 1M-context experiments (YaRN 4×), both cards | 🧪 experimental, quality UNPROVEN >262K — **candidate to deprecate**; ⚠ **8057 clashes with apex-vision-ik** |
| 7 | **litellm** · 4000 | MANUAL: `start-litellm.bat` or `cd services/litellm && docker compose up -d` | **ENTIRELY custom** (not in repo master); config + `custom_hooks.py` (omni max_tokens cap + modalities force + agents-a1 thinking) | Unified OpenAI gateway → ONE Hermes provider, all models | ✅ keep — always up, CPU-only, no GPU |
| 8 | **apex-35b-vision-ik** · 8057 | `./serve.sh apex-vision-ik` | **NEW** ik vision compose (mudler I-Compact + unsloth mmproj-BF16); shares apex `.env` | ⭐ **daily driver + VISION** (screenshots) — ~90 TPS, thinking-ON, 131K, single card. Keeps ik speed | ✅ keep — screenshot driver |
| 9 | **deckard-40b** · 8199 (**+vision** · 8200) | `./serve.sh deckard` / `deckard-vision` | **NEW** mainline llama.cpp; MTP n=2, thinking-ON+preserve; vision = +mradermacher mmproj (MTP kept) | ⭐ **deliberation / hard reasoning** — uncensored dense 40B, ~41 TPS, 131K. Vision variant = same + images | ✅ keep — the "think hard" model |
| 10 | **hauhau-35b** · 8073 (**+vision swap**) | `./serve.sh hauhau` / `hauhau-vision` | **NEW** mainline llama.cpp; MTP n=3, thinking-ON, uncensored; vision = +unsloth mmproj | Uncensored 35B-A3B MoE, 262K, both cards. Vision swap adds images | 🧪 keep — uncensored MoE flavor |
| 11 | **agents-a1** · 8072 | `cd models/agents-a1/… && docker compose up -d` | vLLM fp8 (generated compose) + LiteLLM thinking hook (`AGENTS_A1_THINKING`) | Agentic thinking model, thinking forced ON, both cards | 🧪 eval — weights ~36 GB; niche agentic |

**Redundancy / deprecation:**
- **The 35B MoE exists 3 ways** — apex-single (#1, daily), 35b-a3b vLLM dual (#3), apex-1M (#6). #1 is the driver;
  #3 only earns its slot for MoE **vision + concurrency**; #6 is an unvalidated 1M experiment. **To trim: park #6**,
  and drop #3 unless you actually use MoE subagents/vision.
- **Vision options (lots now)** — single-card fast: **#8 apex-vision-ik** (screenshots on the daily driver).
  Dual: #2 (27b 262K), #3 (35b-a3b 262K), #9 deckard-vision, #10 hauhau-vision. Media (audio/video): #5 omni
  via a light UI (§11). Quick screenshot → **#8**; big-ctx image analysis → **#2**; uncensored + images → #9/#10.
- **mmproj + MTP coexist** (llama.cpp b9570 & ik) — vision variants keep their drafter; only apex leaves MTP
  off (net-negative on that MoE, not a vision limit). See §12 / the compose headers.
- Not run by you but in the registry (ignore unless needed): gemma-4-31b/12b/26b, diffusiongemma.

---

## 11. Light UI for Omni (and all models) — NOT Hermes

Hermes is a full AGENT (loads ~40K of tools/MCP into context, drives a tool loop). That's wrong for
Omni: its 64K window can't hold the overhead + media, and given a video Hermes flails with ffprobe
instead of "seeing" it. For plain multimodal chat you want a THIN OpenAI-compatible UI.

- **Ollama does NOT apply** — it serves its OWN GGUF models; you can't point it at Omni. The UI people
  associate with it is **Open WebUI** (a separate front-end), which is the right answer.
- **⭐ Open WebUI** (Docker): points at any OpenAI endpoint. Aim it at **LiteLLM `:4000`** → every model
  (incl. omni-30b) in one dropdown, image upload for vision, zero agent overhead. Image analysis = attach
  + ask, answer in seconds. Runs as a container alongside litellm.
- Alternatives: **LibreChat** (Docker, similar), or desktop apps **Chatbox / Jan / Cherry Studio** (point
  at `:4000`, key `sk-litellm-master-key`).
- **Video caveat (same as Hermes):** these UIs upload IMAGES, not video frames. For video → extract
  frames first and attach the frames as images. Omni sees images perfectly (validated); it's the
  *video-file* path that no OpenAI UI wires up.
- **Open WebUI setup (done 2026-07-09):** launcher `C:\Users\Paul\.local\bin\open-webui-serve.cmd` edited
  → `--port 8088` (8080 collided) + `OPENAI_API_BASE_URL=http://127.0.0.1:4000/v1` +
  `OPENAI_API_KEY=sk-litellm-master-key`. Open http://localhost:8088, pick omni-30b. Runs on GPU-mutex —
  omni must be the live model.
- ⚠️ **Windows → WSL-published Docker ports: always `127.0.0.1`, never `localhost`.** Windows resolves
  `localhost` to `::1` first; under `networkingMode=mirrored` these ports don't serve IPv6, so every call
  eats a failed `::1` connect first. `curl` hides this (Happy Eyeballs races the families, ~0.2s cost);
  clients that walk `getaddrinfo` serially do not — this was the cause of the Hermes desktop UI hangs
  (fixed 2026-07-10 in its `config.yaml`, and here). Note LiteLLM runs on the **native WSL Docker engine**
  (`default` context, unix socket), *not* Docker Desktop — don't debug this via Docker Desktop.
  Container-internal routes (`host.docker.internal:<port>` in `services/litellm/config.yaml`) are
  unaffected; nothing there needs changing.
- **Open WebUI routes uploads BY FILE TYPE (this is the whole gotcha):**
  - **IMAGE (PNG/JPG)** → sent as VISION → Omni sees it. ✅ Upload the extracted frame, not the clip.
  - **Non-image file (mp4, pdf, …)** → goes to the **document/RAG pipeline**: tries text extraction, a video
    has none → shows **"No sources found"** and passes the model only the FILENAME. NOT a vision failure — it
    never got pixels. (No web-search globe toggle exists; the "Querying/No sources" IS this RAG pipeline.)
  - ⚠️ **DANGER — filename hallucination:** given only a filename, Omni CONFIDENTLY FABRICATES a plausible
    description from the words in the name. **The tell (confirmed 2026-07-09 by Paul):** on a video-file upload
    it got the *technique* RIGHT ("double aerial waveland" — which is IN the filename) but the *character* WRONG
    (a detail NOT in the filename → it guessed and missed). Right on everything the name says, wrong on
    everything that needs eyes = it never saw pixels. (Corroborated: filename-only text prompt → similar
    breakdown but "Wario" not "Fox"; runs disagree = guessing.) A real FRAME instead reads actual on-screen
    content (HUD text). **For real analysis ("which clip has trees") you MUST upload extracted JPG frames —
    a video-file upload gives filename-flavored guesses that look right only when the name already says the answer.**
  - A **GIF** is animated (multi-frame) → extract a still.
- **Video workflow:** on **Windows** (Open WebUI runs on Windows, not WSL), ffmpeg is on PATH via WinGet
  (also `C:\Users\Paul\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe`). `ffmpeg -i "clip.mp4" -vf fps=1
  "frame_%02d.jpg"` → upload the JPGs. Validated: `waveland-f02.jpg` → Omni described the scene + read the
  on-screen HUD player names.
- **⭐ Batch tool — `/home/pawl/media-tools/describe_videos.py`:** describes a whole FOLDER of videos,
  grounded (frames → Omni via litellm → description + tags → CSV/MD report). NON-destructive. No harness
  (Hermes/OpenClaw/etc.) feeds raw video to a model — this is the reliable path. Needs litellm :4000 +
  omni :8042 up. `python3 describe_videos.py "<folder>" [--frames 3]`. Validated 2026-07-09: read player
  names/stage/timer off the HUD (vs the filename-guess "Fox"). Stage-2 organize (copy into category folders,
  originals untouched) = TODO, build on request.

---

## 12. Thinking / preserve-reasoning capability (per model)

**Two separate things:**
- **Thinking** (aka reasoning): the model emits a `<think>…</think>` block before answering. Good for hard
  multi-step logic / debugging / planning; costs latency + output tokens + context. NOT a quality downgrade
  when OFF and NOT a quality upgrade "for free" when ON — it's a *speed vs deliberation* trade.
- **Preserve-thinking**: keep prior turns' `<think>` blocks in the multi-turn context (Qwen
  `preserve_thinking` template kwarg). Only matters when thinking is ON. `true` = better reasoning
  continuity across turns; `false` = saves context. Independent of whether thinking is on.

| Model (Hermes id) | Engine | Thinking | Preserve | Toggle (env in the compose `.env`) | Default |
|---|---|---|---|---|---|
| **apex-35b-compact** · 8056 | ik-llama | ✅ | ✅¹ | `REASONING=off\|on\|auto` + `PRESERVE_THINKING` | thinking **on** / preserve true² |
| **apex-35b-vision-ik** · 8057 | ik-llama | ✅ | ✅¹ | `REASONING` + `PRESERVE_THINKING` | thinking **on** / preserve true² |
| **apex-35b-vision** · 8058 | llama.cpp (mainline) | ✅ | ✅ | `REASONING` + `PRESERVE_THINKING` | off / preserve false |
| **deckard-40b** (+`-vision` :8200) · 8199 | llama.cpp | ✅ | ✅ | `REASONING` + `PRESERVE_THINKING` | thinking **on** / preserve true |
| **hauhau-35b** (+vision swap) · 8073 | llama.cpp | ✅ | ✅ | `REASONING` + `PRESERVE_THINKING` | thinking **on** / preserve true |
| **carnice-v2** · 8070 | beellama | ✅ | ✅ | `ENABLE_THINKING` + `PRESERVE_THINKING` | thinking **on** / preserve true |
| **qwen3.6-27b** · 8010/8021 | vLLM | ✅ | ✅ | `ENABLE_THINKING` + `PRESERVE_THINKING` | thinking **off** / preserve true |
| **qwen3.6-35b-a3b** · 8051 | vLLM | ✅ | ✅³ | `ENABLE_THINKING` + `PRESERVE_THINKING` | thinking **ON** / preserve true³ |
| **agents-a1** · 8072 | vLLM + LiteLLM hook | ✅ | ✅ | `AGENTS_A1_THINKING` in `custom_hooks.py` | **on** (forced by hook) |
| **omni-30b** · 8042 | vLLM-omni | ❌ not a thinking model | — | — | — |
| gemma-4-* (not run by you) | varies | some variants ✅ | varies | — | — |

¹ WIRED + VERIFIED 2026-07-10: ik-llama *does* support `--chat-template-kwargs` (checked `--help`), both
apex templates implement `preserve_thinking` (custom apex template + the GGUF-embedded native), and a
prior-turn recall test with `REASONING=on PRESERVE_THINKING=true` retained the earlier `<think>` (model
recalled an injected codeword). Wired into all four apex ik composes (mtp/long/fit-mtp/vision).
² apex default flipped to thinking-ON + preserve 2026-07-10 (Paul's choice): the thinking on/off A/B was a
quality wash on apex (8/8 both ways, quick probe), so thinking costs latency + tokens, not correctness.
Set `REASONING=off` in the apex `.env` to go back to max-snappy.
³ 35b-a3b vLLM is thinking-**ON**: its `.env` sets `ENABLE_THINKING=true` (comment: "testing whether
reasoning improves agentic comprehension vs the 27b") — the compose *fallback* is false, the `.env`
overrides it. `PRESERVE_THINKING=true` is now WIRED (2026-07-10): fp8.yml's `--default-chat-template-kwargs`
carries both keys (`{"enable_thinking": …, "preserve_thinking": …}`, matching the 27b vLLM pattern), and the
AutoRound weights' embedded `chat_template.jinja` implements `preserve_thinking` on-disk (2 refs, same as
enable_thinking) — so no froggeric-style template mount is needed here. Was a dead no-op before (same
dead-code carnice had pre-fix); no longer.

**The lever is the compose/.env default, NOT the Hermes setting.** Hermes' `agent.reasoning_effort` is
**dropped by LiteLLM** (`drop_params`, see §9) — it does nothing to these backends. To change thinking, edit
the model's `.env` (`REASONING` / `ENABLE_THINKING`) and **recreate** the container (§0). To actually SEE the
`<think>` block in the Hermes UI, also set `display.show_reasoning: true` (it's `false` now → it thinks but
hides it).

**When to turn thinking ON:** hard debugging, multi-file refactors, tricky algorithms/math, planning a task
before doing it. **When OFF:** quick chat, simple edits, latency-sensitive/agentic loops, or when context is
tight (thinking + preserve eat the window). Thinking does *not* "frequently cause weaker code" — on hard
problems it usually reduces errors; its cost is speed + tokens, not correctness. If you like thinking blocks
and don't mind slower replies, turning it on for apex is low-risk (the same-family hauhau 8-pack A/B was a
wash: think-off 103 vs think-on 105 / 150). For an apex-specific number, run `scripts/quality-test.sh --quick`
with `REASONING=on` vs `off`.

**⚠ Port note (2026-07-10):** apex-35b-vision-ik and the older apex-yarn-1m both default to **:8057** — they
never run together (GPU-mutex) but if you want both registered distinctly, move one to :8059.
