# 🎬 VIDEO MODELS — TODO / evaluation notes

**Status:** deferred. Working pipeline today = **frames → 35B-A3B** (`media-tools/describe_videos.py`).
This file tracks the *better* video path to pilot when we come back to it. Not yet cataloged.

---

## The question Paul asked: why not Qwen3-VL-8B?

Short answer: **we probably should.** It's a stronger fit than Qwen3-Omni for video, and
likely better than 35B-A3B-vision too — *if* it serves cleanly on this rig. It just hasn't
been tested here yet.

(Note on naming: there is **no Qwen3-VL-7B**. The small dense sizes are **4B and 8B**
Instruct/Thinking, released 2025-10-15. Also 32B, plus 30B-A3B / 235B-A22B MoE variants.
The "7B" everyone says is really the **8B**.)

### Why Qwen3-VL is the right tool for "understand my videos"

| Property | Qwen3-Omni-30B (tested here) | 35B-A3B vision (shipping now) | **Qwen3-VL-8B (to test)** |
|---|---|---|---|
| Native video input | yes (temporal) | no — stills only, 2-image cap | **yes — purpose-built** |
| Video encoder speed on Ampere | **~14 min/clip** (un-CUDA-graphed ViT) | fast (few sec, but no motion) | **should be fast** (8B, video-optimized) |
| Temporal / event localization | basic | none | **timestamp-grounded** (Text–Timestamp Align, Interleaved-MRoPE) |
| Context | 48K (desktop-shared GPU0 cap) | 262K | **256K native, → 1M** |
| Long video | no | n/a | **up to ~2 h clips** |
| Motion description | yes | no | **yes, precise** |

**The core insight from our Omni test (2026-07-09):** Omni's *quality* was actually the best of
any method — it correctly described both fighters by appearance, read the stage, AND captured
motion ("spinning kick", "overhead strike") without fabricating character names like the frame
runs did. The problem was **purely throughput**: a 3-second clip became 11,583 tokens and took
~14 min to prefill because this `vllm-omni:v0.20.0` build runs the multimodal encoder eager
(`compile_mm_encoder: False`, no CUDA graph) on Ampere (no native FP8 compute).

Qwen3-VL-8B attacks exactly that wall: it's **video-native** (so you get motion/temporal like
Omni) but **small (8B)** and built as a mainline VL model with better-optimized encoder paths —
so the prefill that kills Omni should be far cheaper. Best-of-both: motion understanding at
usable speed. A 32B would be smarter but would re-hit the prefill cost; **start at 8B.**

### Honest caveats before committing

1. **Serving-stack reality > benchmarks.** This repo exists because "benchmarks well" ≠ "serves
   cleanly on 2×3090 PCIe". Must verify Qwen3-VL-8B runs on our vLLM pin (`v0.24.0`) — check
   `supported_model_families`, chat template, and whether video_url prefill is CUDA-graphed in
   this build (if it *also* runs the encoder eager, the Ampere win shrinks).
2. **You may not even need video understanding.** For the actual stated goal — "sort my clips by
   content / which have trees" — content tagging doesn't need motion or temporal reasoning.
   Frames → any decent VLM already solves it. Qwen3-VL matters only if you want real *motion/event*
   descriptions ("he opens the door, then picks up the backpack").
3. Would be a **new catalog entry** (model YAML + `compose_registry.py` + compose + profile-compat),
   not a drop-in. Or serve-locally first via `pull.sh ... --profile-like vllm/minimal` to smoke-test
   before cataloging.

### ChatGPT's take (Paul shared it) — graded

- ✅ Right: "separate perception from reasoning" (scene-detect → keyframes → Whisper → OCR → VLM);
  GIFs work because they fit in sampled frames; don't use Omni for batch video.
- ✅ Right direction: Qwen-VL family is the gold standard for local video. (It said Qwen2.5-VL;
  **Qwen3-VL is newer and better** — timestamp grounding, 256K–1M ctx.)
- ❌ Wrong reason: it claims "Omni loses motion / treats frames as unrelated images." Our test
  showed the opposite — Omni *did* capture motion and was the most accurate. The real problem is
  **Ampere prefill speed**, which ChatGPT never diagnoses.
- ⚠️ Trap: it pitches Qwen2.5-VL-**32B** as the pick. Bigger = *slower* prefill on the same video
  tokens; the bottleneck is token-count × encoder speed, not model IQ. **Go small (8B) first.**

---

## Pilot plan (when we resume)

1. Smoke-test serve: `scripts/pull.sh Qwen/Qwen3-VL-8B-Instruct --profile-like vllm/minimal`
   (or the AWQ/int4 quant), single-card, and confirm it boots on our vLLM pin.
2. Send one native `video_url` clip (reuse `media-tools/describe_video_native.py`, repoint model)
   and **measure prefill time + prompt tokens** — the number that decides everything.
3. If prefill is usable (< ~30 s/clip), it wins for motion-aware cataloging → catalog it properly.
4. If it *also* hits the eager-encoder wall, stick with frames → 35B-A3B for tagging and treat
   any native-video model as "one precious clip, I'll wait."

## Related files
- `media-tools/describe_videos.py` — frames → 35B-A3B batch cataloger (the working path).
- `media-tools/describe_video_native.py` — native video_url → Omni (proof-of-concept; slow).
- `docs/UPSTREAM.md` — where an "Omni encoder not CUDA-graphed on Ampere" upstream row would go.

## Sources
- [Qwen3-VL (GitHub)](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL Technical Report (arXiv 2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL blog — Sharper Vision, Deeper Thought](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef)
- [Ollama: qwen3-vl](https://ollama.com/library/qwen3-vl)
