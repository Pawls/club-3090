# 📸 Personal Media Gallery — project plan

**Vision:** turn a pile of personal photos/videos into a **local, searchable gallery**. Type a
query — a place, an activity, eventually a *person's name* — and get back every clip that matches.
All on-device, no cloud, in keeping with the rest of this rig's local-first ethos.

**Status:** Phase 0 proven (2026-07-09). Everything below Phase 1 is roadmap, not built yet.
This doc is the north star; we pick up phases when we choose to.

---

## Why this is promising

The hard part of a media gallery is *perception* — knowing what's in each file. We just proved
that a local VLM (35B-A3B, frames → description) produces **grounded, useful descriptions** even
on 176×144 2012 phone clips (read a playpen mesh, a disco ball, a floral top), at **~2 min for 26
clips**, fully non-destructive. That perception layer is the foundation everything else builds on:
descriptions + tags → search index → gallery UI → (later) face labels.

---

## Phase 0 — Batch description ✅ DONE (the successful experiment)

- `media-tools/describe_videos.py` — samples frames per clip → VLM → grounded description + tags,
  writes CSV + Markdown. Non-destructive (reads source, writes report only).
- Validated on `E:\...\phone 2-16-12\My_Videos` (26 × `.3g2`) → `F:\AI\VIDEO_CATEGORY_TEST\`.
- Model path chosen: frames → 35B-A3B (fast, clean) over native-video Omni (best quality but
  ~14 min/clip on Ampere — see `VIDEO_MODELS_TODO.md`).

## Phase 1 — Richer, more reliable descriptions (next up)

- [ ] **Frame montage** (immediate) — tile 9 sampled frames into 1 image so 2 "images" = 18 frames
      of coverage, beating 35B-A3B's `limit-mm-per-prompt image=2`. Fixes thin coverage on long clips.
- [ ] **Audio → text** (Whisper, local) — transcribe speech; a huge signal for home videos
      ("happy birthday", names spoken). Separate the audio track with ffmpeg, run whisper.cpp.
- [ ] **Smarter keyframes** — PySceneDetect on scene cuts instead of naive evenly-spaced sampling,
      so we sample *distinct* moments, not near-duplicates.
- [ ] **Photos too** — same pipeline minus frame extraction; images go straight to the VLM.
- [ ] Evaluate **Qwen3-VL-8B** as the perception model (native video + timestamps, small = fast).
      Full rationale + pilot plan already in `VIDEO_MODELS_TODO.md`.

## Phase 2 — Searchable index + gallery UI

- [ ] **Structured store** — move from flat CSV/MD to a small DB (SQLite is plenty) with one row per
      media file: path, description, tags, transcript, duration, date, resolution, thumbnail.
- [ ] **Semantic search** — embed each description (local embedding model) so queries match by
      *meaning* ("kids at a party") not just exact tag text. Vector search over the embeddings.
- [ ] **Gallery front-end** — thumbnail grid + search box. Options, cheapest-first:
      a static HTML page generated from the DB; or a small local web app (the repo already runs
      Open WebUI / has a Textual-TUI pattern in `tools/serve-cockpit/`). Keep it local-only.
- [ ] Click a result → open the original file in place (never moved).

## Phase 3 — People (facial recognition / labeling)  ← the "type my name" goal

- [ ] **Detect + embed faces** — a local face model (e.g. InsightFace/`face_recognition`) produces a
      face embedding per detected face per keyframe.
- [ ] **Cluster** unlabeled faces → "Person A, Person B…"; you attach names once per cluster.
- [ ] **Enrollment** — a few labeled photos per person seed the matcher.
- [ ] **Search by person** — "Paul" → every clip/photo whose face embeddings match Paul's cluster.
- [ ] **Privacy is the whole point of doing it locally** — face data is biometric; it never leaves
      the machine. Store embeddings alongside the media DB, nowhere else.

## Phase 4 — Unified natural-language search

- [ ] One query box spanning description + transcript + tags + people:
      *"videos of Paul and the baby at a birthday party"* → intersect person-match + semantic-match.
- [ ] Optional: a local LLM turns the free-text query into structured filters over the DB.

---

## Data model sketch (Phase 2 target)

```
media(id, path, kind[photo|video], created, duration, width, height, thumb_path)
description(media_id, model, text, tags, embedding)
transcript(media_id, text, embedding)              -- Phase 1 (Whisper)
face(id, media_id, frame_ts, bbox, embedding, person_id?)   -- Phase 3
person(id, name)                                   -- Phase 3
```

## Design principles (carry through every phase)

1. **Non-destructive, always.** Originals are read-only. Reports/index/thumbnails live elsewhere
   (currently `F:\AI\VIDEO_CATEGORY_TEST\`). Copy, never move; the user verifies before any delete.
2. **Local-only.** No cloud APIs. Fits the rig; mandatory once faces/biometrics are involved.
3. **Incremental + resumable.** Re-running skips already-processed files (hash/mtime check) so the
   library can grow without full reprocessing.
4. **Perception model is swappable.** The pipeline shouldn't care if it's 35B-A3B, Qwen3-VL, or
   whatever's best later — it consumes "frames/audio → text+embedding".

## Related files
- `media-tools/describe_videos.py` — Phase 0 batch cataloger (frames → 35B-A3B). **Working.**
- `media-tools/describe_video_native.py` — native video → Omni PoC (slow; reference only).
- `VIDEO_MODELS_TODO.md` — perception-model evaluation (Qwen3-VL-8B pilot, Omni Ampere wall).

## Open questions for later
- Where should the master library DB live? (F: drive alongside output seems natural.)
- Photos and videos in one gallery, or separate? (Lean: one unified library.)
- How much of E:/other drives is in scope? (Start with the test folder, expand once the UI exists.)
