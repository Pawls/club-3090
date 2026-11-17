# Handoff — adopting the LiteLLM route generator (`litellm-emit.sh`, #1078)

> **Scope note:** a decision, not a change request. Nothing here has been implemented.
> Opened as fallout from the `aa45b49b` upstream merge, where the conflict in
> `services/litellm/config.yaml` was resolved **in favour of the hand-maintained table**
> to avoid silent route loss. That resolution was deliberate and is safe to keep
> indefinitely — this note exists so the choice stays a choice. Safe to delete once
> the decision is recorded.

**Subject:** `services/litellm/config.yaml` (30 routes, hand-maintained) vs
`scripts/lib/litellm-emit.sh` (297 lines, generates a marked region from the registry).
**Gate:** `scripts/tests/test-litellm-generate.sh` — currently **red, and red pre-merge too**
(baselined against `backup/pawl-cleanup-premerge-20260901`; not a merge regression).

---

## 1. The question

Upstream moved its whole route table inside a `BEGIN/END GENERATED LOCAL BLOCK` written
from the registry, to kill "the hand-typed *where is the model* drift class." Our table
predates that and uses a **topology-suffix** convention the generator has no concept of.

Two things needed deciding, and only one of them is actually open:

1. **Is the generator's output shape better than ours?** — No. Answered in §3. It emits
   strictly less per route.
2. **Is its *port derivation* better than ours?** — Yes, unambiguously. Answered in §2.
   That is the whole value, and it is capturable without adopting its output shape.

---

## 2. What we SHOULD be taking from it

Three real wins, in descending order of value:

| Win | Why it matters here |
|---|---|
| **Port drift elimination** | We hand-type 30 `api_base` ports. The generator reads `default_port` off the registry entry. Every port we type by hand is a future silent 404 when a compose's `${PORT:-NNNN}` moves. This is the entire point of #1078 and the only reason to care. |
| **`# status: <status>` annotation** | A route whose registry status is not production/caveats emits with a trailing `# status: incubating`, so an operator sees the `--force` gate in the routing table. We annotate none of our 30. |
| **`serve_aliases`** | Models "several names, one port" as first-class. We have four such groups by hand: `:8010` (27b-dual + -autoround), `:8032` (gemma-dual + -autoround), `:8091` (qwen3.8-27b + -fp8), `:8210` (five muse-glimmer routes). |

Also worth knowing: eligibility is **explicit opt-in** — a slug is only generated if its
registry entry carries `gateway: True`. Nothing is inferred. That is what makes the
partial adoption in §4 possible.

---

## 3. What it would DESTROY — why wholesale adoption is wrong

The generated region is **machine-owned and rewritten wholesale**. The emitter produces
exactly five lines per route (`scripts/lib/litellm-emit.sh:239-250`):

```yaml
  - model_name: <name>  # status: <status>
    litellm_params:
      model: openai/<name>
      api_base: http://host.docker.internal:<port>/v1
      api_key: EMPTY
```

That is all it can express. Three of our patterns die inside that region:

**a. Multi-scene models — 12 of 30 routes.** The generator resolves **one canonical route
per model**, walking topologies `dual → single → multi4 → multiN` and taking the first
gateway-flagged hit. Our convention is one route per *(model, topology)*. These pairs
cannot both exist inside the block:

| Model | Our scenes |
|---|---|
| `qwen3.6-27b` | `-dual` :8010 · `-single` :8020 |
| `qwen3.6-35b-a3b` | `-dual` :8051 · `-single` :8060 |
| `muse-glimmer-30b` | base :8210 · `-long` :8211 · `-dual` :8212 · `-max` :8213 |
| `deckard-40b` | `-dual` :8199 · `-dual-vision` :8200 |
| `apex-35b` | `-single-vision` :8057 · `-single-vision-mainline` :8058 |

`serve_aliases` does **not** rescue this — it attaches extra names to the *same* route and
*same* port. Different scenes are different ports by definition.

**b. Enriched routes — 5 of 30.** The emitter has no field for anything past `api_key`:

- `qwen3.8-27b-vision` :8117 — `model_info.max_input_tokens: 262144` **plus** `timeout: 5400`.
  This one is actively guarded by `scripts/tests/test-litellm-context-window.sh`, which
  re-derives the window from the compose and fails on drift. Generating this route would
  strip the declaration and break that gate. It is also the newest deliberate work in this
  file (`4294ccbe`).
- `muse-glimmer-30b-long` :8211, `muse-glimmer-30b-dual` :8212 — `timeout`.
- `qwen3.8-max`, `qwen3.8-max-nothink` — `rpm` / `tpm` / `extra_body`, and **no `api_base`**.
  These are cloud routes; the region contract already excludes them.

**c. Reasoning-effort aliases — 5 routes on :8210.** `muse-glimmer-30b{,-low,-medium,-high,-xhigh}`
depend on `custom_hooks.py §D` reading the effort level off the alias suffix and injecting
it per request. The generator has no concept of a route that exists purely to be
suffix-parsed downstream.

---

## 4. The shape to adopt — hybrid, which is what upstream itself does

Worth noting before anything else: **upstream does not fully generate its own file.**
On `master` at `8abac569`, 13 routes sit inside the block and 3 (`agents-a1`,
`qwen3.8-max`, `qwen3.8-max-nothink`) are hand-maintained below it. The region contract
is explicitly designed for this — everything outside the markers is untouched by the
script. So a hybrid is not a compromise or a fork-ism; it is the intended usage.

**Proposed split of our 30:**

*Generate (~8)* — single-scene, no enrichment, port-drift-exposed:
`deepseek-v4-flash` :8030 · `agents-a1-dual` :8072 · `gemma-4-12b-single` :8038 ·
`hauhau-35b-dual` :8073 · `carnice-v2-dual` :8070 · `omni-30b-dual` :8042 ·
`qwen3.8-27b` + `qwen3.8-27b-fp8` :8091 (upstream already generates this exact pair via
`serve_aliases`, so it is a proven case).

*Keep hand-maintained (~22)* — everything in §3a, §3b, §3c.

**Open sub-question the split does not settle:** our generated routes would come out named
by *served-name*, not by our topology suffix — e.g. `agents-a1`, not `agents-a1-dual`
(upstream's hand-maintained route is named `agents-a1`, confirming the generator produces
the bare name). Either we accept bare names inside the block and keep suffixes only
outside it — which makes the convention inconsistent across one file — or we set a
`served_name` override per entry. **Decide this before writing anything**, because it
determines whether the split is cosmetically worth it at all.

---

## 5. Procedure, when it is worth doing

Not urgent. The gate has been red on this branch since before the merge and nothing
routes incorrectly today.

1. Decide the §4 naming sub-question first. If bare names are unacceptable and
   `served_name` overrides turn out to be per-entry busywork, **stop here** — the
   remaining win is small and the current file is fine.
2. Add `gateway: true` to the ~8 registry entries in the generate list. Nothing else
   becomes eligible; eligibility is opt-in.
3. Insert the `BEGIN`/`END` marker pair around **only** those routes, leaving the other 22
   below the `END` marker.
4. `bash scripts/lib/litellm-emit.sh` to populate, then `--check` for the diff.
5. `bash scripts/tests/test-litellm-generate.sh` — the gate asserts content *outside* the
   markers is byte-identical across a regeneration, which is exactly the property we need
   to trust the hand-maintained 22.
6. `bash scripts/tests/test-litellm-context-window.sh` — confirms `qwen3.8-27b-vision`
   survived untouched.
7. Boot one generated lane and one hand-maintained lane; `GET /v1/models` on the gateway.

## 6. The other red gate in this file's neighbourhood

`scripts/tests/test-switch-registry-parity.sh` is also red, also pre-existing, and
**unrelated** to any of the above — it wants registry entries for seven single-card
composes that exist on disk (`muse-glimmer-30b` long/vision variants, two
`qwen3.6-35b-a3b` vision composes, `qwen3.6-27b` single fp8-mtp). Recorded here only so
the two are not conflated when someone next reads a red test list.
