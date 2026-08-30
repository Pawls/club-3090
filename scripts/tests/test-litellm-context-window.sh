#!/usr/bin/env bash
# Gate: every litellm route that DECLARES model_info.max_input_tokens declares the
# number its backend compose actually serves.
#
# Why this exists: /model/info is how a harness autodetects the loaded context window
# instead of trusting a hand-typed constant (PAWL's backends.probe_context reads
# data[].model_info.max_input_tokens; null/0/missing means "no opinion" and it keeps its
# own declared number). But LiteLLM ECHOES the yaml — it never asks the engine. So a
# declaration that drifts from the backend's real n_ctx is WORSE than declaring nothing:
# null makes the harness fall back, a stale number makes it believe.
#
# Derivation: route api_base port -> the compose whose `${PORT:-<port>}` default matches
# -> `-c ${CTX_SIZE:-N}` and `-np ${NP:-M}`. The per-session window is N/M (what llama.cpp
# reports as /props default_generation_settings.n_ctx), and that is what must be declared.
#
# ⚠️ Scope, deliberately: this checks routes that DO declare, not that every route does.
# Declaring is opt-in — a route whose upstream can't be pinned (the DashScope cloud
# aliases) is correctly left undeclared rather than guessed. A compose-side `.env`
# override of CTX_SIZE/NP would beat the compose default; this gate reads the default and
# says so, because the .env is a user file the repo can't see on another rig.
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CFG="${LITELLM_CONFIG:-services/litellm/config.yaml}"

python3 - "$CFG" <<'PY'
import re, sys, pathlib

cfg_path = pathlib.Path(sys.argv[1])

# --- routes that declare a context window -------------------------------------------
routes = []          # (model_name, port|None, declared)
name = port = declared = None
for ln in cfg_path.read_text(encoding="utf-8").splitlines():
    m = re.match(r"\s*-\s*model_name:\s*(\S+)", ln)
    if m:
        if name and declared is not None:
            routes.append((name, port, declared))
        name, port, declared = m.group(1), None, None
        continue
    a = re.search(r"api_base:\s*https?://(?:host\.docker\.internal|localhost|127\.0\.0\.1):(\d+)", ln)
    if a:
        port = int(a.group(1))
    d = re.match(r"\s*max_input_tokens:\s*(\d+)", ln)
    if d:
        declared = int(d.group(1))
if name and declared is not None:
    routes.append((name, port, declared))

if not routes:
    print("OK: no litellm route declares model_info.max_input_tokens (nothing to drift).")
    sys.exit(0)

# --- compose defaults, indexed by the host port they default to ----------------------
by_port = {}
for f in pathlib.Path("models").rglob("compose/**/*.yml"):
    if "_archive" in f.parts:
        continue
    txt = f.read_text(encoding="utf-8")
    p = re.search(r"\$\{(?:ESTATE_PORT:-\$\{)?PORT:-(\d+)\}", txt)
    if not p:
        continue
    ctx = re.search(r"-c\s+\$\{CTX_SIZE:-(\d+)\}|--ctx-size\s+\$\{CTX_SIZE:-(\d+)\}", txt)
    np_ = re.search(r"-np\s+\$\{NP:-(\d+)\}|--parallel\s+\$\{NP:-(\d+)\}", txt)
    if not ctx:
        continue
    ctx_v = int(next(g for g in ctx.groups() if g))
    np_v = int(next((g for g in np_.groups() if g), 1)) if np_ else 1
    by_port.setdefault(int(p.group(1)), []).append((f, ctx_v, np_v))

bad = []
ok = []
for name, port, declared in routes:
    if port is None:
        bad.append((name, declared, "declares a context window but has no local api_base "
                                    "port — nothing to derive it from"))
        continue
    cands = by_port.get(port)
    if not cands:
        bad.append((name, declared, f"no compose defaults to :{port}, so the declaration "
                                    f"cannot be checked against a backend"))
        continue
    matched = [(f, c // n, c, n) for f, c, n in cands if c // n == declared]
    if matched:
        f, per, c, n = matched[0]
        ok.append(f"    {name} -> :{port}  {declared}  ({f}: -c {c} / -np {n})")
        continue
    detail = "; ".join(f"{f} -> -c {c} / -np {n} = {c // n}" for f, c, n in cands)
    bad.append((name, declared, f"declares {declared} but :{port} serves {detail}"))

if bad:
    print("FAIL: litellm model_info.max_input_tokens drifted from the backend compose:")
    for name, declared, why in bad:
        print(f"    {name}: {why}")
    print("Fix: set max_input_tokens to the compose's per-session window (CTX_SIZE / NP),")
    print("     or drop the model_info block — null makes a harness fall back to its own")
    print("     number, a stale value makes it trust a window the engine does not have.")
    sys.exit(1)

print(f"OK: all {len(ok)} declared litellm context windows match their compose defaults:")
for line in ok:
    print(line)
PY
