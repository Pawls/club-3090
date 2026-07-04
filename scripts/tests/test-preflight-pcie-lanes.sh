#!/usr/bin/env bash
# Unit test for preflight_pcie_lane_width — the proactive PCIe-lane-width warning
# (club-3090 #142). On PCIe-only rigs, TP's per-layer NCCL all-reduce bottlenecks on
# the slowest card's link, so a GPU in a physical x4 slot silently costs ~15% decode
# TPS. The launch-time topology classifier keys only on VRAM+SM and can't see lane
# width, so this preflight surfaces it. Warn-only (never blocks). We mock nvidia-smi
# (a shell function satisfies `command -v`) so the check runs without real GPUs.
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../preflight.sh
source "${ROOT_DIR}/scripts/preflight.sh"

FAILS=0
pass() { echo "  ok:   $1"; }
fail() { echo "  FAIL: $1"; FAILS=$((FAILS + 1)); }

# Mocked GPU lane state: set MOCK_SMI to the CSV rows nvidia-smi should emit
# (index, pcie.link.width.current, pcie.link.width.max).
nvidia-smi() { printf '%s\n' "$MOCK_SMI"; }

# Keep the selector env clean so cases control it explicitly.
unset CLUB3090_GPU CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES PREFLIGHT_NO_PCIE_HINT 2>/dev/null || true

# Dual-card compose: TP=2 header.
tp2=$(mktemp)
cat > "$tp2" <<'YAML'
# Tensor-parallel: 2
# Requires-min-gpu-count: 2
services:
  x:
    command: ["--tensor-parallel-size", "2"]
YAML

# Single-card compose: no TP header (defaults to 1 card).
single=$(mktemp)
cat > "$single" <<'YAML'
services:
  x:
    command: ["--tensor-parallel-size", "1"]
YAML

# 1) ASYMMETRIC x16 + x4 under TP=2 → warns, naming GPU 1 and its narrow width.
MOCK_SMI=$'0, 16, 16\n1, 4, 16'
out=$(preflight_pcie_lane_width "$tp2" 2>&1); rc=$?
[ "$rc" -eq 0 ] && pass "returns 0 (never blocks)" || fail "must return 0, got rc=$rc"
echo "$out" | grep -q "WARN" && pass "warns on x16+x4 asymmetry under TP=2" || fail "should warn on x16+x4"
echo "$out" | grep -q "GPU 1" && pass "names the narrow card (GPU 1)" || fail "should name GPU 1"
echo "$out" | grep -qE "x4( of x16)?" && pass "reports the negotiated x4 width" || fail "should report x4"

# 2) SYMMETRIC x16 + x16 under TP=2 → silent (no penalty to flag).
MOCK_SMI=$'0, 16, 16\n1, 16, 16'
out=$(preflight_pcie_lane_width "$tp2" 2>&1)
[ -z "$out" ] && pass "silent on symmetric x16+x16" || fail "should be silent on x16+x16: $out"

# 3) SYMMETRIC x4 + x4 under TP=2 → warns via the absolute-narrow (current<8) branch,
#    even though current == max (no asymmetry).
MOCK_SMI=$'0, 4, 4\n1, 4, 4'
out=$(preflight_pcie_lane_width "$tp2" 2>&1)
echo "$out" | grep -q "WARN" && pass "warns on symmetric x4+x4 (absolute-narrow branch)" || fail "should warn on x4+x4"

# 4) ASYMMETRIC lanes but SINGLE-CARD compose (TP<2) → silent (gates on TP).
MOCK_SMI=$'0, 16, 16\n1, 4, 16'
out=$(preflight_pcie_lane_width "$single" 2>&1)
[ -z "$out" ] && pass "silent for single-card compose despite slow lane" || fail "single-card should not warn: $out"

# 5) SELECTOR scoping — with only the slow card DESELECTED, stay silent; select both → warn.
MOCK_SMI=$'0, 16, 16\n1, 4, 16'
out=$(CUDA_VISIBLE_DEVICES=0 preflight_pcie_lane_width "$tp2" 2>&1)
[ -z "$out" ] && pass "silent when the slow card is not selected (CUDA_VISIBLE_DEVICES=0)" || fail "should ignore deselected slow card: $out"
out=$(CUDA_VISIBLE_DEVICES=0,1 preflight_pcie_lane_width "$tp2" 2>&1)
echo "$out" | grep -q "WARN" && pass "warns when the slow card is selected (0,1)" || fail "should warn with both selected"

# 6) OPT-OUT — PREFLIGHT_NO_PCIE_HINT=1 → silent.
MOCK_SMI=$'0, 16, 16\n1, 4, 16'
out=$(PREFLIGHT_NO_PCIE_HINT=1 preflight_pcie_lane_width "$tp2" 2>&1)
[ -z "$out" ] && pass "opt-out (PREFLIGHT_NO_PCIE_HINT=1) silences the warning" || fail "opt-out should silence: $out"

# 7) FIX hint points at the docs mitigation + single-card path.
MOCK_SMI=$'0, 16, 16\n1, 4, 16'
out=$(preflight_pcie_lane_width "$tp2" 2>&1)
echo "$out" | grep -q "docs/HARDWARE.md" && pass "Fix hint links docs/HARDWARE.md" || fail "missing docs/HARDWARE.md pointer"
echo "$out" | grep -q "single-card" && pass "Fix hint offers the single-card mitigation" || fail "missing single-card mitigation"

# 8) No nvidia-smi → silent, returns 0 (never blocks a GPU-less / WSL2-degraded rig).
( unset -f nvidia-smi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    out=$(preflight_pcie_lane_width "$tp2" 2>&1); rc=$?
    { [ "$rc" -eq 0 ] && [ -z "$out" ]; } && echo "  ok:   skips silently without nvidia-smi" || echo "  FAIL: should skip silently without nvidia-smi (rc=$rc, out=$out)"
  else
    echo "  ok:   (nvidia-smi present on host — skip-case not exercised)"
  fi )

rm -f "$tp2" "$single"
if [ "$FAILS" -eq 0 ]; then echo "PASS test-preflight-pcie-lanes"; exit 0; else echo "FAIL test-preflight-pcie-lanes ($FAILS)"; exit 1; fi
