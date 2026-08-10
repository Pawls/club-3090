"""Test-suite safety net: no live subprocess, ever.

The cockpit is fully dependency-injected — tests construct ``CockpitData`` with
a ``FakeRunner`` (reads) and a ``FakeWriteRunner`` (writes), so no real process
should ever be spawned.  This autouse fixture turns that convention into an
enforced invariant: it monkeypatches the three real spawn points so that any
accidental escape (a forgotten seam, a regression that bypasses the fake)
raises loudly instead of shelling out / serving / switching on the host.

Specifically blocked for the duration of every test:
  - ``asyncio.create_subprocess_exec``       (RealRunner + core SubprocessRunner)
  - ``services.RealRunner.run``              (the live READ runner)
  - ``SubprocessRunner.start_raw``           (the live WRITE streamer)

A test that needs to assert "a write was attempted" must inject a
``FakeWriteRunner`` — which records ``start_raw`` without spawning — NOT call the
real runner.  This fixture guarantees the real one never runs.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

# Download plumbing must stay hermetic under test: DownloadLog must never touch
# the user's real ~/.config/club-3090, and download_preflight must never depend
# on the HOST's `hf` CLI / token state (a dev box with hf installed would pass
# where CI fails, and vice versa).  Explicit preflight/log tests override these
# per-test via monkeypatch.
os.environ.setdefault("C3_SKIP_DOWNLOAD_PREFLIGHT", "1")
os.environ.setdefault("C3_CONFIG_DIR", tempfile.mkdtemp(prefix="c3-tests-config-"))

from club3090_tui_core.runner import SubprocessRunner
from club3090_cockpit.services import RealRunner


@pytest.fixture(autouse=True)
def _no_live_subprocess(monkeypatch):
    """Hard-block every real subprocess spawn point during tests."""

    async def _blocked_exec(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError(
            "LIVE SUBPROCESS BLOCKED: a test tried to spawn a real process "
            f"({args[:2]}).  Inject a FakeRunner / FakeWriteRunner instead."
        )

    async def _blocked_real_run(self, cmd, *, cwd, timeout=30.0):  # pragma: no cover
        raise AssertionError(
            f"LIVE READ BLOCKED: RealRunner.run was invoked ({cmd[:2]}). "
            "Tests must inject a FakeRunner."
        )

    async def _blocked_start_raw(self, cmd, env, run_type, parser):  # pragma: no cover
        raise AssertionError(
            f"LIVE WRITE BLOCKED: SubprocessRunner.start_raw was invoked ({cmd[:2]}). "
            "Tests must inject a FakeWriteRunner — writes are never executed live."
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _blocked_exec)
    monkeypatch.setattr(RealRunner, "run", _blocked_real_run)
    monkeypatch.setattr(SubprocessRunner, "start_raw", _blocked_start_raw)
    yield


def assert_serve_cmd(cmd: list[str], slug: str, *, force: bool = False) -> None:
    """Assert a serve ActionPlan's cmd matches the two-step serve contract.

    A serve is `bash -c <script> cockpit-serve <slug>`: switch.sh boots the
    model, then preserve-state.sh hands the resolved cross-turn <think> mode to
    the LiteLLM re-inline hook (it reads services/litellm/preserve_state.json,
    which only serve.sh used to write — so cockpit-launched models silently
    inherited the last serve.sh mode).

    Asserted structurally, not as an exact list, so the script's wording can
    change without churning every call site.  The load-bearing invariants:

      * the slug is the LAST element — app.py recovers it via ``plan.cmd[-1]``
        for the pending-serve watch and the problem reporter;
      * the slug is a POSITIONAL arg, never interpolated into the script text
        (no shell injection via a slug);
      * ``--force`` lives inside the script text, so it can't displace the slug
        from the end of cmd.
    """
    assert cmd[:2] == ["bash", "-c"], f"serve should run under `bash -c`, got {cmd[:2]}"
    assert cmd[-1] == slug, f"slug must be LAST (app.py reads cmd[-1]), got {cmd[-1]!r}"
    script = cmd[2]
    assert slug not in script, "slug must be positional, not interpolated into the script"
    assert "scripts/switch.sh" in script, "serve must still boot via switch.sh"
    assert "scripts/preserve-state.sh" in script, "serve must sync the LiteLLM preserve state"
    assert ("--force" in script) is force, f"--force presence should be {force}"
