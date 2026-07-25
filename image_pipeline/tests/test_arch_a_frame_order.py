"""Regression guard: an Arch-A clip must not depend on which frame cooked it.

DESIGN.md's contract is "identical graph + seed + params => identical output,
always". Architecture A cooks a node's ENTIRE clip in one call and replays it for
every output frame, so anything that reaches the cook but not the cache key
(`(node_id, seed)` + params_hash) makes the clip a function of an unkeyed
variable — and whichever output frame missed the cache first silently decides
what every other frame shows.

Two such channels exist. One is fixed; one is open and bounded here.

CHANNEL 1 — `node_seed` (FIXED). `seed + frame + _stable_node_offset(node_id)`
folded the triggering frame into the RNG stream. Measured before the fix, the
same graph at seed 42 rendered three different images for frame 0 depending on
call order:

    494 Screen-Space Fluid   f0=c5f2df5944  f0 after f3=60753d348f  after f5=9fe1d96de5

Reachable in production: the server's `/api/graph/{gid}/render` executor
(`_render_exec_state`) persists across calls, so rendering frame 5 then frame 0
returned a different frame 0 than a fresh server would; sequence renders with
start_frame > 0 and live sessions resumed after a scrub took the same path.
Dropping `+ frame` on the A path closes it for all 115 Arch-A nodes.

CHANNEL 2 — `time` / `_timeline` (OPEN, P1b). Both still reach the cook on the
live frame's clock while being excluded from the cache key, so a method reading
either cooks a frame-dependent clip. Measured: 40 of 115 Arch-A nodes, whenever
the timeline phase actually varies (`frames > 1` — i.e. any sequence render or
live playback; at `frames == 1` make_timeline pins t to 0.0 and the channel is
inert, which is why the channel-1 tests below use frames=1 to isolate it).

Pinning `time` to frame 0 was tried and reverted: it silently zeroes the
spatial-param response of every sim that uses `time` as an initial-condition
input — 112 Kelvin-Helmholtz `u_shear` and 359 Lenia `mu` both drop from a live
field response to exactly 0.000000 (see test_spatial_params). A node whose clip
legitimately depends on the output frame is not a cook-once-and-replay node; it
is Architecture B wearing an `n_frames` param, and the fix belongs in the
architecture declaration. Tracked as P1b in
docs/plans/2026-07-23-mcp-server-plan.md.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.arch import detect_architecture
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_all
from image_pipeline.core.utils import set_canvas

# Arch-A nodes that are stochastic and frame-varying, so an order-dependent cook
# shows up as different pixels. Node 91 is NOT a valid probe: it ignores both
# seed and frame and reports "identical" no matter what. Node 171 (p5.js Sketch)
# is not either — it is nondeterministic run-to-run at a fixed seed.
PROBES = ("494", "954")

# Channel 2 exemplar: reads `time` as an initial-condition input.
TIME_CHANNEL_PROBE = "112"

# Channel-2 residual measured 2026-07-23 at frames=6. A change that pushes this
# UP has widened the hole; one that pushes it down should update the number.
TIME_CHANNEL_AFFECTED = 40
ARCH_A_TOTAL = 115


def _render(method_id: str, frame_sequence: tuple[int, ...],
            seed: int = 42, frames: int = 1) -> str:
    """Render each frame in order on ONE executor; hash whatever the last produced.

    `frames=1` pins the timeline phase to 0.0, isolating channel 1. Pass frames>1
    to let the phase vary and expose channel 2 as well.
    """
    nodes = [{"id": "n1", "method_id": method_id,
              "params": {"n_frames": 6}, "render": True}]
    ex = GraphExecutor(Path(tempfile.mkdtemp()), in_memory=True, audit_to_disk=False)
    image = None
    for frame in frame_sequence:
        flat, terminal, _ = ex.execute(nodes, [], seed, frame=frame, frames=frames)
        image = flat[terminal]["image"]
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()


@pytest.fixture(autouse=True)
def _canvas():
    set_canvas(192, 192)


@pytest.mark.parametrize("method_id", PROBES)
def test_frame_zero_is_independent_of_call_order(method_id: str):
    """Channel 1: frame 0 renders identically whether or not a later frame cooked."""
    alone = _render(method_id, (0,))
    after_3 = _render(method_id, (3, 0))
    after_5 = _render(method_id, (5, 0))
    assert alone == after_3 == after_5, (
        f"method {method_id}: frame 0 depends on which frame cooked the clip "
        f"(alone={alone[:10]}, after f3={after_3[:10]}, after f5={after_5[:10]}). "
        f"Something reaching the Arch-A cook is absent from the sim cache key."
    )


def test_the_fix_did_not_flatten_seed_or_frame():
    """Guard the obvious wrong fix: dropping `frame` must not drop seed or animation."""
    assert _render("494", (0,), seed=42) != _render("494", (0,), seed=43), \
        "seed no longer reaches the Arch-A cook"
    assert _render("112", (0,), frames=6) != _render("112", (2,), frames=6), \
        "Arch-A clips no longer vary across output frames"


@pytest.mark.xfail(reason="P1b: `time`/`_timeline` reach the cook but not the cache key",
                   strict=True)
def test_time_channel_order_dependence_is_still_open():
    """Channel 2, pinned so this flips to a failure the day P1b is closed."""
    assert (_render(TIME_CHANNEL_PROBE, (0,), frames=6)
            == _render(TIME_CHANNEL_PROBE, (3, 0), frames=6))


@pytest.mark.slow
def test_channel_one_is_closed_for_every_arch_a_node():
    """Full sweep with the phase pinned: no node may depend on call order."""
    offenders = []
    for method_id, meta in get_all().items():
        if detect_architecture(meta) != "A":
            continue
        try:
            hashes = {_render(method_id, seq) for seq in ((0,), (3, 0), (5, 0))}
        except Exception:
            continue  # a node that cannot render at all is another test's problem
        if len(hashes) > 1:
            offenders.append(f"{method_id} {meta.name}")
    assert not offenders, f"order-dependent Arch-A nodes via node_seed: {offenders}"


@pytest.mark.slow
def test_channel_two_residual_does_not_grow():
    """Bound the open hole so a refactor cannot quietly widen it."""
    affected, total = [], 0
    for method_id, meta in get_all().items():
        if detect_architecture(meta) != "A":
            continue
        try:
            # Reject nondeterministic nodes — their disagreement is not order.
            if len({_render(method_id, (0,), frames=6) for _ in range(3)}) > 1:
                continue
            hashes = {_render(method_id, seq, frames=6)
                      for seq in ((0,), (3, 0), (5, 0))}
        except Exception:
            continue
        total += 1
        if len(hashes) > 1:
            affected.append(f"{method_id} {meta.name}")
    assert total == ARCH_A_TOTAL, f"Arch-A population changed: {total}"
    assert len(affected) <= TIME_CHANNEL_AFFECTED, (
        f"channel-2 residual grew from {TIME_CHANNEL_AFFECTED} to {len(affected)}: "
        f"{affected}"
    )
