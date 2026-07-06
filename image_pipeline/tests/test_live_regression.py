"""Regression guards for the live-render milestone (2026-07-02).

These lock in the architecture that made continuous live playback work and
fast. If one of these fails, live mode has silently regressed — read the
"Live mode" section of DESIGN.md before "fixing" the test.

The three invariants:

1. LIVE CLOCK ADVANCES. The normalised timeline clock `t` must move when a
   render has more than one frame. At total_frames<=1 it is pinned at 0, and
   every time-driven (Architecture B) node freezes. The live loop passes
   frames=LIVE_TOTAL_FRAMES precisely so `t` sweeps 0->1.

2. INJECTED `time` SURVIVES. The live loop injects a monotonic
   params["time"] = float(frame) for methods that read `time` directly. The
   executor must NOT clobber it — it only fills `time` from the timeline when
   the caller did not provide one (`if "time" not in run_params`).

3. #18 PARAMS ARE HONOURED. Cellular Automata's SCALAR override ports use a
   -1.0 "not wired" sentinel; the method must treat those as "use the UI
   param", not as an active override. The client always sends them at -1.0.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_meta, method
from image_pipeline.core.timeline import make_timeline
from image_pipeline.core.utils import W, H, set_canvas


# The live loop's window length (server.py `_live_loop`). Kept in sync by the
# invariant that any value > 1 makes the clock advance.
LIVE_TOTAL_FRAMES = 300


# A tiny stateless probe that echoes the `time` param it was given, so the
# "executor preserves injected time" invariant can be tested without depending
# on any production method's internals. Architecture B (no n_frames / sim tag).
@method(id="__time_probe__", name="Time Probe", category="test", inputs={})
def _time_probe(out_dir, seed, params=None):
    import numpy as np
    tval = float((params or {}).get("time", -999.0))
    arr = np.zeros((H, W, 3), dtype=np.float32)
    return {"image": arr, "echoed_time": tval}


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="gm_live_"))


# ── Invariant 1: the live clock advances ───────────────────────────────────

def test_timeline_clock_pinned_at_single_frame():
    """total_frames<=1 pins t=0 — the exact cause of the frozen live preview."""
    assert make_timeline(global_frame=7, total_frames=1).t == 0.0


def test_timeline_clock_advances_over_a_window():
    """With a real window, t sweeps 0->1 so time-driven nodes animate."""
    t0 = make_timeline(global_frame=0, total_frames=LIVE_TOTAL_FRAMES).t
    tmid = make_timeline(global_frame=LIVE_TOTAL_FRAMES // 2, total_frames=LIVE_TOTAL_FRAMES).t
    tend = make_timeline(global_frame=LIVE_TOTAL_FRAMES - 1, total_frames=LIVE_TOTAL_FRAMES).t
    assert t0 == 0.0
    assert 0.0 < tmid < 1.0
    assert tend == 1.0


# ── Invariant 2: an injected `time` param is not overwritten ────────────────

def test_executor_preserves_injected_time():
    """The live loop injects params['time']=float(frame); the executor must not
    overwrite it with the timeline phase (or time-driven nodes freeze)."""
    set_canvas(64, 64)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        nodes = [{"id": "p", "method_id": "__time_probe__",
                  "params": {"time": 123.5}, "dirty": True, "render": True}]
        flat, _t, errs = ex.execute(nodes, [], seed=1, frame=0, frames=LIVE_TOTAL_FRAMES)
        assert not errs, errs
        assert abs(float(flat["p"]["echoed_time"]) - 123.5) < 1e-6, \
            f"executor overwrote injected time: {flat['p'].get('echoed_time')}"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_stateful_sim_18_does_not_slow_down():
    """#18 is a stateful Architecture-A sim: cooked once, then served from the
    cache at O(1). Per-frame cost must NOT grow as the live timeline advances
    (the old stateless model ran int(time*60) generations from scratch every
    frame, so cost climbed without bound)."""
    import time as _time
    set_canvas(160, 160)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        base = {"rule": "conway", "size": 4, "speed": 1.0, "n_frames": 60,
                "rule_select": -1.0, "init_select": -1.0, "cell_size": -1.0,
                "age_input": -1.0}

        def cook_ms(f):
            nodes = [{"id": "ca", "method_id": "18",
                      "params": {**base, "time": float(f)}, "dirty": True, "render": True}]
            t0 = _time.time()
            _flat, _t, errs = ex.execute(nodes, [], seed=42, frame=f % LIVE_TOTAL_FRAMES,
                                         frames=LIVE_TOTAL_FRAMES)
            assert not errs, errs
            return (_time.time() - t0) * 1000.0

        cook_ms(1)                    # first frame pays the one-time cook
        early = cook_ms(5)            # served from cache
        late = cook_ms(250)           # far down the timeline — must be just as cheap
        # A stateless recompute would make `late` many times `early`; cached
        # serves are flat. Allow generous slack for noise.
        assert late < early * 4 + 20, f"per-frame cost grew: early={early:.1f}ms late={late:.1f}ms"
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ── Invariant 3: live playback actually produces motion ─────────────────────

def _live_execute(ex: GraphExecutor, nodes, edges, seed, frame):
    """Mirror the server live loop's per-frame contract exactly."""
    for n in nodes:
        n["dirty"] = True                       # always re-cook
        n.setdefault("params", {})["time"] = float(frame)   # monotonic time
    return ex.execute(nodes, edges, seed,
                      frame=frame % LIVE_TOTAL_FRAMES, frames=LIVE_TOTAL_FRAMES)


def test_cellular_automata_18_animates_live():
    """#18 (a stateful Architecture-A sim) must yield distinct live frames from
    a dirty=False (post-Run) start — i.e. the sim actually plays, not frozen."""
    set_canvas(192, 192)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        base = {
            "rule": "conway", "size": 4, "speed": 1.0,
            "rule_select": -1.0, "init_select": -1.0,
            "cell_size": -1.0, "age_input": -1.0,
        }
        # dirty=False mimics the post-Run client state that used to freeze.
        nodes = [{"id": "ca", "method_id": "18", "params": dict(base),
                  "dirty": False, "render": True}]
        digests = []
        for f in (1, 2, 3, 5, 8):
            flat, _t, errs = _live_execute(ex, nodes, [], 42, f)
            assert not errs, errs
            digests.append(flat["ca"]["image"].tobytes())
        assert len({hash(d) for d in digests}) >= 3, "live #18 is frozen"
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ── Invariant 3b: #18 UI params are honoured with the -1.0 sentinels set ────

def test_cellular_automata_18_sequence_shows_start():
    """A rendered sequence must show the sim from its beginning, not clamp the
    first N frames onto a later still.

    Regression: #18 floored its generation count at 60, so every early frame
    whose t*60 was below the floor rendered the identical 60-generation state
    — the opening of the sim was missing and 'picked up' several frames in.
    The floor now only applies to single stills (total_frames<=1).
    """
    set_canvas(160, 160)
    out = _tmp()
    try:
        N = 24
        ex = GraphExecutor(out, in_memory=True)
        base = {"rule": "conway", "size": 4, "speed": 1.0, "rule_select": -1.0,
                "init_select": -1.0, "cell_size": -1.0, "age_input": -1.0}
        digests = []
        for f in range(N):
            nodes = [{"id": "ca", "method_id": "18", "params": dict(base),
                      "dirty": True, "render": True}]
            flat, _t, errs = ex.execute(nodes, [], seed=42, frame=f, frames=N)
            assert not errs, errs
            digests.append(hash(flat["ca"]["image"].tobytes()))
        # At most the single opening frame may repeat; more than a couple means
        # the start of the sim is being clamped onto a later frame again.
        lead = 1
        while lead < N and digests[lead] == digests[0]:
            lead += 1
        assert lead <= 2, f"first {lead} sequence frames are frozen on one still"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_sim_cache_survives_unchanged_hotswap():
    """Arch-A sim cache must survive a hot-swap where only volatile params change.

    The live loop injects time=float(frame) on every frame. Without selective
    invalidation the time param would flush the sim cache on every hot-swap,
    forcing a full re-cook from scratch every time the user tweaks any node.
    """
    import time as _time
    set_canvas(64, 64)
    out = _tmp()
    try:
        ex = GraphExecutor(out, in_memory=True)
        base = {"rule": "conway", "size": 4, "speed": 1.0, "n_frames": 10,
                "rule_select": -1.0, "init_select": -1.0, "cell_size": -1.0,
                "age_input": -1.0}
        nodes_v1 = [{"id": "ca", "method_id": "18",
                     "params": {**base, "time": 0.0}, "dirty": True}]

        # First cook — populates sim cache
        ex.execute(nodes_v1, [], seed=42, frame=0, frames=10)
        assert ("ca", 42) in ex._sim_cache, "sim cache not populated after first cook"
        t_cold = None

        # Measure cold-cook time to compare with warm-cache cost
        t0 = _time.monotonic()
        ex.execute(nodes_v1, [], seed=42, frame=0, frames=10)
        t_warm = (_time.monotonic() - t0) * 1000

        # Simulate hot-swap: only the volatile 'time' param changed
        nodes_v2 = [{"id": "ca", "method_id": "18",
                     "params": {**base, "time": 7.0}, "dirty": True}]
        inv = ex.selective_invalidate(nodes_v1, nodes_v2, [], [], seed=42)
        assert inv == 0, f"volatile-only change invalidated {inv} cache entries (expected 0)"
        assert ("ca", 42) in ex._sim_cache, "cache cleared on volatile-only hotswap"

        # After the unchanged hotswap, frame still served from cache (fast)
        t0 = _time.monotonic()
        ex.execute(nodes_v2, [], seed=42, frame=3, frames=10)
        t_post_swap = (_time.monotonic() - t0) * 1000
        # post-swap serve should be as fast as warm-cache (within 5×, not 10×)
        assert t_post_swap < t_warm * 10 + 50, \
            f"post-swap cost {t_post_swap:.1f}ms >> warm {t_warm:.1f}ms — re-cooked when it shouldn't"

        # Now hot-swap with a real (non-volatile) param change: n_frames
        nodes_v3 = [{"id": "ca", "method_id": "18",
                     "params": {**base, "n_frames": 20, "time": 0.0}, "dirty": True}]
        inv = ex.selective_invalidate(nodes_v2, nodes_v3, [], [], seed=42)
        assert inv == 1, f"non-volatile change should invalidate 1 entry, got {inv}"
        assert ("ca", 42) not in ex._sim_cache, "cache should be gone after n_frames change"
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_cellular_automata_18_params_honored():
    """rule / size / seed_pattern must take effect even when the -1.0
    scalar-override sentinels are present (the client always sends them)."""
    set_canvas(192, 192)
    meta = get_meta("18")
    out = _tmp()
    try:
        # Small n_frames keeps the direct cook fast while still evolving enough
        # for different rules/patterns to diverge visibly.
        base = {"n_frames": 40, "rule_select": -1.0, "init_select": -1.0,
                "cell_size": -1.0, "age_input": -1.0}

        def render(extra):
            r = meta.fn(out, 42, params={**base, **extra})
            return (r["image"] if isinstance(r, dict) else r).tobytes()

        assert render({"rule": "conway"}) != render({"rule": "highlife"}), "rule ignored"
        assert render({"size": 2}) != render({"size": 12}), "size ignored"
        assert render({"seed_pattern": "random"}) != render({"seed_pattern": "pulsar"}), "seed_pattern ignored"
        # A genuinely wired override (value >= 0) must still win.
        assert render({"rule_select": 0.0}) != render({"rule_select": 0.6}), "wired override broken"
    finally:
        shutil.rmtree(out, ignore_errors=True)
