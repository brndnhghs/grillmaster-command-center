"""The live profiler must model the live loop, not a pre-Phase-6 straw man.

``profile_live.py`` used to force ``dirty=True`` on every node every frame and
label the result "Approx live FPS". That is pre-Phase-6 invariant 1, which
DESIGN.md explicitly relaxed — so the headline number described an architecture
the executor had already abandoned. On the reference graph it reported 1.9 fps
where selective recook actually delivers 20.0 fps: a 10.6x understatement, and
it fingered Procedural Noise (#05) as consuming 89% of the frame when live mode
skips that node entirely.

That mattered because it was the project's main performance instrument: work
was being prioritised against a number nobody could act on.

These tests lock the two properties that keep it honest:
  1. the `live` marker reproduces the selective-dirty behaviour (a static
     source with unchanged params stops re-cooking), and
  2. skipping does not freeze the preview — the terminal must still animate.

If (2) ever fails, the skip is wrong and the fps number is meaningless.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas
from image_pipeline.tests.profile_live import (
    EDGES,
    LIVE_TOTAL_FRAMES,
    NODES,
    _mark_dirty_cold,
    _mark_dirty_live,
)

STATIC_SRC = "src"      # method 05, is_time_varying=False, no upstream


def _fresh_nodes():
    nodes = [dict(n) for n in NODES]
    for n in nodes:
        n["params"] = dict(n.get("params") or {})
    return nodes


def test_reference_source_is_actually_static():
    """Guard the fixture: the skip only exists because #05 is not time-varying."""
    meta = get_meta("05")
    assert meta is not None, "node 05 not registered"
    assert meta.is_time_varying is False, (
        "node 05 is now time-varying — it can no longer be skipped, so this "
        "suite no longer exercises the incremental-recook path"
    )


def test_live_marker_stops_recooking_a_stable_static_node():
    """After the first frame, a static source with unchanged params goes clean."""
    nodes = _fresh_nodes()
    last_params: dict[str, dict] = {}

    first = _mark_dirty_live(nodes, EDGES, 0, last_params)
    assert STATIC_SRC in first, "a never-seen node must cook on the first frame"

    for frame in range(1, 5):
        later = _mark_dirty_live(nodes, EDGES, frame, last_params)
        assert STATIC_SRC not in later, (
            f"static source re-cooked on frame {frame} — selective recook is "
            f"not taking effect and the profiler is measuring a cold frame"
        )


def test_live_marker_redirties_on_param_change():
    """A user param edit must bring the static node back into the dirty set."""
    nodes = _fresh_nodes()
    last_params: dict[str, dict] = {}
    _mark_dirty_live(nodes, EDGES, 0, last_params)
    assert STATIC_SRC not in _mark_dirty_live(nodes, EDGES, 1, last_params)

    for n in nodes:
        if n["id"] == STATIC_SRC:
            n["params"]["scale"] = 9.0
    assert STATIC_SRC in _mark_dirty_live(nodes, EDGES, 2, last_params), (
        "param edit did not re-dirty the node — live edits would not apply"
    )


def test_cold_marker_dirties_everything():
    """The cold mode must remain a true worst case, for comparison."""
    nodes = _fresh_nodes()
    for frame in range(3):
        assert _mark_dirty_cold(nodes, EDGES, frame, {}) == {n["id"] for n in nodes}


def test_live_mode_still_animates_while_skipping():
    """The load-bearing check: skipping must not freeze the preview.

    A frame-rate win from selective recook is worthless — actively misleading —
    if the terminal image stops changing. DESIGN.md's invariant 1 existed
    precisely because an over-eager skip froze live mode.
    """
    set_canvas(192, 128)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="prof_anim_")),
                       in_memory=True, audit_to_disk=False)
    nodes = _fresh_nodes()
    last_params: dict[str, dict] = {}

    means = []
    for frame in range(6):
        dirty = _mark_dirty_live(nodes, EDGES, frame, last_params)
        for n in nodes:
            n["dirty"] = n["id"] in dirty
        flat, _term, errs = ex.execute(nodes, EDGES, seed=42, frame=frame,
                                       frames=LIVE_TOTAL_FRAMES)
        assert not errs, f"frame {frame} raised: {errs}"
        means.append(float(np.asarray(flat["xform"]["image"],
                                      dtype=np.float32).mean()))

    assert len({round(m, 8) for m in means}) > 1, (
        "terminal output was identical across 6 frames under selective recook "
        "— the preview is frozen, so any fps figure from it is meaningless"
    )
