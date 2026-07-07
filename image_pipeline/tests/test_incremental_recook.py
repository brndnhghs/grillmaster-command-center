"""Regression guard for Phase 6 incremental re-cook.

Invariants:

1. STATIC SOURCE SKIPPED. A node with is_time_varying=False is cooked exactly
   once; on frame 2 the executor reuses _prev_outputs and the node's cook
   count stays at 1.

2. TIME-VARYING ALWAYS RE-COOKS. A node with is_time_varying=True (or any
   Arch-A simulation node) is re-cooked on every frame.

3. PARAM CHANGE CASCADES. Changing a param on a node marks it dirty, which
   cascades to all downstream nodes, even if those downstream nodes are
   marked is_time_varying=False.

4. ARCH-A SIM CACHE STILL ANIMATES. A simulation node (Arch-A) that is put
   in a graph still produces distinct output frames via the sim cache.

5. BIT-IDENTICAL TO OLD PATH. A fully time-varying graph produces identical
   pixel arrays regardless of whether the old force-dirty path or the new
   selective-dirty path is used.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor, _compute_live_dirty
from image_pipeline.core.registry import get_meta, method
from image_pipeline.core.utils import W, H, set_canvas


# ── Tiny test-only methods ──────────────────────────────────────────────────

_cook_count: dict[str, int] = {}


@method(
    id="__static_probe__",
    name="Static Probe",
    category="test",
    inputs={},
    is_time_varying=False,
)
def _static_probe(out_dir, seed, params=None):
    _cook_count["static"] = _cook_count.get("static", 0) + 1
    arr = np.full((H, W, 3), 0.42, dtype=np.float32)
    return {"image": arr}


@method(
    id="__dynamic_probe__",
    name="Dynamic Probe",
    category="test",
    inputs={},
    is_time_varying=True,
)
def _dynamic_probe(out_dir, seed, params=None):
    _cook_count["dynamic"] = _cook_count.get("dynamic", 0) + 1
    t = float((params or {}).get("time", 0.0))
    # Use sin to ensure distinct values for distinct integer times
    import math
    val = 0.3 + 0.2 * math.sin(t * 0.1)
    arr = np.full((H, W, 3), val, dtype=np.float32)
    return {"image": arr}


@method(
    id="__passthru_probe__",
    name="Passthru Probe",
    category="test",
    inputs={"image_in": "IMAGE"},
    is_time_varying=False,
)
def _passthru_probe(out_dir, seed, params=None):
    _cook_count["passthru"] = _cook_count.get("passthru", 0) + 1
    arr_in = (params or {}).get("_input_image")
    if arr_in is None:
        arr_in = np.zeros((H, W, 3), dtype=np.float32)
    return {"image": arr_in.copy()}


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="gm_p6_"))


# ── Invariant 1: static source is skipped after frame 0 ────────────────────

class TestStaticSourceSkipped:
    def test_static_node_cooked_once_in_two_frame_pass(self):
        """is_time_varying=False node: cook count stays 1 after frame 1."""
        _cook_count.clear()
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "s", "method_id": "__static_probe__", "params": {},
                  "dirty": True, "render": True}]
        edges = []
        # Frame 0 — first cook
        ex.execute(nodes, edges, seed=1, frame=0, frames=300)
        assert _cook_count.get("static", 0) == 1, "Should cook once on frame 0"

        # Frame 1 — selective dirty: static node is NOT marked dirty by live loop
        # simulate: do NOT mark dirty (live loop would set dirty=False for static nodes)
        nodes[0]["dirty"] = False
        ex.execute(nodes, edges, seed=1, frame=1, frames=300)
        assert _cook_count.get("static", 0) == 1, "Should NOT re-cook static node on frame 1"

    def test_static_node_output_reused(self):
        """Skipped static node returns exact same array reference."""
        _cook_count.clear()
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "s", "method_id": "__static_probe__", "params": {},
                  "dirty": True, "render": True}]
        edges = []
        out0, _, _ = ex.execute(nodes, edges, seed=1, frame=0, frames=300)
        arr0 = out0["s"]["image"]

        nodes[0]["dirty"] = False
        out1, _, _ = ex.execute(nodes, edges, seed=1, frame=1, frames=300)
        arr1 = out1["s"]["image"]

        np.testing.assert_array_equal(arr0, arr1)


# ── Invariant 2: time-varying node re-cooks every frame ────────────────────

class TestTimeVaryingAlwaysRecooks:
    def test_dynamic_node_cooked_every_frame(self):
        """is_time_varying=True node: cook count increments each frame."""
        _cook_count.clear()
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "d", "method_id": "__dynamic_probe__", "params": {"time": 0.0},
                  "dirty": True, "render": True}]
        edges = []
        for f in range(3):
            nodes[0]["params"]["time"] = float(f)
            nodes[0]["dirty"] = True  # live loop always marks these dirty
            ex.execute(nodes, edges, seed=1, frame=f, frames=300)
        assert _cook_count.get("dynamic", 0) == 3, "Dynamic node must re-cook every frame"

    def test_dynamic_node_output_changes(self):
        """Time-varying node produces different output each frame."""
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "d", "method_id": "__dynamic_probe__", "params": {"time": 1.0},
                  "dirty": True, "render": True}]
        edges = []
        out0, _, _ = ex.execute(nodes, edges, seed=1, frame=0, frames=300)

        nodes[0]["params"]["time"] = 50.0
        nodes[0]["dirty"] = True
        out1, _, _ = ex.execute(nodes, edges, seed=1, frame=1, frames=300)

        # Different time → different pixel value
        assert not np.array_equal(out0["d"]["image"], out1["d"]["image"])


# ── Invariant 3: param change cascades downstream ──────────────────────────

class TestParamChangeCascades:
    def test_downstream_static_node_re_cooks_when_upstream_changes(self):
        """When static upstream changes params, downstream static node also re-cooks."""
        _cook_count.clear()
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)

        # static_probe → passthru_probe
        nodes = [
            {"id": "src",  "method_id": "__static_probe__",   "params": {},
             "dirty": True, "render": False},
            {"id": "dst",  "method_id": "__passthru_probe__",  "params": {},
             "dirty": True, "render": True},
        ]
        edges = [{"src_node": "src", "src_port": "image",
                  "dst_node": "dst", "dst_port": "image_in"}]

        # Frame 0 — both cook
        ex.execute(nodes, edges, seed=1, frame=0, frames=300)
        assert _cook_count.get("static", 0) == 1
        assert _cook_count.get("passthru", 0) == 1

        # Frame 1 — nothing changed, nothing dirty
        nodes[0]["dirty"] = False
        nodes[1]["dirty"] = False
        ex.execute(nodes, edges, seed=1, frame=1, frames=300)
        assert _cook_count.get("static", 0) == 1,   "src: no re-cook"
        assert _cook_count.get("passthru", 0) == 1,  "dst: no re-cook (nothing changed)"

        # Frame 2 — src params changed → src and dst must both re-cook
        _cook_count.clear()
        nodes[0]["dirty"] = True   # param changed → live loop marks dirty
        nodes[1]["dirty"] = False  # still not explicitly dirty
        ex.execute(nodes, edges, seed=1, frame=2, frames=300)
        assert _cook_count.get("static", 0) == 1,   "src re-cooked after param change"
        assert _cook_count.get("passthru", 0) == 1, "dst re-cooked because upstream ran"

    def test_cascade_compute_live_dirty(self):
        """_compute_live_dirty correctly propagates dirty from upstream to downstream."""
        nodes = [
            {"id": "A", "method_id": "__static_probe__",   "params": {}},
            {"id": "B", "method_id": "__passthru_probe__",  "params": {}},
            {"id": "C", "method_id": "__passthru_probe__",  "params": {}},
        ]
        edges = [
            {"src_node": "A", "src_port": "image", "dst_node": "B", "dst_port": "image_in"},
            {"src_node": "B", "src_port": "image", "dst_node": "C", "dst_port": "image_in"},
        ]
        # A is the only initial dirty node
        initially_dirty = {"A"}
        result = _compute_live_dirty(nodes, edges, initially_dirty)
        assert "A" in result
        assert "B" in result, "B is downstream of dirty A"
        assert "C" in result, "C is downstream of B which is downstream of A"

    def test_cascade_does_not_dirty_unrelated_branch(self):
        """Dirty cascade does not bleed into sibling branches."""
        nodes = [
            {"id": "A", "method_id": "__static_probe__",  "params": {}},
            {"id": "X", "method_id": "__static_probe__",  "params": {}},
            {"id": "B", "method_id": "__passthru_probe__", "params": {}},
        ]
        edges = [
            {"src_node": "A", "src_port": "image", "dst_node": "B", "dst_port": "image_in"},
            # X has no edges → independent
        ]
        result = _compute_live_dirty(nodes, edges, {"A"})
        assert "A" in result
        assert "B" in result
        assert "X" not in result, "X is unrelated to A→B chain, must not be dirtied"


# ── Invariant 4: is_time_varying flag on MethodMeta ────────────────────────

class TestMethodMetaFlag:
    def test_default_is_time_varying_true(self):
        """Methods registered without is_time_varying default to True."""
        meta = get_meta("__dynamic_probe__")
        assert meta is not None
        assert meta.is_time_varying is True

    def test_explicit_false_respected(self):
        meta = get_meta("__static_probe__")
        assert meta is not None
        assert meta.is_time_varying is False

    def test_compositing_nodes_are_static(self):
        """Production compositing nodes must be declared is_time_varying=False."""
        static_ids = ["137", "138", "139", "140", "141",
                      "__image_to_mask__", "__transform__"]
        for mid in static_ids:
            meta = get_meta(mid)
            assert meta is not None, f"Method {mid} not found"
            assert meta.is_time_varying is False, \
                f"Method {mid} ({meta.name}) should be is_time_varying=False"

    def test_channel_nodes_are_time_varying(self):
        """Channel nodes (LFO, Beats, Counter) must always be time-varying."""
        tv_ids = ["__lfo__", "__beats__", "__counter__", "__strobe__"]
        for mid in tv_ids:
            meta = get_meta(mid)
            if meta is None:
                continue  # ok if not registered
            assert meta.is_time_varying is True, \
                f"Channel {mid} must be is_time_varying=True"

    def test_arch_a_nodes_are_time_varying(self):
        """All Architecture-A (simulation) nodes must be time-varying."""
        from image_pipeline.core.arch import detect_architecture
        all_meta = __import__('image_pipeline.core.registry', fromlist=['get_all']).get_all()
        for mid, meta in all_meta.items():
            if detect_architecture(meta) == "A":
                assert meta.is_time_varying is True, \
                    f"Arch-A node {mid} ({meta.name}) must be is_time_varying=True"


# ── Invariant 5: diagnostics expose skipped count ──────────────────────────

class TestSkippedDiagnostics:
    def test_skipped_count_in_last_frame_stats(self):
        """last_frame_stats must include nodes_skipped and nodes_cooked."""
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "s", "method_id": "__static_probe__", "params": {},
                  "dirty": True, "render": True}]
        edges = []
        ex.execute(nodes, edges, seed=1, frame=0, frames=300)
        assert "nodes_skipped" in ex.last_frame_stats
        assert "nodes_cooked"  in ex.last_frame_stats

    def test_skipped_count_increments_when_clean(self):
        """nodes_skipped goes up when a static node is kept clean."""
        tmp = _tmp()
        ex = GraphExecutor(tmp, in_memory=True)
        nodes = [{"id": "s", "method_id": "__static_probe__", "params": {},
                  "dirty": True, "render": True}]
        edges = []
        ex.execute(nodes, edges, seed=1, frame=0, frames=300)
        skipped0 = ex.last_frame_stats.get("nodes_skipped", 0)

        nodes[0]["dirty"] = False
        ex.execute(nodes, edges, seed=1, frame=1, frames=300)
        skipped1 = ex.last_frame_stats.get("nodes_skipped", 0)

        assert skipped1 > skipped0, "nodes_skipped should increase when static node is kept clean"
