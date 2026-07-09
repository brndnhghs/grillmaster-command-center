"""Regression guards for the in-memory transport milestone (2026-07-08).

These lock in the fixes that made the live hot path genuinely disk-free:

1. SAVE CAPTURE WORKS FOR EVERY IMPORT STYLE. Methods bind `save` at import
   time (`from ...core.utils import save`), so the old module-attribute
   monkeypatch never intercepted them — every live frame paid a full PNG
   encode + write + decode read-back per node. The capture sink now lives
   inside utils.save() itself (per-thread), so it fires no matter how the
   method obtained the function.

2. LIVE MODE WRITES NOTHING. audit_to_disk=False must produce zero image /
   sidecar files in the executor's out_dir.

3. FIELD MERGE WIRES DON'T CRASH. `slot.get("field") or src_img` raised
   "truth value of an array is ambiguous" on every field_a/field_b wire and
   killed the whole frame (plan BUG-1).

4. MERGE PORTS ARE IN-MEMORY. Compositing nodes receive image_a/image_b/
   field_a/field_b as ndarrays; the temp-file path is only an audit-mode
   fallback.
"""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — trigger @method registration
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import method, resolve_keys
# Deliberately the same import style the ~150 production method files use:
# a direct name binding taken at import time.
from image_pipeline.core.utils import save, W, H, set_canvas


# A probe that emits its image ONLY via the directly-imported save() —
# the exact pattern the old monkeypatch capture missed.
@method(id="__save_probe__", name="Save Probe", category="test", inputs={})
def _save_probe(out_dir, seed, params=None):
    val = float((params or {}).get("level", 0.25))
    arr = np.full((H, W, 3), val, dtype=np.float32)
    save(arr, "save-probe.png", out_dir)
    # No return value: the executor must read the image from the capture sink.


# A probe that outputs a real (H, W) field sidecar for merge-wire tests.
@method(id="__field_probe__", name="Field Probe", category="test", inputs={},
        outputs={"field": "FIELD"})
def _field_probe(out_dir, seed, params=None):
    field = np.linspace(0.0, 1.0, int(H) * int(W), dtype=np.float32).reshape(int(H), int(W))
    img = np.stack([field] * 3, axis=-1)
    return {"image": img, "field": field}


@pytest.fixture()
def out_dir():
    d = Path(tempfile.mkdtemp(prefix="gm_transport_"))
    set_canvas(256, 192)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _image_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix in (".png", ".npy", ".json")]


def test_save_capture_intercepts_direct_import(out_dir):
    """save() output must land on the in-memory bus even when the method
    holds a direct reference from `from utils import save`."""
    ex = GraphExecutor(out_dir, in_memory=True, audit_to_disk=False)
    nodes = [{"id": "p", "method_id": "__save_probe__",
              "params": {"level": 0.5}, "dirty": True, "render": True}]
    flat, terminal, errs = ex.execute(nodes, [], 42, frame=0, frames=1)
    assert not errs
    img = flat["p"]["image"]
    assert img is not None, "capture sink did not deliver the saved image"
    assert img.dtype == np.float32
    assert abs(float(img.mean()) - 0.5) < 1e-3


def test_live_mode_writes_no_files(out_dir):
    """audit_to_disk=False = zero image/sidecar disk writes on the hot path."""
    ex = GraphExecutor(out_dir, in_memory=True, audit_to_disk=False)
    nodes = [{"id": "p", "method_id": "__save_probe__",
              "params": {"level": 0.3}, "dirty": True, "render": True}]
    for f in range(3):
        _, _, errs = ex.execute(nodes, [], 42, frame=f, frames=3)
        assert not errs
    leftovers = _image_files(out_dir)
    assert leftovers == [], f"live mode wrote files: {leftovers}"


def test_render_mode_keeps_audit_trail(out_dir):
    """Default audit_to_disk=True must still write the per-node PNG that the
    wire inspector and cross-job dirty-skip cache read."""
    ex = GraphExecutor(out_dir, in_memory=True)  # audit on by default
    nodes = [{"id": "p", "method_id": "__save_probe__",
              "params": {"level": 0.3}, "dirty": True, "render": True}]
    _, _, errs = ex.execute(nodes, [], 42, frame=0, frames=1)
    assert not errs
    pngs = list((out_dir / "p").glob("*.png"))
    assert pngs, "render mode lost its on-disk audit trail"


def test_field_merge_wire_executes(out_dir):
    """BUG-1: a FIELD output wired into field_a/field_b crashed the frame
    with 'truth value of an array is ambiguous'."""
    ex = GraphExecutor(out_dir, in_memory=True, audit_to_disk=False)
    nodes = [
        {"id": "fa", "method_id": "__field_probe__", "params": {}, "dirty": True},
        {"id": "fb", "method_id": "__field_probe__", "params": {}, "dirty": True},
        {"id": "combine", "method_id": "139", "params": {"operation": "add"},
         "dirty": True, "render": True},
    ]
    edges = [
        {"src_node": "fa", "src_port": "field", "dst_node": "combine", "dst_port": "field_a"},
        {"src_node": "fb", "src_port": "field", "dst_node": "combine", "dst_port": "field_b"},
    ]
    flat, terminal, errs = ex.execute(nodes, edges, 42, frame=0, frames=1)
    assert not errs, f"field merge wire failed: {errs}"
    combined = flat["combine"].get("field")
    assert isinstance(combined, np.ndarray)
    # add of two identical ramps → doubled ramp
    assert abs(float(combined.max()) - 2.0) < 1e-3


def test_image_blend_receives_in_memory_arrays(out_dir):
    """Blend must work with zero disk files in live mode (in-memory image_a/b)."""
    ex = GraphExecutor(out_dir, in_memory=True, audit_to_disk=False)
    nodes = [
        {"id": "a", "method_id": "__save_probe__", "params": {"level": 1.0}, "dirty": True},
        {"id": "b", "method_id": "__save_probe__", "params": {"level": 0.0}, "dirty": True},
        {"id": "blend", "method_id": "137",
         "params": {"mode": "normal", "opacity": 0.5}, "dirty": True, "render": True},
    ]
    edges = [
        {"src_node": "a", "src_port": "image", "dst_node": "blend", "dst_port": "image_a"},
        {"src_node": "b", "src_port": "image", "dst_node": "blend", "dst_port": "image_b"},
    ]
    flat, terminal, errs = ex.execute(nodes, edges, 42, frame=0, frames=1)
    assert not errs, f"blend failed: {errs}"
    img = flat["blend"]["image"]
    assert img is not None
    # 50% blend of white over white base at opacity .5 → mean 1.0*(1-.5)+1.0*.5… use a/b means
    assert 0.4 < float(img.mean()) < 0.6, f"unexpected blend mean {img.mean()}"
    assert _image_files(out_dir) == [], "blend used disk transport in live mode"


def test_expr_cache_returns_same_values():
    """Compiled-expression caching must not change results across calls."""
    from image_pipeline.core.expr import eval_param, _COMPILED_CACHE
    _COMPILED_CACHE.clear()
    v1 = eval_param("sin(frame * 0.1) + t", frame=10, seed=1, total_frames=100)
    assert "sin(frame * 0.1) + t" in _COMPILED_CACHE
    v2 = eval_param("sin(frame * 0.1) + t", frame=10, seed=1, total_frames=100)
    assert v1 == v2
    # Rejected expressions are cached as None and stay rejected
    assert eval_param("__import__('os')", frame=0, seed=0) == 0.0
    assert eval_param("__import__('os')", frame=0, seed=0) == 0.0


def test_resolve_keys_handles_dunder_ids():
    """BUG-7: registry ids like __counter__ crashed the numeric sort."""
    keys = resolve_keys("all")
    assert keys, "resolve_keys('all') returned nothing"
    numeric = [k for k in keys if k.isdigit()]
    assert numeric == sorted(numeric, key=int), "numeric ids not sorted numerically"


def test_group_subexecutor_is_reused(out_dir):
    """BUG-6: group nodes must keep one sub-executor across frames so
    feedback/sim state inside the group survives."""
    ex = GraphExecutor(out_dir, in_memory=True, audit_to_disk=False)
    group = {
        "id": "grp", "type": "group", "params": {}, "dirty": True, "render": True,
        "subgraph": {
            "nodes": [{"id": "inner", "method_id": "__save_probe__",
                       "params": {"level": 0.7}, "dirty": True, "render": True}],
            "edges": [],
        },
        "exposed_inputs": [], "exposed_outputs": [],
    }
    for f in range(2):
        _, _, errs = ex.execute([dict(group)], [], 42, frame=f, frames=2)
        assert not errs
    assert "grp" in ex._group_executors
    sub = ex._group_executors["grp"]
    assert sub._in_memory and not sub.audit_to_disk, \
        "group sub-executor did not inherit live transport flags"
