"""Per-pixel luminance is computed only when something actually wires it.

DESIGN.md: "in `flat_outputs` the executor computes `luminance` as a per-pixel
H×W grayscale array (so it can drive FIELD consumers); when wired to a scalar
param it collapses to `float(mean)`."

The array form was being built eagerly for EVERY node on EVERY frame — a full
image reduction costing 5.15 ms per node per frame at 768×512, versus 0.41 ms
for the scalar. In the great majority of graphs nothing consumes the array:
every implicit-inheritance path collapses it straight back to `float(mean)`.
That single line was the bulk of the executor's ~2.6 ms/node overhead, which is
what caps graph complexity in live mode (the executor alone would eat a 30 fps
frame budget at ~13 nodes).

The executor now builds the array only for nodes with an outgoing edge whose
``src_port`` is ``luminance`` — the only consumer that can need the H,W form,
since it may drive a FIELD input. Everyone else gets the scalar.

These tests lock both halves: the array must still be there when wired (or
FIELD-driving breaks), and must NOT be there when it isn't (or the cost comes
straight back).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.registry import get_meta
from image_pipeline.core.utils import set_canvas

SRC = "310"


def _defaults(mid):
    return {k: (v.get("default") if isinstance(v, dict) else v)
            for k, v in (get_meta(mid).params or {}).items()}


def _run(edges, extra_nodes=()):
    set_canvas(128, 96)
    nodes = [{"id": "s", "method_id": SRC, "params": _defaults(SRC)}]
    nodes.extend(extra_nodes)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="lum_")),
                       in_memory=True, audit_to_disk=False)
    flat, _term, errs = ex.execute(nodes=nodes, edges=list(edges), seed=3,
                                   frame=1, frames=8)
    assert not errs, f"execute raised: {errs}"
    return flat


def test_unwired_luminance_is_a_scalar():
    """No consumer -> no full-image reduction."""
    flat = _run([])
    lum = flat["s"]["luminance"]
    assert not isinstance(lum, np.ndarray), (
        "luminance was materialised as an array with nothing wired to it — "
        "the per-node full-image reduction is back on the hot path"
    )
    assert isinstance(lum, float) and 0.0 <= lum <= 1.0


def test_explicitly_wired_luminance_is_per_pixel():
    """An explicit `luminance` edge must still get the H,W array.

    This is the form that can drive a FIELD input; collapsing it to a scalar
    would silently degrade those graphs to a flat constant.
    """
    tgt = {"id": "t", "method_id": "__transform__",
           "params": _defaults("__transform__")}
    flat = _run(
        [{"src_node": "s", "src_port": "luminance",
          "dst_node": "t", "dst_port": "rotate"}],
        extra_nodes=[tgt],
    )
    lum = flat["s"]["luminance"]
    assert isinstance(lum, np.ndarray), (
        "explicitly wired luminance came back as a scalar — FIELD consumers "
        "driven from luminance would collapse to a constant"
    )
    assert lum.ndim == 2, f"expected H,W grayscale, got shape {lum.shape}"


def test_scalar_matches_the_array_mean():
    """The cheap path must agree with the expensive one."""
    tgt = {"id": "t", "method_id": "__transform__",
           "params": _defaults("__transform__")}
    wired = _run(
        [{"src_node": "s", "src_port": "luminance",
          "dst_node": "t", "dst_port": "rotate"}],
        extra_nodes=[tgt],
    )["s"]["luminance"]
    unwired = _run([])["s"]["luminance"]
    assert np.isclose(float(np.mean(wired)), float(unwired), atol=1e-5), (
        f"scalar luminance {unwired} disagrees with array mean "
        f"{float(np.mean(wired))} — the two paths have diverged"
    )


def test_node_dirs_are_created_once_not_per_frame():
    """mkdir is memoised; the directory must still exist and be reused."""
    set_canvas(96, 64)
    ex = GraphExecutor(Path(tempfile.mkdtemp(prefix="lum_dirs_")),
                       in_memory=True, audit_to_disk=False)
    nodes = [{"id": "s", "method_id": SRC, "params": _defaults(SRC)}]
    for frame in range(3):
        _flat, _t, errs = ex.execute(nodes=nodes, edges=[], seed=3,
                                     frame=frame, frames=8)
        assert not errs, f"frame {frame} raised: {errs}"
    node_dir = ex.out_dir / "s"
    assert node_dir.exists(), "node dir was not created at all"
    assert node_dir in ex._ensured_dirs, "mkdir memo did not record the dir"
    assert len(ex._ensured_dirs) == 1, (
        f"expected exactly one memoised dir, got {ex._ensured_dirs}"
    )
