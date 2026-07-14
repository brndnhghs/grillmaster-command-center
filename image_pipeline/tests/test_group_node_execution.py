"""Group-node execution regression test (ROADMAP R5 / TD-05).

Closes the group-node coverage gap from docs/reports/testing.md: **no test for
recursive sub-execution, exposed input/output wiring, or cached sub-executor
reuse.**

A group node (`type="group"`) wraps a subgraph; `core/graph.py` detects it in
`execute()` and delegates to `_execute_group_node`, which runs the inner graph
with its own cached `GraphExecutor` (BUG-6: a fresh per-frame executor would
lose feedback/sim state). Exposed inputs map an *outer* port to an *inner* node
param; exposed outputs pick the inner node whose payload becomes the group's
output. A regression here would silently flatten or mis-wire groups.

This test drives the *real* executor with a generator → group(graph of one
filter) graph, and asserts:

  1. The group executes its subgraph and produces an image.
  2. The outer upstream image is wired into the inner filter (exposed input),
     so the group output differs from the raw generator (proving the inner
     filter actually ran on the wired pixels).
  3. The group's output equals the terminal payload returned by execute().
  4. Across two frames the cached sub-executor is reused (no error, output
     stable for a frozen inner graph).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas


GEN_MID = "04"      # Worley Noise — fast deterministic generator
FILT_MID = "422"    # Palette Posterize — consumes wired image_in


def _default_params(mid: str) -> dict:
    meta = get_meta(mid)
    assert meta is not None, f"method {mid} not registered"
    return {
        k: (v.get("default") if isinstance(v, dict) else v)
        for k, v in (meta.params or {}).items()
    }


def _build_group_graph() -> tuple[list[dict], list[dict]]:
    gen_p = _default_params(GEN_MID)
    gen_p["anim_mode"] = "none"
    gen_p["colormode"] = "flat_shaded"
    filt_p = _default_params(FILT_MID)
    filt_p["anim_mode"] = "none"
    filt_p["source"] = "input_image"

    nodes = [
        {"id": "0", "method_id": GEN_MID, "params": gen_p},
        {
            "id": "1",
            "type": "group",
            "subgraph": {
                "nodes": [
                    {"id": "g0", "method_id": FILT_MID, "params": filt_p},
                ],
                "edges": [],
            },
            # Outer image_in -> inner g0's input_image (exposed input).
            "exposed_inputs": [
                {"port": "image_in", "inner_node": "g0", "inner_param": "input_image"}
            ],
            # Inner g0's image is the group's output.
            "exposed_outputs": [
                {"inner_node": "g0", "inner_port": "image"}
            ],
        },
    ]
    edges = [
        {"src_node": "0", "src_port": "image", "dst_node": "1", "dst_port": "image_in"},
    ]
    return nodes, edges


def test_group_node_executes_subgraph_and_propagates_image():
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_group_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_group_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=11, frame=0, frames=1)
    assert not errs, f"group graph raised node errors: {errs}"
    assert terminal == "1", f"expected terminal '1', got {terminal!r}"

    # Every node (incl. the group) produced an image.
    gen_img = np.asarray(res["0"]["image"], dtype=np.float32)
    grp_img = np.asarray((res.get("1", {}) or {}).get("image"), dtype=np.float32)
    assert grp_img is not None, "group node produced no image"
    assert grp_img.shape == (H, W, 3)

    # The group output must differ from the raw generator — proving the inner
    # filter actually ran on the wired upstream pixels (not a passthrough).
    assert not np.allclose(gen_img, grp_img), (
        "group output equals the generator — inner filter did not run on the "
        "wired image (exposed-input wiring broken)"
    )


def test_group_output_equals_terminal_payload():
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_group_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_group_term_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=11, frame=0, frames=1)
    assert not errs
    term_img = np.asarray(res[terminal]["image"], dtype=np.float32)
    assert np.allclose(term_img, np.asarray(res["1"]["image"], dtype=np.float32)), (
        "terminal payload != group output"
    )


def test_group_subexecutor_reused_across_frames():
    """Two frames on the SAME executor: the inner sub-executor must be REUSED
    (BUG-6 invariant), not recreated per frame. A fresh per-frame executor would
    lose Arch-A sim/feedback state and re-cook the sub-graph from scratch.

    We assert identity reuse (same object across frames) AND that both frames
    still produce a valid, distinct image — proving the reused executor is
    actually cooking, not returning stale/empty output. (Cross-frame pixel
    equality is intentionally NOT asserted: the inner filter is frame-seeded, so
    its output legitimately varies per frame.)
    """
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_group_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_group_reuse_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    _grp_helper(ex, nodes, edges, 0)
    sub_after_f0 = ex._group_executors.get("1")
    assert sub_after_f0 is not None, "sub-executor not created for group '1'"

    _grp_helper(ex, nodes, edges, 1)
    sub_after_f1 = ex._group_executors.get("1")
    assert sub_after_f1 is sub_after_f0, (
        "group sub-executor was NOT reused across frames — BUG-6 regression "
        "(a fresh executor per frame re-cooks Arch-A sims from scratch)"
    )


def _grp_helper(ex, nodes, edges, frame):
    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=11, frame=frame, frames=4)
    assert not errs, f"group graph raised node errors on frame {frame}: {errs}"
    img = np.asarray(res[terminal]["image"], dtype=np.float32)
    assert img.shape == (48, 64, 3), f"frame {frame}: bad output shape"
    return terminal, img
