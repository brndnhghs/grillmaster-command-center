"""Feedback-edge regression test (ROADMAP R2 / TD-02).

Closes the second-highest testing gap from docs/reports/testing.md: **no test
validates feedback edges** (cycles that carry the previous frame's output, with
a black-image fallback on frame 0).

A regression in the feedback branch of ``core/graph.py`` (the
``edge.feedback`` handling at lines ~1086–1095, and the exclusion of feedback
edges from the topological sort at lines ~1660–1663) would let cyclic graphs
either crash or silently drop the loop. This test:

  1. Renders a cyclic graph (gen → filter, filter → filter self-feedback)
     across several frames without error — proving feedback edges do not trip
     the cycle guard and that frame 0's black-image fallback works.
  2. Asserts the feedback loop has an *observable* effect: with the previous
     frame's pixels fed back in, frame 1 differs from the equivalent graph
     *without* the feedback edge. This catches a silent "feedback ignored"
     regression.

Deterministic, tiny canvas, fast filters → belongs in the default suite.
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


def _graph(with_feedback: bool) -> tuple[list[dict], list[dict]]:
    gen_p = _default_params(GEN_MID)
    gen_p["anim_mode"] = "none"          # freeze generator → frames are static
    gen_p["colormode"] = "flat_shaded"
    filt_p = _default_params(FILT_MID)
    filt_p["anim_mode"] = "none"
    filt_p["source"] = "input_image"     # force use of the wired upstream image

    nodes = [
        {"id": "0", "method_id": GEN_MID, "params": gen_p},
        {"id": "1", "method_id": FILT_MID, "params": filt_p},
    ]
    edges = [
        {"src_node": "0", "src_port": "image", "dst_node": "1", "dst_port": "image_in"},
    ]
    if with_feedback:
        # Self-feedback: node 1's previous-frame output feeds its own image_in.
        edges.append(
            {"src_node": "1", "src_port": "image", "dst_node": "1",
             "dst_port": "image_in", "feedback": True}
        )
    return nodes, edges


def _render(ex, nodes, edges, seed=11, frame=0, frames=4):
    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=seed, frame=frame, frames=frames)
    assert not errs, f"feedback graph raised node errors: {errs}"
    assert terminal == "1", f"expected terminal '1', got {terminal!r}"
    img = (res.get("1", {}) or {}).get("image")
    assert img is not None, "terminal produced no image"
    return np.asarray(img, dtype=np.float32)


def test_feedback_graph_renders_across_frames():
    """A cyclic graph with a feedback edge must render every frame, no crash."""
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _graph(with_feedback=True)
    out_dir = Path(tempfile.mkdtemp(prefix="graph_fb_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    for fr in range(4):
        img = _render(ex, nodes, edges, frame=fr, frames=4)
        assert img.shape == (H, W, 3), f"frame {fr}: bad output shape"
        assert img.min() >= 0.0 and img.max() <= 1.0 + 1e-6, f"frame {fr}: out of range"


def test_feedback_has_observable_crossframe_effect():
    """Feedback must carry the previous frame's pixels; output differs vs no-loop.

    With feedback, node 1 receives screen_blend(gen, prev_frame_output) on
    frame 1+, whereas without feedback it receives only gen. If the feedback
    branch were broken (previous-frame pixels dropped), the two graphs would
    render identically on frame 1 — this assertion catches that regression.
    """
    W, H = 64, 48
    set_canvas(W, H)

    nodes_fb, edges_fb = _graph(with_feedback=True)
    nodes_nf, edges_nf = _graph(with_feedback=False)

    out_fb = Path(tempfile.mkdtemp(prefix="graph_fb_on_"))
    out_nf = Path(tempfile.mkdtemp(prefix="graph_fb_off_"))
    ex_fb = GraphExecutor(out_fb, fps=24, in_memory=True, audit_to_disk=False)
    ex_nf = GraphExecutor(out_nf, fps=24, in_memory=True, audit_to_disk=False)

    # Frame 0: feedback falls back to a black image, which is a screen-blend
    # identity (1-(1-a)*(1-0) == a), so frame 0 must match the no-feedback graph.
    f0_fb = _render(ex_fb, nodes_fb, edges_fb, frame=0, frames=4)
    f0_nf = _render(ex_nf, nodes_nf, edges_nf, frame=0, frames=4)
    assert np.allclose(f0_fb, f0_nf), (
        "frame 0 with feedback differs from no-feedback — black-image fallback "
        "is not an identity blend (feedback fallback branch broken)"
    )

    # Frame 1+: feedback carries frame 0's output, so the result must differ.
    f1_fb = _render(ex_fb, nodes_fb, edges_fb, frame=1, frames=4)
    f1_nf = _render(ex_nf, nodes_nf, edges_nf, frame=1, frames=4)
    assert not np.allclose(f1_fb, f1_nf), (
        "frame 1 with feedback == without feedback — previous-frame pixels are "
        "NOT fed back through the feedback edge (core/graph.py feedback wiring "
        "may be broken)"
    )
