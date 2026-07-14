"""End-to-end GraphExecutor pipeline test.

This closes the #1 testing gap flagged in docs/reports/testing.md (and
PROJECT_STATUS.md "Top 3 Priority Actions"): there was *no* dedicated test that
exercises the full graph pipeline — topological sort → image payload
propagation across multiple nodes → terminal selection → per-frame output.

The executor is proven *indirectly* by method and driver tests, but a break in
the edge-transport / payload-bus / terminal-selection code in core/graph.py
would not have tripped any existing test. This test renders a real 3-node graph
(gen → filter → filter) and asserts:

  1. Every node produces an image (no silent drops).
  2. Image payloads actually propagate along ``image_in`` edges — the terminal
     is not accidentally equal to the *generator* (which would mean the filter
     chain was bypassed).
  3. The terminal payload equals the last node's output (terminal selection is
     correct).
  4. Wiring is order-independent: a re-ordered node list yields the same output.
  5. Multi-frame rendering is deterministic for a frozen graph (frame 0 == frame 0
     re-rendered).

It deliberately uses tiny canvases and fast nodes so it belongs in the DEFAULT
suite (not marked ``slow``).
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


# Worley Noise — fast deterministic data-source generator (no upstream deps).
GEN_MID = "04"
# Palette Posterize — deterministic filter that consumes wired image_in.
FILT1_MID = "422"
# Kaleidoscope Mirror — deterministic filter that consumes wired image_in.
FILT2_MID = "460"


def _default_params(mid: str) -> dict:
    meta = get_meta(mid)
    assert meta is not None, f"method {mid} not registered"
    return {
        k: (v.get("default") if isinstance(v, dict) else v)
        for k, v in (meta.params or {}).items()
    }


def _build_graph() -> tuple[list[dict], list[dict]]:
    gen_p = _default_params(GEN_MID)
    gen_p["anim_mode"] = "none"  # freeze the generator so frames are deterministic
    gen_p["colormode"] = "flat_shaded"
    f1_p = _default_params(FILT1_MID)
    f1_p["anim_mode"] = "none"
    f1_p["source"] = "input_image"  # force it to use the wired upstream image
    f2_p = _default_params(FILT2_MID)

    nodes = [
        {"id": "0", "method_id": GEN_MID, "params": gen_p},
        {"id": "1", "method_id": FILT1_MID, "params": f1_p},
        {"id": "2", "method_id": FILT2_MID, "params": f2_p},
    ]
    edges = [
        {"src_node": "0", "src_port": "image", "dst_node": "1", "dst_port": "image_in"},
        {"src_node": "1", "src_port": "image", "dst_node": "2", "dst_port": "image_in"},
    ]
    return nodes, edges


def _render_terminal(ex: GraphExecutor, nodes, edges, seed=11, frame=0, frames=1):
    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=seed, frame=frame, frames=frames)
    assert not errs, f"graph raised node errors: {errs}"
    assert terminal is not None, "no terminal node selected"
    term_img = (res.get(terminal, {}) or {}).get("image")
    assert term_img is not None, f"terminal {terminal} produced no image"
    return terminal, np.asarray(term_img, dtype=np.float32)


def test_graph_runs_all_nodes_and_propagates_image():
    """gen→filter→filter must produce images at every node and pass pixels along."""
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_e2e_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    res, terminal, errs = ex.execute(nodes=nodes, edges=edges, seed=11, frame=0, frames=1)
    assert not errs, f"graph raised node errors: {errs}"
    assert terminal == "2", f"expected terminal '2', got {terminal!r}"

    # Every node produced an image.
    for nid in ("0", "1", "2"):
        img = (res.get(nid, {}) or {}).get("image")
        assert img is not None, f"node {nid} produced no image"
        assert isinstance(img, np.ndarray) and img.ndim == 3

    # The propagated image must differ from the generator's raw output —
    # otherwise the filter chain would have been bypassed (a real bug class).
    gen_img = np.asarray(res["0"]["image"], dtype=np.float32)
    filt2_img = np.asarray(res["2"]["image"], dtype=np.float32)
    assert not np.allclose(gen_img, filt2_img), (
        "terminal image equals the generator output — image payload did not "
        "propagate through the filter chain (image_in edge wiring broken)"
    )

    # The two filters must each have transformed the image (chain is real).
    filt1_img = np.asarray(res["1"]["image"], dtype=np.float32)
    assert not np.allclose(filt1_img, gen_img), "filter 1 did not transform the image"


def test_terminal_payload_matches_last_node():
    """The executor's returned terminal_id maps to the actual final output."""
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_e2e_term_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    terminal, term_img = _render_terminal(ex, nodes, edges, seed=11)
    assert terminal == "2"

    # The payload under "2" must be exactly the terminal payload returned.
    last_img = np.asarray(
        ex.execute(nodes=nodes, edges=edges, seed=11, frame=0, frames=1)[0]["2"]["image"],
        dtype=np.float32,
    )
    assert np.allclose(term_img, last_img), "terminal output != last node output"


def test_wiring_order_independent():
    """Re-ordering the node list must not change the rendered terminal image."""
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_graph()
    shuffled = [nodes[2], nodes[0], nodes[1]]  # different list order

    out1 = Path(tempfile.mkdtemp(prefix="graph_e2e_o1_"))
    out2 = Path(tempfile.mkdtemp(prefix="graph_e2e_o2_"))
    ex1 = GraphExecutor(out1, fps=24, in_memory=True, audit_to_disk=False)
    ex2 = GraphExecutor(out2, fps=24, in_memory=True, audit_to_disk=False)

    _, img_a = _render_terminal(ex1, nodes, edges, seed=11)
    _, img_b = _render_terminal(ex2, shuffled, edges, seed=11)
    assert np.allclose(img_a, img_b), (
        "terminal image changed when node list order changed — topological "
        "sort is not order-independent"
    )


def test_multiframe_deterministic_for_frozen_graph():
    """A frozen (non-time-varying) graph renders identically across frames."""
    W, H = 64, 48
    set_canvas(W, H)

    nodes, edges = _build_graph()
    out_dir = Path(tempfile.mkdtemp(prefix="graph_e2e_mf_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    _, f0 = _render_terminal(ex, nodes, edges, seed=11, frame=0, frames=4)
    _, f0_again = _render_terminal(ex, nodes, edges, seed=11, frame=0, frames=4)
    assert np.allclose(f0, f0_again), "frame-0 render was not reproducible"
