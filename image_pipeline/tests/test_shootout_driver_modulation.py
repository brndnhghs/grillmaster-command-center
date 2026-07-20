"""Headless regression: CHOP driver modulation must reach the rendered pixels.

Route 8 (2026-07-14) deliverable. The shootout liveness gate culls ~65% of
genomes as dead; an early hypothesis was that driver (CHOP) nodes — __lfo__,
__counter__, __noise1d__, __ramp__, __strobe__, __envelope__ — were not
actually modulating their target node's params at render time, leaving the clip
frozen and culling it as ``static``.

This test PROVES the wiring works end-to-end (no render-server, no browser):
for each driver type it builds

    [static noise source] -> [Transform(rotate)] <- [driver.value]

renders a short clip with GraphExecutor, and asserts:

  1. the driver's SCALAR output actually varies across frames (the generator
     advances with the timeline's global_frame), AND
  2. the terminal frame-stack temporal_var is ABOVE the liveness floor
     (modulation reached the pixels), AND
  3. the same graph with the driver DISCONNECTED is essentially static
     (temporal_var ~ 0), isolating the driver as the cause.

If a future refactor breaks the CHOP->param injection path, this test fails
loudly instead of silently inflating the dead-clip rate.

Run:  pytest image_pipeline/tests/test_shootout_driver_modulation.py
"""
from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — registers the node catalog
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas
from image_pipeline.shootout.config import ShootoutConfig
from image_pipeline.shootout.evaluator import (
    LivenessAccumulator,
    _terminal_image,
)

W = H = 160
FRAMES = 16
CFG = ShootoutConfig()

# (driver method_id, params) — each must expose a SCALAR "value" output that
# advances with the timeline's global_frame.
DRIVERS = {
    "__lfo__": {
        "waveform": "sine", "min": -120.0, "max": 120.0,
        "rate": 1.0, "phase": 0.0,
    },
    "__counter__": {
        "start": 0, "end": 360, "step_size": 15, "mode": "loop",
    },
    "__noise1d__": {
        "min": -120.0, "max": 120.0, "rate": 0.5, "smooth": 1,
    },
}


def _build(drive: bool, driver_id: str | None, driver_params: dict | None):
    nodes = [
        {"id": "src", "method_id": "05", "params": {"seed": 7},
         "render": False, "dirty": True},
        {"id": "out", "method_id": "__transform__", "params": {"rotate": 0.0},
         "render": True, "dirty": True},
    ]
    edges = [
        {"src_node": "src", "src_port": "image", "dst_node": "out",
         "dst_port": "image_in", "feedback": False},
    ]
    if drive:
        nodes.insert(1, {"id": "drv", "method_id": driver_id,
                         "params": dict(driver_params),
                         "render": False, "dirty": True})
        edges.append({"src_node": "drv", "src_port": "value", "dst_node": "out",
                      "dst_port": "rotate", "feedback": False})
    return nodes, edges


def _render(nodes, edges, seed=42):
    wd = Path(tempfile.mkdtemp(prefix="drvmod-"))
    ex = GraphExecutor(wd, fps=CFG.fps, in_memory=True, audit_to_disk=False)
    acc = LivenessAccumulator(CFG)
    drv_vals = []
    try:
        for frame in range(FRAMES):
            flat, terminal_id, _errs = ex.execute(
                nodes, edges, seed, frame=frame, frames=FRAMES)
            arr = _terminal_image(flat, terminal_id, nodes)
            acc.add(arr)
            if "drv" in flat and isinstance(flat["drv"].get("value"), (int, float)):
                drv_vals.append(float(flat["drv"]["value"]))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return acc.stats()["temporal_var"], drv_vals


@pytest.mark.parametrize("driver_id", list(DRIVERS.keys()))
def test_driver_modulation_reaches_pixels(driver_id):
    set_canvas(W, H)
    tv, drv_vals = _render(*_build(True, driver_id, DRIVERS[driver_id]))

    # 1) The driver actually advanced across frames.
    assert len(drv_vals) == FRAMES, "driver must run every frame"
    span = max(drv_vals) - min(drv_vals)
    assert span > 1.0, f"{driver_id} output did not vary (span={span:.3f})"

    # 2) Modulation reached the terminal pixels: above the liveness floor.
    assert tv > CFG.temporal_var_min, (
        f"{driver_id}: terminal temporal_var {tv:.6f} <= floor "
        f"{CFG.temporal_var_min} — driver did NOT reach pixels")

    # 3) Control: same graph, driver disconnected -> essentially static.
    tv_ctrl, _ = _render(*_build(False, None, None))
    assert tv_ctrl < CFG.temporal_var_min, (
        f"control (no driver) unexpectedly animated: tv={tv_ctrl:.6f}")


def test_born_animated_floor_reaches_slider_terminal():
    """Route 8 (2026-07-19): the born-animated floor must animate a single-node
    Architecture-A terminal whose ONLY tunables are min/max slider params.

    Prior to the fix, ``GenePool.driver_targets`` drew targets only from
    ``param_ports`` (UI wire ports) + declared scalar inputs. The UI auto-detect
    deliberately keeps min/max-bounded params OUT of ``param_ports`` (they render
    as sliders), so a slider-only terminal (e.g. node 39 posterize:
    ``n_colors``/``anim_speed`` are sliders, none wireable) had an EMPTY
    driver-target set.
    The floor silently no-op'd, the graph was genuinely frozen, and the liveness
    gate culled it as ``static`` — ~59 such genomes (34% of all flat/static
    deaths) at last scan.

    This test builds that exact single-node graph, runs the floor, asserts a
    driver edge was injected onto a slider param, then renders the floor-applied
    graph and confirms temporal_var clears the liveness floor (modulation reaches
    pixels). If a future refactor re-narrows driver_targets, this fails loudly.
    """
    import random
    import image_pipeline.shootout.generator as gen
    from image_pipeline.shootout.config import DEFAULT_CONFIG

    pool = gen.build_gene_pool(DEFAULT_CONFIG)
    # A real terminal whose tunables are sliders only (not wire ports).
    # Node 39 (posterize): n_colors/anim_speed are min/max sliders, none wireable;
    # driving n_colors per-frame strongly animates the clip (tv~0.24).
    term_mid = "39"
    assert not pool.driver_targets(term_mid) == [], "precondition: must have slider targets"
    assert pool.wireable_params(term_mid) == [], "precondition: term has no UI wire ports"

    # Materialize concrete default param VALUES (not the {min,max,default} specs).
    def _defaults(mid):
        out = {}
        for pname, spec in (pool.defs[mid].get("params") or {}).items():
            out[pname] = spec.get("default") if isinstance(spec, dict) else spec
        return out

    graph = {
        "version": 1, "name": "t",
        "nodes": [{"id": "n1", "method_id": term_mid,
                   "params": _defaults(term_mid),
                   "x": 0, "y": 0, "render": True}],
        "edges": [],
    }
    assert not gen._graph_has_animation_source(graph, pool), \
        "precondition: genuinely frozen before floor"

    rng = random.Random(7)
    g2 = gen._ensure_animated(graph, pool, DEFAULT_CONFIG, rng, None)

    assert gen._graph_has_animation_source(g2, pool), \
        "floor must inject an animation source"
    mid_by_id = {n["id"]: n["method_id"] for n in g2["nodes"]}
    drv_edges = [e for e in g2["edges"]
                 if mid_by_id.get(e["src_node"]) in pool.scalar_drivers]
    assert drv_edges, "floor must add a driver edge"
    assert drv_edges[0]["dst_port"] in pool.driver_targets(term_mid), \
        "driver wired to a slider target"

    # Render the floor-applied graph and confirm it actually animates.
    nodes = [{"id": n["id"], "method_id": n["method_id"],
              "params": dict(n.get("params", {})), "render": n.get("render", False),
              "dirty": True}
             for n in g2["nodes"]]
    edges = [dict(e) for e in g2["edges"]]
    set_canvas(W, H)
    tv, _ = _render(nodes, edges)
    assert tv > CFG.temporal_var_min, (
        f"floor-applied slider-terminal clip did NOT animate: tv={tv:.6f} "
        f"<= floor {CFG.temporal_var_min}")


def test_driver_less_static_than_control():
    """Sanity: a driven clip must be strictly more alive than its control."""
    set_canvas(W, H)
    tv_d, _ = _render(*_build(True, "__lfo__", DRIVERS["__lfo__"]))
    tv_c, _ = _render(*_build(False, None, None))
    assert tv_d > tv_c + CFG.temporal_var_min
