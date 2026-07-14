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


def test_driver_less_static_than_control():
    """Sanity: a driven clip must be strictly more alive than its control."""
    set_canvas(W, H)
    tv_d, _ = _render(*_build(True, "__lfo__", DRIVERS["__lfo__"]))
    tv_c, _ = _render(*_build(False, None, None))
    assert tv_d > tv_c + CFG.temporal_var_min
