"""Headless regression: every CHOP driver generator emits a time-varying output.

Route 8 (2026-07-14) follow-up. The shootout liveness gate culls a large
fraction of genomes as dead, and the dead-genome method histogram is dominated
by the CHOP control nodes (__lfo__, __counter__, __noise1d__, __ramp__,
__strobe__, __envelope__). An early hypothesis was that a driver *generator*
might not actually advance with the timeline's global_frame, leaving any clip
it drives frozen.

Probe (run headlessly, no render-server, no browser) confirmed all six do
advance. This test LOCKS THAT IN: for each driver it renders a short clip and
asserts the SCALAR ``value`` output varies across frames. If a future refactor
breaks a driver's time advancement (e.g. it stops reading global_frame, or
collapses to a constant), this test fails loudly instead of silently inflating
the dead-clip rate.

Note: this guards the *generator* half of the driver path. The
*modulation-reaches-pixels* half is covered by
``test_shootout_driver_modulation.py`` (lfo/counter/noise1d -> __transform__
rotate). Together they prove the full CHOP -> param -> pixel chain.

Run:  pytest image_pipeline/tests/test_shootout_driver_generators_vary.py
"""
from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import pytest

import image_pipeline.methods  # noqa: F401 — registers the node catalog
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas
from image_pipeline.shootout.config import ShootoutConfig

W = H = 64
FRAMES = 16
CFG = ShootoutConfig()

# (driver method_id, params) — chosen so the generator clearly advances over
# FRAMES frames. min/max are widened to degrees-like ranges where the driver
# feeds a rotate-style param, but here we only assert the output *varies*.
DRIVERS = {
    "__lfo__": {"waveform": "sine", "min": -1.0, "max": 1.0,
                "rate": 1.0, "phase": 0.0},
    "__counter__": {"start": 0, "end": 10, "step_size": 1, "mode": "loop"},
    "__noise1d__": {"min": -1.0, "max": 1.0, "rate": 0.5, "smooth": 1},
    "__ramp__": {"start": -1.0, "end": 1.0, "duration_frames": 48,
                 "easing": "linear", "mode": "loop"},
    "__strobe__": {"rate": 2.0, "duty_cycle": 0.5,
                   "on_value": 1.0, "off_value": 0.0},
    "__envelope__": {"attack": 4, "decay": 6, "sustain": 0.7, "release": 6,
                     "sustain_level": 0.7, "loop": True},
}


def _driver_values(driver_id: str, params: dict) -> list[float]:
    nodes = [{"id": "drv", "method_id": driver_id, "params": dict(params),
              "render": False, "dirty": True}]
    edges: list[dict] = []
    wd = Path(tempfile.mkdtemp(prefix="drvrgn-"))
    ex = GraphExecutor(wd, fps=CFG.fps, in_memory=True, audit_to_disk=False)
    vals: list[float] = []
    try:
        for frame in range(FRAMES):
            flat, _tid, _errs = ex.execute(nodes, edges, 42,
                                           frame=frame, frames=FRAMES)
            v = flat.get("drv", {}).get("value")
            vals.append(0.0 if v is None else float(v))
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return vals


@pytest.mark.parametrize("driver_id", list(DRIVERS.keys()))
def test_driver_generator_advances(driver_id):
    set_canvas(W, H)
    vals = _driver_values(driver_id, DRIVERS[driver_id])
    assert len(vals) == FRAMES, f"{driver_id}: must emit a value every frame"
    finite = [v for v in vals if v == v]  # drop NaN just in case
    assert finite, f"{driver_id}: emitted no finite values"
    span = max(finite) - min(finite)
    # Strobe is intentionally binary (on/off) — its span is still 1.0 and it
    # flips state, so require a real spread for the continuous generators and
    # at least two distinct states for the strobe.
    n_unique = len({round(v, 4) for v in finite})
    if driver_id == "__strobe__":
        assert n_unique >= 2, f"{driver_id}: must toggle on/off across frames"
    else:
        assert span > 0.05, (
            f"{driver_id}: value did not vary across frames (span={span:.4f})")
        assert n_unique >= 4, (
            f"{driver_id}: too few distinct values ({n_unique}) — "
            f"generator may be stuck")
