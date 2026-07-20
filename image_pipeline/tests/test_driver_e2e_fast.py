"""Fast end-to-end regression: CHOP driver SCALAR output must reach pixels.

Route 8 (2026-07-13): the shootout liveness gate culled ~66% of genomes,
dominated by ``static``/``flat`` deaths whose graphs wired a driver
(``__lfo__``/``__counter__``/``__noise1d__``/``__ramp__``/``__strobe__``/
``__envelope__``) into a target node's numeric param. The driver fn-level
advance is already locked by ``test_chop_drivers_advance.py``, BUT that only
proves the node produces a varying SCALAR — it does NOT prove the
GraphExecutor actually *injects* that SCALAR into the target's param every
frame. The driver→pixel wiring lives in ``core/graph.py`` (the edge
transport + ``run_params[dst_port] = value`` path), and a regression there
would freeze every driver-driven clip without tripping the fn-level test.

This test closes that gap with a FULL executor render of a tiny driver→target
graph and asserts the terminal clip's temporal variance clears the shootout
liveness floor. It is marked ``slow``: its chosen target (node 952, Blue-Noise
Dither) is a genuinely expensive generator (~25 s per 8-frame batch), so the
full test costs ~50 s. That made the DEFAULT ``-m "not slow"`` suite stall for
a minute, looking hung. The SCALAR→param executor wiring it validates is still
covered in the default suite by ``test_shootout_driver_generators_vary`` (~1 s)
and ``test_shootout_driver_modulation`` (~2 s), which render cheaper targets —
so this heavier check runs only under the ``slow`` marker.

If this test ever fails, a refactor broke the SCALAR→param edge wiring and
driver-driven animation would silently die again.
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
from image_pipeline.shootout.config import DEFAULT_CONFIG


# Cheap target: node 952 (Blue-Noise Dither) exposes ``matrix_size`` (float
# default, no min/max → wireable SCALAR port). It is a fast per-pixel filter,
# ideal for a tiny end-to-end probe.
TARGET_MID = "952"
TARGET_PARAM = "matrix_size"

FLOOR = DEFAULT_CONFIG.temporal_var_min


@pytest.mark.slow
@pytest.mark.parametrize("driver_mid", ("__lfo__", "__counter__"))
def test_driver_scalar_reaches_pixels_e2e(driver_mid):
    """A driver wired into a target's numeric param must animate the clip."""
    tgt_meta = get_meta(TARGET_MID)
    assert tgt_meta is not None, f"{TARGET_MID} not registered"
    assert TARGET_PARAM in (tgt_meta.params or {}), (
        f"{TARGET_MID} missing wireable param {TARGET_PARAM}"
    )

    # Tiny canvas + few frames → fast enough for the default suite.
    W, H = 96, 64
    set_canvas(W, H)

    tgt_params = {k: (v.get("default") if isinstance(v, dict) else v)
                  for k, v in (tgt_meta.params or {}).items()}
    tgt_params["anim_mode"] = "none"
    drv_params = (
        {"waveform": "sine", "min": 0.0, "max": 1.0, "rate": 0.6}
        if driver_mid == "__lfo__" else {}
    )

    nodes = [
        {"id": "0", "method_id": driver_mid, "params": drv_params},
        {"id": "1", "method_id": TARGET_MID, "params": tgt_params},
    ]
    edges = [{"src_node": "0", "src_port": "value",
              "dst_node": "1", "dst_port": TARGET_PARAM}]

    out_dir = Path(tempfile.mkdtemp(prefix="driver_e2e_fast_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)

    lum: list[float] = []
    for fr in range(8):
        res, _term, errs = ex.execute(nodes=nodes, edges=edges,
                                      seed=7, frame=fr, frames=8)
        assert not errs, f"{driver_mid}→{TARGET_MID} raised: {errs}"
        img = (res.get("1", {}) or {}).get("image")
        assert img is not None, f"{driver_mid}→{TARGET_MID} produced no image"
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        lum.append(float(arr.reshape(-1).mean()))

    # Temporal variance of the terminal clip must clear the liveness floor.
    tvar = float(np.std(lum))
    assert tvar > FLOOR, (
        f"{driver_mid}→{TARGET_MID}.{TARGET_PARAM} did NOT reach pixels "
        f"(temporal_var={tvar:.5f} <= floor={FLOOR}) — driver SCALAR "
        f"edge wiring may be broken in core/graph.py"
    )
