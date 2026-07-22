"""Fast DEFAULT-SUITE guard: CHOP driver SCALAR output must reach pixels.

Companion to ``test_driver_e2e_fast.py``, which validates the same
``GraphExecutor`` SCALAR→param wiring but is marked ``slow``: its target (node
952, Blue-Noise Dither) costs ~25 s per 8-frame batch, so it does NOT run under
the default ``-m "not slow"`` suite. The two cheap tests that used to cover this
path unmarked — ``test_shootout_driver_generators_vary`` and
``test_shootout_driver_modulation`` — were deleted with the evolutionary
generator, leaving the path with no fast guard.

This test restores that coverage. A driver (``__lfo__`` / ``__counter__``)
wired into a target node's numeric param must actually modulate pixels. The
fn-level driver advance is already locked by ``test_chop_drivers_advance.py``,
but that only proves the node emits a varying SCALAR — it does NOT prove the
executor *injects* that SCALAR into the target's param every frame. That
wiring lives in ``core/graph.py`` (the edge transport + ``_inject_typed`` →
``run_params[param] = val`` path), and a regression there would freeze every
driver-driven clip without tripping the fn-level test.

The target is node 417 (Chromatic Aberration): a cheap per-pixel filter whose
``amount`` param is a plain int, driven against a STATIC ``gradient`` source so
the driver is the only possible motion source. Each 8-frame clip renders in
~0.03 s, so the whole module costs well under a second.

Both halves are asserted:
  1. the DRIVEN clip's temporal variance clears the motion floor (the SCALAR
     reached the pixels), AND
  2. the same graph with the driver edge REMOVED is static (variance ~0),
     isolating the driver as the cause rather than incidental node animation.

Deliberately imports nothing from the shootout package (deleted) — the motion
floor is inlined below.

If this test fails, a refactor broke the SCALAR→param edge wiring and
driver-driven animation would silently die again.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401  (registers @method nodes)
from image_pipeline.core.registry import get_meta
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas


# Cheap per-pixel filter: node 417 (Chromatic Aberration). ``amount`` is an int
# param (min 0 / max 60) with no ``choices`` list, so it exercises the plain
# SCALAR→int rounding branch of ``_inject_typed``.
TARGET_MID = "417"
TARGET_PARAM = "amount"

# Motion floor (formerly ShootoutConfig.temporal_var_min). Measured driven
# variance is 0.014 (lfo) / 0.035 (counter) — roughly 5-12x this floor — while
# the undriven control measures exactly 0.0, so the margin is wide either way.
FLOOR = 3e-3

W, H = 96, 64
N_FRAMES = 8

DRIVERS = {
    "__lfo__": {"waveform": "sine", "min": 0.0, "max": 60.0,
                "rate": 1.0, "phase": 0.0},
    "__counter__": {"start": 0, "end": 60, "step_size": 8, "mode": "loop"},
}


def _clip_variance(driver_mid: str, driven: bool) -> float:
    """Render an N-frame clip of driver→target and return the std of mean
    luminance across frames. With ``driven=False`` the driver edge is omitted
    (the control), so any variance would have to come from the target itself."""
    tgt_meta = get_meta(TARGET_MID)
    tgt_params = {k: (v.get("default") if isinstance(v, dict) else v)
                  for k, v in (tgt_meta.params or {}).items()}
    # Pin the target static: no self-animation, and a fixed procedural source
    # so the driver is the sole motion source.
    tgt_params["anim_mode"] = "none"
    tgt_params["source"] = "gradient"

    nodes = [
        {"id": "0", "method_id": driver_mid,
         "params": dict(DRIVERS[driver_mid])},
        {"id": "1", "method_id": TARGET_MID, "params": tgt_params},
    ]
    edges = [{"src_node": "0", "src_port": "value",
              "dst_node": "1", "dst_port": TARGET_PARAM}] if driven else []

    out_dir = Path(tempfile.mkdtemp(prefix="driver_e2e_filter_"))
    ex = GraphExecutor(out_dir, fps=24, in_memory=True, audit_to_disk=False)
    lum: list[float] = []
    try:
        for fr in range(N_FRAMES):
            res, _term, errs = ex.execute(nodes=nodes, edges=edges,
                                          seed=7, frame=fr, frames=N_FRAMES)
            assert not errs, f"{driver_mid}→{TARGET_MID} raised: {errs}"
            img = (res.get("1", {}) or {}).get("image")
            assert img is not None, (
                f"{driver_mid}→{TARGET_MID} produced no image")
            arr = np.asarray(img).astype(np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            lum.append(float(arr.reshape(-1).mean()))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    return float(np.std(lum))


@pytest.mark.parametrize("driver_mid", tuple(DRIVERS))
def test_driver_scalar_reaches_pixels_fast(driver_mid):
    """A driver wired into a cheap filter's numeric param must animate it."""
    tgt_meta = get_meta(TARGET_MID)
    assert tgt_meta is not None, f"{TARGET_MID} not registered"
    assert TARGET_PARAM in (tgt_meta.params or {}), (
        f"{TARGET_MID} missing wireable param {TARGET_PARAM}")

    set_canvas(W, H)

    driven_var = _clip_variance(driver_mid, driven=True)
    assert driven_var > FLOOR, (
        f"{driver_mid}→{TARGET_MID}.{TARGET_PARAM} did NOT reach pixels "
        f"(temporal_var={driven_var:.5f} <= floor={FLOOR}) — driver SCALAR "
        f"edge wiring may be broken in core/graph.py")

    # Control: identical graph minus the driver edge must be static, proving
    # the motion above came from the driver and not from the target node.
    static_var = _clip_variance(driver_mid, driven=False)
    assert static_var < FLOOR, (
        f"control (no driver edge) unexpectedly animated "
        f"(temporal_var={static_var:.5f}) — the target node is not static, so "
        f"this test can no longer isolate the driver as the motion source")
