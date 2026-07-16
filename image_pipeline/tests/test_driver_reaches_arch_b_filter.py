"""Route 8 #1b — driver path reaches Architecture-B (filter / pattern) nodes.

The committed ``test_driver_reaches_arch_a_sim`` locks in the fix for the
Arch-A (simulation) driver path, but the *Arch-B* (per-frame-rendered filter /
pattern) driver path had NO regression test. The corpus depends on BOTH: a CHOP
driver wired into a filter's numeric param (e.g. LFO -> Chromatic Aberration
amount) must modulate the rendered pixels each frame, or the clip is culled as
static and the dead-rate climbs.

A headless probe (2026-07-16) confirmed the path works: node 417 driven by an
LFO on ``amount`` goes from static mean-luminance var ≈ 0.0 to driven var ≈
0.030, clearing the liveness floor. This test LOCKS THAT IN: iterate candidate
Arch-B nodes with a wireable numeric param, render a 4-frame clip with a STATIC
source so the driver is the sole motion source, and assert the driven clip
clears the liveness floor while the undriven control stays static — isolating
the driver as the cause.

Marked ``slow`` (renders real frames); excluded from the default
``-m "not slow"`` run.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import image_pipeline.methods  # noqa: F401 — registers the node catalog
from image_pipeline.core.graph import GraphExecutor
from image_pipeline.core.utils import set_canvas
from image_pipeline.core.registry import get_all, get_meta


FLOOR = 3e-3
W = H = 96
N_FRAMES = 4


def _clip_var(tgt_mid: str, tgt_param: str, driven: bool,
              lo: float, hi: float) -> float | None:
    """Render a (driven or static) 4-frame clip of an Arch-B node with a STATIC
    source (anim_mode='none', source='gradient') so the node output does not
    vary on its own; return the std of mean-luminance across frames, or None on
    exec/image error."""
    out_dir = Path(tempfile.mkdtemp(prefix="archb_drv_"))
    drv_params = {"waveform": "sine", "min": float(lo), "max": float(hi),
                  "rate": 2.0}
    _pmeta = get_meta(tgt_mid).params or {}
    default_val = _pmeta.get(tgt_param, {})
    default_val = default_val.get("default") if isinstance(default_val, dict) else 0.0
    tgt_params = {tgt_param: default_val, "anim_mode": "none"}
    # Static source so the only frame-to-frame variation comes from the driver.
    if "source" in (_pmeta or {}):
        tgt_params["source"] = "gradient"
    nodes = [
        {"id": "0", "method_id": "__lfo__", "params": dict(drv_params)},
        {"id": "1", "method_id": tgt_mid, "params": dict(tgt_params)},
    ]
    edges = [
        {"src_node": "0", "src_port": "value", "dst_node": "1",
         "dst_port": tgt_param}
    ] if driven else []
    ex = GraphExecutor(out_dir=out_dir, fps=24)
    lum: list[float] = []
    for fr in range(N_FRAMES):
        try:
            res, _term, errs = ex.execute(
                nodes=nodes, edges=edges, seed=7, frame=fr, frames=24)
        except Exception:
            return None
        if errs:
            return None
        img = (res.get("1", {}) or {}).get("image")
        if img is None:
            return None
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        lum.append(float(arr.astype(np.float32).reshape(-1).mean()))
    if len(lum) < N_FRAMES:
        return None
    return float(np.std(lum))


@pytest.mark.slow
def test_driver_reaches_arch_b_filter():
    set_canvas(W, H)
    # Collect Arch-B nodes (no n_frames param) in the visual categories with a
    # wireable numeric param.
    candidates: list[tuple[str, str]] = []
    for mid, meta in get_all().items():
        if mid.startswith("__"):
            continue
        params = meta.params or {}
        if "n_frames" in params:
            continue  # Arch-A marker
        if meta.category not in ("filters", "patterns", "fractals", "math_art"):
            continue
        for pname, spec in params.items():
            if not isinstance(spec, dict):
                continue
            d = spec.get("default")
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                if pname in ("time", "phase", "frames", "seed", "anim_speed"):
                    continue
                candidates.append((mid, pname))
                break
    assert candidates, "no Arch-B node with a wireable scalar param found"

    last = None
    for tgt_mid, tgt_param in candidates[:40]:
        spec = (get_meta(tgt_mid).params or {}).get(tgt_param, {})
        lo = float(spec.get("min", 0.0)) if isinstance(spec, dict) else 0.0
        hi = float(spec.get("max", 1.0)) if isinstance(spec, dict) else 1.0
        if not (hi > lo):
            lo, hi = 0.0, 1.0
        static_var = _clip_var(tgt_mid, tgt_param, driven=False, lo=lo, hi=hi)
        driven_var = _clip_var(tgt_mid, tgt_param, driven=True, lo=lo, hi=hi)
        last = (tgt_mid, tgt_param, static_var, driven_var)
        if static_var is None or driven_var is None:
            continue
        # Control must be static (isolate the driver as the cause); the driven
        # clip must clear the liveness floor (driver reached the Arch-B render).
        if static_var < FLOOR and driven_var > FLOOR:
            return  # PASS — driver reached this Arch-B node

    pytest.fail(
        f"driver never reached an Arch-B node across {len(candidates[:40])} "
        f"candidates; last tried {last}"
    )
