"""Route 8 #1 — driver path reaches Architecture-A (simulation) nodes.

The generic driver/keyframe injection in GraphExecutor runs *after* the
Arch-A ``continue`` into the sim-cook branch, so a CHOP driver feeding an
Arch-A terminal was silently dropped: the sim cooked from ``node.params``
defaults and the clip was culled as static by the shootout liveness gate.
That is the dominant dead-genome signal (drivers __lfo__/__counter__/...
appear in ~1000 dead genomes).

This test LOCKS IN the fix: a driver wired to a static (anim_mode='none')
Arch-A sim must modulate the cooked clip (temporal variance above the
liveness floor), while the undriven control stays static — isolating the
driver as the cause.

It iterates candidate Arch-A sims and accepts the first where the driver
clearly modulates output, so individual sim quirks (range clamping, NaN on
out-of-range input) don't make the whole test flaky.

Marked ``slow`` (renders real sim frames); excluded from the default
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
W = H = 64
N_FRAMES = 4


def _clip_var(tgt_mid: str, tgt_param: str, driven: bool, lo: float, hi: float) -> float | None:
    """Render a (driven or static) 4-frame clip of an Arch-A sim; return the
    std of mean-luminance across frames, or None on exec/image error."""
    out_dir = Path(tempfile.mkdtemp(prefix="archa_drv_"))
    drv_params = {"waveform": "sine", "min": float(lo), "max": float(hi), "rate": 5.0}
    _pmeta = get_meta(tgt_mid).params or {}
    default_val = _pmeta.get(tgt_param, {})
    default_val = default_val.get("default") if isinstance(default_val, dict) else 0.0
    sim_params = {tgt_param: default_val, "anim_mode": "none"}
    nodes = [
        {"id": "0", "method_id": "__lfo__", "params": drv_params},
        {"id": "1", "method_id": tgt_mid, "params": dict(sim_params)},
    ]
    edges = [
        {"src_node": "0", "src_port": "value", "dst_node": "1", "dst_port": tgt_param}
    ] if driven else []
    ex = GraphExecutor(out_dir, fps=24)
    lum: list[float] = []
    for fr in range(N_FRAMES):
        try:
            res, _term, errs = ex.execute(nodes=nodes, edges=edges, seed=7, frame=fr, frames=24)
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
def test_driver_reaches_arch_a_sim():
    set_canvas(W, H)
    # Collect Arch-A sims (n_frames param) with a wireable numeric param.
    candidates: list[tuple[str, str]] = []
    for mid, meta in get_all().items():
        if mid.startswith("__"):
            continue
        params = meta.params or {}
        if "n_frames" not in params:
            continue  # Arch-A marker
        for pname, spec in params.items():
            if not isinstance(spec, dict):
                continue
            if "min" in spec or "max" in spec:
                continue  # has slider constraints — not driver-target wireable
            d = spec.get("default")
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                candidates.append((mid, pname))
                break
    assert candidates, "no Arch-A sim with a wireable scalar param found"

    last = None
    for tgt_mid, tgt_param in candidates[:10]:
        # Use the param's own declared range if present, else [0,1].
        spec = (get_meta(tgt_mid).params or {}).get(tgt_param, {})
        lo = float(spec.get("min", 0.0)) if isinstance(spec, dict) else 0.0
        hi = float(spec.get("max", 1.0)) if isinstance(spec, dict) else 1.0
        if hi <= lo:
            lo, hi = 0.0, 1.0
        static_var = _clip_var(tgt_mid, tgt_param, driven=False, lo=lo, hi=hi)
        driven_var = _clip_var(tgt_mid, tgt_param, driven=True, lo=lo, hi=hi)
        last = (tgt_mid, tgt_param, static_var, driven_var)
        if static_var is None or driven_var is None:
            continue
        # Control must be static (isolate the driver as the cause); the driven
        # clip must clear the liveness floor (driver reached the sim cook).
        if static_var < FLOOR and driven_var > FLOOR:
            return  # PASS — driver reached this Arch-A sim

    pytest.fail(
        f"driver never reached an Arch-A sim across {len(candidates[:10])} "
        f"candidates; last tried {last}"
    )
