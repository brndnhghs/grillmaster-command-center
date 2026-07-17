"""Regression tests for Stable Fluids node #517 (Stam 2D Navier–Stokes).

Locks the Fedkiw 2001 vorticity-confinement feature added in commit
c0284c4 against silent dead-param regressions (Skill 8-step audit, Step 7:
a param-sweep Δ probe catches a slider that does nothing).

These import the method function directly (no server / no graph executor),
so they run fast and headlessly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from image_pipeline.methods.simulations.stable_fluids import method_stable_fluids


def _run(vort: float, color_mode: str = "density", steps: int = 22, res: int = 64) -> tuple:
    out_dir = Path(tempfile.mkdtemp())
    params = {
        "resolution": res,
        "steps": steps,
        "vorticity_confinement": vort,
        "color_mode": color_mode,
        "anim_mode": "none",
        "splats": 3,
        "force": 5.0,
    }
    out = np.asarray(method_stable_fluids(out_dir, 7, params), dtype=float)
    scalars_path = out_dir / "scalars.json"
    scalars = json.loads(scalars_path.read_text()) if scalars_path.exists() else {}
    return out, scalars


def test_vorticity_confinement_param_is_live():
    """vort=0 vs vort=6 must visibly change the rendered frame (not a dead slider)."""
    out_off, _ = _run(0.0)
    out_on, _ = _run(6.0)
    delta = float(np.mean(np.abs(out_on - out_off)))
    assert delta > 0.02, f"vorticity_confinement looks inert (Δ={delta:.4f})"


def test_vorticity_confinement_zero_is_noop_safe():
    """vort=0 must still produce a non-blank fluid frame (the feature is additive)."""
    out, _ = _run(0.0)
    std = float(out.reshape(-1, 3).std())
    assert std > 0.005, f"vort=0 produced a blank frame (std={std:.4f})"


def test_color_mode_vorticity_renders_nonblack():
    """The new 'vorticity' colour mode must render a meaningful (non-black) frame."""
    out, _ = _run(0.6, color_mode="vorticity")
    assert out.reshape(-1, 3).std() > 0.01, "color_mode='vorticity' rendered black"


def test_scalars_expose_vorticity_fields():
    """write_scalars must carry max_vorticity and vorticity_confinement."""
    _, scalars = _run(0.6)
    assert "max_vorticity" in scalars, "max_vorticity missing from scalars output"
    assert scalars.get("vorticity_confinement") == 0.6, "vorticity_confinement value not recorded"
