"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

@method(id="__age_heat__", name="AgeHeat", category="channels",
        tags=["chop", "color", "age", "generator"],
        inputs={"age": "SCALAR", "max_age": "SCALAR"},
        outputs={"value": "SCALAR", "r": "SCALAR", "g": "SCALAR", "b": "SCALAR"},
        params={
            "mode": {"description": "age coloring mode",
                     "choices": ["heat", "cool", "rainbow", "mono"],
                     "default": "heat"},
            "max_age_default": {"description": "max age for normalization", "default": 100.0},
        })
def method_age_heat(out_dir: Path, seed: int, params=None):
    """Maps a scalar age value to a color output.

    Replaces the f2l (frames-to-live) animation mode. Wire this into
    hue_shift on the CA node to get age-based coloring.

    Outputs:
        value (SCALAR): normalized age 0-1
        r (SCALAR): red channel 0-1
        g (SCALAR): green channel 0-1
        b (SCALAR): blue channel 0-1
    """
    if params is None:
        params = {}
    seed_all(seed)

    age = float(params.get("age", 0.0))
    max_age = float(params.get("max_age", params.get("max_age_default", 100.0)))
    mode = params.get("mode", "heat")

    # SCALAR overrides
    age_override = params.get("age")
    if age_override is not None:
        age = float(age_override)
    max_age_override = params.get("max_age")
    if max_age_override is not None:
        max_age = float(max_age_override)

    norm = max(0.0, min(1.0, age / max_age)) if max_age > 0 else 0.0

    if mode == "heat":
        r = min(1.0, norm * 2.0)
        g = max(0.0, min(1.0, norm * 2.0 - 1.0))
        b = max(0.0, norm * 3.0 - 2.0)
    elif mode == "cool":
        r = max(0.0, norm * 3.0 - 2.0)
        g = max(0.0, min(1.0, norm * 2.0 - 1.0))
        b = min(1.0, norm * 2.0)
    elif mode == "rainbow":
        h = norm * 0.5
        r = 0.5 + 0.5 * math.sin(h * 2 * math.pi)
        g = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 2.094)
        b = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 4.189)
    else:  # mono
        r = g = b = norm

    return {"value": float(norm), "r": float(r), "g": float(g), "b": float(b)}
