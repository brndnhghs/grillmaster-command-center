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

@method(id="__noise1d__", name="Noise1D", category="channels",
        tags=["chop", "time", "noise", "generator"],
        inputs={"rate": "SCALAR", "seed_offset": "SCALAR"},
        outputs={"value": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            }
        },
        signal={
            "rate": "numeric",
            "seed_offset": "numeric",
            "value": "output"
        },
        params={
            "min": {"description": "minimum output value", "default": 0.0},
            "max": {"description": "maximum output value", "default": 1.0},
            "rate": {"description": "noise rate (higher = faster variation)", "default": 0.5},
            "smooth": {"description": "interpolation smoothing (0=step, 1=linear, 2=smoothstep)", "default": 2},
        })
def method_noise1d(out_dir: Path, seed: int, params=None):
    """1D Perlin-like noise generator — smooth random values over time.

    Outputs:
        value (SCALAR): noise value in [min, max]
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    rate = float(params.get("rate", 0.5))
    smooth = int(params.get("smooth", 2))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject a `time` for CHOP generators. Derive the live phase from
    # the Timeline's global_frame (which advances every rendered frame) so the
    # noise advances instead of staying pinned at t=0 (which froze
    # driver-driven graphs and culled them as static — see __counter__ /
    # __lfo__ / __strobe__). NOTE: use global_frame, not the Timeline's `phase`
    # attribute, because make_timeline() does not set phase (it stays 0).
    if t == 0.0:
        _tl = params.get("_timeline")
        if _tl is not None:
            _gf = int(getattr(_tl, "global_frame", 0))
            _tf = int(getattr(_tl, "total_frames", 24))
            t = (_gf / max(1, _tf - 1)) * (2.0 * math.pi)

    # SCALAR overrides
    rate_override = params.get("rate")
    if rate_override is not None:
        rate = float(rate_override)
    seed_override = params.get("seed_offset")
    if seed_override is not None:
        seed = seed + int(seed_override * 1000)

    # Value noise: interpolate between random values at integer positions
    p = t * rate
    idx_a = int(math.floor(p))
    idx_b = idx_a + 1
    fade = p - idx_a

    if smooth == 0:
        # Step
        rng = random.Random(seed + idx_a)
        val = rng.uniform(min_val, max_val)
    elif smooth == 1:
        # Linear
        rng_a = random.Random(seed + idx_a)
        rng_b = random.Random(seed + idx_b)
        va = rng_a.uniform(min_val, max_val)
        vb = rng_b.uniform(min_val, max_val)
        val = va + (vb - va) * fade
    else:
        # Smoothstep
        fade = fade * fade * (3 - 2 * fade)
        rng_a = random.Random(seed + idx_a)
        rng_b = random.Random(seed + idx_b)
        va = rng_a.uniform(min_val, max_val)
        vb = rng_b.uniform(min_val, max_val)
        val = va + (vb - va) * fade

    return {"value": float(val)}
