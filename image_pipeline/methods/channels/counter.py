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

@method(id="__counter__", name="Counter", category="channels",
        tags=["chop", "time", "integer", "generator"],
        inputs={"reset": "SCALAR", "step": "SCALAR"},
        outputs={"value": "SCALAR", "phase": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            },
            "phase": {
                "type": "output",
                "label": "Phase",
                "observable": True
            }
        },
        signal={
            "reset": "control",
            "step": "numeric",
            "value": "output",
            "phase": "output"
        },
        params={
            "start": {"description": "counter start value", "default": 0},
            "end": {"description": "counter end value (inclusive)", "default": 100},
            "step_size": {"description": "increment per frame", "default": 1},
            "mode": {"description": "counter mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
        })
def method_counter(out_dir: Path, seed: int, params=None):
    """Integer counter that advances per frame.

    Counts from start to end, then wraps or reverses based on mode.
    Can be wired into simulation n_frames to control sub-stepping.

    Outputs:
        value (SCALAR): current count
        phase (SCALAR): normalized position 0->1 between start and end
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    start = int(params.get("start", 0))
    end = int(params.get("end", 100))
    step_size = int(params.get("step_size", 1))
    mode = params.get("mode", "loop")

    # SCALAR overrides
    reset_val = params.get("reset")
    if reset_val is not None:
        frame = int(reset_val)

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject an integer `frame` for CHOP generators. Derive the live
    # frame from the Timeline so the counter advances on every rendered frame
    # instead of staying pinned at frame 0 (which froze driver-driven graphs
    # and culled them as static in the liveness gate).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    step_override = params.get("step")
    if step_override is not None:
        step_size = max(1, int(round(step_override)))

    total = end - start
    if total <= 0:
        return {"value": float(start), "phase": 0.0}

    raw = frame * step_size
    if mode == "once":
        val = min(start + raw, end)
    elif mode == "pingpong":
        cycle = raw % (total * 2)
        val = start + (cycle if cycle <= total else total * 2 - cycle)
    else:  # loop
        val = start + (raw % (total + 1))

    phase = (val - start) / total if total > 0 else 0.0
    return {"value": float(val), "phase": float(phase)}
