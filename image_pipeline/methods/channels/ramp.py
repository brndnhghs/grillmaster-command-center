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

@method(id="__ramp__", name="Ramp", category="channels",
        tags=["chop", "time", "float", "generator"],
        inputs={"trigger": "SCALAR", "speed": "SCALAR"},
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
            "trigger": "event",
            "speed": "numeric",
            "value": "output",
            "phase": "output"
        },
        params={
            "start": {"description": "ramp start value", "default": 0.0},
            "end": {"description": "ramp end value", "default": 1.0},
            "duration_frames": {"description": "frames for one full ramp", "min": 1, "max": 10000, "default": 48},
            "easing": {"description": "ramp easing function",
                       "choices": ["linear", "ease_in", "ease_out", "smoothstep"],
                       "default": "linear"},
            "mode": {"description": "ramp mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
        })
def method_ramp(out_dir: Path, seed: int, params=None):
    """Float ramp that sweeps from start to end over duration_frames.

    Outputs:
        value (SCALAR): current ramp value
        phase (SCALAR): normalized position 0->1
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    start = float(params.get("start", 0.0))
    end = float(params.get("end", 1.0))
    duration = max(1, int(params.get("duration_frames", 48)))
    easing = params.get("easing", "linear")
    mode = params.get("mode", "loop")

    # SCALAR overrides
    trigger_val = params.get("trigger")
    if trigger_val is not None:
        frame = int(trigger_val)

    # Derive the live frame from the injected Timeline (see Counter for why).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    speed_override = params.get("speed")
    if speed_override is not None:
        duration = max(1, int(duration / max(0.01, float(speed_override))))

    raw_phase = (frame % duration) / duration if mode != "once" else min(frame / duration, 1.0)
    if mode == "pingpong":
        cycle = frame % (duration * 2)
        raw_phase = cycle / duration if cycle <= duration else (duration * 2 - cycle) / duration

    # Apply easing
    p = raw_phase
    if easing == "ease_in":
        p = p * p
    elif easing == "ease_out":
        p = 1 - (1 - p) * (1 - p)
    elif easing == "smoothstep":
        p = p * p * (3 - 2 * p)

    val = start + (end - start) * p
    return {"value": float(val), "phase": float(raw_phase)}
