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

@method(id="__burst__", name="Burst", category="channels",
        tags=["chop", "time", "pulse", "generator"],
        inputs={"trigger": "SCALAR", "rate": "SCALAR"},
        outputs={"value": "SCALAR", "active": "SCALAR"},
        params={
            "n_pulses": {"description": "number of pulses per burst", "min": 1, "max": 100, "default": 5},
            "pulse_interval": {"description": "frames between pulses in a burst", "min": 1, "max": 100, "default": 6},
            "pulse_width": {"description": "frames each pulse stays high", "min": 1, "max": 20, "default": 1},
            "amplitude": {"description": "pulse amplitude", "default": 1.0},
            "loop": {"description": "auto-retrigger when burst ends", "default": True},
        })
def method_burst(out_dir: Path, seed: int, params=None):
    """Generates a burst of pulses on trigger.

    Replaces glider_stream animation mode — wire Burst.value -> inject_rate
    to create periodic glider-like injections.

    Outputs:
        value (SCALAR): pulse amplitude when active, 0 otherwise
        active (SCALAR): 1.0 during burst, 0 otherwise
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    n_pulses = int(params.get("n_pulses", 5))
    interval = int(params.get("pulse_interval", 6))
    width = int(params.get("pulse_width", 1))
    amp = float(params.get("amplitude", 1.0))
    loop = params.get("loop", True)
    if isinstance(loop, str):
        loop = loop.lower() in ("true", "1", "yes")

    # SCALAR overrides
    trigger_val = params.get("trigger")
    rate_override = params.get("rate")
    if rate_override is not None:
        interval = max(1, int(interval / max(0.01, float(rate_override))))

    # Determine if we're in a burst
    burst_duration = n_pulses * interval
    burst_start = 0

    if trigger_val is not None and trigger_val > 0:
        burst_start = frame
    elif loop:
        burst_start = (frame // burst_duration) * burst_duration

    elapsed = frame - burst_start
    if 0 <= elapsed < burst_duration:
        pulse_idx = elapsed // interval
        within_pulse = (elapsed % interval) < width
        val = amp if within_pulse else 0.0
        active = 1.0
    else:
        val = 0.0
        active = 0.0

    return {"value": float(val), "active": float(active)}
