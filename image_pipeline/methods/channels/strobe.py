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

@method(id="__strobe__", name="Strobe", category="channels",
        tags=["chop", "time", "gate", "generator"],
        inputs={"rate": "SCALAR", "duty_cycle": "SCALAR"},
        outputs={"value": "SCALAR", "trigger": "SCALAR"},
        params={
            "rate": {"description": "strobe rate in Hz", "default": 2.0},
            "duty_cycle": {"description": "fraction of cycle that is on (0-1)", "default": 0.5},
            "on_value": {"description": "value when gate is open", "default": 1.0},
            "off_value": {"description": "value when gate is closed", "default": 0.0},
        })
def method_strobe(out_dir: Path, seed: int, params=None):
    """Periodic on/off gate — like a square wave with adjustable duty cycle.

    Replaces freeze_frame, spark, and pulse animation modes.
    Wire Strobe.value -> inject_rate for periodic life injection.
    Wire Strobe.value -> speed for freeze-frame strobe effect.

    Outputs:
        value (SCALAR): on_value when gate open, off_value when closed
        trigger (SCALAR): 1.0 on rising edge, 0 otherwise
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    rate = float(params.get("rate", 2.0))
    duty = float(params.get("duty_cycle", 0.5))
    on_val = float(params.get("on_value", 1.0))
    off_val = float(params.get("off_value", 0.0))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject a `time` for CHOP generators. Derive the live phase from
    # the Timeline's global_frame so the strobe advances every rendered frame
    # instead of staying pinned at t=0 (which froze driver-driven graphs and
    # culled them as static — see __counter__ / __lfo__ / __noise1d__).
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
    duty_override = params.get("duty_cycle")
    if duty_override is not None:
        duty = float(duty_override)

    duty = max(0.01, min(0.99, duty))
    phase = (t * rate) % 1.0
    gate_open = phase < duty
    val = on_val if gate_open else off_val

    # Trigger on rising edge
    prev_phase = ((t - 1.0 / 24.0) * rate) % 1.0
    trigger = 1.0 if (prev_phase >= duty and gate_open) else 0.0

    return {"value": float(val), "trigger": float(trigger)}
