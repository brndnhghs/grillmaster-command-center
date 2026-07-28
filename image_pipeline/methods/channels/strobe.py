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
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            },
            "trigger": {
                "type": "event",
                "label": "Trigger",
                "observable": True
            }
        },
        signal={
            "rate": "numeric",
            "duty_cycle": "numeric",
            "value": "output",
            "trigger": "event"
        },
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

    rate = float(params.get("rate", 2.0))
    duty = float(params.get("duty_cycle", 0.5))
    on_val = float(params.get("on_value", 1.0))
    off_val = float(params.get("off_value", 0.0))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject `frame` nor `fps` for CHOP generators.  Derive the live
    # frame from the Timeline's global_frame so the strobe advances every
    # rendered frame.
    frame = int(params.get("frame", 0))
    fps = float(params.get("fps", 24.0))
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", fps))

    duty = max(0.01, min(0.99, duty))

    # Phase advances by rate cycles per second of elapsed real time.
    # t_seconds = frame / fps gives the correct seconds-based time base,
    # so rate=2Hz produces exactly 2 full cycles per second.
    t_seconds = frame / max(1.0, fps)
    phase = (t_seconds * rate) % 1.0
    gate_open = phase < duty
    val = on_val if gate_open else off_val

    # Trigger on rising edge: previous frame was before the gate, now inside
    prev_phase = ((t_seconds - 1.0 / max(1.0, fps)) * rate) % 1.0
    trigger = 1.0 if (prev_phase >= duty and gate_open) else 0.0

    return {"value": float(val), "trigger": float(trigger)}
