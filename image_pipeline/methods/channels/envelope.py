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

@method(id="__envelope__", name="Envelope", category="channels",
        tags=["chop", "time", "adsr", "generator"],
        inputs={"trigger": "SCALAR", "gate": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "attack": {"description": "attack time in frames", "min": 0, "max": 1000, "default": 10},
            "decay": {"description": "decay time in frames", "min": 0, "max": 1000, "default": 20},
            "sustain": {"description": "sustain level 0-1", "default": 0.7},
            "release": {"description": "release time in frames", "min": 0, "max": 1000, "default": 50},
            "sustain_level": {"description": "sustain level (alias)", "default": 0.7},
            "loop": {"description": "loop the envelope", "default": False},
        })
def method_envelope(out_dir: Path, seed: int, params=None):
    """ADSR envelope generator — triggered by a SCALAR input.

    When trigger goes from 0->1, the envelope starts its attack phase.
    When gate goes to 0, the envelope enters release phase.

    Outputs:
        value (SCALAR): envelope amplitude 0->1
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    attack = int(params.get("attack", 10))
    decay = int(params.get("decay", 20))
    sustain = float(params.get("sustain", 0.7))
    release = int(params.get("release", 50))
    loop = params.get("loop", False)
    if isinstance(loop, str):
        loop = loop.lower() in ("true", "1", "yes")

    # Use sustain param, fall back to sustain_level
    sustain = float(params.get("sustain_level", sustain))

    # SCALAR overrides
    trigger_val = params.get("trigger")
    gate_val = params.get("gate")

    # Derive the live frame from the injected Timeline (see Counter for why).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    # Simple model: trigger starts attack, gate holds sustain
    if trigger_val is not None and trigger_val > 0:
        trigger_frame = frame
    else:
        trigger_frame = 0

    if gate_val is not None and gate_val <= 0:
        # Release phase
        elapsed = frame - trigger_frame - attack - decay
        if elapsed < 0:
            elapsed = 0
        if elapsed >= release:
            val = 0.0
        else:
            val = sustain * (1 - elapsed / release)
    else:
        # Attack -> Decay -> Sustain
        elapsed = frame - trigger_frame
        if elapsed < attack:
            val = elapsed / attack
        elif elapsed < attack + decay:
            val = 1 - (1 - sustain) * (elapsed - attack) / decay
        else:
            val = sustain

    if loop and val <= 0:
        val = 0.0  # Hold at zero until next trigger

    return {"value": float(val)}
