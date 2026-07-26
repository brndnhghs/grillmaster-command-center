"""
CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

# Per-node trigger latch for hysteresis
_COUNTER_STATE: dict[str, dict] = {}


@method(id="__counter__", name="Counter", category="channels",
        tags=["chop", "time", "integer", "generator"],
        inputs={"reset": "SCALAR", "step": "SCALAR", "signal": "SCALAR"},
        outputs={"value": "SCALAR", "phase": "SCALAR", "triggered": "SCALAR"},
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
            },
            "triggered": {
                "type": "numeric",
                "label": "Triggered",
                "observable": True
            }
        },
        signal={
            "reset": "control",
            "step": "numeric",
            "signal": "numeric",
            "value": "output",
            "phase": "output",
            "triggered": "output"
        },
        params={
            "start": {"description": "counter start value", "default": 0},
            "end": {"description": "counter end value (inclusive)", "default": 100},
            "step_size": {"description": "increment per frame", "default": 1},
            "mode": {"description": "counter mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
            "threshup": {"description": "Trigger Threshold", "default": 0.5},
            "threshdown": {"description": "Release Threshold", "default": 0.3},
            "release": {"description": "If on, use trigger threshold also as release", "default": True},
        })
def method_counter(out_dir: Path, seed: int, params=None):
    """Integer counter that advances per frame.

    Counts from start to end, then wraps or reverses based on mode.
    Emits triggered=1 when signal input exceeds threshup.

    Outputs:
        value (SCALAR): current count
        phase (SCALAR): normalized position 0->1 between start and end
        triggered (SCALAR): 1 if signal input exceeds trigger threshold
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

    # Derive the live frame from the injected Timeline so the counter
    # advances on every rendered frame instead of staying pinned at 0.
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    step_override = params.get("step")
    if step_override is not None:
        step_size = max(1, int(round(step_override)))

    total = end - start
    if total <= 0:
        return {"value": float(start), "phase": 0.0, "triggered": 0.0}

    raw = frame * step_size
    if mode == "once":
        val = min(start + raw, end)
    elif mode == "pingpong":
        cycle = raw % (total * 2)
        val = start + (cycle if cycle <= total else total * 2 - cycle)
    else:  # loop
        val = start + (raw % (total + 1))

    phase = (val - start) / total if total > 0 else 0.0

    # Schmitt-trigger hysteresis for the triggered output
    # Latch state keyed by _node_id so threshold crossings are stable
    signal_val = params.get("signal")
    threshup = float(params.get("threshup", 0.5))
    threshdown = float(params.get("threshdown", 0.3))
    release_shared = str(params.get("release", "True")).lower() in ("true", "1", "yes")
    eff_thr_dn = threshup if release_shared else threshdown
    node_id = params.get("_node_id", "")

    if signal_val is not None and float(signal_val) > threshup:
        triggered = 1.0
    elif signal_val is not None and float(signal_val) < eff_thr_dn:
        triggered = 0.0
    elif signal_val is not None and node_id and node_id in _COUNTER_STATE:
        # Within the hysteresis window — hold the last state
        triggered = 1.0 if _COUNTER_STATE[node_id].get("triggered", False) else 0.0
    else:
        # Unwired or no signal: green while counting (val > start)
        triggered = 1.0 if val > start else 0.0

    # Persist latch for next frame
    if node_id:
        _COUNTER_STATE[node_id] = {"triggered": triggered > 0.5, "frame": frame}

    return {"value": float(val), "phase": float(phase), "triggered": float(triggered)}
