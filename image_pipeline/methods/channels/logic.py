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

@method(id="__logic__", name="Logic", category="channels",
        tags=["chop", "logic", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR", "control": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "operation": {"description": "logic operation",
                          "choices": ["greater", "less", "equal", "not_equal",
                                      "select", "gate", "hold", "toggle", "pulse"],
                          "default": "greater"},
            "true_value": {"description": "value when condition is true", "default": 1.0},
            "false_value": {"description": "value when condition is false", "default": 0.0},
            "threshold": {"description": "comparison threshold", "default": 0.5},
        })
def method_logic(out_dir: Path, seed: int, params=None):
    """Logic operations — comparison, selection, gating.

    Accepts wired SCALAR inputs A, B, and Control.

    Outputs:
        value (SCALAR): result of the logic operation
    """
    if params is None:
        params = {}
    seed_all(seed)

    op = params.get("operation", "greater")
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    control = float(params.get("control", 0.0))
    true_val = float(params.get("true_value", 1.0))
    false_val = float(params.get("false_value", 0.0))
    threshold = float(params.get("threshold", 0.5))

    # SCALAR overrides
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)
    control_wired = params.get("control")
    if control_wired is not None:
        control = float(control_wired)

    if op == "greater":
        val = true_val if a > b else false_val
    elif op == "less":
        val = true_val if a < b else false_val
    elif op == "equal":
        val = true_val if abs(a - b) < threshold else false_val
    elif op == "not_equal":
        val = true_val if abs(a - b) >= threshold else false_val
    elif op == "select":
        val = a if control > threshold else b
    elif op == "gate":
        val = a if control > threshold else 0.0
    elif op == "hold":
        # Hold last value when control is above threshold
        val = a if control > threshold else float(params.get("_held", a))
        params["_held"] = val
    elif op == "toggle":
        # Toggle between true_val and false_val on each control pulse
        prev = float(params.get("_prev_control", 0.0))
        state = float(params.get("_toggle_state", false_val))
        if control > threshold and prev <= threshold:
            state = true_val if state == false_val else false_val
        params["_prev_control"] = control
        params["_toggle_state"] = state
        val = state
    elif op == "pulse":
        # Output true_val for one frame when control crosses threshold
        prev = float(params.get("_prev_pulse", 0.0))
        val = true_val if control > threshold and prev <= threshold else false_val
        params["_prev_pulse"] = control
    else:
        val = 0.0

    return {"value": float(val)}
