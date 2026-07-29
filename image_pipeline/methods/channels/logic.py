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

# ── Per-node state for stateful logic ops ───────────────────────────────
_LOGIC_STATE: dict[str, dict] = {}
_LOGIC_PRUNE_COUNTER = 0

# ── Per-operation layout: which params and inputs to show for each op ──
_LOG_AB = {"show_params": ["a_default", "b_default", "true_value", "false_value", "threshold"],
           "show_inputs": ["a", "b"]}
_LOG_A  = {"show_params": ["a_default", "true_value", "false_value", "threshold"],
           "show_inputs": ["a"]}
_LOG_CTRL = {"show_params": ["true_value", "false_value", "threshold"],
             "show_inputs": ["control"]}
_LOG_AB_CTRL = {"show_params": ["a_default", "b_default", "true_value", "false_value", "threshold"],
                "show_inputs": ["a", "b", "control"]}

_LOGIC_OP_LAYOUTS: dict[str, dict] = {
    **{op: _LOG_AB for op in ["greater", "less", "equal", "not_equal",
                               "greater_equal", "less_equal", "near",
                               "and", "or", "xor", "nand", "nor",
                               "select", "gate"]},
    "not": {"show_params": ["a_default", "true_value", "false_value", "threshold"],
            "show_inputs": ["a"]},
    **{op: {"show_params": ["a_default", "b_default", "true_value", "false_value",
                            "range_lo", "range_hi"],
            "show_inputs": ["a", "b"]}
       for op in ["within_range", "outside_range"]},
    **{op: {"show_params": ["a_default", "true_value", "false_value", "threshold",
                            "event_debounce"],
            "show_inputs": ["a"]}
       for op in ["change", "rising_edge", "falling_edge", "any_edge"]},
    **{op: _LOG_AB_CTRL for op in ["hold", "toggle", "pulse"]},
    "oneshot": {"show_params": ["true_value", "false_value", "threshold"],
                "show_inputs": ["control"]},
    "repeat": {"show_params": ["true_value", "false_value", "threshold", "repeat_interval"],
               "show_inputs": ["control"]},
    "timer_delay": {"show_params": ["true_value", "false_value", "threshold", "timer_duration"],
                    "show_inputs": ["control"]},
}


@method(id="__logic__", name="Logic", category="channels",
        tags=["chop", "logic", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR", "control": "SCALAR"},
        outputs={"value": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True,
            }
        },
        signal={
            "a": "numeric",
            "b": "numeric",
            "control": "control",
            "value": "output",
        },
        params={
            "operation": {
                "description": "logic operation",
                "choices": [
                    # ── Comparisons (legacy) ──
                    "greater", "less", "equal", "not_equal",
                    # ── Boolean ──
                    "and", "or", "xor", "not", "nand", "nor",
                    # ── Expanded comparisons ──
                    "greater_equal", "less_equal",
                    "within_range", "outside_range", "near",
                    # ── Selection / routing ──
                    "select", "gate",
                    # ── Event detection ──
                    "change", "rising_edge", "falling_edge", "any_edge",
                    # ── Event generation ──
                    "hold", "toggle", "pulse", "oneshot", "repeat", "timer_delay",
                ],
                "default": "greater",
            },
            "true_value": {"description": "value when condition is True", "default": 1.0},
            "false_value": {"description": "value when condition is False", "default": 0.0},
            "threshold": {"description": "comparison / edge threshold", "default": 0.5},
            # ── Range comparisons ──
            "range_lo": {"description": "within_range/outside_range: low bound", "default": 0.3},
            "range_hi": {"description": "within_range/outside_range: high bound", "default": 0.7},
            # ── Event detection ──
            "event_debounce": {"description": "near/debounce: tolerance", "default": 0.05},
            # ── Event generation ──
            "timer_duration": {"description": "timer_delay: duration in frames", "default": 24},
            "repeat_interval": {"description": "repeat: interval in frames", "default": 12},
        },
        is_time_varying=False,
        description=(
            "Logic operations including comparisons, boolean algebra, "
            "event detection, and event generation."
        ),
        op_layouts=_LOGIC_OP_LAYOUTS,
    )
def method_logic(out_dir: Path, seed: int, params=None):
    """Logic operations on SCALAR inputs — comparisons, boolean, events.

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

    # ── Stateful op helpers ─────────────────────────────────────────────
    node_id = params.get("_node_id", "")
    global _LOGIC_PRUNE_COUNTER

    # ── Comparisons (stateless) ────────────────────────────────────────
    if op == "greater":
        val = true_val if a > b else false_val
    elif op == "less":
        val = true_val if a < b else false_val
    elif op == "equal":
        val = true_val if abs(a - b) < threshold else false_val
    elif op == "not_equal":
        val = true_val if abs(a - b) >= threshold else false_val
    elif op == "greater_equal":
        val = true_val if a >= b else false_val
    elif op == "less_equal":
        val = true_val if a <= b else false_val
    elif op == "within_range":
        lo = float(params.get("range_lo", 0.3))
        hi = float(params.get("range_hi", 0.7))
        val = true_val if lo <= a <= hi else false_val
    elif op == "outside_range":
        lo = float(params.get("range_lo", 0.3))
        hi = float(params.get("range_hi", 0.7))
        val = true_val if a < lo or a > hi else false_val
    elif op == "near":
        tol = float(params.get("event_debounce", 0.05))
        val = true_val if abs(a - b) <= tol else false_val

    # ── Boolean ────────────────────────────────────────────────────────
    # 0.0 → False, anything else → True; output true_val/false_val
    elif op == "and":
        a_bool = a > threshold
        b_bool = b > threshold
        val = true_val if a_bool and b_bool else false_val
    elif op == "or":
        a_bool = a > threshold
        b_bool = b > threshold
        val = true_val if a_bool or b_bool else false_val
    elif op == "xor":
        a_bool = a > threshold
        b_bool = b > threshold
        val = true_val if a_bool != b_bool else false_val
    elif op == "not":
        a_bool = a > threshold
        val = false_val if a_bool else true_val
    elif op == "nand":
        a_bool = a > threshold
        b_bool = b > threshold
        val = false_val if a_bool and b_bool else true_val
    elif op == "nor":
        a_bool = a > threshold
        b_bool = b > threshold
        val = true_val if not (a_bool or b_bool) else false_val

    # ── Selection / routing (stateless) ─────────────────────────────────
    elif op == "select":
        val = a if control > threshold else b
    elif op == "gate":
        val = a if control > threshold else 0.0

    # ── Event detection (stateful) ─────────────────────────────────────
    elif op == "change":
        # True when a differs from previous value by more than debounce
        debounce = float(params.get("event_debounce", 0.05))
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {"prev_a": a})
            prev = state.get("prev_a", a)
            val = true_val if abs(a - prev) > debounce else false_val
            state["prev_a"] = a
        else:
            val = false_val
    elif op == "rising_edge":
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {"prev_a": 0.0})
            prev = state.get("prev_a", 0.0)
            val = true_val if a > threshold >= prev else false_val
            state["prev_a"] = a
        else:
            val = false_val
    elif op == "falling_edge":
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {"prev_a": 0.0})
            prev = state.get("prev_a", 0.0)
            val = true_val if a <= threshold < prev else false_val
            state["prev_a"] = a
        else:
            val = false_val
    elif op == "any_edge":
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {"prev_a": 0.0})
            prev = state.get("prev_a", 0.0)
            rising = a > threshold >= prev
            falling = a <= threshold < prev
            val = true_val if rising or falling else false_val
            state["prev_a"] = a
        else:
            val = false_val

    # ── Event generation (stateful) ────────────────────────────────────
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
    elif op == "oneshot":
        # Fire true_val for exactly one frame on rising edge of control,
        # then hold false_val until next rising edge.
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {"prev_ctrl": 0.0, "armed": True})
            prev_ctrl = state["prev_ctrl"]
            rising = control > threshold >= prev_ctrl
            if rising:
                # Fire this frame
                val = true_val
                state["armed"] = False
            elif state["armed"]:
                # Still waiting for first edge
                val = false_val
                state["armed"] = False
            else:
                val = false_val
            state["prev_ctrl"] = control
        else:
            val = false_val
    elif op == "repeat":
        # Fire true_val for one frame, then every repeat_interval frames,
        # as long as control is above threshold.
        dur = int(params.get("repeat_interval", 12))
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {
                "prev_ctrl": 0.0, "elapsed": 0, "fired": False
            })
            rising = control > threshold >= state["prev_ctrl"]
            if rising:
                val = true_val
                state["elapsed"] = 0
                state["fired"] = True
            elif control > threshold:
                state["elapsed"] += 1
                if dur > 0 and state["elapsed"] >= dur:
                    val = true_val
                    state["elapsed"] = 0
                    state["fired"] = True
                else:
                    val = false_val
            else:
                val = false_val
                state["elapsed"] = 0
                state["fired"] = False
            state["prev_ctrl"] = control
        else:
            val = false_val
    elif op == "timer_delay":
        # Output true_val for `timer_duration` frames when control goes high,
        # then fall to false_val.
        dur = int(params.get("timer_duration", 24))
        if node_id:
            state = _LOGIC_STATE.setdefault(node_id, {
                "prev_ctrl": 0.0, "remaining": 0, "active": False,
            })
            rising = control > threshold >= state["prev_ctrl"]
            if rising:
                state["active"] = True
                state["remaining"] = dur
                val = true_val
            elif state["active"]:
                state["remaining"] -= 1
                if state["remaining"] > 0:
                    val = true_val
                else:
                    val = false_val
                    state["active"] = False
            else:
                val = false_val
            state["prev_ctrl"] = control
        else:
            val = false_val
    else:
        val = 0.0

    # ── Lazy prune ──────────────────────────────────────────────────────
    _LOGIC_PRUNE_COUNTER += 1
    if _LOGIC_PRUNE_COUNTER % 1000 == 0:
        frame = int(params.get("frame", 0))
        _cutoff = frame - 7200
        for _nid in list(_LOGIC_STATE):
            if _LOGIC_STATE[_nid].get("frame", 0) < _cutoff:
                del _LOGIC_STATE[_nid]

    return {"value": float(val)}
