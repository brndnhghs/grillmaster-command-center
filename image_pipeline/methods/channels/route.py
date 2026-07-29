"""Route — signal routing: merge, split, select, switch, fan-out, fan-in."""
from __future__ import annotations
from pathlib import Path

from ...core.registry import method


@method(
    id="__route__",
    name="Route",
    category="channels",
    tags=["chop", "route", "switch", "merge", "select", "mux"],
    inputs={
        "a": "SCALAR",
        "b": "SCALAR",
        "c": "SCALAR",
        "d": "SCALAR",
        "control": "SCALAR",
    },
    outputs={
        "out": "SCALAR",
        "a_out": "SCALAR",
        "b_out": "SCALAR",
    },
    runtime={
        "out": {"type": "numeric", "label": "Out", "observable": True},
        "a_out": {"type": "output", "label": "A Out", "observable": True},
        "b_out": {"type": "output", "label": "B Out", "observable": True},
    },
    signal={
        "a": "numeric",
        "b": "numeric",
        "c": "numeric",
        "d": "numeric",
        "control": "control",
        "out": "output",
        "a_out": "output",
        "b_out": "output",
    },
    params={
        "mode": {
            "description": "Routing mode",
            "choices": [
                "select",      # Select one of A/B/C/D by control level or index
                "switch",      # Toggle between A and B on each control pulse
                "merge",       # Sum/avg A and B
                "fan_out",     # Route A to both A_out and B_out
                "gate",        # Pass A through when control is high, 0 when low
                "priority",    # Output the first non-zero / above-threshold input
            ],
            "default": "select",
        },
        "select_index": {
            "description": "Fixed index for select mode (0=A, 1=B, 2=C, 3=D)",
            "min": 0,
            "max": 3,
            "default": 0,
        },
        "threshold": {
            "description": "Control / switch threshold",
            "default": 0.5,
        },
        "merge_mode": {
            "description": "Merge mode — sum or average",
            "choices": ["sum", "average"],
            "default": "average",
        },
        "priority_threshold": {
            "description": "Minimum value to consider an input 'present' in priority mode",
            "default": 0.01,
        },
    },
    is_time_varying=False,
    description=(
        "Signal router — select, switch, merge, or fan-out SCALAR inputs. "
        "Use select to choose one of four inputs by index or control voltage; "
        "switch to toggle between inputs; merge to sum or average; fan-out to "
        "duplicate a signal; gate to pass-through conditionally."
    ),
    op_layouts={
        "select":   {"show_params": ["select_index", "threshold"], "show_inputs": ["a", "b", "c", "d", "control"]},
        "switch":   {"show_params": ["threshold"], "show_inputs": ["a", "b", "control"]},
        "merge":    {"show_params": ["merge_mode"], "show_inputs": ["a", "b"]},
        "fan_out":  {"show_params": [], "show_inputs": ["a"]},
        "gate":     {"show_params": ["threshold"], "show_inputs": ["a", "control"]},
        "priority": {"show_params": ["priority_threshold"], "show_inputs": ["a", "b", "c", "d"]},
    },
)
def method_route(out_dir: Path, seed: int, params=None):
    """Signal routing — select, switch, merge, fan-out, gate, priority.

    Four SCALAR inputs (A, B, C, D) and one control input.
    Outputs out (the primary output), a_out, and b_out (for fan-out).

    Outputs:
        out (SCALAR): primary routed value
        a_out (SCALAR): copy of A (useful in fan-out mode)
        b_out (SCALAR): copy of B (useful in fan-out mode)
    """
    if params is None:
        params = {}

    mode = params.get("mode", "select")
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 0.0))
    d = float(params.get("d", 0.0))
    control = float(params.get("control", 0.0))
    thresh = float(params.get("threshold", 0.5))

    # ── Defaults ────────────────────────────────────────────────────────
    out = 0.0
    a_out = a
    b_out = b

    if mode == "select":
        # Select by control voltage or fixed index
        if control > thresh:
            # Map control level to which input: 0-1 → A/B/C/D
            idx = int((control - thresh) / (1.0 - thresh) * 4) if (1.0 - thresh) > 0 else 0
            idx = max(0, min(3, idx))
        else:
            idx = int(params.get("select_index", 0))
            idx = max(0, min(3, idx))
        inputs = [a, b, c, d]
        out = inputs[idx]

    elif mode == "switch":
        # Toggle between A and B on each control rising edge
        prev_ctrl = float(params.get("_prev_switch_ctrl", 0.0))
        state = float(params.get("_switch_state", a))
        if control > thresh and prev_ctrl <= thresh:
            state = b if abs(state - a) < 1e-6 else a
        params["_prev_switch_ctrl"] = control
        params["_switch_state"] = state
        out = state

    elif mode == "merge":
        merge_mode = params.get("merge_mode", "average")
        if merge_mode == "average":
            out = (a + b) * 0.5
        else:
            out = a + b

    elif mode == "fan_out":
        # A goes to both out and a_out/b_out
        out = a
        a_out = a
        b_out = a

    elif mode == "gate":
        out = a if control > thresh else 0.0

    elif mode == "priority":
        pthresh = float(params.get("priority_threshold", 0.01))
        for val in (a, b, c, d):
            if abs(val) > pthresh:
                out = val
                break
        else:
            out = 0.0

    return {
        "out": float(out),
        "a_out": float(a_out),
        "b_out": float(b_out),
    }
