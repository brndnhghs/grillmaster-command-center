"""State — universal signal state: hold, accumulate, running average, min/max tracker."""
from __future__ import annotations
from pathlib import Path

from ...core.registry import method


# ── Per-node state ──────────────────────────────────────────────────────
_STATE_STATE: dict[str, dict] = {}
_STATE_PRUNE = 0


@method(
    id="__state__",
    name="State",
    category="channels",
    tags=["chop", "state", "memory", "accumulator", "hold"],
    inputs={"input": "SCALAR", "trigger": "SCALAR", "reset": "SCALAR"},
    outputs={
        "value": "SCALAR",
        "min": "SCALAR",
        "max": "SCALAR",
        "count": "SCALAR",
    },
    runtime={
        "value": {"type": "numeric", "label": "Value", "observable": True},
        "min": {"type": "numeric", "label": "Min", "observable": True},
        "max": {"type": "numeric", "label": "Max", "observable": True},
        "count": {"type": "numeric", "label": "Count", "observable": True},
    },
    signal={
        "input": "numeric",
        "trigger": "event",
        "reset": "event",
        "value": "output",
        "min": "output",
        "max": "output",
        "count": "output",
    },
    params={
        "mode": {
            "description": "State mode",
            "choices": [
                "hold",
                "accumulate",
                "running_avg",
                "track_min",
                "track_max",
                "track_both",
            ],
            "default": "hold",
        },
        "init_value": {
            "description": "Initial / reset value",
            "default": 0.0,
        },
        "trigger_thresh": {
            "description": "Rising-edge threshold for trigger input",
            "default": 0.5,
        },
        "clamp_min": {
            "description": "Accumulator / output minimum clamp",
            "default": None,
        },
        "clamp_max": {
            "description": "Accumulator / output maximum clamp",
            "default": None,
        },
    },
    is_time_varying=True,
    description=(
        "Universal stateful signal processor.  Hold samples and keeps them, "
        "accumulate adds each input, running_avg tracks a running mean, "
        "track_min/max/both track extrema."
    ),
)
def method_state(out_dir: Path, seed: int, params=None):
    """Stateful signal processor — hold, accumulate, running average, extrema.

    Use ``trigger`` to gate updates (rising edge triggers a sample/accumulate),
    and ``reset`` to restore to initial condition.

    Outputs:
        value (SCALAR): primary output (held value, accumulator, running avg, min, or max)
        min (SCALAR): tracked minimum (only meaningful in track_both mode)
        max (SCALAR): tracked maximum (only meaningful in track_both mode)
        count (SCALAR): number of samples accumulated (running_avg mode)
    """
    if params is None:
        params = {}

    node_id = params.get("_node_id", "")
    input_val = float(params.get("input", 0.0))
    mode = params.get("mode", "hold")
    init_value = float(params.get("init_value", 0.0))
    thresh = float(params.get("trigger_thresh", 0.5))

    clamp_min_raw = params.get("clamp_min")
    clamp_min = float(clamp_min_raw) if clamp_min_raw is not None else None
    clamp_max_raw = params.get("clamp_max")
    clamp_max = float(clamp_max_raw) if clamp_max_raw is not None else None

    trigger_raw = params.get("trigger")
    reset_raw = params.get("reset")

    frame = int(params.get("frame", 0))
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    # ── Default outputs ─────────────────────────────────────────────────
    value = init_value
    min_out = init_value
    max_out = init_value
    count_out = 0.0

    def _clamp(v: float) -> float:
        if clamp_min is not None:
            v = max(clamp_min, v)
        if clamp_max is not None:
            v = min(clamp_max, v)
        return v

    if node_id:
        state = _STATE_STATE.setdefault(node_id, {
            "held_value": init_value,
            "accum_value": init_value,
            "running_sum": 0.0,
            "running_count": 0,
            "track_min": float("inf"),
            "track_max": float("-inf"),
            "prev_trigger": 0.0,
            "prev_reset": 0.0,
            "prev_frame": frame,
        })

        # Detect timeline regression
        if frame < state.get("prev_frame", frame):
            state["held_value"] = init_value
            state["accum_value"] = init_value
            state["running_sum"] = 0.0
            state["running_count"] = 0
            state["track_min"] = float("inf")
            state["track_max"] = float("-inf")
        state["prev_frame"] = frame

        # ── Reset edge ───────────────────────────────────────────────────
        if reset_raw is not None:
            rv = float(reset_raw)
            if rv >= 0.5 > state.get("prev_reset", 0.0):
                state["held_value"] = init_value
                state["accum_value"] = init_value
                state["running_sum"] = 0.0
                state["running_count"] = 0
                state["track_min"] = float("inf")
                state["track_max"] = float("-inf")
            state["prev_reset"] = rv

        # ── Trigger edge / unconditional update ──────────────────────────
        triggered = False
        if trigger_raw is not None:
            tv = float(trigger_raw)
            triggered = tv >= thresh > state.get("prev_trigger", 0.0)
            state["prev_trigger"] = tv
        else:
            triggered = True  # No trigger wired = continuous update

        if triggered:
            if mode == "hold":
                state["held_value"] = input_val
            elif mode == "accumulate":
                state["accum_value"] = _clamp(state["accum_value"] + input_val)
            elif mode == "running_avg":
                state["running_sum"] += input_val
                state["running_count"] += 1
            elif mode == "track_min":
                if input_val < state["track_min"]:
                    state["track_min"] = input_val
            elif mode == "track_max":
                if input_val > state["track_max"]:
                    state["track_max"] = input_val
            elif mode == "track_both":
                if input_val < state["track_min"]:
                    state["track_min"] = input_val
                if input_val > state["track_max"]:
                    state["track_max"] = input_val

        # ── Build outputs ────────────────────────────────────────────────
        if mode == "hold":
            value = state["held_value"]
        elif mode == "accumulate":
            value = state["accum_value"]
        elif mode == "running_avg":
            if state["running_count"] > 0:
                value = state["running_sum"] / state["running_count"]
            else:
                value = init_value
            count_out = float(state["running_count"])
        elif mode == "track_min":
            value = state["track_min"] if state["track_min"] != float("inf") else init_value
        elif mode == "track_max":
            value = state["track_max"] if state["track_max"] != float("-inf") else init_value
        elif mode == "track_both":
            value = state["track_min"] if state["track_min"] != float("inf") else init_value

        min_out = state["track_min"] if state["track_min"] != float("inf") else init_value
        max_out = state["track_max"] if state["track_max"] != float("-inf") else init_value

        # ── Lazy prune ──────────────────────────────────────────────────
        global _STATE_PRUNE
        _STATE_PRUNE += 1
        if _STATE_PRUNE % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_STATE_STATE):
                if _STATE_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _STATE_STATE[_nid]

    return {
        "value": float(value),
        "min": float(min_out),
        "max": float(max_out),
        "count": float(count_out),
    }
