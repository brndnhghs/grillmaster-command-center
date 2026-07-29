"""Timer — delay / countdown / stopwatch with trigger."""
from __future__ import annotations
from pathlib import Path

from ...core.registry import method


# ── Per-node state ──────────────────────────────────────────────────────
_TIMER_STATE: dict[str, dict] = {}
_TIMER_PRUNE = 0


@method(
    id="__timer__",
    name="Timer",
    category="channels",
    tags=["chop", "time", "timer", "delay", "pulse"],
    inputs={"trigger": "SCALAR", "reset": "SCALAR", "duration": "SCALAR"},
    outputs={
        "value": "SCALAR",
        "progress": "SCALAR",
        "remaining": "SCALAR",
        "done": "SCALAR",
    },
    runtime={
        "value": {"type": "numeric", "label": "Value", "observable": True},
        "progress": {"type": "output", "label": "Progress", "observable": True},
        "remaining": {"type": "output", "label": "Remaining", "observable": True},
        "done": {"type": "event", "label": "Done", "observable": True},
    },
    signal={
        "trigger": "event",
        "reset": "event",
        "duration": "numeric",
        "value": "output",
        "progress": "output",
        "remaining": "output",
        "done": "event",
    },
    params={
        "mode": {
            "description": "Timer mode",
            "choices": ["countdown", "stopwatch", "pulse_on_done"],
            "default": "countdown",
        },
        "duration": {
            "description": "Timer duration in frames",
            "min": 1,
            "max": 10000,
            "default": 60,
        },
        "auto_reset": {
            "description": "Auto-reset when timer completes (continuous cycling)",
            "default": False,
        },
        "on_value": {
            "description": "value output while timer is active",
            "default": 1.0,
        },
        "off_value": {
            "description": "value output when timer is idle/expired",
            "default": 0.0,
        },
    },
    is_time_varying=True,
    description=(
        "Configurable timer — countdown, stopwatch, or pulse-on-complete. "
        "Use trigger to start, reset to restart from zero."
    ),
)
def method_timer(out_dir: Path, seed: int, params=None):
    """Timer node — countdown, stopwatch, or pulse-on-done.

    On a rising edge on the trigger input, the timer starts counting.
    Outputs value, progress (0→1), remaining (1→0), and done (1 on expiry frame).

    Outputs:
        value (SCALAR): on_value while active, off_value when idle/expired
        progress (SCALAR): 0→1 elapsed fraction
        remaining (SCALAR): 1→0 remaining fraction
        done (SCALAR): 1.0 on the frame the timer expires, 0 otherwise
    """
    if params is None:
        params = {}

    node_id = params.get("_node_id", "")
    trigger = params.get("trigger")
    reset = params.get("reset")
    duration_override = params.get("duration")

    mode = params.get("mode", "countdown")
    duration = float(duration_override) if duration_override is not None else float(params.get("duration", 60.0))
    duration = max(1.0, duration)
    auto_reset = params.get("auto_reset", False)
    if isinstance(auto_reset, str):
        auto_reset = auto_reset.lower() in ("true", "1", "yes")
    on_val = float(params.get("on_value", 1.0))
    off_val = float(params.get("off_value", 0.0))

    # ── Frame ────────────────────────────────────────────────────────────
    frame = int(params.get("frame", 0))
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    # ── Defaults ────────────────────────────────────────────────────────
    value = off_val
    progress = 0.0
    remaining = 0.0
    done = 0.0

    if node_id:
        state = _TIMER_STATE.setdefault(node_id, {
            "prev_trigger": 0.0,
            "prev_reset": 0.0,
            "start_frame": -1,
            "prev_frame": frame,
            "completed_this_frame": False,
        })

        # ── Reset edge ───────────────────────────────────────────────────
        if reset is not None:
            rv = float(reset)
            if rv >= 0.5 > state.get("prev_reset", 0.0):
                state["start_frame"] = -1
                state["completed_this_frame"] = False
            state["prev_reset"] = rv

        # ── Trigger rising edge ──────────────────────────────────────────
        trigger_fired = False
        if trigger is not None:
            tv = float(trigger)
            if tv >= 0.5 > state.get("prev_trigger", 0.0):
                state["start_frame"] = frame
                state["completed_this_frame"] = False
                trigger_fired = True
            state["prev_trigger"] = tv

        # ── Compute state ────────────────────────────────────────────────
        sf = state["start_frame"]
        if sf >= 0:
            elapsed = frame - sf
            if elapsed >= duration:
                if auto_reset:
                    state["start_frame"] = frame
                    elapsed = 0
                    if not state.get("completed_this_frame", False):
                        done = 1.0
                        state["completed_this_frame"] = True
                else:
                    value = off_val
                    progress = 1.0
                    remaining = 0.0
                    if not state.get("completed_this_frame", False):
                        done = 1.0
                        state["completed_this_frame"] = True
            else:
                value = on_val
                progress = elapsed / duration
                remaining = 1.0 - progress
                state["completed_this_frame"] = False

        state["prev_frame"] = frame

        # ── Lazy prune ──────────────────────────────────────────────────
        global _TIMER_PRUNE
        _TIMER_PRUNE += 1
        if _TIMER_PRUNE % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_TIMER_STATE):
                if _TIMER_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _TIMER_STATE[_nid]

    return {
        "value": float(value),
        "progress": float(progress),
        "remaining": float(remaining),
        "done": float(done),
    }
