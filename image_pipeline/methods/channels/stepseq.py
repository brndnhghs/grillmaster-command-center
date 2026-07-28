"""
Step Sequencer — CHOP-like channel generator.

Outputs a value from a user-defined step pattern, advancing through
the sequence at a configurable rate or on external trigger edges.

Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
from pathlib import Path

from ...core.registry import method


# ── Per-node state ─────────────────────────────────────────────────────
_STEPSEQ_STATE: dict[str, dict] = {}
_STEPSEQ_PRUNE_COUNTER = 0


def _parse_step_values(raw: str | None) -> list[float]:
    """Parse a comma-separated string of float values into a list."""
    if not raw:
        return [0.0, 1.0, 0.5, 0.8, 0.2, 0.6, 0.3, 0.7]
    try:
        vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
        if not vals:
            return [0.0, 1.0]
        return vals
    except (ValueError, TypeError):
        return [0.0, 1.0]


def _parse_step_bools(raw: str | None, expected: int) -> list[int]:
    """Parse a comma-separated string of 0/1 flags into a list of given length."""
    if not raw:
        return [1] * expected
    try:
        vals = [1 if float(v.strip()) > 0.5 else 0
                for v in raw.split(",") if v.strip()]
        while len(vals) < expected:
            vals.append(1)
        return vals[:expected]
    except (ValueError, TypeError):
        return [1] * expected


def _next_step(current_index: int, n_steps: int, mode: str,
               direction: int) -> tuple[int, int]:
    """Return (next_index, next_direction) given current state and mode."""
    if mode == "once":
        nxt = min(current_index + 1, n_steps - 1)
        return nxt, direction
    elif mode == "pingpong":
        nxt = current_index + direction
        if nxt < 0:
            nxt = 1
            direction = 1
        elif nxt >= n_steps:
            nxt = n_steps - 2
            direction = -1
        return nxt, direction
    else:  # loop
        return (current_index + 1) % n_steps, direction


@method(id="__stepseq__", name="Step Sequencer", category="channels",
        tags=["chop", "time", "pattern", "generator"],
        inputs={"rate": "SCALAR", "reset": "SCALAR", "trigger": "SCALAR",
                "step": "SCALAR"},
        outputs={"value": "SCALAR", "index": "SCALAR", "phase": "SCALAR",
                 "triggered": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True,
            },
            "index": {
                "type": "output",
                "label": "Index",
                "observable": True,
            },
            "phase": {
                "type": "output",
                "label": "Phase",
                "observable": True,
            },
            "triggered": {
                "type": "event",
                "label": "Triggered",
                "observable": True,
            },
        },
        signal={
            "rate": "numeric",
            "reset": "event",
            "trigger": "event",
            "step": "numeric",
            "value": "output",
            "index": "output",
            "phase": "output",
            "triggered": "event",
        },
        params={
            "step_values": {
                "description": "Comma-separated step values (0-1 each). "
                               "Example: 0.1,0.3,0.5,0.7,1.0,0.7,0.5,0.3",
                "default": "0.1,0.3,0.5,0.7,1.0,0.7,0.5,0.3",
            },
            "step_active": {
                "description": "Comma-separated 0/1 flags — 0=muted (hold last value), 1=active",
                "default": "1,1,1,1,1,1,1,1",
            },
            "rate": {
                "description": "Advance rate — steps per second (Hz)",
                "default": 1.0,
            },
            "mode": {
                "description": "Wrap mode",
                "choices": ["loop", "once", "pingpong"],
                "default": "loop",
            },
            "smooth": {
                "description": "Smooth interpolation between steps (0=none, 1=full)",
                "default": 0.0,
            },
            "min": {"description": "Output minimum", "default": 0.0},
            "max": {"description": "Output maximum", "default": 1.0},
            "play": {
                "description": "Advance when 1, hold when 0",
                "default": True,
            },
        })
def method_stepseq(out_dir: Path, seed: int, params=None):
    """Step Sequencer — outputs a value from a user-defined step pattern.

    Advances through the sequence of ``step_values`` at a configurable rate
    (Hz).  When the ``trigger`` SCALAR input is wired, advancement is driven
    by rising edges on that port instead of the internal timer — use this
    to sync steps to a beat clock or LFO.

    The ``reset`` SCALAR input resets to step 0 on a rising edge.

    Wrap modes:
      - **loop**: wrap around to step 0
      - **once**: hold at last step
      - **pingpong**: reverse direction at each boundary

    Outputs:
        value (SCALAR): current step value (range-mapped)
        index (SCALAR): current step index (0-based)
        phase (SCALAR): normalized position 0→1 between start and end of sequence
        triggered (SCALAR): 1 on the frame a step advances, 0 otherwise
    """
    if params is None:
        params = {}

    frame = int(params.get("frame", 0))
    fps = float(params.get("fps", 24.0))
    node_id = params.get("_node_id", "")

    # The executor does NOT inject `frame`/`fps` for CHOP generators.
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", fps))

    # ── Parse params ───────────────────────────────────────────────────
    raw_step_values = params.get("step_values", "")
    if isinstance(raw_step_values, (list, tuple)):
        vals = [float(v) for v in raw_step_values]
    else:
        vals = _parse_step_values(str(raw_step_values) if raw_step_values else None)

    n_steps = len(vals)
    mode = params.get("mode", "loop")
    smooth = float(params.get("smooth", 0.0))
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    play = params.get("play", True)
    if isinstance(play, str):
        play = play.lower() in ("true", "1", "yes")

    # SCALAR overrides
    rate_override = params.get("rate")
    rate = float(rate_override) if rate_override is not None else float(params.get("rate", 1.0))
    rate = max(0.001, rate)

    # ── Reset handling ──────────────────────────────────────────────────
    reset_raw = params.get("reset")

    # ── Trigger edge detection ──────────────────────────────────────────
    trigger_raw = params.get("trigger")
    external_trigger: bool | None = None
    if trigger_raw is not None:
        _tv = float(trigger_raw)
        external_trigger = _tv >= 0.5

    step_override = params.get("step")
    step_size = 1
    if step_override is not None:
        step_size = max(1, int(round(float(step_override))))

    # ── State ───────────────────────────────────────────────────────────
    global _STEPSEQ_PRUNE_COUNTER
    _STEPSEQ_PRUNE_COUNTER += 1

    triggered = 0.0
    active = True      # default: all steps active (for stateless / backward compat)
    state: dict | None = None

    if node_id:
        state = _STEPSEQ_STATE.setdefault(node_id, {
            "current_index": 0,
            "direction": 1,
            "accum_frame": 0.0,
            "prev_frame": frame,
            "prev_trigger": 0.0,
            "prev_reset": 0.0,
            "prev_index": 0,
            "prev_output": 0.0,
        })

        # Accumulate elapsed frames
        _delta = max(0, frame - state["prev_frame"])
        state["prev_frame"] = frame

        # Reset on rising edge
        reset_fired = False
        prev_reset = state.get("prev_reset", 0.0)
        if reset_raw is not None:
            _rv = float(reset_raw)
            if _rv >= 0.5 > prev_reset:
                reset_fired = True
            state["prev_reset"] = _rv
        if reset_fired:
            state["current_index"] = 0
            state["direction"] = 1
            state["accum_frame"] = 0.0

        # Advance logic
        prev_index = state.get("current_index", 0)
        advanced = False

        if external_trigger is not None:
            # ── External trigger mode ──
            # Advance one step per rising edge on trigger input
            prev_trig = state.get("prev_trigger", 0.0)
            if external_trigger and not (prev_trig >= 0.5):
                new_idx, state["direction"] = _next_step(
                    state["current_index"], n_steps, mode, state["direction"])
                state["current_index"] = new_idx
                advanced = True
            state["prev_trigger"] = 1.0 if external_trigger else 0.0
        elif play:
            # ── Internal rate mode ──
            # Accumulate time and advance when enough has passed
            state["accum_frame"] += _delta
            frames_per_step = fps / rate
            if frames_per_step > 0 and state["accum_frame"] >= frames_per_step:
                steps_to_advance = int(state["accum_frame"] / frames_per_step)
                state["accum_frame"] -= steps_to_advance * frames_per_step
                for _ in range(steps_to_advance):
                    new_idx, state["direction"] = _next_step(
                        state["current_index"], n_steps, mode, state["direction"])
                    state["current_index"] = new_idx
                    advanced = True
        # else: play=False, hold position

        current_index = state["current_index"]
        if advanced:
            triggered = 1.0
        state["prev_index"] = current_index

        # Parse step_active flags
        step_active = _parse_step_bools(params.get("step_active"), n_steps)
        active = step_active[current_index] if current_index < len(step_active) else 1

        # Suppress triggered when advancing to an inactive step
        if triggered and not active:
            triggered = 0.0

        # Lazy prune
        if _STEPSEQ_PRUNE_COUNTER % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_STEPSEQ_STATE):
                if _STEPSEQ_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _STEPSEQ_STATE[_nid]

    else:
        # ── Stateless fallback (no _node_id) ──
        t_seconds = frame / max(1.0, fps)
        # When trigger is wired in stateless context, use it per-frame
        if external_trigger is not None:
            current_index = 0 if not external_trigger else 1
        else:
            raw_idx = int((t_seconds * rate) % n_steps)
            if mode == "once":
                current_index = min(raw_idx, n_steps - 1)
            elif mode == "pingpong":
                cycle = raw_idx % (n_steps * 2)
                current_index = (cycle if cycle < n_steps
                                 else n_steps * 2 - cycle - 1)
            else:
                current_index = raw_idx % n_steps

    # ── Compute output value ────────────────────────────────────────────
    current_index = max(0, min(current_index, n_steps - 1))

    # Step-active hold logic: when the current step is muted, keep the
    # last output value instead of jumping to 0.
    if node_id and not active:
        value = state.get("prev_output", 0.0)
    else:
        raw_val = vals[current_index]
        raw_val = max(0.0, min(1.0, raw_val))

        # Smooth interpolation to next step
        if smooth > 0.0 and n_steps > 1:
            next_idx = (current_index + 1) % n_steps
            t = smooth  # scalar blend
            blended = raw_val * (1 - t) + vals[next_idx] * t
        else:
            blended = raw_val

        # Map from [0, 1] to [min, max]
        _range = max_val - min_val
        if _range == 0:
            value = min_val
        else:
            value = min_val + blended * _range

    if node_id:
        state["prev_output"] = value

    # Phase = normalized position in the sequence
    if mode == "once" and n_steps > 1:
        phase = current_index / (n_steps - 1)
    else:
        phase = current_index / n_steps if n_steps > 0 else 0.0

    # ── Lazy prune (stateless path too) ─────────────────────────────────
    if not node_id and _STEPSEQ_PRUNE_COUNTER % 1000 == 0:
        for _nid in list(_STEPSEQ_STATE):
            if _STEPSEQ_STATE[_nid].get("prev_frame", 0) < (frame - 7200):
                del _STEPSEQ_STATE[_nid]

    return {
        "value": float(value),
        "index": float(current_index),
        "phase": float(phase),
        "triggered": float(triggered),
    }
