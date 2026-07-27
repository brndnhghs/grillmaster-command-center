"""
CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
from pathlib import Path
from ...core.registry import method
from ...core.utils import seed_all

# Per-node state for trigger edge detection and count accumulation
_COUNTER_STATE: dict[str, dict] = {}
_COUNTER_PRUNE_COUNTER = 0


@method(id="__counter__", name="Counter", category="channels",
        tags=["chop", "time", "integer", "generator"],
        inputs={"reset": "SCALAR", "step": "SCALAR", "signal": "SCALAR",
                "trigger": "SCALAR"},
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
            "trigger": "event",
            "value": "output",
            "phase": "output",
            "triggered": "output"
        },
        params={
            "start": {"description": "counter start value", "default": 0},
            "end": {"description": "counter end value (inclusive)", "default": 100},
            "step_size": {"description": "increment per trigger edge", "default": 1},
            "advance_mode": {
                "description": "How the counter advances",
                "choices": ["trigger", "free"],
                "default": "trigger",
            },
            "mode": {"description": "counter wrap mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
            "threshup": {"description": "Trigger Threshold", "default": 0.5},
            "threshdown": {"description": "Release Threshold", "default": 0.3},
            "release": {"description": "If on, use trigger threshold also as release", "default": True},
        })
def method_counter(out_dir: Path, seed: int, params=None):
    """Integer counter with configurable advance mode.

    Two advance modes (``advance_mode`` param):

      - **trigger** (default): the counter increments by ``step_size`` on
        each rising edge (>0.5 transition) of the ``trigger`` SCALAR input.
        When unwired the counter holds at ``start`` — it does NOT advance
        by frame.  The count is accumulated statefully.

      - **free**: backward-compatible frame-based counting.  The counter
        advances by ``step_size`` every rendered frame, computed from
        ``frame * step_size``.  No trigger input needed.

    ``trigger`` (SCALAR input, event) drives the count increment, while
    ``signal`` (SCALAR input) drives the Schmitt-trigger ``triggered`` output
    — they are independent functions.

    Wrap modes:
      - once: clamp at ``end``
      - loop: wrap around to ``start``
      - pingpong: reverse direction at each boundary

    Outputs:
        value (SCALAR): current count
        phase (SCALAR): normalized position 0→1 between start and end
        triggered (SCALAR): 1 when signal input exceeds threshup (Schmitt)
    """
    if params is None:
        params = {}
    seed_all(seed)

    start = int(params.get("start", 0))
    end = int(params.get("end", 100))
    step_size = int(params.get("step_size", 1))
    mode = params.get("mode", "loop")

    if step_size < 1:
        step_size = 1

    total = end - start
    if total <= 0:
        return {"value": float(start), "phase": 0.0, "triggered": 0.0}

    advance_mode = params.get("advance_mode", "trigger")

    # ── Detect whether trigger port is wired ────────────────────────────
    trigger_input: float | None = None
    if "trigger" in params:
        _tv = params["trigger"]
        if _tv is not None:
            trigger_input = float(_tv)

    # ── SCALAR overrides (shared) ───────────────────────────────────────
    reset_val: int | None = None
    _rv = params.get("reset")
    if _rv is not None:
        reset_val = int(round(float(_rv)))

    step_override = params.get("step")
    if step_override is not None:
        step_size = max(1, int(round(float(step_override))))

    node_id = params.get("_node_id", "")

    # ── Compute value ───────────────────────────────────────────────────
    if advance_mode == "free":
        # ── Free-run on frame (backward compat) ─────────────────────────
        frame = int(params.get("frame", 0))
        if "frame" not in params:
            _tl = params.get("_timeline")
            if _tl is not None:
                frame = int(getattr(_tl, "global_frame", 0))

        raw = frame * step_size

        if reset_val is not None:
            raw = reset_val

        if mode == "once":
            raw_val = min(start + raw, end)
        elif mode == "pingpong":
            cycle = raw % (total * 2)
            raw_val = start + (cycle if cycle <= total else total * 2 - cycle)
        else:  # loop
            raw_val = start + (raw % (total + 1))

    elif trigger_input is not None:
        # ── Trigger-driven (edge-detected, stateful) ────────────────────
        # Always create a local state dict even without node_id, so the
        # code path doesn't KeyError on state["current_value"].
        state = _COUNTER_STATE.setdefault(node_id, {
            "current_value": float(start),
            "prev_trigger": 0.0,
            "frame": 0,
        }) if node_id else {
            "current_value": float(start),
            "prev_trigger": 0.0,
        }

        # Reset takes precedence (level-sensitive, like original)
        if reset_val is not None:
            state["current_value"] = float(reset_val)
        else:
            # Rising-edge detection on trigger
            prev_trigger = state.get("prev_trigger", 0.0)
            if trigger_input >= 0.5 > prev_trigger:
                state["current_value"] += step_size

        state["prev_trigger"] = trigger_input
        if node_id:
            state["frame"] = int(params.get("frame", state.get("frame", 0)))

        raw_val = state["current_value"]
        count = raw_val - start

        # Apply mode and sync state so next edge continues from wrapped value
        if mode == "once":
            raw_val = min(max(raw_val, start), end)
        elif mode == "pingpong":
            cycle = int(count) % (total * 2)
            raw_val = start + (cycle if cycle <= total else total * 2 - cycle)
        else:  # loop
            raw_val = start + (int(count) % (total + 1))
        if node_id:
            state["current_value"] = float(raw_val)

    else:
        # ── Trigger mode, unwired — hold at start ───────────────────────
        raw_val = float(start)

    val = raw_val
    phase = (val - start) / total if total > 0 else 0.0

    # ── Schmitt-trigger hysteresis ──────────────────────────────────────
    signal_val = params.get("signal")
    threshup = float(params.get("threshup", 0.5))
    threshdown = float(params.get("threshdown", 0.3))
    release_shared = str(params.get("release", "True")).lower() in ("true", "1", "yes")
    eff_thr_dn = threshup if release_shared else threshdown
    prev_state = _COUNTER_STATE.get(node_id, {}) if node_id else {}

    if signal_val is not None and float(signal_val) > threshup:
        triggered = 1.0
    elif signal_val is not None and float(signal_val) < eff_thr_dn:
        triggered = 0.0
    elif signal_val is not None and node_id and node_id in _COUNTER_STATE:
        # Within the hysteresis window — hold the last state
        triggered = 1.0 if prev_state.get("triggered", False) else 0.0
    else:
        # Unwired or no signal: not triggered
        triggered = 0.0

    if node_id:
        _COUNTER_STATE.setdefault(node_id, {})["triggered"] = triggered > 0.5

    # ── Lazy prune ──────────────────────────────────────────────────────
    global _COUNTER_PRUNE_COUNTER
    _COUNTER_PRUNE_COUNTER += 1
    if _COUNTER_PRUNE_COUNTER % 1000 == 0:
        _cutoff = int(params.get("frame", 0)) - 7200
        for _nid in list(_COUNTER_STATE):
            if _COUNTER_STATE[_nid].get("frame", 0) < _cutoff:
                del _COUNTER_STATE[_nid]

    return {"value": float(val), "phase": float(phase), "triggered": float(triggered)}
