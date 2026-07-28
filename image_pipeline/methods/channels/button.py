"""Button node — UI-interactive trigger source.

When the user clicks the button widget on the node body, the value output goes
to 1.0. In momentary mode it returns to 0.0 on release; in toggle mode each
click toggles between 0.0 and 1.0.

A ``trigger`` SCALAR input fires the button externally: in momentary mode a
rising edge on trigger produces a one-frame 1.0 pulse; in toggle mode a rising
edge toggles the output state.

Outputs:
    value (SCALAR): 1.0 when pressed/on, 0.0 when released/off
"""

from __future__ import annotations
from pathlib import Path
from ...core.registry import method

# Per-node state for trigger edge detection and toggle tracking
_BUTTON_STATE: dict[str, dict] = {}
_BUTTON_PRUNE_COUNTER = 0


@method(id="__button__", name="Button", category="channels",
        tags=["chop", "trigger", "control", "source"],
        inputs={"trigger": "SCALAR"},
        outputs={"value": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True,
            },
        },
        signal={
            "trigger": "event",
            "value": "output",
        },
        params={
            "mode": {
                "description": "momentary (1 while held) / toggle (click on/off)",
                "choices": ["momentary", "toggle"],
                "default": "momentary",
            },
            "button_pressed": {
                "hidden": True,
                "default": False,
                "description": "Button state — set by the UI button widget",
            },
        },
        is_time_varying=True)
def method_button(out_dir: Path, seed: int, params=None):
    """UI-interactive button — manual trigger source.

    The ``trigger`` SCALAR input provides external control:
      - **momentary** mode: a rising edge (0→1) on trigger produces a single
        frame of 1.0 output, then self-clears.
      - **toggle** mode: a rising edge on trigger toggles the output state
        (the same as clicking the UI button).

    Two separate sources (UI button widget and external trigger wire) can
    fire independently — either one can set the output to 1.0 in momentary
    mode, or toggle the state in toggle mode.

    Outputs:
        value (SCALAR): 1.0 when pressed/on, 0.0 when released/off
    """
    if params is None:
        params = {}

    node_id = params.get("_node_id", "")
    mode = params.get("mode", "momentary")

    # ── UI button state ─────────────────────────────────────────
    pressed = params.get("button_pressed", False)
    if isinstance(pressed, str):
        pressed = pressed.lower() in ("true", "1", "yes", "on")

    # ── Trigger rising edge detection ────────────────────────────
    trigger_val = params.get("trigger")
    trigger_rising = False
    if trigger_val is not None and node_id:
        try:
            tv = float(trigger_val)
            prev = _BUTTON_STATE.get(node_id, {}).get("prev_trigger", 0.0)
            trigger_rising = tv >= 0.5 > prev
            _BUTTON_STATE.setdefault(node_id, {})["prev_trigger"] = tv
        except (ValueError, TypeError):
            pass

    # ── Compute output ───────────────────────────────────────────
    val = 0.0
    if mode == "momentary":
        # UI hold OR trigger pulse (single frame — self-clearing next frame)
        val = 1.0 if pressed or trigger_rising else 0.0
    else:  # toggle
        # Internal toggle state, synced with UI and toggleable by trigger
        state = _BUTTON_STATE.setdefault(node_id, {}) if node_id else {}
        toggle_state = state.get("toggle_state", pressed)

        if trigger_rising:
            toggle_state = not toggle_state
        elif node_id:
            # Sync with UI: detect button_pressed transition
            prev_ui = state.get("prev_ui_pressed", False)
            if pressed != prev_ui:
                toggle_state = pressed
            state["prev_ui_pressed"] = pressed

        if node_id:
            state["toggle_state"] = toggle_state
        val = 1.0 if toggle_state else 0.0

    # ── Lazy prune ──────────────────────────────────────────────
    if node_id:
        global _BUTTON_PRUNE_COUNTER
        _BUTTON_PRUNE_COUNTER += 1
        if _BUTTON_PRUNE_COUNTER % 1000 == 0:
            _cutoff = int(params.get("frame", 0)) - 7200
            for _nid in list(_BUTTON_STATE):
                if _BUTTON_STATE[_nid].get("frame", 0) < _cutoff:
                    del _BUTTON_STATE[_nid]

    return {"value": val}
