"""Sample & Hold — samples the input on each rising edge of the trigger."""
from __future__ import annotations
from pathlib import Path
from ...core.registry import method
from ...core.utils import seed_all

# Per-node state: the held value + previous trigger for edge detection
_SH_STATE: dict[str, dict] = {}
_SH_PRUNE_COUNTER = 0


@method(id="__sample_hold__", name="Sample & Hold", category="channels",
        tags=["chop", "sample", "hold", "trigger", "samplehold"],
        inputs={
            "input": "SCALAR",
            "trigger": "SCALAR",
        },
        outputs={
            "value": "SCALAR",
        },
        runtime={
            "value": {"type": "numeric", "label": "Value", "observable": True},
        },
        signal={
            "input": "numeric",
            "trigger": "event",
            "value": "output",
        },
        params={
            "init_value": {
                "description": "Initial held value before first trigger",
                "default": 0.0,
            },
            "thresh": {
                "description": "Trigger threshold — rising edge crosses this to sample",
                "default": 0.5,
            },
        })
def method_sample_hold(out_dir: Path, seed: int, params=None):
    """Sample & Hold — captures the ``input`` value on each rising edge of the
    ``trigger`` signal and holds it steady until the next trigger.

    Modelled after Eurorack / TouchDesigner S&H modules.  The ``trigger`` input
    is edge-detected using a configurable threshold (``thresh`` param, default
    0.5).  On each rising edge (current ≥ threshold AND previous < threshold),
    the current ``input`` value is recorded and held.

    Initial output before any trigger is ``init_value`` (default 0.0).

    Outputs:
        value (SCALAR): the last sampled input, held constant between triggers
    """
    if params is None:
        params = {}
    seed_all(seed)

    node_id = params.get("_node_id", "")

    # ── Read inputs ──────────────────────────────────────────────────────
    input_val = float(params.get("input", 0.0))
    trigger_val: float | None = None
    _tv = params.get("trigger")
    if _tv is not None:
        trigger_val = float(_tv)

    thresh = float(params.get("thresh", 0.5))
    init_value = float(params.get("init_value", 0.0))

    # ── Default output (unwired / no-id fallback) ────────────────────────
    output = init_value

    if node_id and trigger_val is not None:
        state = _SH_STATE.setdefault(node_id, {
            "held_value": init_value,
            "prev_trigger": 0.0,
            "frame": 0,
        })

        # Timeline regression / reset: if frame went backward, re-init
        frame = int(params.get("frame", 0))
        if "frame" not in params:
            _tl = params.get("_timeline")
            if _tl is not None:
                frame = int(getattr(_tl, "global_frame", 0))

        prev_frame = state.get("frame", 0)
        if frame < prev_frame:
            state["held_value"] = init_value
            state["prev_trigger"] = 0.0
        state["frame"] = frame

        # ── Rising-edge detection ────────────────────────────────────────
        prev_trigger = state.get("prev_trigger", 0.0)
        if trigger_val >= thresh > prev_trigger:
            state["held_value"] = input_val
        state["prev_trigger"] = trigger_val

        output = state["held_value"]

        # ── Lazy prune stale state ───────────────────────────────────────
        global _SH_PRUNE_COUNTER
        _SH_PRUNE_COUNTER += 1
        if _SH_PRUNE_COUNTER % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_SH_STATE):
                if _SH_STATE[_nid].get("frame", 0) < _cutoff:
                    del _SH_STATE[_nid]

    elif trigger_val is None:
        # No trigger wired — just pass the input through
        output = input_val

    return {"value": float(output)}
