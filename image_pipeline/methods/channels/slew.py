"""Slew — attack/release rate limiter for SCALAR signals."""
from __future__ import annotations
import math
from pathlib import Path

from ...core.registry import method


# ── Per-node state ──────────────────────────────────────────────────────
_SLEW_STATE: dict[str, dict] = {}
_SLEW_PRUNE = 0

_DEFAULT_FPS = 24.0


@method(
    id="__slew__",
    name="Slew",
    category="channels",
    tags=["chop", "filter", "slew", "rate", "attack", "release"],
    inputs={"input": "SCALAR"},
    outputs={"value": "SCALAR", "active": "SCALAR"},
    runtime={
        "value": {"type": "numeric", "label": "Value", "observable": True},
        "active": {"type": "numeric", "label": "Active", "observable": True},
    },
    signal={
        "input": "numeric",
        "value": "output",
        "active": "output",
    },
    params={
        "attack": {
            "description": "Attack time — how fast the output can rise (in frames)",
            "min": 0,
            "max": 1000,
            "default": 10,
        },
        "release": {
            "description": "Release time — how fast the output can fall (in frames)",
            "min": 0,
            "max": 1000,
            "default": 10,
        },
        "slewunit": {
            "description": "Slew unit — Frames or Seconds",
            "choices": ["frames", "seconds"],
            "default": "frames",
        },
        "mode": {
            "description": "Slew mode — separate attack/release (ar) or follow the faster (faster) or slower (slower) rate",
            "choices": ["ar", "faster", "slower"],
            "default": "ar",
        },
    },
    is_time_varying=True,
    description=(
        "Rate-of-change limiter — controls how fast the output can rise (attack) "
        "and fall (release).  Models the Slew Limiter / Attack-Release concept."
    ),
)
def method_slew(out_dir: Path, seed: int, params=None):
    """Attack/Release rate limiter.

    Smooths an input signal by limiting the maximum rate of change per frame.
    Attack controls the maximum upward step; release controls the maximum
    downward step.

    Outputs:
        value (SCALAR): slew-limited output
        active (SCALAR): 1.0 when output is still slewing toward input, 0.0 when settled
    """
    if params is None:
        params = {}

    node_id = params.get("_node_id", "")
    input_val = float(params.get("input", 0.0))
    attack = float(params.get("attack", 10.0))
    release = float(params.get("release", 10.0))
    unit = params.get("slewunit", "frames")
    mode = params.get("mode", "ar")

    # ── Frame derivation ─────────────────────────────────────────────────
    frame = int(params.get("frame", 0))
    fps = _DEFAULT_FPS
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", _DEFAULT_FPS))

    # ── Default pass-through ────────────────────────────────────────────
    output = input_val
    active = 0.0

    if node_id:
        state = _SLEW_STATE.setdefault(node_id, {
            "prev_output": 0.0,
            "prev_input": input_val,
            "prev_frame": frame,
        })

        _delta = frame - state["prev_frame"]
        if _delta < 0:
            state["prev_output"] = input_val
            state["prev_input"] = input_val
            _delta = 1
        _delta = max(1, _delta)
        state["prev_frame"] = frame

        prev_out = state["prev_output"]

        # Convert to per-frame step limits
        def _max_step(amount: float, unit_str: str) -> float:
            if unit_str == "seconds":
                return amount * _delta / fps if fps > 0 else amount * _delta
            # frames
            return amount * _delta if amount > 0 else amount

        max_up = _max_step(1.0 / max(attack, 0.001), unit) if attack > 0 else float("inf")
        max_down = _max_step(1.0 / max(release, 0.001), unit) if release > 0 else float("inf")

        if mode == "faster":
            max_up = max(max_up, max_down)
            max_down = max_up
        elif mode == "slower":
            max_up = min(max_up, max_down)
            max_down = max_up

        # Apply slew
        diff = input_val - prev_out
        if diff > 0:
            output = prev_out + min(diff, max_up)
        elif diff < 0:
            output = prev_out + max(diff, -max_down)
        else:
            output = prev_out

        active = 0.0 if abs(output - input_val) < 1e-6 else 1.0

        state["prev_output"] = output
        state["prev_input"] = input_val

        # ── Lazy prune ──────────────────────────────────────────────────
        global _SLEW_PRUNE
        _SLEW_PRUNE += 1
        if _SLEW_PRUNE % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_SLEW_STATE):
                if _SLEW_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _SLEW_STATE[_nid]

    return {"value": float(output), "active": float(active)}
