"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
from collections import deque
from pathlib import Path
from ...core.registry import method
from ...core.utils import seed_all

# ── Per-node state ──────────────────────────────────────────────────────
_LAG_STATE: dict[str, dict] = {}
_LAG_PRUNE_COUNTER = 0

_DEFAULT_FPS = 24.0
_LN10 = math.log(10.0)  # ≈2.3026 — used for 90% response time constant


@method(id="__lag__", name="Lag", category="channels",
        tags=["chop", "time", "filter", "smooth", "lag"],
        inputs={
            "signal": "SCALAR",
            "reset_in": "SCALAR",
        },
        outputs={
            "value": "SCALAR",
            "velocity": "SCALAR",
            "acceleration": "SCALAR",
        },
        runtime={
            "value": {"type": "numeric", "label": "Value", "observable": True},
            "velocity": {"type": "numeric", "label": "Velocity", "observable": True},
            "acceleration": {"type": "numeric", "label": "Acceleration", "observable": True},
        },
        signal={
            "signal": "numeric",
            "reset_in": "event",
            "value": "output",
            "velocity": "output",
            "acceleration": "output",
        },
        params={
            "delay": {"description": "Delay in seconds before lag begins", "default": 0.0},
            "lagmethod": {"description": "LagMethod — The method by which lag is applied",
                          "choices": ["exponential", "spring"], "default": "exponential"},
            "lag_up": {"description": "Lag ↑ — time to follow 90% of a step upward", "default": 0.1},
            "lag_down": {"description": "Lag ↓ — time to follow 90% of a step downward", "default": 0.1},
            "lagunit": {"description": "Lag Unit — Samples, Frames, or Seconds",
                        "choices": ["samples", "frames", "seconds"], "default": "seconds"},
            "overshoot_up": {"description": "Overshoot ↑ — overshoot strength", "default": 0.0},
            "overshoot_down": {"description": "Overshoot ↓ — overshoot strength", "default": 0.0},
            "overshootunit": {"description": "Overshoot Unit — Samples, Frames, or Seconds",
                              "choices": ["samples", "frames", "seconds"], "default": "seconds"},
            "clamp_slope": {"description": "Clamp Slope — clamp slope to Max Slope", "default": False},
            "max_slope_up": {"description": "Max Slope ↑ (value/unit)", "default": 1.0},
            "max_slope_down": {"description": "Max Slope ↓ (value/unit)", "default": 1.0},
            "clamp_accel": {"description": "Clamp Acceleration — clamp accel to Max Acceleration", "default": False},
            "max_accel_up": {"description": "Max Acceleration ↑ (value/unit²)", "default": 1.0},
            "max_accel_down": {"description": "Max Acceleration ↓ (value/unit²)", "default": 1.0},
            "lagsamples": {"description": "Lag per Sample — per-sample lag (single-channel: no-op)", "default": False},
            "snap": {"description": "Snap — snap output to input if within threshold", "default": False},
            "threshold": {"description": "Threshold — snap threshold", "default": 0.001},
            "reset": {"description": "Reset — bypass the lag effect", "default": False},
            "resetpulse": {"description": "Reset Pulse — instantly reset", "default": False},
        })
def method_lag(out_dir: Path, seed: int, params=None):
    """Temporal lag/smoothing filter — applies exponential lag to a signal.

    Models TouchDesigner's Lag CHOP: exponentially smooths an input signal
    with separate time constants for rising and falling, plus overshoot,
    slope/acceleration clamping, snap, and reset behavior.
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Frame derivation (anti-culling) ──────────────────────────────────
    frame = int(params.get("frame", 0))
    fps = _DEFAULT_FPS
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", _DEFAULT_FPS))

    node_id = params.get("_node_id", "")

    # ── Read params ──────────────────────────────────────────────────────
    delay = float(params.get("delay", 0.0))
    lagmethod = params.get("lagmethod", "exponential")
    lag_up = float(params.get("lag_up", 0.1))
    lag_down = float(params.get("lag_down", 0.1))
    lagunit = params.get("lagunit", "seconds")
    overshoot_up = float(params.get("overshoot_up", 0.0))
    overshoot_down = float(params.get("overshoot_down", 0.0))
    overshootunit = params.get("overshootunit", "seconds")

    clamp_slope = params.get("clamp_slope", False)
    if isinstance(clamp_slope, str):
        clamp_slope = clamp_slope.lower() in ("true", "1", "yes")
    max_slope_up = float(params.get("max_slope_up", 1.0))
    max_slope_down = float(params.get("max_slope_down", 1.0))

    clamp_accel = params.get("clamp_accel", False)
    if isinstance(clamp_accel, str):
        clamp_accel = clamp_accel.lower() in ("true", "1", "yes")
    max_accel_up = float(params.get("max_accel_up", 1.0))
    max_accel_down = float(params.get("max_accel_down", 1.0))

    snap = params.get("snap", False)
    if isinstance(snap, str):
        snap = snap.lower() in ("true", "1", "yes")
    threshold_val = float(params.get("threshold", 0.001))

    reset_param = params.get("reset", False)
    if isinstance(reset_param, str):
        reset_param = reset_param.lower() in ("true", "1", "yes")
    resetpulse_param = params.get("resetpulse", False)
    if isinstance(resetpulse_param, str):
        resetpulse_param = resetpulse_param.lower() in ("true", "1", "yes")

    # ── SCALAR input overrides ───────────────────────────────────────────
    input_val = params.get("signal")
    if input_val is not None:
        input_val = float(input_val)
    else:
        input_val = 0.0

    reset_in_val = params.get("reset_in")

    # ── Unit conversion helpers ──────────────────────────────────────────
    def _to_frames(val: float, unit: str) -> float:
        if unit == "seconds":
            return val * fps
        elif unit == "samples":
            return val
        else:
            return val

    def _rate_to_pf(val: float, unit: str) -> float:
        """Convert a rate (value/unit) to value/frame."""
        return val / fps if unit == "seconds" else val

    def _accel_to_pf2(val: float, unit: str) -> float:
        """Convert an acceleration (value/unit²) to value/frame²."""
        return val / (fps * fps) if unit == "seconds" else val

    lag_up_frames = max(0.0, _to_frames(lag_up, lagunit))
    lag_down_frames = max(0.0, _to_frames(lag_down, lagunit))
    delay_frames = max(0, int(round(delay * fps)))
    # Overshoot is dimensionless gain — no unit conversion (TD convention)
    overshoot_up_gain = max(0.0, overshoot_up)
    overshoot_down_gain = max(0.0, overshoot_down)

    # ── Default output (standalone/test fallback) ────────────────────────
    output = input_val
    velocity = 0.0
    acceleration = 0.0

    # ── Stateful mode (graph executor) ───────────────────────────────────
    global _LAG_PRUNE_COUNTER
    _LAG_PRUNE_COUNTER += 1

    if node_id:
        buf_len = max(delay_frames, 1)
        state = _LAG_STATE.setdefault(node_id, {
            "prev_output": 0.0,       # start from 0: first frame shows step
            "prev_velocity": 0.0,
            "prev_frame": frame,
            "prev_resetpulse": 0.0,
            "prev_reset_in": 0.0,
            "delay_buf": deque([0.0] * buf_len, maxlen=buf_len),
            "initialized": True,
        })

        # Update delay buffer length if delay changed at runtime
        if state["delay_buf"].maxlen != buf_len:
            state["delay_buf"] = deque([0.0] * buf_len, maxlen=buf_len)

        # ── Frame delta ──────────────────────────────────────────────────
        _delta = max(1, frame - state["prev_frame"])
        state["prev_frame"] = frame

        # ── Reset pulse detection (button — rising edge) ─────────────────
        reset_active = False
        if resetpulse_param and not state.get("prev_resetpulse", 0.0):
            reset_active = True
        state["prev_resetpulse"] = 1.0 if resetpulse_param else 0.0

        # ── Reset from SCALAR input (rising edge) ────────────────────────
        if not reset_active and reset_in_val is not None:
            prev_reset_in = state.get("prev_reset_in", 0.0)
            if prev_reset_in <= 0.5 < float(reset_in_val):
                reset_active = True
            state["prev_reset_in"] = float(reset_in_val)

        # ── Reset toggle param ───────────────────────────────────────────
        if not reset_active and reset_param:
            reset_active = True

        # ── Push input into delay buffer ─────────────────────────────────
        state["delay_buf"].append(input_val)
        delayed_input = state["delay_buf"][0]

        # ── Apply reset or lag ───────────────────────────────────────────
        prev_out = state["prev_output"]
        prev_vel = state["prev_velocity"]

        if reset_active:
            output = input_val
            velocity = 0.0
            acceleration = 0.0
            state["delay_buf"].clear()
            state["delay_buf"].append(input_val)
        elif lag_up_frames <= 0 and lag_down_frames <= 0:
            output = delayed_input
            velocity = output - prev_out
            acceleration = velocity - prev_vel
        else:
            rising = delayed_input >= prev_out
            lag_frames = lag_up_frames if rising else lag_down_frames

            if lag_frames <= 0:
                alpha = 1.0
            else:
                alpha = 1.0 - math.exp(-_delta * _LN10 / max(lag_frames, 1e-8))
                alpha = min(1.0, max(0.0, alpha))

            target = delayed_input
            output = prev_out + alpha * (target - prev_out)

            overshoot_amt = overshoot_up_gain if rising else overshoot_down_gain

            if lagmethod == "spring" and overshoot_amt > 0 and lag_frames > 0:
                # Second-order spring-damper (only path that produces real overshoot)
                _omega = _LN10 / max(lag_frames, 1e-8)
                _os_gain = max(0.01, min(0.99, overshoot_amt))
                _zeta = -math.log(_os_gain) / math.sqrt(
                    math.pi**2 + math.log(_os_gain) ** 2
                )
                if _zeta < 0.01:
                    _zeta = 0.01
                # Spring MUST start from prev_out with stored velocity, not from
                # the exponentially-interpolated output — otherwise the
                # oscillation amplitude is killed at birth.
                output = prev_out
                raw_vel = prev_vel
                for _step in range(_delta):
                    accel = (_omega * _omega * (target - output)
                             - 2.0 * _zeta * _omega * raw_vel)
                    raw_vel += accel
                    output += raw_vel
                # Safety clamp — prevent numerical blowup
                if not math.isfinite(output):
                    output = target
                    raw_vel = 0.0
                raw_vel = output - prev_out
                raw_accel = raw_vel - prev_vel
            elif overshoot_amt > 0:
                # Velocity-momentum boost for exponential mode.
                # NOTE: a first-order exponential filter CANNOT overshoot past the
                # target — this only accelerates convergence. Real overshoot
                # requires the second-order spring-damper (lagmethod="spring").
                raw_vel = output - prev_out
                output += raw_vel * overshoot_amt
                raw_vel = output - prev_out
                raw_accel = raw_vel - prev_vel
            else:
                raw_vel = output - prev_out
                raw_accel = raw_vel - prev_vel

            # ── Slope clamp ──────────────────────────────────────────────
            if clamp_slope:
                max_slope_val = _rate_to_pf(
                    max_slope_up if rising else max_slope_down, lagunit
                )
                if max_slope_val > 0 and abs(raw_vel) > max_slope_val:
                    raw_vel = math.copysign(max_slope_val, raw_vel)
                    output = prev_out + raw_vel

            # ── Acceleration clamp ───────────────────────────────────────
            raw_accel = raw_vel - prev_vel
            if clamp_accel:
                max_accel_val = _accel_to_pf2(
                    max_accel_up if rising else max_accel_down, lagunit
                )
                if max_accel_val > 0 and abs(raw_accel) > max_accel_val:
                    raw_accel = math.copysign(max_accel_val, raw_accel)
                    output = prev_out + prev_vel + raw_accel
                    raw_vel = output - prev_out

            # ── Snap ─────────────────────────────────────────────────────
            if snap and abs(output - target) < threshold_val:
                output = target
                raw_vel = 0.0
                raw_accel = 0.0

            velocity = raw_vel
            acceleration = raw_accel

        state["prev_output"] = output
        state["prev_velocity"] = velocity

        # ── Lazy prune ──────────────────────────────────────────────────
        if _LAG_PRUNE_COUNTER % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_LAG_STATE):
                if _LAG_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _LAG_STATE[_nid]

    return {
        "value": float(output),
        "velocity": float(velocity),
        "acceleration": float(acceleration),
    }
