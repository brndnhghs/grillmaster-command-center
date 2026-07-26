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
# Keyed by _node_id (injected by GraphExecutor into run_params).
# Stores per-instance tracking: prev_output, prev_velocity, delay_buf, etc.
_LAG_STATE: dict[str, dict] = {}
_LAG_PRUNE_COUNTER = 0

_DEFAULT_FPS = 24.0
_LN10 = math.log(10.0)  # ≈2.3026 — used for 90% response time constant


@method(id="__lag__", name="Lag", category="channels",
        tags=["chop", "time", "filter", "smooth", "lag"],
        inputs={
            "signal": "SCALAR",     # the value to lag
            "reset_in": "SCALAR",   # external reset pulse (rising edge)
        },
        outputs={
            "value": "SCALAR",         # the lagged/smoothed output
            "velocity": "SCALAR",      # per-frame slope (output change)
            "acceleration": "SCALAR",  # per-frame acceleration (velocity change)
        },
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True,
            },
            "velocity": {
                "type": "numeric",
                "label": "Velocity",
                "observable": True,
            },
            "acceleration": {
                "type": "numeric",
                "label": "Acceleration",
                "observable": True,
            },
        },
        signal={
            "signal": "numeric",
            "reset_in": "event",
            "value": "output",
            "velocity": "output",
            "acceleration": "output",
        },
        params={
            "delay": {
                "description": "Delay in seconds before lag begins",
                "default": 0.0,
            },
            "lagmethod": {
                "description": "LagMethod ⊞ — The method by which lag is applied",
                "choices": ["exponential", "spring"],
                "default": "exponential",
            },
            "lag_up": {
                "description": "Lag ↑ — time to follow 90% of a step upward",
                "default": 0.1,
            },
            "lag_down": {
                "description": "Lag ↓ — time to follow 90% of a step downward",
                "default": 0.1,
            },
            "lagunit": {
                "description": "Lag Unit — Samples, Frames, or Seconds",
                "choices": ["samples", "frames", "seconds"],
                "default": "seconds",
            },
            "overshoot_up": {
                "description": "Overshoot ↑ — overshoot strength while moving up",
                "default": 0.0,
            },
            "overshoot_down": {
                "description": "Overshoot ↓ — overshoot strength while moving down",
                "default": 0.0,
            },
            "overshootunit": {
                "description": "Overshoot Unit — Samples, Frames, or Seconds",
                "choices": ["samples", "frames", "seconds"],
                "default": "seconds",
            },
            "clamp_slope": {
                "description": "Clamp Slope ⊞ — clamp the slope to Max Slope values",
                "default": False,
            },
            "max_slope_up": {
                "description": "Max Slope ↑ — limits rising slope (value/unit)",
                "default": 1.0,
            },
            "max_slope_down": {
                "description": "Max Slope ↓ — limits falling slope (value/unit)",
                "default": 1.0,
            },
            "clamp_accel": {
                "description": "Clamp Acceleration ⊞ — clamp acceleration to Max Acceleration values",
                "default": False,
            },
            "max_accel_up": {
                "description": "Max Acceleration ↑ — limits rising acceleration (value/unit²)",
                "default": 1.0,
            },
            "max_accel_down": {
                "description": "Max Acceleration ↓ — limits falling acceleration (value/unit²)",
                "default": 1.0,
            },
            "lagsamples": {
                "description": "Lag per Sample ⊞ — apply lag per sample (single-channel: no-op)",
                "default": False,
            },
            "snap": {
                "description": "Snap ⊞ — snap output to input if within threshold",
                "default": False,
            },
            "threshold": {
                "description": "Threshold — snap threshold value",
                "default": 0.001,
            },
            "reset": {
                "description": "Reset — when On, bypass the lag effect",
                "default": False,
            },
            "resetpulse": {
                "description": "Reset Pulse — instantly reset the lag effect",
                "default": False,
            },
        })
def method_lag(out_dir: Path, seed: int, params=None):
    """Temporal lag/smoothing filter — applies exponential lag to a signal.

    Models TouchDesigner's Lag CHOP: exponentially smooths an input signal
    with separate time constants for rising and falling, plus overshoot,
    slope/acceleration clamping, snap, and reset behavior.

    Inputs:
        signal (SCALAR): the value to lag
        reset_in (SCALAR): external reset (rising edge resets state)

    Outputs:
        value (SCALAR): the lagged/smoothed output
        velocity (SCALAR): per-frame slope (output change)
        acceleration (SCALAR): per-frame acceleration (velocity change)
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Frame derivation (anti-culling) ──────────────────────────────────
    # Channel nodes MUST derive their live frame from the injected Timeline
    # to prevent the executor from culling them as "static" (documented in
    # lfo.py / counter.py / noise1d.py — uses global_frame, NOT phase).
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
    lag_per_sample = params.get("lagsamples", False)
    if isinstance(lag_per_sample, str):
        lag_per_sample = lag_per_sample.lower() in ("true", "1", "yes")

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

    # ── SCALAR input overrides (from wired inputs) ───────────────────────
    input_val = params.get("signal")
    if input_val is not None:
        input_val = float(input_val)
    else:
        input_val = 0.0  # default when unwired

    reset_in_val = params.get("reset_in")

    # ── Unit conversion helpers ──────────────────────────────────────────
    def _to_frames(val: float, unit: str) -> float:
        if unit == "seconds":
            return val * fps
        elif unit == "samples":
            return val  # samples ≈ frames for single-channel
        else:  # frames
            return val

    lag_up_frames = max(0.0, _to_frames(lag_up, lagunit))
    lag_down_frames = max(0.0, _to_frames(lag_down, lagunit))
    delay_frames = max(0, int(round(delay * fps)))
    # Overshoot is a dimensionless gain multiplier (0 = none, 0.5 = moderate, 1.0+ = strong).
    # No unit conversion — the Unit selector exists in the UI spec but the value is
    # always a pure ratio (TouchDesigner Lag CHOP convention).
    overshoot_up_gain = max(0.0, overshoot_up)
    overshoot_down_gain = max(0.0, overshoot_down)

    # ── Default output (standalone/test fallback) ────────────────────────
    output = input_val
    velocity = 0.0
    acceleration = 0.0

    # ── Stateful mode (graph executor: _node_id present) ─────────────────
    global _LAG_PRUNE_COUNTER
    _LAG_PRUNE_COUNTER += 1

    if node_id:
        state = _LAG_STATE.setdefault(node_id, {
            "prev_output": input_val,
            "prev_velocity": 0.0,
            "prev_frame": frame,
            "prev_resetpulse": 0.0,
            "prev_reset_in": 0.0,
            "delay_buf": deque(maxlen=delay_frames if delay_frames > 0 else 1),
            "initialized": True,
        })

        # Update delay buffer maxlen if delay changed at runtime
        if delay_frames > 0 and state["delay_buf"].maxlen != delay_frames:
            state["delay_buf"] = deque(
                (list(state["delay_buf"])[-delay_frames:]),
                maxlen=delay_frames,
            )
        elif delay_frames <= 0 and state["delay_buf"].maxlen != 1:
            state["delay_buf"] = deque(state["delay_buf"], maxlen=1)

        # ── Frame delta (guarantee at least 1 frame of advancement) ──────
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
        # The effective delayed input (front of deque when delay > 0)
        delayed_input = state["delay_buf"][0]

        # ── Apply reset or lag ───────────────────────────────────────────
        prev_out = state["prev_output"]
        prev_vel = state["prev_velocity"]

        if reset_active:
            # Reset: output = current input (bypass both lag AND delay)
            output = input_val
            velocity = 0.0
            acceleration = 0.0
            # Clear delay buffer so lag resumes from current value
            state["delay_buf"].clear()
            state["delay_buf"].append(input_val)
        elif lag_up_frames <= 0 and lag_down_frames <= 0:
            # No lag — pass-through (still honors delay)
            output = delayed_input
            velocity = output - prev_out
            acceleration = velocity - prev_vel
        else:
            # ── Determine lag direction ──
            rising = delayed_input >= prev_out
            lag_frames = lag_up_frames if rising else lag_down_frames

            if lag_frames <= 0:
                alpha = 1.0
            else:
                # Exponential lag: alpha = 1 - exp(-dt * ln(10) / lag_frames)
                # Where lag_frames = time for ~90% response (3.32 time constants)
                # For dt frames, alpha reaches 1 - exp(-dt * ln(10) / lag_frames)
                alpha = 1.0 - math.exp(-_delta * _LN10 / max(lag_frames, 1e-8))
                alpha = min(1.0, max(0.0, alpha))

            # ── Apply exponential lag ──
            target = delayed_input
            output = prev_out + alpha * (target - prev_out)

            # ── Overshoot (spring model: proper second-order oscillator) ──
            overshoot_amt = overshoot_up_gain if rising else overshoot_down_gain
            raw_vel = output - prev_out
            raw_accel = raw_vel - prev_vel

            if lagmethod == "spring" and overshoot_amt > 0 and lag_frames > 0:
                # Second-order spring-damper: map lag_frames→natural frequency,
                # overshoot→damping ratio.
                # omega = LN10/lag_frames gives ~90% step response in lag_frames.
                # Tuned ~1.2× so the undamped period ≈ 2*lag_frames.
                _omega = _LN10 / max(lag_frames, 1e-8)
                # damping_ratio zeta ∈ (0, 1] maps overshoot_gain ≅ overshoot_pct/100
                # zeta = -ln(M)/sqrt(pi² + ln(M)²) where M = gain.
                _os_gain = max(0.01, min(0.99, overshoot_amt))
                _zeta = -math.log(_os_gain) / math.sqrt(math.pi**2 + math.log(_os_gain)**2)
                if _zeta < 0.01:
                    _zeta = 0.01  # prevent blowup
                # Semi-implicit Euler sub-stepping
                _dt_sub = max(1, _delta)
                for _step in range(_dt_sub):
                    dt = 1.0
                    accel = _omega * _omega * (target - output) - 2.0 * _zeta * _omega * raw_vel
                    raw_vel += accel * dt
                    output += raw_vel * dt
                raw_vel = output - prev_out
                raw_accel = raw_vel - prev_vel
            elif overshoot_amt > 0:
                # Dimensionless-gain overshoot (added after lag, decays naturally).
                # The gain is a fraction of the remaining step that gets added as
                # extra push past the target. Decays with the same alpha as the lag.
                os_frac = overshoot_amt * (1.0 - alpha)
                output += (target - prev_out) * os_frac
                # Don't allow overshoot to retrograde (always toward then past)
                raw_vel = output - prev_out
                raw_accel = raw_vel - prev_vel

            # ── Slope clamp (applied after lag + overshoot) ──────────────
            raw_vel = output - prev_out
            if clamp_slope:
                max_slope_val = max_slope_up if rising else max_slope_down
                if max_slope_val > 0 and abs(raw_vel) > max_slope_val:
                    raw_vel = math.copysign(max_slope_val, raw_vel)
                    output = prev_out + raw_vel

            # ── Acceleration clamp ──────────────────────────────────────
            raw_accel = raw_vel - prev_vel
            if clamp_accel:
                max_accel_val = max_accel_up if rising else max_accel_down
                if max_accel_val > 0 and abs(raw_accel) > max_accel_val:
                    raw_accel = math.copysign(max_accel_val, raw_accel)
                    output = prev_out + prev_vel + raw_accel
                    raw_vel = output - prev_out

            # ── Snap ────────────────────────────────────────────────────
            if snap and abs(output - target) < threshold_val:
                output = target
                raw_vel = 0.0
                raw_accel = 0.0

            velocity = raw_vel
            acceleration = raw_accel

        # ── Update state ─────────────────────────────────────────────────
        state["prev_output"] = output
        state["prev_velocity"] = velocity

        # ── Lazy prune: every ~1000 calls, drop stale states ─────────────
        if _LAG_PRUNE_COUNTER % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_LAG_STATE):
                if _LAG_STATE[_nid].get("prev_frame", 0) < _cutoff:
                    del _LAG_STATE[_nid]

    # ── Return SCALAR outputs ────────────────────────────────────────────
    return {
        "value": float(output),
        "velocity": float(velocity),
        "acceleration": float(acceleration),
    }
