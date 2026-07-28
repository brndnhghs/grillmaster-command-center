"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
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
            "input": "SCALAR",
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
            "value": "output",
            "velocity": "output",
            "acceleration": "output",
        },
        params={
            "lag_up": {"description": "Lag ↑ — time to follow 90% of a step upward", "default": 0.1},
            "lag_down": {"description": "Lag ↓ — time to follow 90% of a step downward", "default": 0.1},
            "lagunit": {"description": "Lag Unit — Samples, Frames, or Seconds",
                        "choices": ["samples", "frames", "seconds"], "default": "seconds"},
            "lagmethod": {"description": "LagMethod — The method by which lag is applied",
                          "choices": ["exponential", "spring"], "default": "exponential"},
            "overshoot_up": {"description": "Overshoot ↑ — spring overshoot fraction upward (0=critical, 0.3=30%)", "default": 0.0},
            "overshoot_down": {"description": "Overshoot ↓ — spring overshoot fraction downward (0=critical, 0.3=30%)", "default": 0.0},
        })
def method_lag(out_dir: Path, seed: int, params=None):
    """Temporal lag/smoothing filter — applies exponential or spring lag to a signal.

    Models TouchDesigner's Lag CHOP: smooths an input signal with separate time
    constants for rising and falling.

    The connected upstream signal is injected by the GraphExecutor into
    ``params["input"]`` (the declared SCALAR input port, not a param). When
    unwired, falls back to 0.0.

    Velocity and acceleration are per-frame deltas (first and second differences),
    consistent with TouchDesigner convention — NOT time-normalised.

    Lag units:
        ``frames``  — raw frame count (FPS-independent).
        ``samples`` — equivalent to frames; this engine has no sample-rate
                      distinction separate from frame rate.
        ``seconds`` — scaled by FPS (FPS-dependent).
    """
    if params is None:
        params = {}
    seed_all(seed)

    # ── Frame derivation ─────────────────────────────────────────────────
    frame = int(params.get("frame", 0))
    fps = _DEFAULT_FPS
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", _DEFAULT_FPS))

    node_id = params.get("_node_id", "")

    # ── Read input signal & params ───────────────────────────────────────
    # The executor injects the upstream SCALAR value into params["input"]
    # (the declared input port). When unwired, fall back to 0.0.
    input_val = float(params.get("input", 0.0))

    lag_up = float(params.get("lag_up", 0.1))
    lag_down = float(params.get("lag_down", 0.1))
    lagunit = params.get("lagunit", "seconds")
    lagmethod = params.get("lagmethod", "exponential")
    overshoot_up = float(params.get("overshoot_up", 0.0))
    overshoot_down = float(params.get("overshoot_down", 0.0))

    # ── Unit conversion ──────────────────────────────────────────────────
    def _to_frames(val: float, unit: str) -> float:
        if unit == "seconds":
            return val * fps
        # samples and frames are equivalent in this engine (no oversampling)
        return val

    lag_up_frames = max(0.0, _to_frames(lag_up, lagunit))
    lag_down_frames = max(0.0, _to_frames(lag_down, lagunit))

    # ── Default output ───────────────────────────────────────────────────
    output = input_val
    velocity = 0.0
    acceleration = 0.0

    # ── Stateful smoothing ───────────────────────────────────────────────
    global _LAG_PRUNE_COUNTER
    _LAG_PRUNE_COUNTER += 1

    if node_id:
        state = _LAG_STATE.setdefault(node_id, {
            "prev_output": 0.0,
            "prev_input": 0.0,
            "prev_velocity": 0.0,
            "spring_velocity": 0.0,
            "prev_frame": frame,
        })

        _delta = frame - state["prev_frame"]

        # ── Timeline regression / reset ──────────────────────────────────
        # If frame goes backward (seek/rewind/loop), reset state so the
        # filter restarts from the current input rather than carrying stale
        # state from a different timeline position.
        if _delta < 0:
            state["prev_output"] = input_val
            state["prev_input"] = input_val
            state["prev_velocity"] = 0.0
            state["spring_velocity"] = 0.0
            _delta = 1

        _delta = max(1, _delta)
        state["prev_frame"] = frame

        prev_out = state["prev_output"]
        prev_input = state["prev_input"]
        prev_vel = state["prev_velocity"]
        spring_vel = state["spring_velocity"]

        if lag_up_frames <= 0 and lag_down_frames <= 0:
            # No lag — pass through
            output = input_val
            velocity = output - prev_out
            acceleration = velocity - prev_vel
        elif lagmethod == "spring":
            # ── Spring (second-order) mode ───────────────────────────────
            # Uses a spring-mass-damper second-order system.
            # lag_up/lag_down control 90% settling time.
            # overshoot controls damping ratio: 0 = critical, >0 = underdamped.
            output = prev_out  # start from previous filtered position
            lag_frames = lag_up_frames if input_val >= prev_input else lag_down_frames
            lag_frames = max(lag_frames, 0.001)

            # Convert overshoot fraction to damping ratio ζ
            os = overshoot_up if input_val >= prev_input else overshoot_down
            if os <= 0.0:
                zeta = 1.0  # critical damping
            else:
                zeta = -math.log(os) / math.sqrt(math.pi**2 + math.log(os)**2)
                zeta = max(0.05, min(1.0, zeta))

            # Natural frequency for 90% settling time
            # Envelope of underdamped response: 1 - exp(-ζ·ω·t)
            # 90% → ζ·ω·lag_frames = ln(10)
            # For critical damping (ζ=1): same formula works
            omega = _LN10 / (max(zeta, 1e-10) * lag_frames)
            k = omega * omega       # stiffness
            c = 2.0 * zeta * omega  # damping coefficient

            for _step in range(_delta):
                a = k * (input_val - output) - c * spring_vel
                spring_vel += a
                output += spring_vel

            # Limit spring velocity to prevent explosive divergence
            max_spring_vel = abs(input_val - prev_out) * 10.0 + 1.0
            spring_vel = max(-max_spring_vel, min(max_spring_vel, spring_vel))

            # Per-frame delta outputs (TD convention)
            velocity = output - prev_out
            acceleration = velocity - prev_vel

        else:
            # ── Exponential mode ─────────────────────────────────────────
            # Direction from actual input movement, NOT output position.
            rising = input_val >= prev_input
            lag_frames = lag_up_frames if rising else lag_down_frames

            if lag_frames <= 0:
                alpha = 1.0
            else:
                alpha = 1.0 - math.exp(-_delta * _LN10 / max(lag_frames, 1e-8))
                alpha = min(1.0, max(0.0, alpha))

            output = prev_out + alpha * (input_val - prev_out)

            # Per-frame delta outputs (TD convention)
            velocity = output - prev_out
            acceleration = velocity - prev_vel

        state["prev_output"] = output
        state["prev_input"] = input_val
        state["prev_velocity"] = velocity
        state["spring_velocity"] = spring_vel

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
