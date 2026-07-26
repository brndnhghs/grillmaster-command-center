"""CHOP-like channel generator nodes.
Auto-imported by channels/__init__.py.
"""
from __future__ import annotations
import math
import random
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import seed_all

# ── Per-node state for play/pause/reset ──────────────────────────────────
# Keyed by _node_id (injected by GraphExecutor into run_params).
# Stores per-instance tracking: {"playing_frame": int, "prev_resetpulse": float,
#                                 "prev_reset": float, "prev_frame": int}.
# Populated whenever _node_id is available (i.e. when running through the
# graph executor — always in graph/live mode, never in standalone batch/test
# calls where backward-compat frame-based computation is used).
_LFO_STATE: dict[str, dict] = {}
# Lazy-prune counter — module-level to avoid Pyright function-attribute error
_LFO_PRUNE_COUNTER = 0


@method(id="__lfo__", name="LFO", category="channels",
        tags=["chop", "time", "oscillator", "generator"],
        inputs={"rate": "SCALAR", "phase_offset": "SCALAR", "amplitude": "SCALAR",
                "reset_in": "SCALAR", "octave": "SCALAR"},
        outputs={"value": "SCALAR", "bipolar": "SCALAR", "phase": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            },
            "bipolar": {
                "type": "output",
                "label": "Bipolar",
                "observable": True
            },
            "phase": {
                "type": "output",
                "label": "Phase",
                "observable": True
            }
        },
        signal={
            "rate": "numeric",
            "phase_offset": "numeric",
            "amplitude": "numeric",
            "reset_in": "control",
            "octave": "numeric",
            "value": "output",
            "bipolar": "output",
            "phase": "output"
        },
        params={
            "waveform": {"description": "WaveType ⊞ — The shape of the waveform",
                         "choices": ["sine", "triangle", "saw", "square",
                                     "random", "noise", "gaussian"],
                         "default": "sine"},
            "min": {"description": "minimum output value", "default": 0.0},
            "max": {"description": "maximum output value", "default": 1.0},
            "rate": {"description": "Frequency — cycles per second (Hz)", "default": 0.5},
            "phase": {"description": "initial phase offset 0-1", "default": 0.0},
            "bipolar": {"description": "output -1 to 1 instead of min to max",
                        "default": False},
            "play": {"description": "Oscillate when 1, stop when 0 (Play/Pause)",
                     "default": True},
            "offset": {"description": "Additive offset added to output", "default": 0.0},
            "amp": {"description": "Scale multiplier applied to output amplitude",
                    "default": 1.0},
            "bias": {"description": "Shape control: triangle=peak pos, square=duty",
                     "default": 0.0},
            "resetcondition": {
                "description": "ResetCondition ⊞ — How reset triggers",
                "choices": ["rising_edge", "falling_edge", "high", "low"],
                "default": "rising_edge"},
            "reset": {"description": "Reset output to 0 while On", "default": False},
            "resetpulse": {"description": "Instantly reset output to 0 (button)",
                           "default": False},
        })
def method_lfo(out_dir: Path, seed: int, params=None):
    """Low Frequency Oscillator — generates periodic waveforms.

    Outputs:
        value (SCALAR): waveform output in [min, max] or [-1, 1] if bipolar,
                        then scaled by amp, then offset added
        bipolar (SCALAR): always -1 to 1 before amp/offset
        phase (SCALAR): normalized cycle position 0..1
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    frame = int(params.get("frame", 0))
    fps = float(params.get("fps", 24.0))
    node_id = params.get("_node_id", "")

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject an integer `frame` (nor a `time`) for CHOP generators.
    # Derive the live frame from the Timeline's global_frame (which advances
    # every rendered frame) so the LFO advances instead of staying pinned at
    # frame 0.
    total_frames_for_phase = int(params.get("total_frames", 24))
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", fps))
            total_frames_for_phase = int(getattr(_tl, "total_frames",
                                                  total_frames_for_phase))

    # ── Params (accept both old and new names for backward compat) ──────
    waveform = params.get("waveform", "sine")
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    rate = float(params.get("rate", 0.5))
    phase_offset = float(params.get("phase", 0.0))
    bipolar_mode = params.get("bipolar", False)
    if isinstance(bipolar_mode, str):
        bipolar_mode = bipolar_mode.lower() in ("True", "1", "yes")

    # New params with safe defaults
    play = params.get("play", True)
    if isinstance(play, str):
        play = play.lower() in ("True", "1", "yes")
    offset = float(params.get("offset", 0.0))
    amp_scale = float(params.get("amp", 1.0))
    bias = float(params.get("bias", 0.0))
    resetcondition = params.get("resetcondition", "rising_edge")
    reset_val = params.get("reset", False)
    if isinstance(reset_val, str):
        reset_val = reset_val.lower() in ("True", "1", "yes")
    resetpulse_val = params.get("resetpulse", False)
    if isinstance(resetpulse_val, str):
        resetpulse_val = resetpulse_val.lower() in ("True", "1", "yes")

    # SCALAR input overrides
    rate_override = params.get("rate")
    if rate_override is not None:
        rate = float(rate_override)
    phase_override = params.get("phase_offset")
    if phase_override is not None:
        phase_offset = float(phase_override)
    amp_override = params.get("amplitude")
    if amp_override is not None:
        max_val = min_val + float(amp_override)

    # Octave Control input: alters rate exponentially (rate *= 2^octave)
    octave_override = params.get("octave")
    if octave_override is not None:
        rate *= 2.0 ** float(octave_override)

    # ── Stateful play/pause/reset ───────────────────────────────────────
    # When _node_id is present (graph executor), use accumulated state so
    # play/pause/reset interact correctly. Without _node_id (standalone
    # / test calls), fall back to pure frame-based computation (identical
    # to original behavior).
    global _LFO_PRUNE_COUNTER
    _LFO_PRUNE_COUNTER += 1

    reset_active = False
    if node_id:
        state = _LFO_STATE.setdefault(node_id, {
            "playing_frame": frame,
            "prev_resetpulse": 0.0,
            "prev_reset": 0.0,
            "prev_frame": frame,
        })

        # ── Frame delta — only accumulate when playing ──
        # NOTE: if the timeline frame regresses (scrubbing backward) we detect
        # it here and resync playing_frame so the LFO doesn't freeze.  The
        # `max(0, ...)` guard handles single-frame skips; the regression check
        # handles multi-frame backward jumps.
        _delta = max(0, frame - state["prev_frame"]) if play else 0
        state["prev_frame"] = frame
        if _delta <= 0 and not play and frame < state.get("prev_regression_check", frame):
            state["playing_frame"] = frame
        state["prev_regression_check"] = frame

        # ── Reset pulse (button — rising edge detection) ──
        if resetpulse_val and not state.get("prev_resetpulse", 0.0):
            reset_active = True
            state["playing_frame"] = 0
        state["prev_resetpulse"] = 1.0 if resetpulse_val else 0.0

        # ── Reset condition from SCALAR input ──
        if not reset_active:
            _reset_in = params.get("reset_in", 0.0)
            if isinstance(_reset_in, (int, float)):
                prev_reset = state.get("prev_reset", 0.0)
                if resetcondition == "rising_edge" and prev_reset <= 0.5 < _reset_in:
                    reset_active = True
                    state["playing_frame"] = 0
                elif resetcondition == "falling_edge" and prev_reset >= 0.5 > _reset_in:
                    reset_active = True
                    state["playing_frame"] = 0
                elif resetcondition == "high" and _reset_in > 0.5 and prev_reset <= 0.5:
                    reset_active = True
                    state["playing_frame"] = 0
                elif resetcondition == "low" and _reset_in < 0.5 and prev_reset >= 0.5:
                    reset_active = True
                    state["playing_frame"] = 0
                state["prev_reset"] = float(_reset_in)

        # ── Reset toggle ──
        if not reset_active and reset_val:
            reset_active = True
            state["playing_frame"] = 0

        # ── Advance frame only when playing — no jump on resume ──
        if not play:
            # Paused — hold playing_frame where it is
            pass
        elif reset_active:
            state["playing_frame"] = 0
        else:
            state["playing_frame"] += _delta

        # Use stateful frame
        frame = state["playing_frame"]

        # Lazy prune: only every ~1000 invocations
        if _LFO_PRUNE_COUNTER % 1000 == 0:
            _cutoff = frame - 7200
            for _nid in list(_LFO_STATE):
                if _LFO_STATE[_nid].get("playing_frame", 0) < _cutoff:
                    del _LFO_STATE[_nid]


    # ── Compute phase ───────────────────────────────────────────────────
    # `rate` is documented as cycles-per-second (Hz): one full cycle spans
    # `fps / rate` frames, so phase advances by `2*pi*rate/fps` radians PER
    # FRAME (angular frequency omega). True Hz makes low-rate LFOs sweep.
    _omega = 2.0 * math.pi * rate / max(1.0, fps)
    phase = (frame * _omega + phase_offset * 2 * math.pi) % (2 * math.pi)
    phase_norm = phase / (2 * math.pi)  # [0, 1) for frontend playhead

    # ── Compute waveform value in [-1, 1] ───────────────────────────────
    if waveform == "sine":
        bipolar = math.sin(phase)
    elif waveform == "triangle":
        # Standard triangle: phase_norm in [0, 1)
        # With bias: shift the peak position
        p = phase_norm
        peak = 0.5 + bias * 0.45  # bias ∈ [-1,1] maps peak to [0.05, 0.95]
        peak = max(0.05, min(0.95, peak))
        if p < peak:
            bipolar = -1 + 2 * (p / peak)
        else:
            bipolar = 1 - 2 * ((p - peak) / (1 - peak))
    elif waveform == "saw":
        bipolar = 2 * (phase / (2 * math.pi)
                       - math.floor(phase / (2 * math.pi) + 0.5))
    elif waveform == "square":
        # With bias: duty cycle control
        duty = 0.5 + bias * 0.45  # bias ∈ [-1,1] maps duty to [0.05, 0.95]
        duty = max(0.05, min(0.95, duty))
        p = phase_norm
        bipolar = 1.0 if p < duty else -1.0
    elif waveform == "random":
        # Step random: rate-responsive (same as current code)
        clip_seconds = max(1e-3, total_frames_for_phase / max(1.0, fps))
        n_steps = max(1, int(round(rate * clip_seconds * 4.0)))
        step_idx = int(frame * n_steps / max(1, total_frames_for_phase))
        rng = random.Random(seed + step_idx)
        bipolar = rng.uniform(-1, 1)
    elif waveform == "noise":
        # Perlin-like smooth random (same as current code)
        rng = random.Random(seed)
        p = phase / (2 * math.pi)
        idx_a = int(p * 10) % 10
        idx_b = (idx_a + 1) % 10
        fade = (p * 10) % 1
        fade = fade * fade * (3 - 2 * fade)  # smoothstep
        va = rng.uniform(-1, 1)
        rng = random.Random(seed + idx_b)
        vb = rng.uniform(-1, 1)
        bipolar = va + (vb - va) * fade
    elif waveform == "gaussian":
        # Smooth bell curve; bias shifts the mean
        p = phase_norm
        mean = 0.5 + bias * 0.4
        mean = max(0.05, min(0.95, mean))
        sigma = 0.15
        g = math.exp(-((p - mean) ** 2) / (2 * sigma * sigma))
        bipolar = g * 2 - 1  # map [0,1] to [-1,1]
    else:
        bipolar = 0.0

    # ── Apply amp scaling and offset ──
    # NOTE: offset is added AFTER range mapping (not scaled by _range), matching
    # the User Spec for LFO: "Values output from the CHOP can be scaled (amp)
    # and have an offset added to them."  The SCALAR `amplitude` input port
    # (legacy) sets max_val = min_val + amplitude and is distinct from `amp`.
    # The `reset_in` SCALAR input port was renamed from `reset` to avoid
    # collision with the `reset` bool toggle param.
    bipolar_amp = bipolar * amp_scale
    if reset_active:
        bipolar = 0.0
        bipolar_amp = 0.0
        phase_norm = 0.0
        val = 0.0
    elif bipolar_mode:
        val = bipolar_amp + offset
    else:
        mid = (min_val + max_val) / 2
        _range = (max_val - min_val) / 2
        val = mid + bipolar_amp * _range + offset

    return {"value": float(val), "bipolar": float(bipolar),
            "phase": float(phase_norm)}
