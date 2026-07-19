"""CHOP-like channel generator nodes — Counter, Ramp, LFO, Beats, Noise1D, Envelope, Math, Logic, Blend.

These nodes generate time-based values that can be wired into any param
on any other node, replacing the built-in anim_mode system. They work like
TouchDesigner's CHOPs: a Counter drives simulation steps, an LFO drives
font_size, a Ramp drives gradient direction, etc.

All nodes output SCALAR values. The executor handles SCALAR→FIELD cross-wiring
automatically, so these can drive FIELD inputs too.
"""
from __future__ import annotations
import math
import random
from pathlib import Path

import numpy as np

from ..core.registry import method
from ..core.utils import seed_all


# ═══════════════════════════════════════════════════════════════════════════
# 1. Counter — integer, counts up/down per frame
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__counter__", name="Counter", category="channels",
        tags=["chop", "time", "integer", "generator"],
        inputs={"reset": "SCALAR", "step": "SCALAR"},
        outputs={"value": "SCALAR", "phase": "SCALAR"},
        params={
            "start": {"description": "counter start value", "default": 0},
            "end": {"description": "counter end value (inclusive)", "default": 100},
            "step_size": {"description": "increment per frame", "default": 1},
            "mode": {"description": "counter mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
        })
def method_counter(out_dir: Path, seed: int, params=None):
    """Integer counter that advances per frame.

    Counts from start to end, then wraps or reverses based on mode.
    Can be wired into simulation n_frames to control sub-stepping.

    Outputs:
        value (SCALAR): current count
        phase (SCALAR): normalized position 0→1 between start and end
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    start = int(params.get("start", 0))
    end = int(params.get("end", 100))
    step_size = int(params.get("step_size", 1))
    mode = params.get("mode", "loop")

    # SCALAR overrides
    reset_val = params.get("reset")
    if reset_val is not None:
        frame = int(reset_val)

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject an integer `frame` for CHOP generators. Derive the live
    # frame from the Timeline so the counter advances on every rendered frame
    # instead of staying pinned at frame 0 (which froze driver-driven graphs
    # and culled them as static in the shootout liveness gate).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    step_override = params.get("step")
    if step_override is not None:
        step_size = max(1, int(round(step_override)))

    total = end - start
    if total <= 0:
        return {"value": float(start), "phase": 0.0}

    raw = frame * step_size
    if mode == "once":
        val = min(start + raw, end)
    elif mode == "pingpong":
        cycle = raw % (total * 2)
        val = start + (cycle if cycle <= total else total * 2 - cycle)
    else:  # loop
        val = start + (raw % (total + 1))

    phase = (val - start) / total if total > 0 else 0.0
    return {"value": float(val), "phase": float(phase)}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ramp — float, sweeps 0→1 over N frames
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__ramp__", name="Ramp", category="channels",
        tags=["chop", "time", "float", "generator"],
        inputs={"trigger": "SCALAR", "speed": "SCALAR"},
        outputs={"value": "SCALAR", "phase": "SCALAR"},
        params={
            "start": {"description": "ramp start value", "default": 0.0},
            "end": {"description": "ramp end value", "default": 1.0},
            "duration_frames": {"description": "frames for one full ramp", "min": 1, "max": 10000, "default": 48},
            "easing": {"description": "ramp easing function",
                       "choices": ["linear", "ease_in", "ease_out", "smoothstep"],
                       "default": "linear"},
            "mode": {"description": "ramp mode",
                     "choices": ["once", "loop", "pingpong"],
                     "default": "loop"},
        })
def method_ramp(out_dir: Path, seed: int, params=None):
    """Float ramp that sweeps from start to end over duration_frames.

    Outputs:
        value (SCALAR): current ramp value
        phase (SCALAR): normalized position 0→1
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    start = float(params.get("start", 0.0))
    end = float(params.get("end", 1.0))
    duration = max(1, int(params.get("duration_frames", 48)))
    easing = params.get("easing", "linear")
    mode = params.get("mode", "loop")

    # SCALAR overrides
    trigger_val = params.get("trigger")
    if trigger_val is not None:
        frame = int(trigger_val)

    # Derive the live frame from the injected Timeline (see Counter for why).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    speed_override = params.get("speed")
    if speed_override is not None:
        duration = max(1, int(duration / max(0.01, float(speed_override))))

    raw_phase = (frame % duration) / duration if mode != "once" else min(frame / duration, 1.0)
    if mode == "pingpong":
        cycle = frame % (duration * 2)
        raw_phase = cycle / duration if cycle <= duration else (duration * 2 - cycle) / duration

    # Apply easing
    p = raw_phase
    if easing == "ease_in":
        p = p * p
    elif easing == "ease_out":
        p = 1 - (1 - p) * (1 - p)
    elif easing == "smoothstep":
        p = p * p * (3 - 2 * p)

    val = start + (end - start) * p
    return {"value": float(val), "phase": float(raw_phase)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. LFO — oscillates between min and max
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__lfo__", name="LFO", category="channels",
        tags=["chop", "time", "oscillator", "generator"],
        inputs={"rate": "SCALAR", "phase_offset": "SCALAR", "amplitude": "SCALAR"},
        outputs={"value": "SCALAR", "bipolar": "SCALAR"},
        params={
            "waveform": {"description": "LFO waveform",
                         "choices": ["sine", "triangle", "saw", "square", "random", "noise"],
                         "default": "sine"},
            "min": {"description": "minimum output value", "default": 0.0},
            "max": {"description": "maximum output value", "default": 1.0},
            "rate": {"description": "cycles per second (Hz)", "default": 0.5},
            "phase": {"description": "initial phase offset 0-1", "default": 0.0},
            "bipolar": {"description": "output -1 to 1 instead of min to max", "default": False},
        })
def method_lfo(out_dir: Path, seed: int, params=None):
    """Low Frequency Oscillator — generates periodic waveforms.

    Outputs:
        value (SCALAR): waveform output in [min, max] or [-1, 1] if bipolar
        bipolar (SCALAR): always -1 to 1
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    frame = int(params.get("frame", 0))
    fps = float(params.get("fps", 24.0))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject an integer `frame` (nor a `time`) for CHOP generators.
    # Derive the live frame from the Timeline's global_frame (which advances
    # every rendered frame) so the LFO advances instead of staying pinned at
    # frame 0. NOTE: we use global_frame, not the Timeline's `phase` attribute,
    # because the executor's make_timeline() does not set phase (it stays 0),
    # whereas global_frame is always correct. The other CHOP nodes (__counter__,
    # __ramp__, __beats__, __envelope__) already derive frame this way.
    total_frames_for_phase = int(params.get("total_frames", 24))
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))
            fps = float(getattr(_tl, "fps", fps))
            total_frames_for_phase = int(getattr(_tl, "total_frames", total_frames_for_phase))
    # Derive the cyclic phase from the live frame so t advances per frame.
    # `rate` is documented as cycles-per-second (Hz) ... [see note below]
    t = (frame / max(1, total_frames_for_phase - 1)) * (2.0 * math.pi)

    waveform = params.get("waveform", "sine")
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    rate = float(params.get("rate", 0.5))
    phase_offset = float(params.get("phase", 0.0))
    bipolar_mode = params.get("bipolar", False)
    if isinstance(bipolar_mode, str):
        bipolar_mode = bipolar_mode.lower() in ("true", "1", "yes")

    # SCALAR overrides
    rate_override = params.get("rate")
    if rate_override is not None:
        rate = float(rate_override)
    phase_override = params.get("phase_offset")
    if phase_override is not None:
        phase_offset = float(phase_override)
    amp_override = params.get("amplitude")
    if amp_override is not None:
        max_val = min_val + float(amp_override)

    # Compute phase.
    # `rate` is documented as cycles-per-second (Hz): one full cycle spans
    # `fps / rate` frames, so phase advances by `2*pi*rate/fps` radians PER
    # FRAME (angular frequency omega). The legacy `phase = t*rate` (with
    # t = frame/total*2pi) made `rate` span cycles-per-CLIP, so any rate < 0.5
    # completed < half a cycle over the clip and square/saw/triangle collapsed
    # to DC (constant) output — the dominant cause of "static"/"flat" shootout
    # deaths for LFO-driven graphs. True Hz makes low-rate LFOs actually sweep.
    _omega = 2.0 * math.pi * rate / max(1.0, fps)
    phase = (frame * _omega + phase_offset * 2 * math.pi) % (2 * math.pi)

    if waveform == "sine":
        bipolar = math.sin(phase)
    elif waveform == "triangle":
        bipolar = 2 * abs(2 * (phase / (2 * math.pi) - math.floor(phase / (2 * math.pi) + 0.5))) - 1
    elif waveform == "saw":
        bipolar = 2 * (phase / (2 * math.pi) - math.floor(phase / (2 * math.pi) + 0.5))
    elif waveform == "square":
        bipolar = 1.0 if math.sin(phase) >= 0 else -1.0
    elif waveform == "random":
        # Step random: a new random value every few frames. The step cadence is
        # driven by `rate` (Hz, cycles-per-second — SAME semantics as the
        # continuous waveforms above, where omega = 2*pi*rate/fps) so the
        # `rate` control is LIVE. Previously this branch hardcoded
        # `frame // 6`, which made `rate` have NO effect whatsoever — a silent
        # dead param that inflated the shootout dead-clip rate for
        # random-LFO-driven graphs (the #1 dead-genome method is __lfo__).
        # We lay `n_steps` evenly across the clip and advance the random seed
        # once per step, so a higher rate yields more, faster random flips.
        clip_seconds = max(1e-3, total_frames_for_phase / max(1.0, fps))
        n_steps = max(1, int(round(rate * clip_seconds * 4.0)))  # ~4 random flips per Hz-second
        step_idx = int(frame * n_steps / max(1, total_frames_for_phase))
        rng = random.Random(seed + step_idx)
        bipolar = rng.uniform(-1, 1)
    elif waveform == "noise":
        # Perlin-like smooth random
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
    else:
        bipolar = 0.0

    if bipolar_mode:
        val = bipolar
    else:
        mid = (min_val + max_val) / 2
        amp = (max_val - min_val) / 2
        val = mid + bipolar * amp

    return {"value": float(val), "bipolar": float(bipolar)}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Beats/Clock — trigger pulses at musical intervals
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__beats__", name="Beats", category="channels",
        tags=["chop", "time", "music", "generator"],
        inputs={"reset": "SCALAR", "swing": "SCALAR"},
        outputs={"beat": "SCALAR", "bar": "SCALAR", "trigger": "SCALAR"},
        params={
            "bpm": {"description": "beats per minute", "min": 20, "max": 300, "default": 120},
            "beats_per_bar": {"description": "beats per bar / time signature numerator", "min": 1, "max": 16, "default": 4},
            "swing": {"description": "swing amount 0-1", "default": 0.0},
            "fps": {"description": "frames per second for beat calculation", "min": 1, "max": 120, "default": 24},
        })
def method_beats(out_dir: Path, seed: int, params=None):
    """Musical beat generator — outputs beat phase, bar phase, and triggers.

    Outputs:
        beat (SCALAR): 0→1 phase within current beat
        bar (SCALAR): 0→1 phase within current bar
        trigger (SCALAR): 1.0 on first frame of each beat, 0 otherwise
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    bpm = float(params.get("bpm", 120))
    beats_per_bar = int(params.get("beats_per_bar", 4))
    swing = float(params.get("swing", 0.0))
    fps = float(params.get("fps", 24))

    # SCALAR overrides
    reset_val = params.get("reset")
    if reset_val is not None:
        frame = int(reset_val)
    swing_override = params.get("swing")
    if swing_override is not None:
        swing = float(swing_override)

    # Derive the live frame from the injected Timeline (see Counter for why).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    frames_per_beat = fps * 60.0 / bpm
    total_beats = frame / frames_per_beat

    beat_phase = (total_beats % 1.0)
    bar_phase = (total_beats % beats_per_bar) / beats_per_bar

    # Swing: delay every other beat
    if swing > 0:
        beat_idx = int(total_beats) % 2
        if beat_idx == 1:
            beat_phase = (beat_phase + swing) % 1.0

    # Trigger: 1 on first frame of each beat
    prev_beat = (frame - 1) / frames_per_beat
    trigger = 1.0 if int(prev_beat) != int(total_beats) else 0.0

    return {"value": float(beat_phase), "beat": float(beat_phase), "bar": float(bar_phase), "trigger": float(trigger)}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Noise1D — Perlin noise over time
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__noise1d__", name="Noise1D", category="channels",
        tags=["chop", "time", "noise", "generator"],
        inputs={"rate": "SCALAR", "seed_offset": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "min": {"description": "minimum output value", "default": 0.0},
            "max": {"description": "maximum output value", "default": 1.0},
            "rate": {"description": "noise rate (higher = faster variation)", "default": 0.5},
            "smooth": {"description": "interpolation smoothing (0=step, 1=linear, 2=smoothstep)", "default": 2},
        })
def method_noise1d(out_dir: Path, seed: int, params=None):
    """1D Perlin-like noise generator — smooth random values over time.

    Outputs:
        value (SCALAR): noise value in [min, max]
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    min_val = float(params.get("min", 0.0))
    max_val = float(params.get("max", 1.0))
    rate = float(params.get("rate", 0.5))
    smooth = int(params.get("smooth", 2))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject a `time` for CHOP generators. Derive the live phase from
    # the Timeline's global_frame (which advances every rendered frame) so the
    # noise advances instead of staying pinned at t=0 (which froze
    # driver-driven graphs and culled them as static — see __counter__ /
    # __lfo__ / __strobe__). NOTE: use global_frame, not the Timeline's `phase`
    # attribute, because make_timeline() does not set phase (it stays 0).
    if t == 0.0:
        _tl = params.get("_timeline")
        if _tl is not None:
            _gf = int(getattr(_tl, "global_frame", 0))
            _tf = int(getattr(_tl, "total_frames", 24))
            t = (_gf / max(1, _tf - 1)) * (2.0 * math.pi)

    # SCALAR overrides
    rate_override = params.get("rate")
    if rate_override is not None:
        rate = float(rate_override)
    seed_override = params.get("seed_offset")
    if seed_override is not None:
        seed = seed + int(seed_override * 1000)

    # Value noise: interpolate between random values at integer positions
    p = t * rate
    idx_a = int(math.floor(p))
    idx_b = idx_a + 1
    fade = p - idx_a

    if smooth == 0:
        # Step
        rng = random.Random(seed + idx_a)
        val = rng.uniform(min_val, max_val)
    elif smooth == 1:
        # Linear
        rng_a = random.Random(seed + idx_a)
        rng_b = random.Random(seed + idx_b)
        va = rng_a.uniform(min_val, max_val)
        vb = rng_b.uniform(min_val, max_val)
        val = va + (vb - va) * fade
    else:
        # Smoothstep
        fade = fade * fade * (3 - 2 * fade)
        rng_a = random.Random(seed + idx_a)
        rng_b = random.Random(seed + idx_b)
        va = rng_a.uniform(min_val, max_val)
        vb = rng_b.uniform(min_val, max_val)
        val = va + (vb - va) * fade

    return {"value": float(val)}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Envelope — ADSR-style shape
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__envelope__", name="Envelope", category="channels",
        tags=["chop", "time", "adsr", "generator"],
        inputs={"trigger": "SCALAR", "gate": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "attack": {"description": "attack time in frames", "min": 0, "max": 1000, "default": 10},
            "decay": {"description": "decay time in frames", "min": 0, "max": 1000, "default": 20},
            "sustain": {"description": "sustain level 0-1", "default": 0.7},
            "release": {"description": "release time in frames", "min": 0, "max": 1000, "default": 50},
            "sustain_level": {"description": "sustain level (alias)", "default": 0.7},
            "loop": {"description": "loop the envelope", "default": False},
        })
def method_envelope(out_dir: Path, seed: int, params=None):
    """ADSR envelope generator — triggered by a SCALAR input.

    When trigger goes from 0→1, the envelope starts its attack phase.
    When gate goes to 0, the envelope enters release phase.

    Outputs:
        value (SCALAR): envelope amplitude 0→1
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    attack = int(params.get("attack", 10))
    decay = int(params.get("decay", 20))
    sustain = float(params.get("sustain", 0.7))
    release = int(params.get("release", 50))
    loop = params.get("loop", False)
    if isinstance(loop, str):
        loop = loop.lower() in ("true", "1", "yes")

    # Use sustain param, fall back to sustain_level
    sustain = float(params.get("sustain_level", sustain))

    # SCALAR overrides
    trigger_val = params.get("trigger")
    gate_val = params.get("gate")

    # Derive the live frame from the injected Timeline (see Counter for why).
    if frame == 0:
        _tl = params.get("_timeline")
        if _tl is not None:
            frame = int(getattr(_tl, "global_frame", 0))

    # Simple model: trigger starts attack, gate holds sustain
    if trigger_val is not None and trigger_val > 0:
        trigger_frame = frame
    else:
        trigger_frame = 0

    if gate_val is not None and gate_val <= 0:
        # Release phase
        elapsed = frame - trigger_frame - attack - decay
        if elapsed < 0:
            elapsed = 0
        if elapsed >= release:
            val = 0.0
        else:
            val = sustain * (1 - elapsed / release)
    else:
        # Attack → Decay → Sustain
        elapsed = frame - trigger_frame
        if elapsed < attack:
            val = elapsed / attack
        elif elapsed < attack + decay:
            val = 1 - (1 - sustain) * (elapsed - attack) / decay
        else:
            val = sustain

    if loop and val <= 0:
        val = 0.0  # Hold at zero until next trigger

    return {"value": float(val)}


# ═══════════════════════════════════════════════════════════════════════════
# 7. Math — arithmetic operations on two SCALAR inputs
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__math__", name="Math", category="channels",
        tags=["chop", "math", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "operation": {"description": "math operation",
                          "choices": ["add", "sub", "mul", "div", "mod", "pow",
                                      "min", "max", "map_range", "clamp", "abs", "round",
                                      "floor", "ceil", "negate", "reciprocal"],
                          "default": "add"},
            "a_default": {"description": "default value for input A when not wired", "default": 0.0},
            "b_default": {"description": "default value for input B when not wired", "default": 1.0},
            "map_src_min": {"description": "map_range: source range min", "default": 0.0},
            "map_src_max": {"description": "map_range: source range max", "default": 1.0},
            "map_dst_min": {"description": "map_range: destination range min", "default": 0.0},
            "map_dst_max": {"description": "map_range: destination range max", "default": 1.0},
            "clamp_min": {"description": "clamp: minimum value", "default": 0.0},
            "clamp_max": {"description": "clamp: maximum value", "default": 1.0},
        })
def method_math(out_dir: Path, seed: int, params=None):
    """Math operations on two SCALAR inputs.

    Accepts wired SCALAR inputs A and B, with fallback defaults.
    Supports 16 operations including map_range and clamp.

    Outputs:
        value (SCALAR): result of the operation
    """
    if params is None:
        params = {}
    seed_all(seed)

    op = params.get("operation", "add")
    a = float(params.get("a", params.get("a_default", 0.0)))
    b = float(params.get("b", params.get("b_default", 1.0)))

    # SCALAR overrides (from wired inputs)
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)

    if op == "add":
        val = a + b
    elif op == "sub":
        val = a - b
    elif op == "mul":
        val = a * b
    elif op == "div":
        val = a / b if b != 0 else 0.0
    elif op == "mod":
        val = a % b if b != 0 else 0.0
    elif op == "pow":
        val = a ** b
    elif op == "min":
        val = min(a, b)
    elif op == "max":
        val = max(a, b)
    elif op == "map_range":
        src_min = float(params.get("map_src_min", 0.0))
        src_max = float(params.get("map_src_max", 1.0))
        dst_min = float(params.get("map_dst_min", 0.0))
        dst_max = float(params.get("map_dst_max", 1.0))
        if src_max != src_min:
            norm = (a - src_min) / (src_max - src_min)
        else:
            norm = 0.0
        val = dst_min + norm * (dst_max - dst_min)
    elif op == "clamp":
        cmin = float(params.get("clamp_min", 0.0))
        cmax = float(params.get("clamp_max", 1.0))
        val = max(cmin, min(cmax, a))
    elif op == "abs":
        val = abs(a)
    elif op == "round":
        val = round(a)
    elif op == "floor":
        val = math.floor(a)
    elif op == "ceil":
        val = math.ceil(a)
    elif op == "negate":
        val = -a
    elif op == "reciprocal":
        val = 1.0 / a if a != 0 else 0.0
    else:
        val = 0.0

    return {"value": float(val)}


# ═══════════════════════════════════════════════════════════════════════════
# 8. Logic — comparison and selection
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__logic__", name="Logic", category="channels",
        tags=["chop", "logic", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR", "control": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "operation": {"description": "logic operation",
                          "choices": ["greater", "less", "equal", "not_equal",
                                      "select", "gate", "hold", "toggle", "pulse"],
                          "default": "greater"},
            "true_value": {"description": "value when condition is true", "default": 1.0},
            "false_value": {"description": "value when condition is false", "default": 0.0},
            "threshold": {"description": "comparison threshold", "default": 0.5},
        })
def method_logic(out_dir: Path, seed: int, params=None):
    """Logic operations — comparison, selection, gating.

    Accepts wired SCALAR inputs A, B, and Control.

    Outputs:
        value (SCALAR): result of the logic operation
    """
    if params is None:
        params = {}
    seed_all(seed)

    op = params.get("operation", "greater")
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    control = float(params.get("control", 0.0))
    true_val = float(params.get("true_value", 1.0))
    false_val = float(params.get("false_value", 0.0))
    threshold = float(params.get("threshold", 0.5))

    # SCALAR overrides
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)
    control_wired = params.get("control")
    if control_wired is not None:
        control = float(control_wired)

    if op == "greater":
        val = true_val if a > b else false_val
    elif op == "less":
        val = true_val if a < b else false_val
    elif op == "equal":
        val = true_val if abs(a - b) < threshold else false_val
    elif op == "not_equal":
        val = true_val if abs(a - b) >= threshold else false_val
    elif op == "select":
        val = a if control > threshold else b
    elif op == "gate":
        val = a if control > threshold else 0.0
    elif op == "hold":
        # Hold last value when control is above threshold
        val = a if control > threshold else float(params.get("_held", a))
        params["_held"] = val
    elif op == "toggle":
        # Toggle between true_val and false_val on each control pulse
        prev = float(params.get("_prev_control", 0.0))
        state = float(params.get("_toggle_state", false_val))
        if control > threshold and prev <= threshold:
            state = true_val if state == false_val else false_val
        params["_prev_control"] = control
        params["_toggle_state"] = state
        val = state
    elif op == "pulse":
        # Output true_val for one frame when control crosses threshold
        prev = float(params.get("_prev_pulse", 0.0))
        val = true_val if control > threshold and prev <= threshold else false_val
        params["_prev_pulse"] = control
    else:
        val = 0.0

    return {"value": float(val)}


# ═══════════════════════════════════════════════════════════════════════════
# 9. Blend — mix/lerp between two values
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__blend__", name="Blend", category="channels",
        tags=["chop", "mix", "operator"],
        inputs={"a": "SCALAR", "b": "SCALAR", "mix": "SCALAR"},
        outputs={"value": "SCALAR"},
        params={
            "mode": {"description": "blend mode",
                     "choices": ["lerp", "add", "multiply", "screen", "overlay"],
                     "default": "lerp"},
            "a_default": {"description": "default value for input A", "default": 0.0},
            "b_default": {"description": "default value for input B", "default": 1.0},
            "mix_default": {"description": "default mix factor", "default": 0.5},
        })
def method_blend(out_dir: Path, seed: int, params=None):
    """Blend between two SCALAR values using various modes.

    Outputs:
        value (SCALAR): blended result
    """
    if params is None:
        params = {}
    seed_all(seed)

    mode = params.get("mode", "lerp")
    a = float(params.get("a", params.get("a_default", 0.0)))
    b = float(params.get("b", params.get("b_default", 1.0)))
    mix = float(params.get("mix", params.get("mix_default", 0.5)))

    # SCALAR overrides
    a_wired = params.get("a")
    if a_wired is not None:
        a = float(a_wired)
    b_wired = params.get("b")
    if b_wired is not None:
        b = float(b_wired)
    mix_wired = params.get("mix")
    if mix_wired is not None:
        mix = float(mix_wired)

    mix = max(0.0, min(1.0, mix))

    if mode == "lerp":
        val = a + (b - a) * mix
    elif mode == "add":
        val = a + b * mix
    elif mode == "multiply":
        val = a * (b * mix + (1 - mix))
    elif mode == "screen":
        val = 1 - (1 - a) * (1 - b * mix)
    elif mode == "overlay":
        val = 2 * a * b * mix if a < 0.5 else 1 - 2 * (1 - a) * (1 - b * mix)
    else:
        val = a

    return {"value": float(val)}


# ═══════════════════════════════════════════════════════════════════════════
# 10. Strobe — periodic on/off gating
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__strobe__", name="Strobe", category="channels",
        tags=["chop", "time", "gate", "generator"],
        inputs={"rate": "SCALAR", "duty_cycle": "SCALAR"},
        outputs={"value": "SCALAR", "trigger": "SCALAR"},
        params={
            "rate": {"description": "strobe rate in Hz", "default": 2.0},
            "duty_cycle": {"description": "fraction of cycle that is on (0-1)", "default": 0.5},
            "on_value": {"description": "value when gate is open", "default": 1.0},
            "off_value": {"description": "value when gate is closed", "default": 0.0},
        })
def method_strobe(out_dir: Path, seed: int, params=None):
    """Periodic on/off gate — like a square wave with adjustable duty cycle.

    Replaces freeze_frame, spark, and pulse animation modes.
    Wire Strobe.value → inject_rate for periodic life injection.
    Wire Strobe.value → speed for freeze-frame strobe effect.

    Outputs:
        value (SCALAR): on_value when gate open, off_value when closed
        trigger (SCALAR): 1.0 on rising edge, 0 otherwise
    """
    if params is None:
        params = {}
    seed_all(seed)

    t = float(params.get("time", 0.0))
    rate = float(params.get("rate", 2.0))
    duty = float(params.get("duty_cycle", 0.5))
    on_val = float(params.get("on_value", 1.0))
    off_val = float(params.get("off_value", 0.0))

    # The GraphExecutor injects a per-frame Timeline (params["_timeline"]) but
    # does NOT inject a `time` for CHOP generators. Derive the live phase from
    # the Timeline's global_frame so the strobe advances every rendered frame
    # instead of staying pinned at t=0 (which froze driver-driven graphs and
    # culled them as static — see __counter__ / __lfo__ / __noise1d__).
    if t == 0.0:
        _tl = params.get("_timeline")
        if _tl is not None:
            _gf = int(getattr(_tl, "global_frame", 0))
            _tf = int(getattr(_tl, "total_frames", 24))
            t = (_gf / max(1, _tf - 1)) * (2.0 * math.pi)

    # SCALAR overrides
    rate_override = params.get("rate")
    if rate_override is not None:
        rate = float(rate_override)
    duty_override = params.get("duty_cycle")
    if duty_override is not None:
        duty = float(duty_override)

    duty = max(0.01, min(0.99, duty))
    phase = (t * rate) % 1.0
    gate_open = phase < duty
    val = on_val if gate_open else off_val

    # Trigger on rising edge
    prev_phase = ((t - 1.0 / 24.0) * rate) % 1.0
    trigger = 1.0 if (prev_phase >= duty and gate_open) else 0.0

    return {"value": float(val), "trigger": float(trigger)}


# ═══════════════════════════════════════════════════════════════════════════
# 11. Burst — generates a burst of pulses
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__burst__", name="Burst", category="channels",
        tags=["chop", "time", "pulse", "generator"],
        inputs={"trigger": "SCALAR", "rate": "SCALAR"},
        outputs={"value": "SCALAR", "active": "SCALAR"},
        params={
            "n_pulses": {"description": "number of pulses per burst", "min": 1, "max": 100, "default": 5},
            "pulse_interval": {"description": "frames between pulses in a burst", "min": 1, "max": 100, "default": 6},
            "pulse_width": {"description": "frames each pulse stays high", "min": 1, "max": 20, "default": 1},
            "amplitude": {"description": "pulse amplitude", "default": 1.0},
            "loop": {"description": "auto-retrigger when burst ends", "default": True},
        })
def method_burst(out_dir: Path, seed: int, params=None):
    """Generates a burst of pulses on trigger.

    Replaces glider_stream animation mode — wire Burst.value → inject_rate
    to create periodic glider-like injections.

    Outputs:
        value (SCALAR): pulse amplitude when active, 0 otherwise
        active (SCALAR): 1.0 during burst, 0 otherwise
    """
    if params is None:
        params = {}
    seed_all(seed)

    frame = int(params.get("frame", 0))
    n_pulses = int(params.get("n_pulses", 5))
    interval = int(params.get("pulse_interval", 6))
    width = int(params.get("pulse_width", 1))
    amp = float(params.get("amplitude", 1.0))
    loop = params.get("loop", True)
    if isinstance(loop, str):
        loop = loop.lower() in ("true", "1", "yes")

    # SCALAR overrides
    trigger_val = params.get("trigger")
    rate_override = params.get("rate")
    if rate_override is not None:
        interval = max(1, int(interval / max(0.01, float(rate_override))))

    # Determine if we're in a burst
    burst_duration = n_pulses * interval
    burst_start = 0

    if trigger_val is not None and trigger_val > 0:
        burst_start = frame
    elif loop:
        burst_start = (frame // burst_duration) * burst_duration

    elapsed = frame - burst_start
    if 0 <= elapsed < burst_duration:
        pulse_idx = elapsed // interval
        within_pulse = (elapsed % interval) < width
        val = amp if within_pulse else 0.0
        active = 1.0
    else:
        val = 0.0
        active = 0.0

    return {"value": float(val), "active": float(active)}


# ═══════════════════════════════════════════════════════════════════════════
# 12. AgeHeat — maps cell age to heat color
# ═══════════════════════════════════════════════════════════════════════════

@method(id="__age_heat__", name="AgeHeat", category="channels",
        tags=["chop", "color", "age", "generator"],
        inputs={"age": "SCALAR", "max_age": "SCALAR"},
        outputs={"value": "SCALAR", "r": "SCALAR", "g": "SCALAR", "b": "SCALAR"},
        params={
            "mode": {"description": "age coloring mode",
                     "choices": ["heat", "cool", "rainbow", "mono"],
                     "default": "heat"},
            "max_age_default": {"description": "max age for normalization", "default": 100.0},
        })
def method_age_heat(out_dir: Path, seed: int, params=None):
    """Maps a scalar age value to a color output.

    Replaces the f2l (frames-to-live) animation mode. Wire this into
    hue_shift on the CA node to get age-based coloring.

    Outputs:
        value (SCALAR): normalized age 0-1
        r (SCALAR): red channel 0-1
        g (SCALAR): green channel 0-1
        b (SCALAR): blue channel 0-1
    """
    if params is None:
        params = {}
    seed_all(seed)

    age = float(params.get("age", 0.0))
    max_age = float(params.get("max_age", params.get("max_age_default", 100.0)))
    mode = params.get("mode", "heat")

    # SCALAR overrides
    age_override = params.get("age")
    if age_override is not None:
        age = float(age_override)
    max_age_override = params.get("max_age")
    if max_age_override is not None:
        max_age = float(max_age_override)

    norm = max(0.0, min(1.0, age / max_age)) if max_age > 0 else 0.0

    if mode == "heat":
        r = min(1.0, norm * 2.0)
        g = max(0.0, min(1.0, norm * 2.0 - 1.0))
        b = max(0.0, norm * 3.0 - 2.0)
    elif mode == "cool":
        r = max(0.0, norm * 3.0 - 2.0)
        g = max(0.0, min(1.0, norm * 2.0 - 1.0))
        b = min(1.0, norm * 2.0)
    elif mode == "rainbow":
        h = norm * 0.5
        r = 0.5 + 0.5 * math.sin(h * 2 * math.pi)
        g = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 2.094)
        b = 0.5 + 0.5 * math.sin(h * 2 * math.pi + 4.189)
    else:  # mono
        r = g = b = norm

    return {"value": float(norm), "r": float(r), "g": float(g), "b": float(b)}
