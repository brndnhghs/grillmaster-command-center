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

@method(id="__lfo__", name="LFO", category="channels",
        tags=["chop", "time", "oscillator", "generator"],
        inputs={"rate": "SCALAR", "phase_offset": "SCALAR", "amplitude": "SCALAR"},
        outputs={"value": "SCALAR", "bipolar": "SCALAR"},
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
            }
        },
        signal={
            "rate": "numeric",
            "phase_offset": "numeric",
            "amplitude": "numeric",
            "value": "output",
            "bipolar": "output"
        },
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
        bipolar_mode = bipolar_mode.lower() in ("True", "1", "yes")

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
    # to DC (constant) output — the dominant cause of "static"/"flat" render
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
        # dead param that inflated the dead-clip rate for
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
