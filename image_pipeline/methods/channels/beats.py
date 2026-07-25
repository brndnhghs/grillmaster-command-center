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

@method(id="__beats__", name="Beats", category="channels",
        tags=["chop", "time", "music", "generator"],
        inputs={"reset": "SCALAR", "swing": "SCALAR"},
        outputs={"value": "SCALAR", "beat": "SCALAR", "bar": "SCALAR", "trigger": "SCALAR"},
        runtime={
            "value": {
                "type": "numeric",
                "label": "Value",
                "observable": True
            },
            "beat": {
                "type": "output",
                "label": "Beat",
                "observable": True
            },
            "bar": {
                "type": "output",
                "label": "Bar",
                "observable": True
            },
            "trigger": {
                "type": "event",
                "label": "Trigger",
                "observable": True
            }
        },
        signal={
            "reset": "control",
            "swing": "numeric",
            "value": "output",
            "beat": "output",
            "bar": "output",
            "trigger": "event"
        },
        params={
            "bpm": {"description": "beats per minute", "min": 20, "max": 300, "default": 120},
            "beats_per_bar": {"description": "beats per bar / time signature numerator", "min": 1, "max": 16, "default": 4},
            "swing": {"description": "swing amount 0-1", "default": 0.0},
            "fps": {"description": "frames per second for beat calculation", "min": 1, "max": 120, "default": 24},
        })
def method_beats(out_dir: Path, seed: int, params=None):
    """Musical beat generator — outputs beat phase, bar phase, and triggers.

    Outputs:
        beat (SCALAR): 0->1 phase within current beat
        bar (SCALAR): 0->1 phase within current bar
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
