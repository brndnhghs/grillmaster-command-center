"""Timeline Node — global animation clock source.

Provides t, phase, speed, beat, and segment as SCALAR output ports.
This is the single source of truth for animation timing in the graph.
When present, all other nodes inherit their timing from this node.

Outputs:
  t:        SCALAR — normalized position [0, 1]
  phase:    SCALAR — cyclic phase [0, 2π)
  speed:    SCALAR — speed multiplier (default 1.0)
  beat:     SCALAR — 1.0 on beat frames, 0.0 otherwise
  segment:  SCALAR — which segment index (0, 1, 2, ...)

No image output — this is a pure data source node.
"""

from __future__ import annotations
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...core.registry import method
from ...core.utils import save, mn, seed_all, W, H, write_scalars


@method(
    id="__timeline__",
    name="Timeline",
    description="Timeline — system node.",
    category="system",
    tags=["system", "timeline", "clock", "animation"],
    timeout=10,
    inputs={},  # No image input — pure data source
    outputs={
        "t":       "SCALAR",
        "phase":   "SCALAR",
        "speed":   "SCALAR",
        "beat":    "SCALAR",
        "segment": "SCALAR",
    },
    params={
        "total_frames": {
            "description": "Total frames in the animation",
            "min": 1, "max": 600, "default": 120,
        },
        "fps": {
            "description": "Frames per second",
            "min": 1, "max": 60, "default": 24,
        },
        "speed": {
            "description": "Speed multiplier (methods opt in)",
            "min": 0.1, "max": 5.0, "default": 1.0,
        },
        "loop": {
            "description": "Loop animation (wrap t back to 0 at end)",
            "choices": ["true", "false"],
            "default": "true",
        },
        "beats_per_cycle": {
            "description": "Number of beats per full cycle",
            "min": 1, "max": 32, "default": 4,
        },
        "segments": {
            "description": "Number of named segments",
            "min": 1, "max": 16, "default": 4,
        },
    }
)
def method_timeline(out_dir: Path, seed: int, params=None):
    """Timeline node — writes scalars and a minimal placeholder image.

    The real work happens in GraphExecutor which injects _timeline into
    every node's run_params. This method just writes the scalars so they
    appear as wirable output ports, and creates a minimal placeholder
    image so the executor doesn't error on the PNG readback.
    """
    if params is None:
        params = {}

    tl = params.get("_timeline")
    if tl is not None:
        t_val = tl.t
        phase_val = tl.phase
        speed_val = tl.speed
        total_frames = tl.total_frames
        fps_val = tl.fps
    else:
        t_val = float(params.get("t", 0.0))
        phase_val = float(params.get("phase", t_val * 2.0 * math.pi))
        speed_val = float(params.get("speed", 1.0))
        total_frames = int(params.get("total_frames", 120))
        fps_val = int(params.get("fps", 24))

    loop = str(params.get("loop", "true")).lower() in ("true", "1", "yes")
    beats_per_cycle = int(params.get("beats_per_cycle", 4))
    n_segments = int(params.get("segments", 4))

    # Beat detection: 1.0 on beat frames, 0.0 otherwise
    beat_val = 0.0
    if beats_per_cycle > 0 and total_frames > 0:
        beat_interval = total_frames / beats_per_cycle
        frame = tl.global_frame if tl is not None else int(t_val * total_frames)
        if beat_interval > 0 and abs((frame % beat_interval) - 0) < 0.5:
            beat_val = 1.0

    # Segment index
    segment_val = min(n_segments - 1, int(t_val * n_segments))

    # Write scalars so they appear as wirable output ports
    write_scalars(
        out_dir,
        t=t_val,
        phase=phase_val,
        speed=speed_val,
        beat=beat_val,
        segment=float(segment_val),
    )

    # Write a minimal placeholder image (dark background with frame number)
    # so the executor doesn't error on PNG readback.
    img = np.full((H, W, 3), [10, 10, 15], dtype=np.uint8)
    save(img, mn(0, "Timeline"), out_dir)
    return img
