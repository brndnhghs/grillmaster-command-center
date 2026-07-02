"""Detect whether a method is Architecture A (simulation) or B (stateless).

Architecture A methods:
  - Have an 'n_frames' param
  - Run their own internal simulation loop
  - Call capture_frame() to emit intermediate frames
  - Examples: Dynamic Fracture (145), Gray-Scott (32), Metaballs (53),
    Boids (34), Particle Flow (35), DLA (36), Reaction-Diffusion

Architecture B methods:
  - Are stateless generators driven by 'time' or '_timeline' or 'anim_mode'
  - One call = one frame
  - Examples: Fractals (07), Noise (05), Dither (13), Glitch (17)
"""
from __future__ import annotations

from .registry import MethodMeta


def detect_architecture(meta: MethodMeta) -> str:
    """Return 'A' for simulation methods, 'B' for stateless methods.

    Heuristics (checked in order):
    1. Has 'n_frames' param → Architecture A
    2. Has 'anim_mode' param with non-'none' default → Architecture A
    3. Tags contain 'simulation' or 'sim' → Architecture A
    4. Has 'time' or '_timeline' in params → Architecture B
    5. Default → Architecture B
    """
    params = meta.params or {}
    tags = set(t.lower() for t in (meta.tags or []))

    # 1. n_frames param is the strongest signal
    if "n_frames" in params:
        return "A"

    # 2. anim_mode with non-none default indicates internal animation loop
    anim_mode_spec = params.get("anim_mode")
    if isinstance(anim_mode_spec, dict):
        default = anim_mode_spec.get("default", "none")
        if default != "none":
            return "A"

    # 3. Tags
    if "simulation" in tags or "sim" in tags:
        return "A"

    # 4. Has time/timeline param → stateless, driven externally
    if "time" in params or "_timeline" in params:
        return "B"

    # 5. Default: stateless
    return "B"