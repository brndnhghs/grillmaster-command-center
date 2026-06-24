"""
GPU Procedural Shaders — method #82.
Generates imagery from scratch using GLSL fragment shaders on the GPU.
25 shaders available: fractals, noise, cellular, fire, smoke, terrain, etc.
Requires ModernGL (Apple M1 Metal, GL 4.1 core).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image

from ..core.registry import method
from ..core.utils import save, seed_all
from ..core.animation import capture_frame
from ..core.shaders import render_procedural, list_shaders, SHADERS

SHADER_NAMES = sorted([k for k, v in SHADERS.items() if v["type"] == "procedural"])


@method(
    id="82",
    name="GPU Procedural Shaders",
    category="ml_models",
    tags=["gpu", "glsl", "fast", "expanded"],
    params={
        "shader": {
            "description": f"shader name: {', '.join(SHADER_NAMES)}",
            "default": "domain_warp",
        },
        "p1": {"description": "generic float param 1", "min": 0.0, "max": 1.0, "default": 0.5},
        "p2": {"description": "generic float param 2", "min": 0.0, "max": 1.0, "default": 0.5},
        "p3": {"description": "generic float param 3", "min": 0.0, "max": 1.0, "default": 0.5},
        "p4": {"description": "generic float param 4", "min": 0.0, "max": 1.0, "default": 0.5},
        "anim_mode": {"description": "animation mode", "choices": ["none", "animate"], "default": "none"},
        "anim_speed": {"description": "animation speed multiplier", "min": 0.0, "max": 5.0, "default": 1.0},
    },
)
def method_gpu_procedural(out_dir: Path, seed: int, params=None):
    """GPU Procedural Shaders — generate imagery from GLSL fragment shaders on the GPU.

    Renders procedural textures, fractals, noise, cellular patterns, fire, smoke,
    terrain, and more using 25+ GLSL fragment shaders via ModernGL (Apple M1 Metal).

    Parameters:
        shader (str): Shader name (domain_warp, mandelbrot, julia, fire, smoke, terrain, etc.)
        p1 (float): Generic float param 1 (0-1, default 0.5)
        p2 (float): Generic float param 2 (0-1, default 0.5)
        p3 (float): Generic float param 3 (0-1, default 0.5)
        p4 (float): Generic float param 4 (0-1, default 0.5)
        time (float): Animation time offset (0-6.28, default 0.0)
        anim_mode (str): Animation mode (none, animate)
        anim_speed (float): Animation speed multiplier (0-5, default 1.0)
    """
    if params is None:
        params = {}

    raw_time = float(params.get("time", 0.0))
    anim_mode = params.get("anim_mode", "none")
    anim_speed = float(params.get("anim_speed", 1.0))
    t = raw_time * anim_speed
    seed_all(seed)

    shader_name = params.get("shader", "domain_warp")
    if shader_name not in SHADERS or SHADERS[shader_name]["type"] != "procedural":
        # Fallback
        shader_name = "domain_warp"

    p = (
        float(params.get("p1", 0.5)),
        float(params.get("p2", 0.5)),
        float(params.get("p3", 0.5)),
        float(params.get("p4", 0.5)),
    )

    result = render_procedural(shader_name, resolution=(1024, 1024), params=p, time=t)
    arr = np.array(result, dtype=np.uint8)

    # Generate filename
    filename = f"82_{shader_name}_{seed:04d}.png"
    capture_frame("82", arr.astype(np.float32) / 255.0)
    save(arr, filename, out_dir)
    return out_dir / filename