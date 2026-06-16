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
        "time": {"description": "animation time offset", "min": 0.0, "max": 100.0, "default": 0.0},
    },
)
def method_gpu_procedural(out_dir: Path, seed: int, params=None):
    if params is None:
        params = {}

    t = float(params.get("time", 0.0))
    seed_all(seed + int(t * 100))

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