"""Legacy combined GPU procedural node."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ...core.registry import method
from ...core.utils import save, seed_all, get_canvas
from ...core.animation import capture_frame
from ...core.shaders import render_shader, SHADERS
from ._shared import SHADER_NAMES


@method(
    id="82",
    name="GPU Procedural Shaders",
    category="ml_models",
    new_image_contract=True,
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
    """GPU Procedural Shaders — generate imagery from GLSL fragment shaders on the GPU."""
    if params is None:
        params = {}

    raw_time = float(params.get("time", 0.0))
    anim_speed = float(params.get("anim_speed", 1.0))
    t = raw_time * anim_speed
    seed_all(seed)

    shader_name = params.get("shader", "domain_warp")
    if shader_name not in SHADERS or SHADERS[shader_name]["type"] != "procedural":
        shader_name = "domain_warp"

    p = (
        float(params.get("p1", 0.5)),
        float(params.get("p2", 0.5)),
        float(params.get("p3", 0.5)),
        float(params.get("p4", 0.5)),
    )

    cw, ch = get_canvas()
    result = render_shader(shader_name, (cw, ch), p, t)
    arr = np.array(result, dtype=np.uint8)
    capture_frame("82", arr.astype(np.float32) / 255.0)
    save(arr, f"82_{shader_name}_{seed:04d}.png", out_dir)
    return out_dir / f"82_{shader_name}_{seed:04d}.png"
