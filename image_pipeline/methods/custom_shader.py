"""Custom GLSL Shader node — live GLSL editor with hot-reload.

The user writes raw GLSL (just the void main(){} body).
The standard prologue (uniforms u_resolution, u_time, u_params, u_texture,
plus helpers rot/hash21/noise/fbm) is injected automatically.

Works as both a procedural generator (no input) and a filter (image_in wired).
Compile errors propagate as exceptions so the executor surfaces them in
node_errors, which the UI picks up and displays in the glsl-err panel.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from ..core.registry import method
from ..core.utils import get_canvas
from ..core.shaders import render_custom_shader, CUSTOM_SHADER_TEMPLATE


@method(
    id="__custom_shader__",
    name="Custom GLSL Shader",
    category="gpu_shaders",
    new_image_contract=True,
    tags=["gpu", "glsl", "custom", "fast"],
    inputs={"image_in": "IMAGE"},
    params={
        "glsl_code": {
            "content": True,
            # Audit tool reads content_probe to pick GLSL-valid probe strings
            # (prose would be rejected by the compiler and report a false ERROR).
            "content_probe": [
                "void main() { f_color = vec4(vec3(v_uv.x), 1.0); }",
                "void main() { f_color = vec4(vec3(1.0 - v_uv.y), 1.0); }",
            ],
            "description": "GLSL fragment shader body (void main)",
            "default": CUSTOM_SHADER_TEMPLATE,
            "multiline": True,
        },
        "p1": {"description": "param 1 → u_params.x", "min": 0.0, "max": 1.0, "default": 0.5},
        "p2": {"description": "param 2 → u_params.y", "min": 0.0, "max": 1.0, "default": 0.5},
        "p3": {"description": "param 3 → u_params.z", "min": 0.0, "max": 1.0, "default": 0.5},
        "p4": {"description": "param 4 → u_params.w", "min": 0.0, "max": 1.0, "default": 0.5},
    },
)
def method_custom_shader(out_dir: Path, seed: int, params=None):
    """Compile and run a user-written GLSL fragment shader.

    Raises RuntimeError on GLSL compile failure — the executor catches it,
    shows a red error placeholder, and surfaces the message in node_errors.
    """
    if params is None:
        params = {}

    glsl_body = params.get("glsl_code", CUSTOM_SHADER_TEMPLATE)
    t = float(params.get("time", 0.0))
    p = tuple(float(params.get(f"p{i}", 0.5)) for i in range(1, 5))
    inp = params.get("_input_image")
    cw, ch = get_canvas()

    pil_img = render_custom_shader(glsl_body, (cw, ch), p, t, inp)
    arr = np.array(pil_img, dtype=np.uint8).astype(np.float32) / 255.0
    return {"image": arr}
