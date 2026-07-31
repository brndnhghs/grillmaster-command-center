"""barnsley — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("barnsley", 'Barnsley fern approximation', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0 - 1.5;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 200; i++) {\n        if (i >= u_iterations) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 0.5 + t), cos(fi * 0.3 + t * 0.7));\n        float dx = uv.x - p.x * 0.5;\n        float dy = uv.y - p.y * 0.8 - 0.5;\n        v += 0.001 / (dx*dx + dy*dy + 0.001);\n    }\n    f_color = vec4(v * 0.2, v * 0.8, v * 0.2, 1.0);\n}\n',
          uniforms={
  "iterations": {
    "glsl": "int",
    "min": 20,
    "max": 200,
    "default": 100,
    "description": "sample iterations"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "sway speed"
  }
})