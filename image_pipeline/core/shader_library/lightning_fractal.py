"""lightning_fractal — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("lightning_fractal", 'Fractal lightning branching', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    vec2 p = vec2(0.0);\n    float v = 0.0;\n    for (int i = 0; i < 128; i++) {\n        if (i >= u_segments) break;\n        float fi = float(i);\n        p += vec2(sin(fi * 0.3 + t), cos(fi * 0.7 + t * 0.5)) * 0.02;\n        float d = length(uv - p);\n        v += 0.02 / (d + 0.01);\n    }\n    f_color = vec4(v * 0.3, v * 0.5, v, 1.0);\n}\n',
          uniforms={
  "segments": {
    "glsl": "int",
    "min": 8,
    "max": 128,
    "default": 64,
    "description": "branch segments"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.2,
    "description": "flicker speed"
  }
})