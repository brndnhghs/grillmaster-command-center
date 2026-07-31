"""bubble_chamber — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("bubble_chamber", 'Simulated bubble chamber trails', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 40; i++) {\n        if (i >= u_count) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 1.7 + t * 0.5), cos(fi * 2.3 + t * 0.7)) * 0.8;\n        float d = length(uv - p) - 0.03;\n        v += 0.005 / (d * d + 0.001);\n    }\n    f_color = vec4(v * 0.5, v * 0.8, v, 1.0);\n}\n',
          uniforms={
  "count": {
    "glsl": "int",
    "min": 1,
    "max": 40,
    "default": 20,
    "description": "number of trails"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.3,
    "description": "drift speed"
  }
})