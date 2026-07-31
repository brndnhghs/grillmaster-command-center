"""voronoise — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("voronoise", 'Smooth voronoi layers', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * 0.02;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(3.7, 1.2) + t));\n    vec2 r = vec2(fbm(uv + 4.0 * q + vec2(1.7, 9.2)),\n                  fbm(uv + 4.0 * q + vec2(8.3, 2.8)));\n    float v = fbm(uv + 4.0 * r);\n    f_color = vec4(0.5 + 0.5 * cos(v * 4.0 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 4.0,
    "description": "layer frequency"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})