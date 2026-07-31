"""domain_warp — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("domain_warp", 'Domain-warped fractal noise', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0;\n    float t = u_time * 0.05;\n    float w = 2.0 + u_warp * 3.0;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));\n    vec2 r = vec2(fbm(uv + w * q + vec2(1.7, 9.2) + t * 0.3),\n                  fbm(uv + w * q + vec2(8.3, 2.8) + t * 0.4));\n    float v = fbm(uv + w * r);\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "warp": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "warp strength"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})