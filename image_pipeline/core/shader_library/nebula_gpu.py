"""nebula_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("nebula_gpu", 'Space nebula gas clouds', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * u_speed;\n    vec2 q = vec2(fbm(uv + t), fbm(uv + vec2(5.2, 1.3) + t * 0.7));\n    vec2 r = vec2(fbm(uv + 3.0 * q + vec2(1.7, 9.2) + t * 0.3),\n                  fbm(uv + 3.0 * q + vec2(8.3, 2.8) + t * 0.4));\n    float v = fbm(uv + 3.0 * r);\n    float mask = 1.0 - abs(v_uv.y - 0.5) * 2.0;\n    vec3 col = 0.3 + 0.7 * (0.5 + 0.5 * cos(v * 4.0 + vec3(0, 1, 2)));\n    col *= mask;\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 6.0,
    "default": 2.0,
    "description": "cloud scale"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.03,
    "description": "drift speed"
  }
})