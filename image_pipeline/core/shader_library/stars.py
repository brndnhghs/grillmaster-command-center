"""stars — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("stars", 'Starfield with parallax', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    vec3 col = vec3(0.0);\n    for (int i = 0; i < 120; i++) {\n        if (i >= u_count) break;\n        float fi = float(i);\n        vec2 p = fract(vec2(sin(fi * 127.1 + t), cos(fi * 311.7 + t * 0.7)));\n        float d = length(uv - p);\n        float brightness = 0.003 / (d * d);\n        vec3 star_col = 0.5 + 0.5 * cos(fi + vec3(0, 2, 4));\n        col += brightness * star_col;\n    }\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "count": {
    "glsl": "int",
    "min": 10,
    "max": 120,
    "default": 50,
    "description": "star count"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.05,
    "description": "parallax speed"
  }
})