"""voronoi — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("voronoi", 'Voronoi/Worley noise cells', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv);\n    float md = 1.0;\n    for (int y = -1; y <= 1; y++) {\n        for (int x = -1; x <= 1; x++) {\n            vec2 n = vec2(float(x), float(y));\n            vec2 p = hash21(i + n) * vec2(1.0);\n            float d = length(n + p - f);\n            md = min(md, d);\n        }\n    }\n    f_color = vec4(md, md * 0.5, 1.0 - md, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 15.0,
    "default": 7.5,
    "description": "cell density"
  }
})