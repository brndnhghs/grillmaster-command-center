"""cells — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("cells", 'Cellular growth simulation', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv);\n    float md = 8.0;\n    vec2 mp = vec2(0.0);\n    for (int y = -1; y <= 1; y++) {\n        for (int x = -1; x <= 1; x++) {\n            vec2 n = vec2(float(x), float(y));\n            vec2 p = hash21(i + n) * vec2(1.0);\n            float d = length(n + p - f);\n            if (d < md) { md = d; mp = n + p; }\n        }\n    }\n    float c = hash21(i + mp);\n    vec3 col = 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4));\n    col *= 1.0 - md * 1.2;\n    col += vec3(0.05) / (md * md + 0.01);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 16.0,
    "default": 8.0,
    "description": "cell scale"
  }
})