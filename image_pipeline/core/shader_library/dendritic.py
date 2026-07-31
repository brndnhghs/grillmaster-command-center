"""dendritic — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("dendritic", 'Dendritic / tree-like branching', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float d = length(uv);\n    float a = atan(uv.y, uv.x) * u_branches;\n    float branch = sin(a * 8.0 + log(d + 0.001) * 10.0 + t) * 0.5 + 0.5;\n    float v = branch * exp(-d * 2.0);\n    f_color = vec4(v * 0.3, v * 0.6, v * 0.2, 1.0);\n}\n',
          uniforms={
  "branches": {
    "glsl": "float",
    "min": 2.0,
    "max": 16.0,
    "default": 8.0,
    "description": "branch count"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "growth speed"
  }
})