"""kaleidoscope_fractal — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("kaleidoscope_fractal", 'Kaleidoscope IFS fractal', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    for (int i = 0; i < 20; i++) {\n        if (i >= u_iterations) break;\n        uv = abs(uv);\n        float a = sin(t + float(i) * 0.5);\n        uv = rot(a) * uv;\n        uv = uv * 1.5 - vec2(0.5);\n    }\n    float v = length(uv);\n    f_color = vec4(0.5 + 0.5 * cos(v * 10.0 + vec3(0, 2, 4)), 1.0);\n}\n',
          uniforms={
  "iterations": {
    "glsl": "int",
    "min": 3,
    "max": 20,
    "default": 10,
    "description": "IFS iterations"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "rotation speed"
  }
})