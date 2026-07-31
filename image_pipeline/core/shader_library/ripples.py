"""ripples — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ripples", 'Concentric ripple pattern', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float d = length(uv);\n    float ph = d * u_frequency - u_time * u_speed;\n    float r = sin(ph) * 0.5 + 0.5;\n    float g = sin(ph + 2.0) * 0.5 + 0.5;\n    float b = sin(ph + 4.0) * 0.5 + 0.5;\n    f_color = vec4(r, g, b, 1.0) * (1.0 - d);\n}\n',
          uniforms={
  "frequency": {
    "glsl": "float",
    "min": 5.0,
    "max": 60.0,
    "default": 30.0,
    "description": "ripple frequency"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 5.0,
    "default": 2.0,
    "description": "ripple speed"
  }
})