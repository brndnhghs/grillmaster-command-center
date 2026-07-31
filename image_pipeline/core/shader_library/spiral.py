"""spiral — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spiral", 'Logarithmic spiral galaxy', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float a = atan(uv.y, uv.x);\n    float r = length(uv);\n    float spiral = sin(a * u_arms - r * u_tightness + u_time * u_speed) * 0.5 + 0.5;\n    float fade = exp(-r * 3.0);\n    float col = spiral * fade;\n    f_color = vec4(col * 1.2, col * 0.8, col * fade + 0.1, 1.0);\n}\n',
          uniforms={
  "arms": {
    "glsl": "float",
    "min": 1.0,
    "max": 12.0,
    "default": 4.0,
    "description": "spiral arms"
  },
  "tightness": {
    "glsl": "float",
    "min": 5.0,
    "max": 30.0,
    "default": 15.0,
    "description": "winding tightness"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "rotation speed"
  }
})