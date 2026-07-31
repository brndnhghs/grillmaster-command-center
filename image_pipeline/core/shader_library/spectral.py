"""spectral — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("spectral", 'Spectral / rainbow interference', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 2.0 - 1.0;\n    float t = u_time * u_speed;\n    float a = atan(uv.y, uv.x);\n    float r = length(uv);\n    float v = sin(r * u_rings - t) + cos(a * u_arms + t * 0.5);\n    v = v * 0.25 + 0.5;\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0, 2, 4)), 1.0);\n}\n',
          uniforms={
  "rings": {
    "glsl": "float",
    "min": 5.0,
    "max": 40.0,
    "default": 20.0,
    "description": "radial rings"
  },
  "arms": {
    "glsl": "float",
    "min": 2.0,
    "max": 10.0,
    "default": 5.0,
    "description": "angular arms"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "animation speed"
  }
})