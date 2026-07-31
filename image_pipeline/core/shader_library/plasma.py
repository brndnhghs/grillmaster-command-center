"""plasma — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("plasma", 'Multi-octave colored plasma', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * 0.1;\n    float v = sin(uv.x * u_scale + t) * cos(uv.y * u_scale * 0.75 + t * 0.7);\n    v += sin(uv.x * u_scale * 2.0 - t * 1.2) * cos(uv.y * u_scale * 1.5 + t * 0.5) * 0.5;\n    v += sin((uv.x + uv.y) * u_scale * 3.0 + t * 0.3) * 0.25;\n    v = v * 0.5 + 0.5;\n    f_color = vec4(0.5 + 0.5 * cos(v * 6.28 + vec3(0.0, 2.0, 4.0) + u_hue_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 32.0,
    "default": 8.0,
    "description": "spatial frequency"
  },
  "hue_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.0,
    "description": "hue rotation"
  }
})