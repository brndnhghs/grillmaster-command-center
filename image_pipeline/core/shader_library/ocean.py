"""ocean — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("ocean", 'Procedural ocean waves', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * 3.0;\n    float t = u_time * u_speed;\n    float v = sin(uv.x * 5.0 + t) * cos(uv.y * 3.0 + t * 0.7);\n    v += sin(uv.x * 8.0 - t * 1.3) * sin(uv.y * 6.0 + t) * 0.5 * u_choppiness;\n    v += sin((uv.x + uv.y) * 12.0 + t * 0.5) * 0.25;\n    v = v * 0.5 + 0.5;\n    vec3 col = mix(vec3(0.0, 0.2, 0.5), vec3(0.1, 0.6, 0.8), v);\n    col += vec3(0.3, 0.4, 0.5) * pow(v, 4.0);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "choppiness": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "wave amplitude"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.3,
    "description": "wave speed"
  }
})