"""smoke_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("smoke_gpu", 'Rising smoke / steam', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    float v = fbm(uv * 3.0 + vec2(0.0, t));\n    v = v * (1.0 - uv.y) * 0.8 * u_density;\n    vec3 col = mix(vec3(0.8, 0.8, 0.85), vec3(0.2, 0.2, 0.25), v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "density": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "smoke density"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.1,
    "description": "rise speed"
  }
})