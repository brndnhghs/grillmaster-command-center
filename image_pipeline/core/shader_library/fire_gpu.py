"""fire_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("fire_gpu", 'Animated fire/flame', "procedural", '\nvoid main() {\n    vec2 uv = v_uv;\n    float t = u_time * u_speed;\n    float v = fbm(vec2(uv.x * 3.0, (1.0 - uv.y) * 5.0 + t));\n    v = v * (1.0 - uv.y) * u_intensity;\n    vec3 col = mix(vec3(1.0, 0.9, 0.4), vec3(0.8, 0.2, 0.0), v);\n    col = mix(col, vec3(0.1, 0.0, 0.0), 1.0 - v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "intensity": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 1.0,
    "description": "flame intensity"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "rise speed"
  }
})