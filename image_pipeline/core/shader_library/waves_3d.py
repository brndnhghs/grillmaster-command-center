"""waves_3d — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("waves_3d", '3D wave interference', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale - 2.0;\n    float t = u_time * u_speed;\n    float v = 0.0;\n    for (int i = 0; i < 20; i++) {\n        if (float(i) >= u_waves) break;\n        float fi = float(i);\n        vec2 p = vec2(sin(fi * 1.3 + t), cos(fi * 1.7 + t * 0.8));\n        v += sin(dot(uv, p) * 3.0 + t) * 0.1;\n    }\n    vec3 col = 0.5 + 0.5 * cos(v * 4.0 + vec3(0, 2, 4));\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "waves": {
    "glsl": "float",
    "min": 2.0,
    "max": 20.0,
    "default": 10.0,
    "description": "wave sources"
  },
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 4.0,
    "description": "field scale"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 3.0,
    "default": 0.5,
    "description": "animation speed"
  }
})