"""terrain — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("terrain", 'Procedural terrain heightmap', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    float t = u_time * u_speed;\n    float h = fbm(uv + t);\n    float h2 = fbm(uv * 2.0 + t * 1.5) * 0.5;\n    float h3 = fbm(uv * 4.0 + t * 2.0) * 0.25;\n    h = h * 0.6 + h2 * 0.3 + h3 * 0.1;\n    vec3 col;\n    if (h < u_sea_level) col = vec3(0.1, 0.3, 0.6);\n    else if (h < u_sea_level + 0.15) col = vec3(0.2, 0.5, 0.2);\n    else if (h < 0.6) col = vec3(0.3, 0.3, 0.1);\n    else if (h < 0.75) col = vec3(0.4, 0.25, 0.1);\n    else col = vec3(0.8, 0.8, 0.9);\n    float shade = 0.5 + 0.5 * cos(h * 20.0);\n    f_color = vec4(col * shade, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 8.0,
    "default": 3.0,
    "description": "terrain scale"
  },
  "sea_level": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.3,
    "description": "water line"
  },
  "speed": {
    "glsl": "float",
    "min": 0.0,
    "max": 2.0,
    "default": 0.02,
    "description": "erosion speed"
  }
})