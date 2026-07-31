"""wood_grain_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("wood_grain_gpu", 'Concentric wood grain rings', "procedural", '\nvoid main() {\n    vec2 uv = v_uv - 0.5;\n    float d = length(uv) * u_rings;\n    float grain = sin(d * u_turbulence + fbm(uv * u_turbulence) * 0.5) * 0.5 + 0.5;\n    vec3 col = mix(vec3(0.3, 0.15, 0.05), vec3(0.6, 0.3, 0.1), grain);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "rings": {
    "glsl": "float",
    "min": 2.0,
    "max": 20.0,
    "default": 10.0,
    "description": "ring density"
  },
  "turbulence": {
    "glsl": "float",
    "min": 1.0,
    "max": 10.0,
    "default": 8.0,
    "description": "grain wobble"
  }
})