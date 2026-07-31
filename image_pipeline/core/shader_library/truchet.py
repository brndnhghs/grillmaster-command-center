"""truchet — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



_register("truchet", 'Truchet tile pattern', "procedural", '\nvoid main() {\n    vec2 uv = v_uv * u_scale;\n    vec2 i = floor(uv); vec2 f = fract(uv) - 0.5;\n    float flip = hash21(i) > 0.5 ? 1.0 : -1.0;\n    float d = length(f * flip);\n    float v = smoothstep(0.4, 0.5, d);\n    float c = hash21(i + vec2(1.0));\n    vec3 col = mix(vec3(0.9, 0.9, 0.95), 0.5 + 0.5 * cos(c * 6.28 + vec3(0, 2, 4)), v);\n    f_color = vec4(col, 1.0);\n}\n',
          uniforms={
  "scale": {
    "glsl": "float",
    "min": 1.0,
    "max": 12.0,
    "default": 6.0,
    "description": "tile scale"
  }
})