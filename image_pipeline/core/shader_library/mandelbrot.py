"""mandelbrot — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ═══════════════════════════════════════════════
#  TYPED-UNIFORM UPGRADE — dedicated GPU procedural nodes 173-197
#  These re-register the same shader names with named, typed `uniforms=`
#  and bodies that read `u_<name>` instead of the legacy `u_params` p-slots.
#  The node factory (methods/gpu_shaders.py) routes these ids through
#  `_make_typed`, so each variable becomes a real param + wireable SCALAR
#  port + typed IMAGE/FIELD outputs. Additive — CPU/fp64 export untouched.
# ═══════════════════════════════════════════════

_register("mandelbrot", 'Mandelbrot set zoom region', "procedural", '\nvoid main() {\n    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);\n    float zoom = exp(u_zoom * 3.0);\n    vec2 c = vec2(-0.5, 0.0) + uv * zoom;\n    vec2 z = vec2(0.0);\n    int n = 0;\n    for (int i = 0; i < 100; i++) {\n        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;\n        if (dot(z, z) > 4.0) break;\n        n++;\n    }\n    float t = float(n) / 100.0;\n    f_color = vec4(0.5 + 0.5 * cos(t * 6.28 + vec3(0.0, 2.0, 4.0) + u_color_shift * 6.28), 1.0);\n}\n',
          uniforms={
  "zoom": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "zoom (0.5 = full view)"
  },
  "color_shift": {
    "glsl": "float",
    "min": 0.0,
    "max": 1.0,
    "default": 0.5,
    "description": "hue rotation"
  }
})