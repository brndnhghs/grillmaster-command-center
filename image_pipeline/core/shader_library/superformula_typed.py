"""superformula_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Categorical coverage pt.9 (typed closed-form patterns, nodes 283-288) ──
# superformula, harmonograph, Maurer rose, magnetic dipole field, star polygon,
# torus-knot ribbon. Each is a pure f(uv, t) → exact CPU/GPU parity (P0.6),
# continuous-time motion only. Six more distinct math_art generators in the
# same family as 265-282.

_register("superformula_typed", "Superformula: Gielis radial curve sweep (typed, node 283)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5);
    p.x *= u_resolution.x / u_resolution.y;
    float t = u_time * 0.1 * u_speed;
    float r = length(p);
    float a = atan(p.y, p.x);
    // Superformula radius for normalized angle a (continuous t rotation).
    float aa = a + t;
    float ca = cos(u_m * aa / 4.0);
    float sa = sin(u_n * aa / 4.0);
    float ra = pow(abs(ca), u_b) + pow(abs(sa), u_c);
    ra = pow(max(ra, 1e-4), -1.0 / u_p);
    float rr = ra * u_scale;
    float d = abs(r - rr);
    float line = smoothstep(u_thick, u_thick * 0.3, d);
    vec3 col = mix(u_bg, inferno(clamp(r / max(u_scale, 1e-3), 0.0, 1.0)), line);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "m":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 6.0,
              "description": "superformula m (symmetry)"},
    "n":     {"glsl": "float", "min": 1.0, "max": 20.0, "default": 8.0,
              "description": "superformula n"},
    "b":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent b"},
    "c":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent c"},
    "p":     {"glsl": "float", "min": 0.2, "max": 6.0, "default": 1.0,
              "description": "exponent p"},
    "scale": {"glsl": "float", "min": 0.3, "max": 1.2, "default": 0.85,
              "description": "curve radius"},
    "thick": {"glsl": "float", "min": 0.006, "max": 0.08, "default": 0.02,
              "description": "line thickness"},
    "speed": {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
              "description": "rotation speed"},
    "bg":    {"glsl": "color", "default": "#04060c", "description": "background"},
})