"""schotter_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════
#  CLOSED-FORM TYPED-UNIFORM NODES — pt.13 (nodes 302-307)
#  Pure f(uv,t) field-eval twins: each variable is a named, typed uniform
#  wired through _make_typed (real param + wireable SCALAR port). No ping-pong
#  state, exact server/browser parity. Additive — CPU/fp64 export untouched.
# ═══════════════════════════════════════════════

# 302 — Schotter (Georg Nees, 1968): a rigid grid of squares whose jitter and
# rotation grow with distance from the centre. The canonical generative-art
# "ordered disorder" piece.
_register("schotter_typed", "Schotter — Georg Nees generative grid of jittered squares (typed, node 302)",
          "procedural", '''void main() {
    float N = max(u_cells, 2.0);
    vec2 g = v_uv * N;
    vec2 id = floor(g);
    vec2 f = fract(g) - 0.5;
    vec2 ctr = (id + 0.5) / N - 0.5;
    float d = length(ctr);
    float amt = u_jitter * smoothstep(0.0, 0.7, d);
    float r1 = hash21(id + 1.3);
    float r2 = hash21(id + 7.7);
    float r3 = hash21(id + 3.1);
    float ang = (r1 - 0.5) * amt * 1.5 + u_time * u_speed * (r2 - 0.5) * 0.3;
    vec2 disp = (vec2(r2 - 0.5, r3 - 0.5)) * amt * 0.6;
    vec2 q = f - disp;
    q = rot(ang) * q;
    float s = 0.5 * u_square;
    vec2 a = abs(q);
    float inside = step(max(a.x, a.y), s);
    vec3 col = mix(u_bg, u_fg, inside);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "cells":  {"glsl": "float", "min": 2.0, "max": 24.0, "default": 11.0, "description": "grid cells per axis"},
    "jitter": {"glsl": "float", "min": 0.0, "max": 1.5, "default": 0.85, "description": "displacement (grows outward)"},
    "square": {"glsl": "float", "min": 0.3, "max": 0.95, "default": 0.72, "description": "square fill fraction"},
    "speed":  {"glsl": "float", "min": 0.0, "max": 3.0, "default": 0.5, "description": "animation speed"},
    "fg":     {"glsl": "color", "default": "#f4c020", "description": "square color"},
    "bg":     {"glsl": "color", "default": "#0e0e16", "description": "background color"},
})