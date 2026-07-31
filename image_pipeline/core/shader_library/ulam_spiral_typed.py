"""ulam_spiral_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Typed math_art pattern nodes (ids 271-276) ───────────────────────────
# Categorical coverage for the math_art family: closed-form visual patterns
# (Ulam-spiral homage, hash maze, circle packing, Fourier epicycles, summed
# waveform, Clifford strange-attractor bands). Each exposes NAMED typed
# controls + wireable SCALAR ports (the _make_typed factory derives them from
# `uniforms`). CPU fns stay authoritative; these are an additive typed-uniform
# live-preview layer. No per-frame seeds — every frame is a pure function of
# (uv, t), so GPU/CPU parity is exact (no seeded-layout divergence).
_register("ulam_spiral_typed", "Ulam-spiral homage: sparse glowing dots along a number spiral (typed, node 271)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 uv = (v_uv - 0.5);
    uv.x *= u_resolution.x / u_resolution.y;
    float rad = length(uv);
    float ang = atan(uv.y, uv.x);
    float turns = max(u_turns, 0.5);
    float idx = rad * turns * 6.28318530;
    vec2 cell = vec2(floor(idx / max(u_cells, 1.0)),
                     floor((ang + 3.14159265) / (6.28318530 / max(u_arms, 1.0))));
    float h = hash21(cell + 0.5);
    float isPrime = step(1.0 - u_density, h);
    float t = u_time * 0.03 * u_speed;
    float glow = 0.5 + 0.5 * sin(idx * 0.25 - t * 6.28318530);
    float v = max(isPrime, glow * 0.22);
    vec3 col = mix(u_bg, u_fg, v);
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "turns":   {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                "description": "spiral turns"},
    "cells":   {"glsl": "float", "min": 4.0, "max": 80.0, "default": 24.0,
                "description": "cells per turn"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "radial arms"},
    "density": {"glsl": "float", "min": 0.02, "max": 0.6, "default": 0.18,
                "description": "prime-dot density"},
    "speed":   {"glsl": "float", "min": 0.0, "max": 6.0, "default": 1.0,
                "description": "animation speed"},
    "bg":      {"glsl": "color", "default": "#05060f", "description": "background"},
    "fg":      {"glsl": "color", "default": "#ffcf5c", "description": "dot color"},
})