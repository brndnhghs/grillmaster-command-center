"""tunnel_typed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register
from ._helpers import _INFERNO_GPU



# ── Typed closed-form patterns pt.10 (ids 289-294) ─────────────────────────
# Categorical coverage continuation (2026-07-11): classic generative-art
# patterns with NAMED typed controls — infinite zoom tunnel, vortex/galaxy
# field, woven fabric, topographic contour map, cross-hatch engraving, and a
# domain-warped grid lattice. All closed-form f(uv,t); additive live-preview
# twins. CPU fns stay authoritative; these are a convenience layer.

_register("tunnel_typed", "Infinite zoom tunnel: polar depth-warp with typed arms/freq/falloff (node 289)",
          "procedural", _INFERNO_GPU + '''void main() {
    vec2 p = (v_uv - 0.5) * 2.0;
    p.x *= u_resolution.x / u_resolution.y;
    float r = max(length(p), 1e-3);
    float a = atan(p.y, p.x);
    float t = u_time * u_speed;
    float depth = u_scale / r + t * 0.5;
    float rings = 0.5 + 0.5 * sin(depth * u_freq);
    float spokes = 0.5 + 0.5 * sin(a * u_arms + depth * 0.5);
    float v = rings * 0.6 + spokes * 0.4;
    vec3 col = inferno(fract(depth) * 0.9 + 0.05);
    col = mix(u_bg, col, smoothstep(0.0, u_falloff, r));
    f_color = vec4(col, 1.0);
}
''', uniforms={
    "speed":   {"glsl": "float", "min": 0.0, "max": 4.0, "default": 1.0,
                "description": "zoom speed"},
    "scale":   {"glsl": "float", "min": 0.05, "max": 1.5, "default": 0.35,
                "description": "tunnel depth scale"},
    "freq":    {"glsl": "float", "min": 1.0, "max": 40.0, "default": 12.0,
                "description": "ring frequency"},
    "arms":    {"glsl": "float", "min": 1.0, "max": 16.0, "default": 6.0,
                "description": "spoke count"},
    "falloff": {"glsl": "float", "min": 0.1, "max": 1.5, "default": 0.55,
                "description": "edge fade"},
    "bg":      {"glsl": "color", "default": "#040610", "description": "vanishing point"},
})