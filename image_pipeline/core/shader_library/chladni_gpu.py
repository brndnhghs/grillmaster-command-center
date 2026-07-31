"""chladni_gpu — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register






# ── P0.6 field-eval client-GPU twins (client-GPU live preview of nodes
# 125/164) ──────────────────────────────────────────────────────────────────
# Additive: the server's CPU numpy nodes stay the authoritative export (two-tier
# precision). These bodies only drive the browser live preview. They reuse the
# prologue helpers so they are auto-covered by test_webgl2_transform_is_valid +
# the gl330 legacy-equivalence parametrized tests. Both nodes render a pure
# per-frame field that is a closed-form function of (uv, t), so the twin is an
# exact preview (no seeded-layout divergence, unlike pattern/generative nodes).
#
# IMPORTANT (pitfall #15): encode 0.5 as NEUTRAL so the default u_params
# (0.5,0.5,0.5,0.5) yields the node's canonical full view, not an extreme.

_register("chladni_gpu",
          "Chladni eigenmode field (client-GPU twin of node 125)",
          "procedural", '''
void main() {
    // u_params.x = m-mode (0.5 -> 3.0 canonical start, range 0.5..11.5),
    // u_params.y = n-mode (0.5 -> 3.0),
    // u_params.z = rotation (0.5 -> 0 rad, range -PI..PI),
    // u_params.w = phase shimmer (0.5 -> 0 rad, range -PI..PI).
    float m = 0.5 + u_m_start * 11.0;
    float n = 0.5 + u_n_start * 11.0;
    float rot_ang = (u_rotation_speed - 0.5) * 6.2831853;
    float ph = (u_phase_speed_x - 0.5) * 6.2831853;

    // Centered, normalized coords in [-1, 1] (matches node: xn = X/(W/2)).
    vec2 p = (v_uv - 0.5) * 2.0;
    // Coordinate rotation (plate spin).
    vec2 pr = rot(rot_ang) * p;

    // u_mn(x,y) = sin(m*PI*(x+1)/2 + φx) * sin(n*PI*(y+1)/2 + φy)
    float u = sin(m * 3.14159265 * (pr.x + 1.0) * 0.5 + ph)
            * sin(n * 3.14159265 * (pr.y + 1.0) * 0.5 + ph);

    // Centered, sharp sigmoid emphasis of zero-crossings (nodal lines).
    float sig = tanh(clamp(u, -4.0, 4.0) * 3.5);
    float gray = (sig + 1.0) * 0.5;
    // Nodal-line bright highlight: gaussian bell centered at u=0.
    float nodal = exp(-u * u * 8.0);
    gray = clamp(gray + nodal * 0.35, 0.0, 1.0);
    f_color = vec4(vec3(gray), 1.0);
}
''',
    uniforms={
    "m_start": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "m mode"},
    "n_start": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "n mode"},
    "rotation_speed": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "plate rotation speed"},
    "phase_speed_x": {"glsl": "float", "min": 0.0, "max": 1.0, "default": 0.5, "description": "shimmer phase"}
}
    )