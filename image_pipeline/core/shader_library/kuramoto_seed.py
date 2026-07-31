"""kuramoto_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Kuramoto coupled-oscillator phase field (client-GPU sim of node 999) ─────
# Client-GPU sim twin of the Arch-A Kuramoto node. The phase field is NOT a
# velocity advection (every flow node does that) — it is self-organized
# synchronization: each pixel is an oscillator whose phase θ is nudged toward
# its neighbours AND toward the global mean phase. State packs:
#   .r = phase θ (wrapped to [0, 2π])
#   .g = natural frequency Ω (frozen at seed — per-oscillator intrinsic rate)
#   .b = RNG carry (see pitfall #6b: renderGpuSim gives step NO u_time, so we
#        carry a per-cell random in state instead of hashing the clock).
# CPU node stays authoritative for export (two-tier precision).
_register("kuramoto_seed",
          "Kuramoto seed: hashed phase + spatially-structured natural frequency Ω (node 999 twin)",
          "procedural", '''
void main() {
    float h1 = hash21(floor(v_uv * u_resolution * 0.37));
    float h2 = hash21(floor(v_uv * u_resolution * 0.91) + 5.3);
    // phase: scattered so the field starts incoherent
    float theta = h1 * 6.2831853;
    // Ω: smooth spatial gradient + mild noise → travelling spiral waves
    vec2 c = v_uv - 0.5;
    float omega = (c.x + c.y) * u_params.z * 1.4 + (h2 - 0.5) * u_params.z * 0.4;
    f_color = vec4(theta, omega, h2, 1.0);
}
''')