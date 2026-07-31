"""wave_eq_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.3 — Wave-equation family (client-GPU sim twins of nodes 100, 144, 166)
#  All three are scalar displacement u + velocity v leapfrog field systems
#  (plus a pump/drive phase accumulator). State packs R=u, G=v, B=pump_phase
#  in RGBA-float ping-pong, stepped `substeps` times per rendered frame. The
#  CPU numpy nodes stay the authoritative export (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 100: Wave Equation ── -------------------------------------------------
# 2D wave equation u_tt = c^2 laplacian(u) via velocity-Verlet on (u, v).
# p1=wave_speed, p2=damping, p3=source_frequency, p4=source_amplitude.
_register("wave_eq_seed",
          "Wave Equation seed: small hashed noise displacement, zero velocity (node 100 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 9.0) * 0.5 + noise(v_uv * 23.0) * 0.25;
    f_color = vec4((n - 0.375) * 0.6, 0.0, 0.0, 1.0);  // R=u (small), G=v=0, B=phase=0
}
''')