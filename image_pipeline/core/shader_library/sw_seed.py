"""sw_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ═══════════════════════════════════════════════════════════════════════════
#  P1.3b — Fluid / surface-growth / lattice sim twins (nodes 132, 135, 150)
#  Same RGBA-float ping-pong contract as P1.3: seed writes initial state,
#  step reads u_texture + u_params (p1..p4) and writes new state, display maps
#  state -> RGB. CPU numpy nodes stay authoritative (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 132: Shallow Water Waves ───────────────────────────────────────────
# 2D shallow-water surrogate: height h + velocity (u,v). A wave-like advection/
# diffusion of h coupled to velocity (gravity g, base depth, viscosity nu,
# source amplitude) gives a faithful live preview of the CPU solver.
# p1=gravity, p2=base_depth, p3=viscosity(nu), p4=source_amplitude.
_register("sw_seed",
          "Shallow Water seed: hashed noise height field, zero velocity (node 132 twin)",
          "procedural", '''
void main() {
    float n = noise(v_uv * 7.0) * 0.5 + noise(v_uv * 17.0) * 0.3
            + noise(v_uv * 31.0) * 0.2;
    f_color = vec4((n - 0.5) * 0.4, 0.0, 0.0, 1.0);  // R=h, G=u, B=v
}
''')