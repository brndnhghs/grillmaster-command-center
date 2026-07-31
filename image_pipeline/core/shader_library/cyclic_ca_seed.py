"""cyclic_ca_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



#  P1.4 — Discrete cellular automata / statistical-mechanics twins
#  Client-GPU sim twins of nodes 87 (Cyclic CA), 96 (Forest Fire), 93 (Ising).
#  All use RGBA-float ping-pong: a single-channel integer CA state in .r,
#  an auxiliary channel for age/aux data in .g, and a per-cell RNG carry in .b
#  (advanced each step via hash21 so the live sim does not require u_time).
#  CPU numpy nodes stay authoritative for export (two-tier precision).
# ═══════════════════════════════════════════════════════════════════════════

# ── Node 87: Cyclic (Rock-Paper-Scissors) CA ──
# State in .r ∈ [0,1) encodes state index = floor(.r * n_states); n_states from
# u_params.x (3-8). .b carries the per-cell RNG seed, advanced each step.
_register("cyclic_ca_seed",
          "Cyclic CA seed: hashed random state in [0,n_states), RNG carry in .b (node 87 twin)",
          "procedural", '''
void main() {
    float ns = clamp(floor(u_params.x + 0.5), 3.0, 8.0);
    float h = hash21(floor(v_uv * u_resolution * 0.5));
    float s = floor(h * ns) / ns;   // quantize into n_states buckets
    float rng = hash21(v_uv * u_resolution + 7.13);
    f_color = vec4(s, 0.0, rng, 1.0);
}
''')