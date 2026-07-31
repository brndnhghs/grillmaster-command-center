"""dendrite_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.5 — Dendritic Solidification (node 122) phase-field ping-pong twin
# ═══════════════════════════════════════════════════════════════════════════
# Allen-Cahn phase field φ (.r) coupled to a passive thermal field u (.g).
# Anisotropic interface width W(θ)=W0(1+ε cos(kθ)) gives the 4-fold dendrite
# branching. Faithful to the CPU node's ACTUAL (simplified) update — it uses the
# W²∇²φ diffusion form, constant driving force, double-well f'(φ). The CPU numpy
# node stays the authoritative export; every param is clamped to its documented
# range so the twin is robust to the client's neutral u_params fallback.
_register("dendrite_seed",
          "Dendritic seed: single tanh nucleus at center + thermal bump (node 122 twin)",
          "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 p = v_uv * res;
    float dist = length(p - 0.5 * res);
    float W0 = 0.5, seedR = 12.0;
    float phi = tanh((seedR - dist) / (W0 * 1.41421356));   // φ=+1 solid core → −1 liquid
    float u0 = clamp(u_params.x, -1.0, -0.1);               // undercooling
    float u  = clamp(u0 + 0.3 * exp(-(dist * dist) / 100.0), -1.0, 1.0);
    f_color = vec4(phi, u, 0.0, 1.0);
}
''')