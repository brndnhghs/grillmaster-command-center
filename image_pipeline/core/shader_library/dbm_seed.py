"""dbm_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
# ── Node 106: Dielectric Breakdown Model (GPU sim twin) ─────────────────────
# Client-GPU sim twin of the Arch-A dielectric-breakdown node (DBM, Niemeyer
# 1984): a Jacobi-relaxed Laplace potential field with stochastic growth
# probability proportional to |grad(phi)|^eta at the tree frontier. State packs
# the potential phi in .r, occupancy in .g (-1 far-field boundary / 0 empty /
# 1 tree), and temperature (brightness) in .b. The CPU numpy node stays
# authoritative for export (two-tier precision).
_register("dbm_seed",
          "Dielectric Breakdown seed: center electrode + fixed far-field boundary (node 106 twin)",
          "procedural", '''
void main() {
    vec2 res = u_resolution;
    vec2 uv = v_uv;
    float occ = 0.0;
    float phi = 0.0;
    float temp = 0.0;
    // Single seed electrode at center (the twin uses n_seeds = 1).
    if (distance(uv, vec2(0.5)) < 1.5 / res.x) {
        occ = 1.0; temp = 1.0; phi = 1.0;
    }
    // Fixed Dirichlet far-field: potential 0, never grows.
    float m = 2.0 / res.x;
    if (uv.x < m || uv.x > 1.0 - m || uv.y < m || uv.y > 1.0 - m) {
        occ = -1.0; phi = 0.0; temp = 0.0;
    }
    f_color = vec4(phi, occ, temp, 1.0);
}
''')