"""burridge_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 131: Burridge-Knopoff Spring-Block (Earthquake Cascades) ── -----------
# 2D grid of frictional blocks slowly driven by a plate. Stress builds until a
# block exceeds its heterogeneous friction threshold and slips, resetting to a
# residual level and redistributing a coupling fraction of released stress to
# its 4 neighbors — which may trigger a branching cascade. The CPU node runs an
# inner while-loop to fully relax a cascade per frame; the GPU twin performs one
# threshold+redistribute relaxation per substep, so a cascade propagates over
# consecutive substeps (many substeps/frame → visually equivalent avalanches).
# State packs: .r = stress, .g = damage (accumulated slip count), .b = strength
# (heterogeneous friction, seeded once and preserved). CPU numpy node stays the
# authoritative export (two-tier precision).
# p1=loading_rate, p2=threshold, p3=residual, p4=coupling(α).
_register("burridge_seed",
          "Burridge-Knopoff seed: heterogeneous strength (.b), near-threshold stress (.r), zero damage (node 131 twin)",
          "procedural", '''
void main() {
    float thr = clamp(u_params.y, 0.5, 5.0);
    // Heterogeneous per-cell strength in [0.7, 1.3] (matches CPU 0.7+0.6*rand).
    float hs = hash21(v_uv * 71.31 + 3.7);
    float strength = 0.7 + 0.6 * hs;
    // Initial stress near each block's own threshold (0.5..1.0 of thr).
    float hr = hash21(v_uv * 137.13 + 0.123);
    float stress = thr * (0.5 + 0.5 * hr);
    f_color = vec4(stress, 0.0, strength, 1.0);  // .r=stress .g=damage .b=strength
}
''')