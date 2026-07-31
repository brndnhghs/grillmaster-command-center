"""erosion_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.6 — Hydraulic Erosion / River Network (node 156) 3-field terrain twin
# ═══════════════════════════════════════════════════════════════════════════
# 3-field grid sim: terrain height h (.r), water w (.g), sediment s (.b). Local
# model (visual-style parity; CPU authoritative for the exact steepest-descent
# routing): rain → water pools down the (h+w) surface gradient → stream-power
# erosion (K_e·w·slope) lifts sediment → deposition (K_d) settles it on flats →
# thermal creep smooths toward the angle of repose → evaporation. Display is the
# CPU's grayscale hillshade + water-channel brightening.
_register("erosion_seed",
          "Hydraulic-erosion seed: fbm fractal terrain, dry (h=.r, w=.g=0, s=.b=0) (node 156)",
          "procedural", '''
void main() {
    float t = fbm(v_uv * 4.0) + 0.5 * fbm(v_uv * 8.0 + 3.1) + 0.25 * fbm(v_uv * 16.0 + 7.7);
    float h = (t - 0.6) * 0.6;               // broad fractal landscape, centered
    f_color = vec4(h, 0.0, 0.0, 1.0);
}
''')