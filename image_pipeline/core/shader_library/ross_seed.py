"""ross_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 162: Coupled Rössler Oscillator Array ─────────────────────────────
# 3-variable chaotic oscillator per cell, diffusively coupled on a 2D grid.
# State packs R=x (slow), G=y (fast), B=z (fold). p1=a, p2=b, p3=c_ross,
# p4=omega. coupling D is fixed (CPU authoritative); live preview approximates.
_register("ross_seed",
          "Rössler array seed: near-fixed-point oscillation, hashed per-cell (node 162 twin)",
          "procedural", '''
void main() {
    float x = -5.7 + 0.5 * (noise(v_uv * 13.0) - 0.5);
    float y = -5.7 + 0.5 * (noise(v_uv * 13.0 + 3.1) - 0.5);
    float z = 5.7  + 0.5 * (noise(v_uv * 13.0 + 7.7) - 0.5);
    f_color = vec4(x, y, z, 1.0);  // R=x, G=y, B=z
}
''')