"""rd_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ═══════════════════════════════════════════════════════════════════════════
#  P1.2 — RD family (Lotka-Volterra, FitzHugh-Nagumo, Turing, Colony)
#  Client-GPU sim twins of nodes 118-121, 133, 143/160, 168, 169.
#  All share the 5-pt toroidal Laplacian + RGBA-float ping-pong; only the
#  reaction term differs. CPU numpy nodes stay authoritative export.
# ═══════════════════════════════════════════════════════════════════════════

# Generic seeded RD state: U~1, V~0 with hashed seed blobs in V (node 118/119/133/169).
_register("rd_seed",
          "Generic RD seed: U~1, V~0 with hashed seed blobs (P1.2 RD twins)",
          "procedural", '''
void main() {
    float U = 1.0; float V = 0.0;
    for (int i = 0; i < 18; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)), hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;
        float d = distance(v_uv, c);
        V += 0.5 * exp(-(d * d) / 0.002);
    }
    V = clamp(V, 0.0, 1.0);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')