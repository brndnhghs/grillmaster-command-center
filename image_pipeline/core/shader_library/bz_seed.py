"""bz_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── BZ Oregonator (client-GPU sim of node 91) ───────────────────────────────
# Two-variable reaction-diffusion with Oregonator kinetics. State packs U in
# .r, V in .g (same channel layout as Gray-Scott). CPU node is Arch-A sim; this
# is the live-preview twin only — server export stays authoritative.
_register("bz_seed",
          "BZ Oregonator initial state: U~1, V~0 with hashed seed blobs (node 91 twin)",
          "procedural", '''
void main() {
    float U = 1.0;
    float V = 0.0;
    for (int i = 0; i < 16; i++) {
        float fi = float(i);
        vec2 c = vec2(hash21(vec2(fi + 0.5, 1.37)),
                      hash21(vec2(fi + 0.5, 7.91)));
        c = 0.05 + 0.90 * c;
        float d = distance(v_uv, c);
        V += 0.6 * exp(-(d * d) / 0.004);
    }
    V = clamp(V, 0.0, 0.9);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')