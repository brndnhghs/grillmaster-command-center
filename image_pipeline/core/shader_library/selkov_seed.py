"""selkov_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Sel'kov Glycolysis (client-GPU sim of node 1003) ─────────────────────────
# Excitable two-variable reaction-diffusion (Sel'kov 1968): substrate-depletion
# kinetics u²v with TWO diffusing species. State packs U in .r, V in .g. The
# medium is *excitable* (not Turing): a perturbation ignites a wavefront that
# travels and curls into spirals — the signature of glycolytic waves and a
# different dynamical regime from Gray-Scott (id 155) and BZ (id 91). CPU node
# stays authoritative for export (two-tier precision); this is the live twin.
_register("selkov_seed",
          "Sel'kov initial state: U~0.6, V~0.25 with a hashed ignition blob (node 1003 twin)",
          "procedural", '''
void main() {
    float U = 0.6;
    float V = 0.25;
    // Ignite one seeded blob near center so the excitable wave actually starts.
    // (The CPU node supports several seed shapes; the twin just needs ONE live
    // ignition to show the same spiral dynamics.)
    vec2 c = vec2(0.5);
    float d = distance(v_uv, c);
    float ign = smoothstep(0.04, 0.0, d);
    U = mix(U, 0.05, ign);
    V = mix(V, 0.85, ign);
    f_color = vec4(U, V, 0.0, 1.0);
}
''')