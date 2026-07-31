"""cahn_hilliard_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Cahn-Hilliard Phase Separation (client-GPU sim of node 1008) ───────────────
# Spinodal decomposition / phase coarsening — the free-energy model behind
# emulsions and alloy decomposition. A distinct regime from the reaction-
# diffusion twins (Gray-Scott 155, Sel'kov 1003, BZ 91): there is
# NO reaction term, only a double-well potential + interfacial energy.
# State packs φ (phase) in .r and μ (chemical potential) in .g (two
# channels). The CPU node (methods/simulations/cahn_hilliard.py) stays
# authoritative for export; this is the live-preview twin.
_register("cahn_hilliard_seed",
          "Cahn-Hilliard initial state: small-noise φ in .r, μ=0 in .g (node 1008 twin)",
          "procedural", '''
void main() {
    float amp = max(u_params.z, 0.05);   // seed_variance (p3)
    float hh = hash21(v_uv * 137.13 + 0.123);
    float phi = (hh - 0.5) * 2.0 * amp;
    f_color = vec4(phi, 0.0, 0.0, 1.0);  // .r = phi, .g = mu(0)
}
''')