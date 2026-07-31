"""pfc_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 170: Phase Field Crystal ───────────────────────────────────────────
# PFC amplitude equation live-preview approximation. Single scalar ψ packed in
# .r; .g phase drives noise. p1=epsilon, p2=dt, p3=noise_amp, p4=r2 (=r/2).
_register("pfc_seed",
          "Phase Field Crystal seed: small hashed amplitude field (node 170 twin)",
          "procedural", '''
void main() {
    float psi = 0.2 * (noise(v_uv * 11.0) - 0.5) + 0.05 * (noise(v_uv * 23.0) - 0.5);
    f_color = vec4(psi, 0.0, 0.0, 1.0);  // R=psi, G=phase
}
''')