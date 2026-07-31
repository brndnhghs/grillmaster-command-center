"""sh128_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── Node 128: Swift-Hohenberg (ε·u − u³ − (1+∇²)²·u) ───────────────────────
# Spectral-style pattern formation. Local approximation of the biharmonic
# operator via a 5-pt stencil. Scalar u packed in .r; .g phase drives noise.
# p1=epsilon, p2=dt, p3=noise_amp, p4=linear_gain (~0.5 ctrl of (1+∇²) weight).
_register("sh128_seed",
          "Swift-Hohenberg (128) seed: small hashed pattern field (node 128 twin)",
          "procedural", '''
void main() {
    float u = 0.2 * (sin(v_uv.x * 12.0) * 0.5 + sin(v_uv.y * 9.0) * 0.5)
            + 0.05 * (noise(v_uv * 7.0) - 0.5);
    f_color = vec4(u, 0.0, 0.0, 1.0);  // R=u, G=phase
}
''')