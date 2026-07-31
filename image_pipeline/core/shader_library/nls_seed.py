"""nls_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── P1.3 complex-field PDE — Nonlinear Schrödinger (node 124). Same R/G complex
# field packing as CGL. NLSE in real space: ψ=a+ib, ∂ψ/∂t = i(β∇²ψ − g|ψ|²ψ + Vψ)
#   → ∂a/∂t = −β·∇²b + g·|ψ|²·b − V·b ;  ∂b/∂t = β·∇²a − g|ψ|²·a + V·a
# Explicit Euler on the 5-pt (toroidal) Laplacian. CPU node is a split-step
# Fourier Arch-A sim; this is the live-preview twin only — server export stays
# authoritative (seeded layout differs, as expected for this PDE family).
_register("nls_seed",
          "NLSE initial state: small random complex noise in RG (node 124 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    float a = (hash21(p + 0.19) - 0.5) * 0.3;
    float b = (hash21(p + 7.31) - 0.5) * 0.3;
    f_color = vec4(a, b, 0.0, 1.0);
}
''')