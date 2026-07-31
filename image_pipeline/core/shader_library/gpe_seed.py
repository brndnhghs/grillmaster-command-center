"""gpe_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register



# ── P1.3 complex-field PDE — Gross-Pitaevskii (node 148). Same R/G complex
# field packing as CGL/NLSE. ψ=a+ib, split-step symplectic Euler: half-step
# nonlinear (kinetic in k-space via a precomputed k² texture) + half-step
# potential. The live twin approximates the spectral kinetic step with a 5-pt
# Laplacian proxy (lapR, lapI) which carries the same smoothing dynamics; the
# CPU node stays authoritative for frame-accurate export (seeded layout +
# full split-step Fourier differ, as expected for this PDE family).
#   Re(k²ψ) = lapR, Im(k²ψ) = lapI ; D = (g·m + V) is real potential.
#   half-nonlin: a' = a·cos(D·dt/2) - b·sin(D·dt/2)
#                b' = b·cos(D·dt/2) + a·sin(D·dt/2)
#   kinetic:     a'' = a' + α·lapI·dt ;  b'' = b' - α·lapR·dt
#                (∂a/∂t = +α·∇²b, ∂b/∂t = -α·∇²a → curl-free rotation)
_register("gpe_seed",
          "GPE initial state: small random complex Gaussian bump in RG (node 148 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    vec2 d = v_uv - 0.5;
    float bump = exp(-dot(d, d) * 8.0);                 // central condensate
    float a = bump * (1.0 + (hash21(p + 0.19) - 0.5) * 0.06);
    float b = bump * (hash21(p + 7.31) - 0.5) * 0.06;
    f_color = vec4(a, b, 0.0, 1.0);                     // .b = accumulated sim-time (0)
}
''')