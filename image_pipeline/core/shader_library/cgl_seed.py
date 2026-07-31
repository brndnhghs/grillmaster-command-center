"""cgl_seed — GPU shader registration (dynamically loaded by core/shaders.py)."""

from ._registry import _register




# ── Complex Ginzburg-Landau (client-GPU sim of node 126) ────────────────────
# Complex field A packed as .r = Re(A), .g = Im(A). Explicit Euler with a
# 5-point Laplacian (toroidal). CGL: dA/dt = A + (1+i*alpha)*lap(A)
#   - (1+i*beta)*|A|^2*A. CPU node is Arch-A sim; this is the live-preview twin
# only — server export stays authoritative (seeded layout differs, as expected).
_register("cgl_seed",
          "CGL initial state: small random complex noise in RG (node 126 twin)",
          "procedural", '''
void main() {
    vec2 p = v_uv * u_resolution;
    float a = (hash21(p + 0.19) - 0.5) * 0.2;
    float b = (hash21(p + 7.31) - 0.5) * 0.2;
    f_color = vec4(a, b, 0.0, 1.0);
}
''')